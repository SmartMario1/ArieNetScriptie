"""Dataset classes for RLAF training with BPG-format graphs.

Two classes are provided:

    RLAFBPGDataset
        Loads a collection of DIMACS CNF files, preprocesses them into BPG
        (Bipartite Problem Graph) objects with local satisfaction percentages,
        and optionally includes literal co-occurrence (COOC) edges.  CNF
        clauses are loaded lazily from disk on first access so that large
        datasets don't exhaust RAM.  Processed BPG tensors are cached on disk.

    RLAFTrainingDataset
        Holds the per-iteration dataset built from sampled variable
        parameterisations and their solver statistics.  Mirrors
        RLAF.src.data.dataset.RLTrainingDataset but uses BPG data attributes.
"""

import gc
import glob
import multiprocessing
import os
from collections.abc import Mapping
from copy import copy
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset

from nsnet.utils.dataset import (
    BPG,
    BPGParamBuilder,
    _compute_c2l_inputs_fast,
    _compute_l2c_inputs_fast,
)
from nsnet.utils.utils import parse_cnf_file, unit_propagation


# ---------------------------------------------------------------------------
# Lazy clause loader — avoids loading all CNF files into RAM at startup
# ---------------------------------------------------------------------------

class _LazyCNFClauses(Mapping):
    """Dict-like object that parses a CNF file only when its clauses are first accessed."""

    def __init__(self, id_to_file: Dict[int, str]) -> None:
        self._id_to_file = id_to_file
        self._cache: Dict[int, List[List[int]]] = {}

    def __getitem__(self, cnf_id: int) -> List[List[int]]:
        if cnf_id not in self._cache:
            _, clauses = parse_cnf_file(self._id_to_file[cnf_id])
            self._cache[cnf_id] = clauses
        return self._cache[cnf_id]

    def __iter__(self):
        return iter(self._id_to_file)

    def __len__(self):
        return len(self._id_to_file)

    def clear_cache(self) -> None:
        """Release all cached clauses to free memory."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Worker: process a single CNF file into a cached BPG tensor file
# ---------------------------------------------------------------------------

def _malloc_trim() -> None:
    """Release free glibc heap pages back to the OS (Linux only, no-op elsewhere)."""
    try:
        import ctypes
        ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _process_rlaf_file(args: tuple) -> tuple:
    """Build and cache BPG tensors for one CNF file (no backbone labels).

    Returns (idx, error_str_or_None).
    """
    (
        idx,
        cnf_path,
        processed_dir,
        use_cooc,
        no_precomputed_local_sat,
        use_up_features,
    ) = args
    try:
        return _process_rlaf_file_inner(
            idx, cnf_path, processed_dir, use_cooc, no_precomputed_local_sat, use_up_features
        )
    except Exception as e:
        return (idx, str(e))
    finally:
        # Release Python heap fragmentation and glibc free-page cache so
        # worker RSS does not grow monotonically over thousands of files.
        gc.collect()
        _malloc_trim()


def _process_rlaf_file_inner(
    idx: int,
    cnf_path: str,
    processed_dir: str,
    use_cooc: bool,
    no_precomputed_local_sat: bool,
    use_up_features: bool,
) -> tuple:
    output_path = os.path.join(processed_dir, f"data_{idx}.pt")
    if os.path.exists(output_path):
        return (idx, None)

    n_vars, clauses = parse_cnf_file(cnf_path)
    if n_vars == 0 or len(clauses) == 0:
        torch.save({"skip": True}, output_path)
        return (idx, None)

    if use_cooc:
        # Apply unit propagation before building BPG (same as train_arienet_cooc.py)
        clauses, _ = unit_propagation(clauses)
        if len(clauses) == 0:
            torch.save({"skip": True}, output_path)
            return (idx, None)

    # Estimate total l2c entries: sum over clauses of 2^k * k * (k-1), capped at
    # k=20 to avoid overflow.  Each of the 3 l2c arrays has this many int32 entries,
    # so the threshold of 1_000_000 ≈ 4 MB per array (12 MB total).
    # k=3 clauses: 48 entries each → typically < 200 k entries — always cache.
    # k=7 clauses: 5376 entries each → 8k clauses ≈ 45 M entries — skip cache.
    _est_l2c = sum(min(2 ** len(c), 2 ** 20) * len(c) * max(len(c) - 1, 1)
                   for c in clauses)
    cache_c2l_l2c = _est_l2c < 1_000_000

    p = BPGParamBuilder(
        clauses,
        n_vars,
        compute_local_satisfaction_percentages=not no_precomputed_local_sat,
        compute_up_features=use_up_features,
        compute_c2l=cache_c2l_l2c,
        compute_l2c=cache_c2l_l2c,
    ).params

    # Always store compact clauses so get() can reconstruct c2l/l2c on-the-fly
    # when they were not cached (large files).
    clause_lens = torch.tensor([len(c) for c in clauses], dtype=torch.int16)
    clause_flat = torch.tensor([lit for c in clauses for lit in c], dtype=torch.int32)

    record = {
        "n_clauses":  p.n_clauses,
        "n_literals": p.n_literals,
        "lipe":  p.literal_indices_per_edge.to(torch.int32),
        "lipo":  p.literal_indices_per_occurence.to(torch.int32),
        "cipo":  p.clause_indices_per_occurence.to(torch.int32),
        "lsppe": (p.local_satisfaction_percentage_per_edge.to(torch.float32)
                  if p.local_satisfaction_percentage_per_edge is not None else None),
        "clause_lens": clause_lens,   # int16, one entry per clause
        "clause_flat": clause_flat,   # int32, all literals concatenated
        "up_feat": (p.up_features_per_literal.to(torch.float32)
                    if p.up_features_per_literal is not None else None),
    }

    if cache_c2l_l2c:
        record["c2l_r"] = p.c2l_msg_receiver_indices.to(torch.int32)
        record["c2l_s"] = p.c2l_msg_sender_indices.to(torch.int32)
        record["l2c_r"] = p.l2c_msg_receiver_indices.to(torch.int32)
        record["l2c_a"] = p.l2c_assignment_indices.to(torch.int32)
        record["l2c_n"] = p.l2c_assignment_neighborhoods.to(torch.int32)

    if use_cooc:
        # Build directed literal co-occurrence edges
        cooc_src_list, cooc_dst_list = [], []
        seen: set[tuple[int, int]] = set()
        for clause in clauses:
            lit_indices = [
                2 * (abs(l) - 1) if l > 0 else 2 * (abs(l) - 1) + 1
                for l in clause
            ]
            for i in range(len(lit_indices)):
                for j in range(len(lit_indices)):
                    if i != j:
                        s, d = lit_indices[i], lit_indices[j]
                        if (s, d) not in seen:
                            seen.add((s, d))
                            cooc_src_list.append(s)
                            cooc_dst_list.append(d)
        record["cooc_src"] = torch.tensor(cooc_src_list, dtype=torch.int32)
        record["cooc_dst"] = torch.tensor(cooc_dst_list, dtype=torch.int32)

    torch.save(record, output_path)
    return (idx, None)


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class RLAFBPGDataset(Dataset):
    """DIMACS CNF → BPG dataset for RLAF training.

    Caching strategy
    ----------------
    Each CNF file is preprocessed exactly once and stored as a compact dict on
    disk (``processed_rlaf*/data_<i>.pt``).  Index tensors are saved as
    ``int32``, floats as ``float32`` — no full PyG ``Data`` wrapper overhead.
    On every ``get()`` call the dict is loaded from disk and a ``BPG`` object
    is reconstructed on the fly; there is intentionally no in-memory object
    cache so that large datasets don't exhaust RAM.

    Args:
        path:
            A glob pattern, a directory path, or a **list** of glob patterns /
            directories pointing to ``.cnf`` files.  Examples::

                "dataRLAF/training/3sat/**/*.cnf"          # single glob
                ["dataRLAF/training/3sat",                 # list of dirs
                 "dataRLAF/training/crypto"]

            All matching files are merged into one dataset.
        root:
            Directory for the on-disk BPG cache.  Defaults to a folder next to
            the first data directory, named
            ``processed_rlaf[_cooc][_nolsp][_up]/``.
        use_cooc:
            If True, build and cache literal co-occurrence edges and expect the
            model to be an ``ArieNetRLAFCooc``.
        no_precomputed_local_sat:
            If True, skip computing local satisfaction percentages (faster
            preprocessing, weaker features).
        use_up_features:
            If True, compute and store per-literal Unit Propagation features.
        num_workers:
            Number of parallel processes for the initial cache-build step.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        root: Optional[str] = None,
        transform=None,
        pre_transform=None,
        use_cooc: bool = False,
        no_precomputed_local_sat: bool = False,
        use_up_features: bool = False,
        num_workers: int = 4,
    ):
        self.use_cooc = use_cooc
        self.no_precomputed_local_sat = no_precomputed_local_sat
        self.use_up_features = use_up_features
        self.num_workers = num_workers

        # Collect CNF file paths from one or multiple patterns / directories
        paths = [path] if isinstance(path, str) else list(path)
        all_files: List[str] = []
        first_dir: Optional[str] = None
        for p in paths:
            if os.path.isdir(p):
                found = sorted(glob.glob(os.path.join(p, "**", "*.cnf"), recursive=True))
                if first_dir is None:
                    first_dir = p
            else:
                found = sorted(glob.glob(p, recursive=True))
                if first_dir is None and found:
                    first_dir = os.path.dirname(found[0])
            all_files.extend(found)
        self.cnf_files = sorted(set(all_files))  # deduplicate, keep deterministic order
        data_dir = first_dir or "."

        if not self.cnf_files:
            raise ValueError(f"No CNF files found for path(s): {paths!r}")
        print(f"Found {len(self.cnf_files)} CNF files.")

        # id → file mapping (for logging)
        self.id_to_file = {i: f for i, f in enumerate(self.cnf_files)}

        # Lazy clause loader — clauses are read from disk only when needed
        self.cnf_clauses: _LazyCNFClauses = _LazyCNFClauses(self.id_to_file)

        # Processed directory name encodes feature configuration
        if root is None:
            suffix = "_cooc" if use_cooc else ""
            suffix += "_nolsp" if no_precomputed_local_sat else ""
            suffix += "_up" if use_up_features else ""
            root = os.path.join(data_dir, f"processed_rlaf{suffix}")

        super().__init__(root, transform, pre_transform)

    # ------------------------------------------------------------------ #

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(len(self.cnf_files))]

    def download(self):
        pass

    def process(self):
        tasks = [
            (
                idx,
                cnf_path,
                self.processed_dir,
                self.use_cooc,
                self.no_precomputed_local_sat,
                self.use_up_features,
            )
            for idx, cnf_path in enumerate(self.cnf_files)
            if not os.path.exists(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        ]

        if not tasks:
            print("All BPG files already processed, skipping.")
            return

        n = self.num_workers
        print(
            f"Processing {len(tasks)}/{len(self.cnf_files)} files "
            f"({'cooc, ' if self.use_cooc else ''}{'no-lsp, ' if self.no_precomputed_local_sat else ''}"
            f"{n} worker{'s' if n != 1 else ''})..."
        )
        completed = 0
        if n == 1:
            for t in tasks:
                idx, err = _process_rlaf_file(t)
                if err is not None:
                    print(f"[ERROR] idx={idx}: {err}")
                else:
                    completed += 1
                    if completed % 50 == 0 or completed == len(tasks):
                        print(f"  [{completed}/{len(tasks)}]", end="\r")
        else:
            # maxtasksperchild: restart each worker after this many tasks so that
            # accumulated Python heap fragmentation is reclaimed by the OS.
            # Workers that process large crypto instances hold ~200–400 MB of
            # freed-but-not-returned memory after a few hundred files; a fresh
            # process starts clean.
            with multiprocessing.Pool(processes=n, maxtasksperchild=100) as pool:
                for idx, err in pool.imap_unordered(_process_rlaf_file, tasks):
                    if err is not None:
                        print(f"[ERROR] idx={idx}: {err}")
                    else:
                        completed += 1
                        if completed % 50 == 0 or completed == len(tasks):
                            print(f"  [{completed}/{len(tasks)}]", end="\r")
        print(f"\nDone: {completed} files processed.")

    # ------------------------------------------------------------------ #

    def len(self) -> int:
        return len(self.cnf_files)

    def get(self, idx: int) -> Optional[BPG]:
        cache_path = os.path.join(self.processed_dir, f"data_{idx}.pt")
        try:
            d = torch.load(cache_path, weights_only=True)
        except Exception:
            d = torch.load(cache_path, weights_only=False)

        if d.get("skip", False):
            return None

        # Reconstruct clauses for c2l/l2c — either from compact storage (new
        # format) or by falling back to on-the-fly reparse (very old format).
        if "clause_flat" in d:
            flat  = d["clause_flat"].numpy()
            lens  = d["clause_lens"].numpy().astype(int)
            ends  = lens.cumsum()
            starts = ends - lens
            clauses = [flat[s:e].tolist() for s, e in zip(starts, ends)]
        else:
            # Legacy: file was processed before the compact-clauses format.
            # Re-parse from disk so we can compute c2l/l2c.
            from nsnet.utils.utils import parse_cnf_file
            _, clauses = parse_cnf_file(self.id_to_file[idx])

        n_vars = d["n_literals"] // 2

        # c2l / l2c: use pre-stored tensors if present (old format), otherwise
        # compute on-the-fly from the raw clauses (new format, saves GB of disk).
        if "c2l_r" in d:
            c2l_r = d["c2l_r"].long()
            c2l_s = d["c2l_s"].long()
            l2c_r = d["l2c_r"].long()
            l2c_a = d["l2c_a"].long()
            l2c_n = d["l2c_n"].long()
        else:
            c2l_r, c2l_s = _compute_c2l_inputs_fast(clauses, n_vars)
            l2c_r, l2c_a, l2c_n = _compute_l2c_inputs_fast(clauses)

        data = BPG(
            n_clauses=d["n_clauses"],
            n_literals=d["n_literals"],
            literal_indices_per_edge=d["lipe"].long(),
            literal_indices_per_occurence=d["lipo"].long(),
            clause_indices_per_occurence=d["cipo"].long(),
            local_satisfaction_percentage_per_edge=d.get("lsppe"),
            c2l_msg_receiver_indices=c2l_r,
            c2l_msg_sender_indices=c2l_s,
            l2c_msg_receiver_indices=l2c_r,
            l2c_assignment_indices=l2c_a,
            l2c_assignment_neighborhoods=l2c_n,
            up_features_per_literal=d.get("up_feat"),
            cooc_src_indices=d["cooc_src"].long() if "cooc_src" in d else None,
            cooc_dst_indices=d["cooc_dst"].long() if "cooc_dst" in d else None,
        )
        data.cnf_id = torch.tensor(idx, dtype=torch.long)
        return data


# ---------------------------------------------------------------------------
# Per-iteration training dataset
# ---------------------------------------------------------------------------

def _collate_skip_none(batch):
    """Drop None items before PyG batching (skipped / trivial instances)."""
    from torch_geometric.data import Batch
    batch = [b for b in batch if b is not None]
    return Batch.from_data_list(batch) if batch else None


class RLAFTrainingDataset(Dataset):
    """Per-iteration dataset for GRPO / DPO training.

    Stores BPG graphs annotated with sampled variable parameterisations and
    their associated solver statistics.  Mirrors
    ``RLAF.src.data.dataset.RLTrainingDataset`` but uses BPG data attributes
    instead of the HeteroData-dict style.

    Args:
        data_list:    List of BPG graphs with .var_params, .y_var_ref, .log_prob.
        solver_stats: DataFrame from compute_solver_stats.
        target_stat:  Column of solver_stats to use as optimisation target.
        objective:    "minimize" (default) or "maximize".
    """

    def __init__(
        self,
        data_list: list,
        solver_stats: pd.DataFrame,
        target_stat: str = "decisions",
        objective: Literal["minimize", "maximize"] = "minimize",
    ):
        self.data = []

        stats_sorted = solver_stats.sort_values(["cnf_id", "sample_id"])

        for data in data_list:
            data = copy(data)
            cnf_id = int(data.cnf_id.item())

            cnf_stats = stats_sorted[stats_sorted["cnf_id"] == cnf_id]
            stats = torch.tensor(cnf_stats[target_stat].to_numpy(), dtype=torch.float32)

            # Sort samples: ascending cost for DPO (lowest cost = most preferred)
            if objective == "minimize":
                idx = torch.argsort(-stats)   # worst → best
            else:
                idx = torch.argsort(stats)    # best → worst (maximise)

            # data.var_params: [n_vars, num_samples, 2]
            data.var_params = data.var_params[:, idx]
            # data.log_prob:   [num_samples]  → [1, num_samples] for batching
            data.log_prob = data.log_prob[idx].unsqueeze(0)
            # data.stats:      [num_samples]  → [1, num_samples] for batching
            data.stats = stats[idx].unsqueeze(0)

            self.data.append(data)

        super().__init__()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int):
        return self.data[idx]

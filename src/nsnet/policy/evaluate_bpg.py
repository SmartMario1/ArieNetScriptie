"""Sampling and solver-evaluation helpers for the RLAF + BPG training loop.

Adapted from RLAF/src/policy/evaluate.py for nsnet's BPG data format.

Key difference: var_batch is derived from data.n_literals (the number of
literals per graph in the batch) rather than data["lit"].batch.
"""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import unbatch

from nsnet.policy import policy_bpg as policy


# ---------------------------------------------------------------------------
# var_batch helper
# ---------------------------------------------------------------------------

def get_var_batch(data: Data, device=None) -> torch.Tensor:
    """Return a 1-D tensor mapping each variable to its graph index in the batch.

    For a batch of graphs with n_literals = [nl_0, nl_1, ...] the result is
    [0,0,...,0, 1,1,...,1, ...] with (nl_i // 2) repeats for graph i.
    """
    nl = data.n_literals
    if isinstance(nl, torch.Tensor) and nl.dim() > 0:
        n_vars = nl // 2  # [B]
        batch_idx = torch.arange(len(n_vars), device=nl.device if device is None else device)
        var_batch = torch.repeat_interleave(batch_idx, n_vars)
    else:
        # Single graph (not batched)
        n_vars = int(nl) // 2
        var_batch = torch.zeros(n_vars, dtype=torch.long)
    if device is not None:
        var_batch = var_batch.to(device)
    return var_batch


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_var_params(
    model: torch.nn.Module,
    loader: DataLoader,
    num_samples: int,
    max_num_batches: int = -1,
    device: Union[torch.device, str] = "cpu",
    use_mode: bool = False,
    scale_sigma: float = 0.1,
) -> List[Data]:
    """Run the model and sample variable parameterisations.

    Returns a flat list of individual BPG graphs (one per CNF formula), each
    annotated with:
        data.var_params  : [n_vars, num_samples, 2]
        data.y_var_ref   : [n_vars, 2]  model output (reference policy)
        data.log_prob    : [num_samples]  total log-prob per sample (for this graph)
    """
    model.to(device)
    model.eval()

    data_list_all = []
    for i, data in enumerate(loader):
        data = data.to(device)
        var_batch = get_var_batch(data, device=device)

        y_var = model(data)  # [total_vars_in_batch, 2]

        if use_mode:
            var_params = policy.mode(y_var, scale_sigma=scale_sigma)
        else:
            var_params = policy.sample(y_var, num_samples=num_samples, scale_sigma=scale_sigma)
        # var_params: [num_samples, total_vars, 2]

        log_prob = policy.log_prob(y_var, var_params, var_batch, scale_sigma=scale_sigma)
        # log_prob: [num_samples, n_graphs_in_batch]

        data = data.to("cpu")
        y_var = y_var.to("cpu")
        var_batch = var_batch.to("cpu")
        var_params = var_params.to("cpu")          # [num_samples, total_vars, 2]
        log_prob = log_prob.to("cpu").transpose(0, 1)  # [n_graphs, num_samples]

        # Unbatch: split y_var and var_params by graph
        y_var_per_graph = unbatch(y_var, var_batch)
        # For var_params: transpose to [total_vars, num_samples, 2] then unbatch
        var_params_t = var_params.transpose(0, 1)  # [total_vars, num_samples, 2]
        var_params_per_graph = unbatch(var_params_t, var_batch)

        graph_list = data.to_data_list()
        for j, g in enumerate(graph_list):
            g.y_var_ref = y_var_per_graph[j]          # [n_vars, 2]
            g.var_params = var_params_per_graph[j]    # [n_vars, num_samples, 2]
            g.log_prob = log_prob[j]                  # [num_samples]
            data_list_all.append(g)

        if max_num_batches > -1 and i + 1 >= max_num_batches:
            break

    return data_list_all


# ---------------------------------------------------------------------------
# Solver evaluation
# ---------------------------------------------------------------------------

def _solver_pool_fn(args: tuple) -> dict:
    cnf_idx, sample_idx, clauses, var_params, solver_params = args
    stats = _solve_cnf(clauses, var_params, **solver_params)
    stats["cnf_id"] = cnf_idx
    stats["sample_id"] = sample_idx
    return stats


_STATS = ["decisions", "conflicts", "propagations", "restarts", "CPU time"]


def _stdout_to_stats(stdout: str) -> dict:
    result = {}
    for line in stdout.splitlines():
        line = line.strip()
        for stat in _STATS:
            if line.startswith(f"c {stat}") or line.startswith(stat):
                _, val = line.split(":", 1)
                result[stat] = float(val.strip().split()[0])
        if line.startswith("s"):
            result["Result"] = line.split()[1]
    return result


def _make_dimacs(clauses: List[List[int]], var_params: Optional[np.ndarray] = None) -> str:
    variables = {abs(lit) for clause in clauses for lit in clause}
    n_vars = max(variables)
    lines = [f"p cnf {n_vars} {len(clauses)}"]
    if var_params is not None:
        assert n_vars == var_params.shape[0]
        params = ["c weight"]
        for i in range(n_vars):
            weight = float(var_params[i, 1])
            sgn = 1 if var_params[i, 0] > 0 else -1
            params.append(f"{sgn * weight:.4f}")
        lines.append(" ".join(params))
    for clause in clauses:
        lines.append(" ".join(map(str, clause)) + " 0")
    return "\n".join(lines)


def _solve_cnf(
    clauses: List[List[int]],
    var_params: Optional[np.ndarray] = None,
    solver_dir: str = ".",
    seed: int = 1,
    solver: str = "march",
    **params: Any,
) -> dict:
    import os, subprocess, tempfile

    # All binaries live in the nsnet directory itself (solver_dir=".").
    solver_bins = {
        "march":               os.path.join(solver_dir, "march_unmodified/march/march_nh"),
        "march_weighted":      os.path.join(solver_dir, "march_weighted/march_nh"),
        "glucose_unmodified":  os.path.join(solver_dir, "glucose_unmodified/simp/glucose"),
        "glucose_weighted":    os.path.join(solver_dir, "glucose_weighted/simp/glucose"),
        # Legacy key kept for back-compatibility:
        "glucose":             os.path.join(solver_dir, "glucose_unmodified/simp/glucose"),
    }

    dimacs = _make_dimacs(clauses, var_params)

    if solver.startswith("march"):
        bin_key = "march_weighted" if var_params is not None else "march"
        call = [solver_bins[bin_key]]
        tmp_dir = os.path.join(solver_dir, "data/tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, f"tmp_{abs(hash(dimacs))}.cnf")
        with open(tmp_path, "w") as f:
            f.write(dimacs)
        call.append(tmp_path)
        proc_timeout = params.pop("timeout", None)
        try:
            result = subprocess.run(call, capture_output=True, text=True,
                                    timeout=proc_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return {"Result": "UNKNOWN", "decisions": float("nan"),
                    "conflicts": float("nan"), "CPU time": float("nan")}
        os.remove(tmp_path)
    else:
        # glucose_weighted when guidance is provided, unmodified glucose otherwise
        bin_key = "glucose_weighted" if var_params is not None else "glucose_unmodified"
        call = [solver_bins[bin_key]]
        if seed is not None and seed > 0:
            call.append(f"-rnd-seed={seed}")
        call += [f"-{k}={v}" for k, v in params.items()]
        with tempfile.TemporaryFile(mode="w+") as f:
            f.write(dimacs)
            f.seek(0)
            result = subprocess.run(call, capture_output=True, text=True, stdin=f)

    return _stdout_to_stats(result.stdout)


def compute_solver_stats(
    data_list: List[Data],
    cnf_clauses: Dict[int, List[List[int]]],
    solver_dir: str = ".",
    num_workers: int = 8,
    solver: str = "march",
    **solver_params: Any,
) -> pd.DataFrame:
    """Run the SAT solver for all (formula, sample) pairs and return a DataFrame.

    Args:
        data_list:    List of BPG graphs with data.var_params and data.cnf_id.
        cnf_clauses:  Mapping from cnf_id to list-of-clauses.
        solver_dir:   Root directory containing the solver binaries (default: ".").
        num_workers:  CPU cores for parallel solving.
        solver:       "march" (default), "glucose", or "glucose_unmodified". All
                      binaries are expected inside solver_dir.
        **solver_params: Extra solver CLI arguments (used by glucose only).
    """

    def iter_inputs():
        for data in data_list:
            cnf_id = int(data.cnf_id.item())
            clauses = cnf_clauses[cnf_id]
            vp_all = data.var_params.numpy()  # [n_vars, num_samples, 2]
            for j in range(vp_all.shape[1]):
                yield (cnf_id, j, clauses, vp_all[:, j, :],
                       {**solver_params, "solver_dir": solver_dir, "solver": solver})

    total = sum(data.var_params.shape[1] for data in data_list)
    t0 = time.time()

    stats_dicts = Parallel(n_jobs=num_workers)(
        delayed(_solver_pool_fn)(inp) for inp in iter_inputs()
    )

    print(f"Solved {total} formulas in {time.time() - t0:.2f}s")
    return pd.DataFrame.from_records(stats_dicts)

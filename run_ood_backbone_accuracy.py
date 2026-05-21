"""
OOD Backbone Accuracy Evaluation Script

For each size bucket in an OOD benchmark directory:
  1. Computes the backbone of every SAT instance (using PySAT/Cadical).
  2. Runs backbone-objective model inference on the same instances.
  3. Reports per-bucket accuracy of the model's backbone predictions.

Only backbone-objective models are supported (Backbone, BackboneCanonical,
BackboneUP, ArieNetCooc, and extras registered via --extra_backbone_checkpoint /
--extra_cooc_checkpoint).  RLAF / NSNet / ArieNet models are intentionally excluded.

Labels:
  0 = variable must be False  (negative backbone)
  1 = variable must be True   (positive backbone)
  2 = free  (not in backbone — excluded from accuracy computation)

Metrics reported per bucket per model:
  instances_computed  – number of SAT instances with a successfully computed backbone
  backbone_rate       – fraction of variables that are backbone (0 or 1)
  accuracy            – % correct predictions among backbone variables
  neg_accuracy        – % correct among negatively-forced variables (label 0)
  pos_accuracy        – % correct among positively-forced variables (label 1)
  backbone_vars       – total backbone variables across all computed instances
  free_vars           – total free variables across all computed instances

Usage:
    python run_ood_backbone_accuracy.py SATSolving/3-sat/ood_test \\
        --backbone_checkpoint models/arienet_backbone/best_model.pt \\
        --extra_backbone_checkpoint BackboneNoLSP=models/arienet_backbone_nolsp/best_model.pt \\
        --canonical_checkpoint models/arienet_backbone_canonical/best_model.pt \\
        --cooc_checkpoint models/arienet_cooc/best_model.pt \\
        --extra_cooc_checkpoint CoocAlt=models/arienet_cooc_alt/best_model.pt
"""

import os
import sys
import glob
import json
import argparse
import pickle
import subprocess
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------------------------
# Default march binary for backbone SAT checks (unmodified, no GNN guidance).
# ---------------------------------------------------------------------------
MARCH_BACKBONE_PATH = "./march_unmodified/march/march_nh"

# ---------------------------------------------------------------------------
# Module-level registries (populated in main()).
# ---------------------------------------------------------------------------
_EXTRA_BACKBONE_NAMES: set = set()
_EXTRA_COOC_NAMES: set = set()


def is_backbone_model(model_name):
    return model_name == "Backbone" or model_name in _EXTRA_BACKBONE_NAMES


def is_cooc_model(model_name):
    return model_name == "ArieNetCooc" or model_name in _EXTRA_COOC_NAMES


def parse_extra_backbone_args(extra_args):
    """
    Parse repeatable --extra_backbone_checkpoint values.
    Accepted formats:
      NAME=PATH  or  PATH  (auto-named Backbone_2, Backbone_3, …)
    Returns dict model_name -> checkpoint_path.
    """
    extra_map = {}
    next_idx = 2
    for raw in extra_args or []:
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
            name, path = name.strip(), path.strip()
            if not name:
                raise ValueError(f"Invalid --extra_backbone_checkpoint '{raw}': empty name")
            if name in extra_map:
                raise ValueError(f"Duplicate extra backbone model name: {name}")
            extra_map[name] = path
        else:
            while True:
                auto_name = f"Backbone_{next_idx}"
                next_idx += 1
                if auto_name not in extra_map:
                    break
            extra_map[auto_name] = item
    return extra_map


def parse_extra_cooc_args(extra_args):
    """
    Parse repeatable --extra_cooc_checkpoint values.
    Accepted formats:
      NAME=PATH  or  PATH  (auto-named ArieNetCooc_2, ArieNetCooc_3, …)
    Returns dict model_name -> checkpoint_path.
    """
    extra_map = {}
    next_idx = 2
    for raw in extra_args or []:
        item = raw.strip()
        if not item:
            continue
        if "=" in item:
            name, path = item.split("=", 1)
            name, path = name.strip(), path.strip()
            if not name:
                raise ValueError(f"Invalid --extra_cooc_checkpoint '{raw}': empty name")
            if name in extra_map:
                raise ValueError(f"Duplicate extra CoOC model name: {name}")
            extra_map[name] = path
        else:
            while True:
                auto_name = f"ArieNetCooc_{next_idx}"
                next_idx += 1
                if auto_name not in extra_map:
                    break
            extra_map[auto_name] = item
    return extra_map


# ---------------------------------------------------------------------------
# Backbone computation helpers
# ---------------------------------------------------------------------------

def _check_sat_march(dimacs_str, march_path, per_call_timeout):
    """Write dimacs_str to a temp file, run march, return 'SAT'/'UNSAT'/'TIMEOUT'."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as f:
        f.write(dimacs_str)
        tmp = f.name
    try:
        result = subprocess.run(
            [march_path, tmp],
            capture_output=True, text=True, timeout=per_call_timeout,
        )
        for line in result.stdout.splitlines():
            upper = line.strip().upper()
            if "UNSATISFIABLE" in upper:
                return "UNSAT"
            if "SATISFIABLE" in upper:
                return "SAT"
        return "UNKNOWN"
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Backbone computation (PySAT)
# ---------------------------------------------------------------------------

def _compute_backbone_worker(args):
    """Worker function for ProcessPoolExecutor — computes backbone for one CNF.

    args = (cnf_path, timeout, solver, march_path)
      solver    : "march" or "pysat"
      march_path: path to march binary (only used when solver=="march")
    """
    cnf_path, timeout, solver, march_path = args

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
        from nsnet.utils.utils import parse_cnf_file
        n_vars, clauses = parse_cnf_file(cnf_path)
    except Exception as e:
        return cnf_path, None, f"parse error: {e}"

    if n_vars == 0 or len(clauses) == 0:
        return cnf_path, None, "empty CNF"

    if solver == "march":
        return _compute_backbone_march(cnf_path, n_vars, clauses, timeout, march_path)
    else:
        return _compute_backbone_pysat(cnf_path, n_vars, clauses)


def _compute_backbone_march(cnf_path, n_vars, clauses, timeout, march_path):
    """Compute backbone using march as the SAT oracle."""
    # Per-call timeout: divide total budget evenly over 2*n_vars+1 calls,
    # clamped to [2s, 60s].
    per_call = max(2.0, min(60.0, timeout / (2 * n_vars + 1)))

    # Pre-build the clause body string once (O(clauses)).
    clause_lines = "\n".join(" ".join(map(str, c)) + " 0" for c in clauses)
    n_c = len(clauses)

    def make_dimacs(extra_lit=None):
        n = n_c + (1 if extra_lit is not None else 0)
        base = f"p cnf {n_vars} {n}\n{clause_lines}"
        return base if extra_lit is None else f"{base}\n{extra_lit} 0"

    # Initial SAT check.
    res = _check_sat_march(make_dimacs(), march_path, min(timeout, 60.0))
    if res == "UNSAT":
        return cnf_path, None, "UNSAT"
    if res != "SAT":
        return cnf_path, None, f"initial SAT check: {res}"

    backbone = {}
    for var in range(1, n_vars + 1):
        # Test if var must be True: is (formula ∧ ¬var) UNSAT?
        res = _check_sat_march(make_dimacs(-var), march_path, per_call)
        if res == "UNSAT":
            backbone[var] = True
            continue
        # Test if var must be False: is (formula ∧ var) UNSAT?
        res = _check_sat_march(make_dimacs(var), march_path, per_call)
        if res == "UNSAT":
            backbone[var] = False
            continue
        backbone[var] = None  # free (or timed-out — treated as free)

    return cnf_path, backbone, None


def _compute_backbone_pysat(cnf_path, n_vars, clauses):
    """Compute backbone using PySAT/Cadical."""
    try:
        from pysat.solvers import Cadical
    except ImportError:
        return cnf_path, None, "pysat not installed"

    try:
        with Cadical(bootstrap_with=clauses) as solver:
            if not solver.solve():
                return cnf_path, None, "UNSAT"

        backbone = {}
        for var in range(1, n_vars + 1):
            with Cadical(bootstrap_with=clauses) as s:
                s.add_clause([-var])
                if not s.solve():
                    backbone[var] = True
                    continue
            with Cadical(bootstrap_with=clauses) as s:
                s.add_clause([var])
                if not s.solve():
                    backbone[var] = False
                    continue
            backbone[var] = None
        return cnf_path, backbone, None
    except Exception as e:
        return cnf_path, None, str(e)


def compute_backbones_for_bucket(cnf_files, timeout, num_workers, solver, march_path,
                                  cache_file=None):
    """
    Compute backbones for all CNF files in a bucket using a process pool.

    If cache_file is given:
      - Already-cached entries are loaded and reused (only missing files are computed).
      - Newly computed entries are merged into the cache and written back.

    Returns:
        backbones   : dict  abs_path -> {var: True/False/None}
        skip_reasons: dict  abs_path -> reason string  (for failed instances)
    """
    # Load existing cache if present.
    cached = {}
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            print(f"    Loaded {len(cached)} cached backbone(s) from {cache_file}")
        except Exception as e:
            print(f"    [WARN] Could not load backbone cache {cache_file}: {e}")
            cached = {}

    abs_files = [os.path.abspath(f) for f in cnf_files]
    missing = [p for p in abs_files if p not in cached]

    backbones = {p: cached[p] for p in abs_files if p in cached}
    skip_reasons = {}

    if missing:
        print(f"    Computing backbone for {len(missing)} new instance(s) "
              f"(solver={solver}, timeout={timeout}s, workers={num_workers}) …")
        task_args = [(p, timeout, solver, march_path) for p in missing]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for cnf_path, backbone, err in executor.map(_compute_backbone_worker, task_args):
                if backbone is not None:
                    backbones[cnf_path] = backbone
                    cached[cnf_path] = backbone
                else:
                    skip_reasons[cnf_path] = err or "unknown error"

        # Persist updated cache.
        if cache_file:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(cached, f)
                print(f"    Saved backbone cache ({len(cached)} entries) to {cache_file}")
            except Exception as e:
                print(f"    [WARN] Could not save backbone cache: {e}")
    else:
        print(f"    All {len(abs_files)} backbone(s) found in cache — skipping computation.")

    return backbones, skip_reasons


def backbone_dict_to_labels(backbone_dict, n_vars):
    """Convert {var: True/False/None} to np.int8 array of length n_vars (labels 0/1/2)."""
    labels = np.full(n_vars, 2, dtype=np.int8)  # default: free
    for var, val in backbone_dict.items():
        if val is not None and 1 <= var <= n_vars:
            labels[var - 1] = 1 if val else 0
    return labels


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, checkpoint_path, device):
    """Load a backbone-objective model checkpoint."""
    from train_arienet_backbone import ArieNetBackbone

    if model_name == "BackboneCanonical":
        model = ArieNetBackbone(device=device, use_subgraph_features=True, subgraph_dim=32)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    elif model_name == "BackboneUP":
        model = ArieNetBackbone(device=device, use_up_features=True)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    elif is_cooc_model(model_name):
        import argparse as _ap
        from src.nsnet.models.arienet import ArieNetCooc
        ckpt = torch.load(checkpoint_path, map_location=device)
        ckpt_args = ckpt["args"]
        cooc_opts = _ap.Namespace(
            dim=ckpt_args.dim,
            n_rounds=ckpt_args.n_rounds,
            n_mlp_layers=ckpt_args.n_mlp_layers,
            activation="relu",
            device=device,
            task="satisfiability",
        )
        model = ArieNetCooc(cooc_opts)
        model.load_state_dict(ckpt["model_state_dict"])

    else:
        # Backbone / extra backbone variants
        ckpt = torch.load(checkpoint_path, map_location=device)
        ckpt_args = ckpt.get("args", {})
        no_lsp = ckpt_args.get("no_precomputed_local_sat", False)
        model = ArieNetBackbone(device=device, no_precomputed_local_sat=no_lsp)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.to(device)
    model.eval()
    return model


def build_dataloader(model_name, cnf_dir, batch_size, num_workers,
                     backbone_process_workers, cooc_process_workers=1):
    """Build a dataloader for the given backbone-objective model variant."""
    if model_name == "BackboneCanonical":
        from train_arienet_backbone_canonical import CanonicalBackboneBPGDataset
        dataset = CanonicalBackboneBPGDataset(cnf_dir, compute_subgraphs=False)
    elif model_name == "BackboneUP":
        from train_arienet_backbone import BackboneBPGDataset
        dataset = BackboneBPGDataset(cnf_dir, use_up_features=True,
                                     num_process_workers=backbone_process_workers)
    elif is_cooc_model(model_name):
        from train_arienet_cooc import CoocBPGDataset
        dataset = CoocBPGDataset(cnf_dir, process_workers=cooc_process_workers)
    else:
        from train_arienet_backbone import BackboneBPGDataset
        dataset = BackboneBPGDataset(cnf_dir, num_process_workers=backbone_process_workers)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=False)
    return loader, [os.path.abspath(f) for f in dataset.cnf_files]


def run_inference(model, model_name, loader, device):
    """
    Run model inference.  Returns a list of np.int8 arrays of length n_vars,
    each containing the argmax class (0 or 1) for every variable.
    """
    predictions = []

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            logits = model(data)  # [total_vars_in_batch, 2]

        raw = data.n_literals
        if isinstance(raw, torch.Tensor):
            v_sizes = (raw / 2).int().tolist()
        elif isinstance(raw, list):
            v_sizes = [int(n / 2) for n in raw]
        else:
            v_sizes = [int(raw / 2)]

        pred_classes = torch.argmax(logits, dim=1).cpu().numpy().astype(np.int8)

        offset = 0
        for v_size in v_sizes:
            predictions.append(pred_classes[offset: offset + v_size])
            offset += v_size

    return predictions


# ---------------------------------------------------------------------------
# Per-bucket evaluation
# ---------------------------------------------------------------------------

def evaluate_bucket(model, model_name, cnf_dir, batch_size, num_workers,
                    device, backbone_timeout, backbone_workers, backbone_process_workers,
                    cooc_process_workers=1, backbone_solver="march", backbone_march_path=None,
                    backbone_cache_file=None):
    """
    1. Build dataloader.
    2. Compute backbone for all instances in cnf_dir.
    3. Run model inference.
    4. Compare predictions to backbone labels.

    Returns a metrics dict or None on failure.
    """
    print(f"    Building dataloader for {model_name} …")
    try:
        loader, cnf_files = build_dataloader(
            model_name, cnf_dir, batch_size, num_workers,
            backbone_process_workers, cooc_process_workers
        )
    except Exception as e:
        print(f"    [ERROR] Could not build dataloader: {e}")
        return None

    n_total = len(cnf_files)
    if n_total == 0:
        print(f"    [WARN] No CNF files found in {cnf_dir}")
        return None

    # ------------------------------------------------------------------
    # Step 1: compute backbones
    # ------------------------------------------------------------------
    backbones, skip_reasons = compute_backbones_for_bucket(
        cnf_files, backbone_timeout, backbone_workers, backbone_solver, backbone_march_path,
        cache_file=backbone_cache_file,
    )
    n_computed = len(backbones)
    n_skipped = n_total - n_computed
    print(f"    Backbone computed: {n_computed}/{n_total}  "
          f"(skipped {n_skipped}: "
          + ", ".join(f"{r}={sum(1 for v in skip_reasons.values() if v == r)}"
                      for r in sorted(set(skip_reasons.values())))
          + ")")

    # ------------------------------------------------------------------
    # Step 2: model inference
    # ------------------------------------------------------------------
    print(f"    Running model inference …")
    try:
        predictions = run_inference(model, model_name, loader, device)
    except Exception as e:
        print(f"    [ERROR] Inference failed: {e}")
        return None

    if len(predictions) != n_total:
        print(f"    [ERROR] Prediction count mismatch: {len(predictions)} vs {n_total}")
        return None

    # ------------------------------------------------------------------
    # Step 3: compare predictions to true backbone labels
    # ------------------------------------------------------------------
    total_backbone_vars = 0
    total_neg_backbone = 0
    total_pos_backbone = 0
    total_free_vars = 0
    correct_backbone = 0
    correct_neg = 0
    correct_pos = 0

    for cnf_path, pred in zip(cnf_files, predictions):
        abs_path = os.path.abspath(cnf_path)
        if abs_path not in backbones:
            continue  # could not compute backbone — skip

        n_vars = len(pred)
        true_labels = backbone_dict_to_labels(backbones[abs_path], n_vars)

        backbone_mask = true_labels != 2
        neg_mask = true_labels == 0
        pos_mask = true_labels == 1

        n_bb = int(backbone_mask.sum())
        n_neg = int(neg_mask.sum())
        n_pos = int(pos_mask.sum())
        n_free = int((~backbone_mask).sum())

        total_backbone_vars += n_bb
        total_neg_backbone += n_neg
        total_pos_backbone += n_pos
        total_free_vars += n_free

        if n_bb > 0:
            correct_backbone += int((pred[backbone_mask] == true_labels[backbone_mask]).sum())
        if n_neg > 0:
            correct_neg += int((pred[neg_mask] == true_labels[neg_mask]).sum())
        if n_pos > 0:
            correct_pos += int((pred[pos_mask] == true_labels[pos_mask]).sum())

    total_vars = total_backbone_vars + total_free_vars
    backbone_rate = total_backbone_vars / total_vars if total_vars > 0 else None
    accuracy = correct_backbone / total_backbone_vars if total_backbone_vars > 0 else None
    neg_accuracy = correct_neg / total_neg_backbone if total_neg_backbone > 0 else None
    pos_accuracy = correct_pos / total_pos_backbone if total_pos_backbone > 0 else None

    return {
        "total_instances": n_total,
        "instances_computed": n_computed,
        "instances_skipped": n_skipped,
        "total_vars": total_vars,
        "backbone_vars": total_backbone_vars,
        "neg_backbone_vars": total_neg_backbone,
        "pos_backbone_vars": total_pos_backbone,
        "free_vars": total_free_vars,
        "backbone_rate": round(backbone_rate, 4) if backbone_rate is not None else None,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "neg_accuracy": round(neg_accuracy, 4) if neg_accuracy is not None else None,
        "pos_accuracy": round(pos_accuracy, 4) if pos_accuracy is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OOD benchmark: evaluate backbone model accuracy on backbone predictions."
    )
    parser.add_argument(
        "ood_dir",
        type=str,
        help="Root OOD benchmark directory (e.g. SATSolving/3-sat/ood_test)",
    )

    # Model checkpoints
    parser.add_argument("--backbone_checkpoint", type=str, default=None,
                        help="Path to plain ArieNetBackbone checkpoint")
    parser.add_argument(
        "--extra_backbone_checkpoint",
        action="append",
        default=[],
        help=(
            "Additional Backbone checkpoint. Repeatable. "
            "Formats: NAME=PATH or PATH (auto-named Backbone_2, Backbone_3, …)."
        ),
    )
    parser.add_argument("--canonical_checkpoint", type=str, default=None,
                        help="Path to BackboneCanonical checkpoint")
    parser.add_argument("--backbone_up_checkpoint", type=str, default=None,
                        help="Path to BackboneUP checkpoint")
    parser.add_argument("--cooc_checkpoint", type=str, default=None,
                        help="Path to ArieNetCooc checkpoint")
    parser.add_argument(
        "--extra_cooc_checkpoint",
        action="append",
        default=[],
        help=(
            "Additional CoOC checkpoint. Repeatable. "
            "Formats: NAME=PATH or PATH (auto-named ArieNetCooc_2, ArieNetCooc_3, …)."
        ),
    )

    parser.add_argument("--backbone_solver", choices=["march", "pysat"], default="march",
                        help="SAT oracle for backbone computation (default: march)")
    parser.add_argument("--backbone_march_path", type=str, default=None,
                        help="Path to march binary used for backbone computation "
                             f"(default: {MARCH_BACKBONE_PATH})")

    # Backbone computation settings
    parser.add_argument("--backbone_timeout", type=int, default=60,
                        help="Per-instance backbone computation timeout in seconds (default: 60)")
    parser.add_argument("--backbone_workers", type=int, default=4,
                        help="Parallel workers for backbone computation (default: 4)")

    # Dataloader / inference settings
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for GNN inference (default: 1)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (default: 0)")
    parser.add_argument("--backbone_process_workers", type=int, default=1,
                        help="Dataset preprocessing workers (default: 1; "
                             "set to 1 to avoid pysat deadlocks under fork)")
    parser.add_argument("--cooc_process_workers", type=int, default=1,
                        help="CoOC dataset preprocessing workers (default: 1)")

    # Output
    parser.add_argument("--out_file", type=str, default=None,
                        help="Output JSON file path (default: auto-timestamped)")

    args = parser.parse_args()

    # Build model checkpoint map
    model_ckpts = {
        "Backbone": args.backbone_checkpoint,
        "BackboneCanonical": args.canonical_checkpoint,
        "BackboneUP": args.backbone_up_checkpoint,
        "ArieNetCooc": args.cooc_checkpoint,
    }

    try:
        extra_backbone_models = parse_extra_backbone_args(args.extra_backbone_checkpoint)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    try:
        extra_cooc_models = parse_extra_cooc_args(args.extra_cooc_checkpoint)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    overlapping = (
        set(model_ckpts).intersection(extra_backbone_models)
        | set(model_ckpts).intersection(extra_cooc_models)
        | set(extra_backbone_models).intersection(extra_cooc_models)
    )
    if overlapping:
        print(f"[ERROR] Duplicate model names: {sorted(overlapping)}")
        sys.exit(1)

    model_ckpts.update(extra_backbone_models)
    model_ckpts.update(extra_cooc_models)
    _EXTRA_BACKBONE_NAMES.update(extra_backbone_models.keys())
    _EXTRA_COOC_NAMES.update(extra_cooc_models.keys())

    active_models = {k: v for k, v in model_ckpts.items() if v is not None}

    if not active_models:
        print("[ERROR] No model checkpoints provided. Use --backbone_checkpoint, "
              "--canonical_checkpoint, --backbone_up_checkpoint, --cooc_checkpoint, "
              "--extra_backbone_checkpoint, or --extra_cooc_checkpoint.")
        sys.exit(1)

    # Discover size buckets
    ood_dir = os.path.abspath(args.ood_dir)
    buckets = sorted(
        [d for d in os.listdir(ood_dir) if os.path.isdir(os.path.join(ood_dir, d))],
        key=lambda x: int(x.lstrip("n")) if x.lstrip("n").isdigit() else 0,
    )
    if not buckets:
        print(f"[ERROR] No size bucket folders found in {ood_dir}")
        sys.exit(1)

    print("=" * 70)
    print("OOD BACKBONE ACCURACY EVALUATION")
    print("=" * 70)
    print(f"  OOD dir             : {ood_dir}")
    print(f"  Size buckets        : {buckets}")
    print(f"  Models              : {list(active_models.keys())}")
    print(f"  Backbone timeout    : {args.backbone_timeout}s per instance")
    print(f"  Backbone solver     : {args.backbone_solver}")
    print(f"  Backbone workers    : {args.backbone_workers}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device              : {device}\n")

    # Determine output file
    if args.out_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_file = f"ood_backbone_accuracy_{ts}.json"

    results = {
        "timestamp": datetime.now().isoformat(),
        "ood_dir": ood_dir,
        "backbone_timeout": args.backbone_timeout,
        "backbone_solver": args.backbone_solver,
        "models": list(active_models.keys()),
        "checkpoints": active_models,
        "sizes": {},
    }

    # Resolve march path
    march_path = args.backbone_march_path or MARCH_BACKBONE_PATH

    # Pre-load all models
    print("Loading models …")
    loaded_models = {}
    for model_name, ckpt_path in active_models.items():
        print(f"  Loading {model_name} from {ckpt_path} …")
        try:
            loaded_models[model_name] = load_model(model_name, ckpt_path, device)
            print(f"  ✓ {model_name} loaded")
        except Exception as e:
            print(f"  ✗ {model_name} FAILED: {e}")
            loaded_models[model_name] = None
    print()

    # Main loop: buckets × models
    for bucket in buckets:
        bucket_dir = os.path.join(ood_dir, bucket)
        cnf_count = len(glob.glob(os.path.join(bucket_dir, "*.cnf")))
        print(f"\n{'='*70}")
        print(f"SIZE BUCKET: {bucket}  ({cnf_count} CNF files)")
        print(f"{'='*70}")

        results["sizes"][bucket] = {}

        for model_name, model in loaded_models.items():
            if model is None:
                print(f"  [{model_name}] Skipped (failed to load).")
                results["sizes"][bucket][model_name] = {"error": "model load failed"}
                continue

            print(f"\n  >> {model_name}")
            bucket_result = evaluate_bucket(
                model=model,
                model_name=model_name,
                cnf_dir=bucket_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                backbone_timeout=args.backbone_timeout,
                backbone_workers=args.backbone_workers,
                backbone_process_workers=args.backbone_process_workers,
                cooc_process_workers=args.cooc_process_workers,
                backbone_solver=args.backbone_solver,
                backbone_march_path=march_path,
                backbone_cache_file=os.path.join(bucket_dir, "backbones.pkl"),
            )

            if bucket_result is None:
                results["sizes"][bucket][model_name] = {"error": "evaluation failed"}
                print(f"  [{model_name}] Evaluation failed.")
            else:
                results["sizes"][bucket][model_name] = bucket_result
                acc_str = f"{bucket_result['accuracy']*100:.1f}%" if bucket_result['accuracy'] is not None else "N/A"
                neg_str = f"{bucket_result['neg_accuracy']*100:.1f}%" if bucket_result['neg_accuracy'] is not None else "N/A"
                pos_str = f"{bucket_result['pos_accuracy']*100:.1f}%" if bucket_result['pos_accuracy'] is not None else "N/A"
                print(f"  [{model_name}] "
                      f"instances={bucket_result['instances_computed']}/{bucket_result['total_instances']} | "
                      f"backbone_rate={bucket_result['backbone_rate']} | "
                      f"accuracy={acc_str}  (neg={neg_str}, pos={pos_str})")

        # Save after every bucket
        with open(args.out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [Saved intermediate results to {args.out_file}]")

    # Final summary table
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY  (backbone accuracy on backbone variables)")
    print("=" * 70)
    col_w = 20
    header = f"{'Bucket':<12}" + "".join(f"{m:<{col_w}}" for m in active_models)
    print(header)
    print("-" * len(header))

    for bucket in buckets:
        row = f"{bucket:<12}"
        for model_name in active_models:
            r = results["sizes"][bucket].get(model_name, {})
            if "error" in r:
                row += f"{'ERROR':<{col_w}}"
            elif r:
                acc = r.get("accuracy")
                acc_str = f"{acc*100:.1f}%" if acc is not None else "N/A"
                rate = r.get("backbone_rate")
                rate_str = f"{rate:.3f}" if rate is not None else "?"
                cell = f"{acc_str} (r={rate_str})"
                row += f"{cell:<{col_w}}"
            else:
                row += f"{'N/A':<{col_w}}"
        print(row)

    print(f"\nFull results saved to: {args.out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

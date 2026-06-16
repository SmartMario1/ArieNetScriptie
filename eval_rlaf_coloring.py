"""Evaluate trained RLAF models on a SAT benchmark.

For every CNF file under --test-dir the script:
  1. Optionally runs unguided glucose (include 'Solver' in --models) to
     establish a baseline and determine SAT/UNSAT.
  2. For each trained model, runs the model + guided glucose_weighted.

Results are reported (and saved) as a table split by size bucket × SAT/UNSAT,
showing per-group average decisions, total decisions, and total CPU time.

Usage
-----
    # 3-coloring
    python eval_rlaf_coloring.py --test-dir dataRLAF/test/coloring \\
        --ckpt-base supercomputer/best3col_gluc.pt \\
        --ckpt-nolsp supercomputer/best3col_nolsp_gluc.pt \\
        --ckpt-cooc supercomputer/best3col_cooc_gluc.pt \\
        --ckpt-coocedge supercomputer/best3col_cooc_edge_gluc.pt

    # SAT (run only COOCEdge, reuse previous baseline)
    python eval_rlaf_coloring.py --test-dir dataRLAF/test/sat \\
        --ckpt-coocedge supercomputer/bestsat_cooc_edge.pt \\
        --models COOCEdge --prev-results results_sat_prev.csv
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nsnet.models.arienet_rlaf import ArieNetRLAF, ArieNetRLAFCooc, ArieNetRLAFCoocEdge
from nsnet.datasets.rlaf_dataset import RLAFBPGDataset, _collate_skip_none
from nsnet.policy.evaluate_bpg import sample_var_params, _solve_cnf
from nsnet.utils.utils import parse_cnf_file


# ─── Configuration ────────────────────────────────────────────────────────

# Default test directory — overridden by --test-dir
TEST_DIR = "dataRLAF/test/coloring"

# Canonical model order (checkpoint paths are supplied via CLI args)
MODEL_ORDER = ["Solver", "noLSP", "Base", "COOC", "COOCEdge"]

# Model meta: (use_cooc, use_cooc_edge, no_precomputed_lsp)
# Checkpoint paths are resolved at runtime from CLI args.
MODEL_META: Dict[str, tuple] = {
    "Solver":   (False, False, False),  # baseline-only, no guided model
    "noLSP":    (False, False, True),
    "Base":     (False, False, False),
    "COOC":     (True,  False, False),
    "COOCEdge": (False, True,  False),
}

# Glucose binaries (relative to the nsnet root, same as during training)
SOLVER_DIR = "."

# Model architecture (must match training config)
MODEL_DIM        = 128
MODEL_N_ROUNDS   = 26
MODEL_N_MLP      = 3
MODEL_ACTIVATION = "relu"
MODEL_SCALE_SIGMA = 0.1


# ─── Helpers ──────────────────────────────────────────────────────────────

def _size_from_path(cnf_path: str) -> str:
    """Infer the size bucket (400/500/600) from the directory name."""
    parts = os.path.normpath(cnf_path).split(os.sep)
    # Walk up until we find the numeric subdirectory name
    for p in reversed(parts[:-1]):
        if p.isdigit():
            return p
    return "unknown"


def _solve_baseline(args):
    cnf_path, solver_dir, timeout, solver_family = args
    try:
        n_vars, clauses = parse_cnf_file(cnf_path)
        extra = {"timeout": timeout} if solver_family == "march" else {"cpu-lim": timeout}
        stats = _solve_cnf(
            clauses,
            var_params=None,
            solver_dir=solver_dir,
            solver=solver_family,
            **extra,
        )
    except Exception as e:
        stats = {"Result": "ERROR", "decisions": float("nan"),
                 "conflicts": float("nan"), "CPU time": float("nan")}
    stats.setdefault("Result", "UNKNOWN")
    stats["cnf_path"] = cnf_path
    return stats


def _solve_guided(args):
    cnf_path, var_params_np, solver_dir, timeout, solver_family = args
    try:
        _, clauses = parse_cnf_file(cnf_path)
        extra = {"timeout": timeout} if solver_family == "march" else {"cpu-lim": timeout}
        stats = _solve_cnf(
            clauses,
            var_params=var_params_np,
            solver_dir=solver_dir,
            solver=solver_family,
            **extra,
        )
    except Exception as e:
        stats = {"Result": "ERROR", "decisions": float("nan"),
                 "conflicts": float("nan"), "CPU time": float("nan")}
    stats.setdefault("Result", "UNKNOWN")
    stats["cnf_path"] = cnf_path
    return stats


def build_model(use_cooc: bool, use_cooc_edge: bool, no_lsp: bool,
                norm_mode: str = None) -> torch.nn.Module:
    if use_cooc_edge:
        # COOCEdge uses combine="add" (sum) and no normalisation (l2c_msg_norm
        # normalisation is bypassed because the input is already summed)
        return ArieNetRLAFCoocEdge(
            dim=MODEL_DIM,
            n_rounds=MODEL_N_ROUNDS,
            n_mlp_layers=MODEL_N_MLP,
            activation=MODEL_ACTIVATION,
            no_precomputed_local_sat=no_lsp,
            combine="add",
            norm_mode=norm_mode,
        )
    elif use_cooc:
        return ArieNetRLAFCooc(
            dim=MODEL_DIM,
            n_rounds=MODEL_N_ROUNDS,
            n_mlp_layers=MODEL_N_MLP,
            activation=MODEL_ACTIVATION,
            no_precomputed_local_sat=no_lsp,
            norm_mode=norm_mode,
        )
    else:
        return ArieNetRLAF(
            dim=MODEL_DIM,
            n_rounds=MODEL_N_ROUNDS,
            n_mlp_layers=MODEL_N_MLP,
            activation=MODEL_ACTIVATION,
            no_precomputed_local_sat=no_lsp,
        )


def load_model(ckpt: str, use_cooc: bool, use_cooc_edge: bool, no_lsp: bool, device: str,
               norm_mode: str = None) -> torch.nn.Module:
    model = build_model(use_cooc, use_cooc_edge, no_lsp, norm_mode=norm_mode)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"  [load_model] Ignoring unexpected state_dict keys: {unexpected}")
    if missing:
        print(f"  [load_model] WARNING — missing state_dict keys: {missing}")
    model.to(device)
    model.eval()
    return model


def collect_test_cnfs(test_dir: str) -> List[str]:
    """Return sorted list of all .cnf files under test_dir (any depth)."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(test_dir, "**", "*.cnf"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .cnf files found under {test_dir!r}")
    return files


def print_summary(df: pd.DataFrame, model_names: List[str], test_dir: str = "") -> None:
    """Print a formatted table split by size × SAT/UNSAT."""
    # Normalise Result column: keep only SATISFIABLE / UNSATISFIABLE
    df = df.copy()
    df["sat_label"] = df["baseline_result"].apply(
        lambda r: "SAT" if str(r).upper().startswith("S") else "UNSAT"
    )

    all_models = ["Baseline"] + model_names
    title = f"EVALUATION RESULTS  ({test_dir or 'test set'})"
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for size in sorted(df["size"].unique()):
        for sat in ["SAT", "UNSAT"]:
            sub = df[(df["size"] == size) & (df["sat_label"] == sat)]
            if sub.empty:
                continue
            print(f"\nSize={size}, {sat}  (n={len(sub)})")
            print(f"  {'Model':<20} {'Avg decisions':>15} {'Total decisions':>16} {'Total CPU time':>15}")
            print(f"  {'-'*20} {'-'*15} {'-'*16} {'-'*15}")

            for model in all_models:
                dec_col  = f"{model}_decisions"
                time_col = f"{model}_cputime"
                if dec_col not in sub.columns:
                    continue
                avg_dec   = sub[dec_col].mean()
                tot_dec   = sub[dec_col].sum()
                tot_time  = sub[time_col].sum() if time_col in sub.columns else float("nan")
                print(f"  {model:<20} {avg_dec:>15.1f} {tot_dec:>16.0f} {tot_time:>14.2f}s")

    print("\n" + "=" * 80)


# ─── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test-dir", default=TEST_DIR,
                        help=f"Root of the test CNF folder (default: {TEST_DIR!r})")
    # Per-model checkpoint paths (None = model will be skipped if selected)
    parser.add_argument("--ckpt-nolsp",    default="supercomputer/best3col_nolsp_gluc.pt",
                        metavar="PATH", help="Checkpoint for noLSP model.")
    parser.add_argument("--ckpt-base",     default="supercomputer/best3col_gluc.pt",
                        metavar="PATH", help="Checkpoint for Base model.")
    parser.add_argument("--ckpt-cooc",     default="supercomputer/best3col_cooc_gluc.pt",
                        metavar="PATH", help="Checkpoint for COOC model.")
    parser.add_argument("--ckpt-coocedge", default="supercomputer/best3col_cooc_edge_gluc.pt",
                        metavar="PATH", help="Checkpoint for COOCEdge model.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Torch device for model forward passes.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="Parallel solver processes.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="DataLoader batch size for model inference.")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-instance CPU time limit in seconds (default: 60).")
    parser.add_argument("--solver", default="glucose", choices=["glucose", "march"],
                        help="Solver family to use for both baseline and guided evaluation "
                             "(default: glucose). 'march' uses march/march_weighted; "
                             "'glucose' uses glucose/glucose_weighted.")
    parser.add_argument("--out", default="results_eval_coloring.csv",
                        help="Output CSV file (default: results_eval_coloring.csv).")
    parser.add_argument("--out-json", default="results_eval_coloring.json",
                        help="Output JSON file (default: results_eval_coloring.json).")
    parser.add_argument(
        "--coocedge-norm-mode", default=None, choices=["lse"],
        metavar="MODE",
        help="L2L aggregation mode for the COOCEdge model. "
             "Set to 'lse' to use log-sum-exp (default: sum)."
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        metavar="NAME",
        help=(
            "Which models to run, in order. Choices: " + ", ".join(MODEL_ORDER) + ". "
            "Default: all, in canonical order (Solver → noLSP → Base → COOC → COOCEdge)."
        ),
    )
    parser.add_argument(
        "--prev-results", default=None, metavar="CSV",
        help=(
            "Path to a CSV produced by a previous run of this script. "
            "Only cnf files whose path appears in that CSV will be evaluated "
            "(files absent from the CSV are assumed to have timed out and are "
            "preemptively skipped)."
        ),
    )
    args = parser.parse_args()

    # Validate --models
    if args.models is not None:
        # Always use a set for membership tests to avoid substring matching
        # (e.g. 'COOC' in 'COOCEdge' would be True for a plain string check)
        requested = set(args.models)
        unknown = requested - set(ALL_MODELS)
        if unknown:
            parser.error(f"Unknown model name(s): {', '.join(sorted(unknown))}. "
                         f"Valid choices: {', '.join(MODEL_ORDER)}")
        # Preserve canonical ordering even if user provided a different order
        run_model_order = [m for m in MODEL_ORDER if m in requested]
    else:
        run_model_order = MODEL_ORDER

    # Build MODELS dict: name → (ckpt, use_cooc, use_cooc_edge, no_lsp)
    ckpt_map = {
        "Solver":   None,
        "noLSP":    args.ckpt_nolsp,
        "Base":     args.ckpt_base,
        "COOC":     args.ckpt_cooc,
        "COOCEdge": args.ckpt_coocedge,
    }
    MODELS = {
        name: (ckpt_map[name],) + MODEL_META[name]
        for name in run_model_order
    }

    # ── Load previous results to restrict the file set ────────────────────
    # Files absent from the previous CSV are assumed to have timed out and
    # will be skipped.  Only files whose path appears in the CSV are run.
    # When "Solver" is not in MODELS, the baseline columns are read from the
    # previous CSV instead of being re-run.
    prev_df: Optional[pd.DataFrame] = None
    prev_allowed_paths: Optional[set] = None
    if args.prev_results is not None:
        if not os.path.isfile(args.prev_results):
            print(f"WARNING: --prev-results file not found: {args.prev_results!r} — ignoring.")
        else:
            prev_df = pd.read_csv(args.prev_results)
            if "cnf_path" not in prev_df.columns:
                print(f"WARNING: {args.prev_results!r} has no 'cnf_path' column — ignoring.")
                prev_df = None
            else:
                prev_allowed_paths = set(prev_df["cnf_path"].dropna().unique())
                print(
                    f"Loaded prev results from {args.prev_results!r}: "
                    f"{len(prev_df)} rows, {len(prev_allowed_paths)} known paths — "
                    f"only these will be evaluated (absent = timed out = skipped)."
                )

    # ── Collect CNF files ──────────────────────────────────────────────────
    print(f"Collecting test CNFs from {args.test_dir!r} ...")
    cnf_files = collect_test_cnfs(args.test_dir)
    print(f"  {len(cnf_files)} files found.")

    if prev_allowed_paths is not None:
        cnf_files = [f for f in cnf_files if f in prev_allowed_paths]
        print(f"  {len(cnf_files)} remaining after restricting to paths present in previous results CSV.")

    run_solver = "Solver" in MODELS

    # ── Baseline (unguided glucose) ────────────────────────────────────────
    n_steps = len(MODELS) + 2
    if run_solver:
        print(f"\n[1/{n_steps}] Running baseline (unguided {args.solver}) on {len(cnf_files)} instances "
              f"with {args.workers} workers ...")
        t0 = time.time()
        baseline_results = list(tqdm(
            Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                delayed(_solve_baseline)((f, SOLVER_DIR, args.timeout, args.solver)) for f in cnf_files
            ),
            total=len(cnf_files),
            desc="  baseline",
            unit="cnf",
        ))
        print(f"  Done in {time.time() - t0:.1f}s")

        baseline_df = pd.DataFrame(baseline_results).rename(columns={
            "decisions": "Baseline_decisions",
            "conflicts": "Baseline_conflicts",
            "CPU time":  "Baseline_cputime",
            "Result":    "baseline_result",
        })
        baseline_df["size"] = baseline_df["cnf_path"].apply(_size_from_path)

        # Drop instances that timed out (glucose returns UNKNOWN) or errored
        solved_mask = baseline_df["baseline_result"].str.upper().isin(
            ["SATISFIABLE", "UNSATISFIABLE"]
        )
        n_timed_out = (~solved_mask).sum()
        if n_timed_out:
            print(f"  Skipping {n_timed_out} timed-out/unsolved instances "
                  f"(baseline_result != SAT/UNSAT) from all evaluations.")
            baseline_df = baseline_df[solved_mask].reset_index(drop=True)
    else:
        # Reuse baseline columns from the previous CSV
        if prev_df is None:
            raise RuntimeError(
                "'Solver' is not in --models so the baseline will be read from "
                "--prev-results, but no valid --prev-results CSV was provided."
            )
        baseline_cols = ["cnf_path"] + [c for c in prev_df.columns
                                         if c in ("baseline_result", "Baseline_decisions",
                                                   "Baseline_conflicts", "Baseline_cputime", "size")]
        baseline_df = prev_df[baseline_cols].copy()
        # Restrict to the cnf_files we're actually evaluating
        baseline_df = baseline_df[baseline_df["cnf_path"].isin(set(cnf_files))].reset_index(drop=True)
        if "size" not in baseline_df.columns:
            baseline_df["size"] = baseline_df["cnf_path"].apply(_size_from_path)
        print(f"\n[1/{n_steps}] Skipping baseline solver (not in --models); "
              f"reusing {len(baseline_df)} baseline rows from {args.prev_results!r}.")

    valid_cnf_paths: set = set(baseline_df["cnf_path"])

    # ── Per-model evaluation ───────────────────────────────────────────────
    all_model_dfs: List[pd.DataFrame] = []
    model_names = list(MODELS.keys())

    for step, (model_name, (ckpt, use_cooc, use_cooc_edge, no_lsp)) in enumerate(MODELS.items(), start=2):
        # "Solver" entry is baseline-only — no guided model to run
        if ckpt is None:
            print(f"\n[{step}/{n_steps}] '{model_name}' is the unguided solver baseline (already done).")
            continue

        if not os.path.isfile(ckpt):
            print(f"\n[{step}/{n_steps}] SKIPPING {model_name}: checkpoint not found at {ckpt!r}")
            continue

        print(f"\n[{step}/{n_steps}] Evaluating model: {model_name}")
        print(f"  checkpoint   : {ckpt}")
        print(f"  use_cooc     : {use_cooc}")
        print(f"  use_cooc_edge: {use_cooc_edge}")
        print(f"  no_lsp       : {no_lsp}")
        print(f"  device       : {args.device}")
        if use_cooc_edge:
            print(f"  combine      : add  (sum, no normalisation)")
            print(f"  norm_mode    : {args.coocedge_norm_mode}")

        # Build dataset (reuses cached BPGs from training if available)
        # num_workers=0 forces single-threaded (main thread) dataset processing
        # to avoid subworker locking issues.
        dataset = RLAFBPGDataset(
            path=args.test_dir,
            use_cooc=use_cooc or use_cooc_edge,
            no_precomputed_local_sat=no_lsp,
            num_workers=1,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=_collate_skip_none,
        )

        # Load model
        norm_mode = args.coocedge_norm_mode if use_cooc_edge else None
        model = load_model(ckpt, use_cooc, use_cooc_edge, no_lsp, args.device,
                           norm_mode=norm_mode)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  parameters : {n_params:,}")

        # Forward passes → var_params (use policy mode for deterministic eval)
        print("  Running model forward passes ...")
        t1 = time.time()
        data_list = sample_var_params(
            model=model,
            loader=loader,
            num_samples=1,
            device=args.device,
            use_mode=True,
            scale_sigma=MODEL_SCALE_SIGMA,
        )
        print(f"  Forward done in {time.time() - t1:.1f}s — {len(data_list)} graphs")

        # Match BPG order back to original cnf_files order via cnf_id
        # (RLAFBPGDataset assigns cnf_id = index in self.cnf_files)
        id_to_path = {i: f for i, f in enumerate(dataset.cnf_files)}

        # Build (cnf_path, var_params_np) list — skip timed-out instances
        guided_inputs = []
        for data in data_list:
            cid = int(data.cnf_id.item())
            path = id_to_path[cid]
            if path not in valid_cnf_paths:
                continue
            vp_np = data.var_params[:, 0, :].numpy()  # [n_vars, 2]  (1 sample, mode)
            guided_inputs.append((path, vp_np, SOLVER_DIR, args.timeout, args.solver))

        # Run guided solver in parallel
        print(f"  Running guided solver ({args.workers} workers) ...")
        t2 = time.time()
        guided_results = list(tqdm(
            Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                delayed(_solve_guided)(inp) for inp in guided_inputs
            ),
            total=len(guided_inputs),
            desc=f"  {model_name}",
            unit="cnf",
        ))
        print(f"  Guided solving done in {time.time() - t2:.1f}s")

        model_df = pd.DataFrame(guided_results).rename(columns={
            "decisions": f"{model_name}_decisions",
            "conflicts": f"{model_name}_conflicts",
            "CPU time":  f"{model_name}_cputime",
            "Result":    f"{model_name}_result",
        })
        all_model_dfs.append(model_df)

        # Free GPU memory
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ── Merge results ──────────────────────────────────────────────────────
    print(f"\n[{n_steps}/{n_steps}] Merging results ...")
    merged = baseline_df.copy()
    for mdf in all_model_dfs:
        merged = merged.merge(mdf, on="cnf_path", how="left")

    # ── Print tables ──────────────────────────────────────────────────────
    evaluated_models = [name for name in model_names
                        if f"{name}_decisions" in merged.columns]
    print_summary(merged, evaluated_models, test_dir=args.test_dir)

    # ── Per-size aggregated summary ────────────────────────────────────────
    all_models_in_df = ["Baseline"] + evaluated_models
    agg_rows = []
    merged_with_sat = merged.copy()
    merged_with_sat["sat_label"] = merged_with_sat["baseline_result"].apply(
        lambda r: "SAT" if str(r).upper().startswith("S") else "UNSAT"
    )

    for size in sorted(merged_with_sat["size"].unique()):
        for sat in ["SAT", "UNSAT"]:
            sub = merged_with_sat[(merged_with_sat["size"] == size) &
                                  (merged_with_sat["sat_label"] == sat)]
            if sub.empty:
                continue
            for model in all_models_in_df:
                dec_col  = f"{model}_decisions"
                time_col = f"{model}_cputime"
                if dec_col not in sub.columns:
                    continue
                agg_rows.append({
                    "size": size,
                    "sat": sat,
                    "model": model,
                    "n": len(sub),
                    "avg_decisions": sub[dec_col].mean(),
                    "total_decisions": sub[dec_col].sum(),
                    "total_cpu_time": sub[time_col].sum() if time_col in sub.columns else float("nan"),
                })

    agg_df = pd.DataFrame(agg_rows)

    # ── Save ──────────────────────────────────────────────────────────────
    merged.to_csv(args.out, index=False)
    agg_df.to_json(args.out_json, orient="records", indent=2)
    print(f"\nFull per-instance results saved to {args.out!r}")
    print(f"Aggregated summary saved to {args.out_json!r}")


if __name__ == "__main__":
    main()

"""Compare approximate vs exact local_sat_percentage computation on crypto files.

Run from the nsnet/ directory:
    python test_local_sat_accuracy.py

The exact method iterates over every literal occurrence and is very slow on
crypto instances (~190 s/file).  The approximate method iterates over variables
instead (~few s/file).  This script measures how much accuracy is lost.

NOTE: both methods use Monte-Carlo sampling, so results have sampling noise.
The comparison evaluates the *expected* feature distribution, not sample-by-sample
equality.
"""

import glob
import sys
import time

sys.path.insert(0, "src")

import numpy as np
from nsnet.utils.utils import parse_cnf_file
from nsnet.utils.dataset import _compute_all_local_sat_percentages

# ── locate crypto training files ──────────────────────────────────────────────
crypto_files = sorted(glob.glob("dataRLAF/training/crypto/**/*.cnf", recursive=True))
if not crypto_files:
    print("No crypto CNF files found under dataRLAF/training/crypto/")
    print("Make sure you run this script from the nsnet/ directory.")
    sys.exit(1)

files_to_test = crypto_files[:5]
print(f"Testing on {len(files_to_test)} files:\n")
for f in files_to_test:
    print(f"  {f}")
print()

# ── per-file comparison ────────────────────────────────────────────────────────
all_mae   = []
all_corr  = []
all_rmse  = []

for path in files_to_test:
    n_vars, clauses = parse_cnf_file(path)
    # Deduplicate clauses (same as BPGParamBuilder)
    all_clauses = [list(c) for c in dict.fromkeys(tuple(c) for c in clauses)]
    n_occ = sum(len(c) for c in all_clauses)

    print(f"{'─'*60}")
    print(f"File : {path}")
    print(f"Stats: {n_vars} vars, {len(all_clauses)} clauses, {n_occ} literal occurrences")

    # ── exact (per occurrence, k2-neighbourhood) ──────────────────────────────
    print("  Running EXACT method (this can take several minutes)…", flush=True)
    t0 = time.time()
    exact = _compute_all_local_sat_percentages(
        all_clauses, n_vars,
        n_samples=500,
        show_progress=True,
        _approx_threshold=10 ** 9,  # force exact path (n_occ never exceeds this)
    )
    t_exact = time.time() - t0
    print(f"  Exact  done in {t_exact:.1f}s  →  {len(exact)} values")

    # ── approximate (per variable) ────────────────────────────────────────────
    print("  Running APPROXIMATE method…", flush=True)
    t0 = time.time()
    approx = _compute_all_local_sat_percentages(
        all_clauses, n_vars,
        n_samples=500,
        show_progress=True,
        _approx_threshold=0,        # force approximate branch (n_occ always > 0)
    )
    t_approx = time.time() - t0
    print(f"  Approx done in {t_approx:.1f}s  →  {len(approx)} values")

    # ── accuracy metrics ──────────────────────────────────────────────────────
    mae  = float(np.abs(approx - exact).mean())
    rmse = float(np.sqrt(((approx - exact) ** 2).mean()))
    corr = float(np.corrcoef(approx, exact)[0, 1])

    # Rank correlation (more meaningful for a feature used in a GNN)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(approx, exact)

    print(f"\n  Accuracy (approx vs exact):")
    print(f"    MAE              = {mae:.4f}")
    print(f"    RMSE             = {rmse:.4f}")
    print(f"    Pearson r        = {corr:.4f}")
    print(f"    Spearman ρ       = {rho:.4f}")
    print(f"    Speedup          = {t_exact / t_approx:.1f}×")

    # Distribution summary
    for label, arr in [("exact ", exact), ("approx", approx)]:
        print(f"    {label} — mean={arr.mean():.3f}  std={arr.std():.3f}"
              f"  min={arr.min():.3f}  max={arr.max():.3f}")

    all_mae.append(mae)
    all_corr.append(corr)
    all_rmse.append(rmse)
    print()

# ── summary ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY across all files:")
print(f"  Mean MAE     = {np.mean(all_mae):.4f}")
print(f"  Mean RMSE    = {np.mean(all_rmse):.4f}")
print(f"  Mean Pearson = {np.mean(all_corr):.4f}")

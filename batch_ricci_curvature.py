"""
Batch Ollivier-Ricci curvature computation over a folder of graphs.

Accepts either:
  - a folder of CNF files  (*.cnf)       → builds BPG on the fly
  - a folder of BPG files  (data_*.pt)   → loads directly

Usage
-----
  # All CNF files in a folder
  python batch_ricci_curvature.py SATSolving/3-sat/test/

  # First 50 files, save to CSV + JSON
  python batch_ricci_curvature.py SATSolving/3-sat/test/ --limit 50 --csv out.csv --json out.json

  # From pre-processed BPG .pt files
  python batch_ricci_curvature.py data/processed/  --alpha 0.5

Requirements
------------
  pip install GraphRicciCurvature
  (networkx, torch, and this repo's src/ must be on PYTHONPATH)
"""

import argparse
import csv
import glob
import json
import os
import sys
import traceback

from typing import Optional

import numpy as np
import torch

# Make sure the project's src/ directory is importable
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from nsnet.utils.utils import parse_cnf_file
from nsnet.utils.dataset import BPG, BPGParamBuilder

from compute_ricci_curvature import bpg_to_networkx, compute_ollivier_ricci, ricci_curvature_stats

# ---------------------------------------------------------------------------
# Optional: tqdm progress bar
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _iter_with_progress(iterable, total, desc=""):
    if _HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, unit="graph")
    return iterable


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(folder: str, limit: Optional[int]):
    """Return a sorted list of (label, path) pairs.

    Tries .cnf first, then data_*.pt.
    """
    cnf_files = sorted(glob.glob(os.path.join(folder, "**", "*.cnf"), recursive=True))
    if cnf_files:
        files = [(os.path.basename(p), p, "cnf") for p in cnf_files]
    else:
        pt_files = sorted(glob.glob(os.path.join(folder, "**", "data_*.pt"), recursive=True))
        files = [(os.path.basename(p), p, "pt") for p in pt_files]

    if not files:
        raise FileNotFoundError(
            f"No *.cnf or data_*.pt files found under {folder!r}"
        )

    if limit is not None:
        files = files[:limit]

    return files


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def bpg_from_cnf(cnf_path: str) -> BPG:
    n_vars, clauses = parse_cnf_file(cnf_path)
    params = BPGParamBuilder(clauses, n_vars).params
    return BPG(*params)


def process_file(name, path, kind, alpha):
    """Load a graph, compute Ricci curvature, return a stats dict."""
    try:
        if kind == "cnf":
            bpg = bpg_from_cnf(path)
        else:
            bpg = torch.load(path, weights_only=False)

        G = bpg_to_networkx(bpg)

        # Skip trivial graphs
        if G.number_of_edges() == 0:
            return {"file": name, "error": "no_edges",
                    "n_literals": int(bpg.n_literals), "n_clauses": int(bpg.n_clauses),
                    "n_nodes": G.number_of_nodes(), "n_edges": 0}

        orc = compute_ollivier_ricci(G, alpha=alpha)
        stats = ricci_curvature_stats(orc)
        stats["file"] = name
        stats["n_literals"] = int(bpg.n_literals)
        stats["n_clauses"] = int(bpg.n_clauses)
        stats["n_nodes"] = G.number_of_nodes()
        return stats

    except Exception as exc:  # noqa: BLE001
        return {"file": name, "error": str(exc), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_stats(results) -> dict:
    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not ok:
        return {"n_total": len(results), "n_ok": 0, "n_errors": len(errors)}

    for key in ("mean", "std", "min", "max", "median"):
        vals = [r[key] for r in ok if key in r]
        if vals:
            arr = np.array(vals)
            print(f"  curvature_{key:6s}  →  "
                  f"mean={np.mean(arr):.5f}  std={np.std(arr):.5f}  "
                  f"min={np.min(arr):.5f}  max={np.max(arr):.5f}")

    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "n_errors": len(errors),
        "errors": [{"file": r["file"], "error": r["error"]} for r in errors],
        "curvature": {
            key: {
                "mean":   float(np.mean([r[key] for r in ok if key in r])),
                "std":    float(np.std( [r[key] for r in ok if key in r])),
                "min":    float(np.min( [r[key] for r in ok if key in r])),
                "max":    float(np.max( [r[key] for r in ok if key in r])),
                "median": float(np.median([r[key] for r in ok if key in r])),
            }
            for key in ("mean", "std", "min", "max", "median")
            if any(key in r for r in ok)
        },
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "file", "n_literals", "n_clauses", "n_nodes", "n_edges",
    "mean", "std", "min", "max", "median", "error",
]


def write_csv(results, path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Per-file CSV written to {path}")


def write_json(results, agg: dict, path: str):
    with open(path, "w") as f:
        json.dump({"aggregate": agg, "per_file": results}, f, indent=2)
    print(f"Full JSON written to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch Ollivier-Ricci curvature over a folder of CNF / BPG files."
    )
    parser.add_argument("folder", help="Folder containing *.cnf or data_*.pt files")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="OllivierRicci laziness parameter in [0,1]  (default: 0.5)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of graphs to process  (default: all)"
    )
    parser.add_argument("--csv",  default=None, help="Save per-file results to CSV")
    parser.add_argument("--json", default=None, help="Save full results to JSON")
    args = parser.parse_args()

    # Discover files
    files = discover_files(args.folder, args.limit)
    print(f"Found {len(files)} graphs in {args.folder!r}  (alpha={args.alpha})")

    results = []
    failed = 0
    iterable = _iter_with_progress(files, total=len(files), desc="Curvature")

    for name, path, kind in iterable:
        result = process_file(name, path, kind, args.alpha)
        results.append(result)
        if "error" in result:
            failed += 1
            if not _HAS_TQDM:
                print(f"  [ERROR] {name}: {result['error']}")

    # Print aggregate
    print(f"\n{'='*60}")
    print(f"Processed: {len(results)}  |  OK: {len(results)-failed}  |  Errors: {failed}")
    print(f"{'='*60}")
    agg = aggregate_stats(results)

    if args.csv:
        write_csv(results, args.csv)
    if args.json:
        write_json(results, agg, args.json)


if __name__ == "__main__":
    main()

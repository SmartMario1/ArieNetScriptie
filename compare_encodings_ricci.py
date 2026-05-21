"""
Compare graph curvature across three CNF graph encodings.

Encodings
---------
  LCG  – Literal-Clause Graph + polarity edges   (our current NSNet encoding)
  VCG  – Variable-Clause Graph
    VCG+REWIRE(e) – VCG with BFC-oriented degree-preserving edge rewiring
                                    (edge alteration budget e)
  WLIG – Weighted Literal-Incidence Graph

Curvature measures
------------------
  orc  – Ollivier-Ricci Curvature   (lazy random walk / Wasserstein-1)
  bfc  – Balanced Forman Curvature  (Topping et al., 2022; combinatorial)
  both – compute both and report side-by-side

For each CNF file the script builds all three graph representations, runs the
selected curvature computation(s) on each, and reports per-encoding
statistics.  Aggregate summaries (averaged over all processed instances) are
printed in a comparison table.

Usage
-----
  # Default: ORC, first 20 files
  python compare_encodings_ricci.py SATSolving/3-sat/test/ --limit 20

  # BFC instead
  python compare_encodings_ricci.py SATSolving/3-sat/test/ --curvature bfc --limit 20

  # Both curvatures, save outputs
  python compare_encodings_ricci.py SATSolving/3-sat/test/ --curvature both \\
      --limit 100 --csv results_encodings.csv --json results_encodings.json

Notes on WLIG
-------------
  WLIG edges carry a ``weight`` attribute (co-occurrence count).  Curvature
  uses *hop distance* (1 per edge, ignoring weights) so the topological
  comparison is on equal footing with LCG and VCG.  Mean edge weight is
  included as a separate column for reference.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))

from nsnet.utils.utils import parse_cnf_file
from compute_ricci_curvature import (
    compute_ollivier_ricci,
    compute_balanced_forman,
    ricci_curvature_stats,
)
from graph_encodings import (
    cnf_to_lcg, cnf_to_vcg, cnf_to_vcg_bfc_augmented,
    cnf_to_lcg_bfc_augmented, cnf_to_lcg_clause_bridge,
    cnf_to_lcg_cooccurrence, cnf_to_wlig,
    cnf_to_lcg_clause_split,
)

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ---------------------------------------------------------------------------
# Encoding registry
# ---------------------------------------------------------------------------

ENCODINGS = ["LCG", "VCG", "LCG_BRIDGE", "LCG_COOC", "WLIG"]

_BASE_BUILDERS = {
    "LCG":        cnf_to_lcg,
    "VCG":        cnf_to_vcg,
    "LCG_BRIDGE": cnf_to_lcg_clause_bridge,
    "LCG_COOC":   cnf_to_lcg_cooccurrence,
    "WLIG":       cnf_to_wlig,
}

_BUILDERS = dict(_BASE_BUILDERS)


def _parse_int_csv(value: str) -> List[int]:
    """Parse comma-separated ints, e.g. '20,100,200' -> [20, 100, 200]."""
    if value is None:
        return []
    text = value.strip()
    if text == "":
        return []
    out = []
    for token in text.split(","):
        token = token.strip()
        if token == "":
            continue
        out.append(int(token))
    return out

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_cnf_files(folder: str, limit: Optional[int]) -> List[str]:
    files = sorted(glob.glob(os.path.join(folder, "**", "*.cnf"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No *.cnf files found under {folder!r}")
    return files[:limit] if limit is not None else files


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _mean_edge_weight(G) -> Optional[float]:
    """Return mean ``weight`` edge attribute, or None if no weights present."""
    weights = [d.get("weight") for _, _, d in G.edges(data=True) if "weight" in d]
    return float(np.mean(weights)) if weights else None


# Which curvature measures to compute, and the edge attribute they write
_CURVATURE_ATTR = {
    "orc": "ricciCurvature",
    "bfc": "bfc",
}


def _omega_star(mean_ric: float, var_ric: float, clause_density: float) -> float:
    """Skenderi (2025) hardness heuristic ω*(G).

    ω(G)  = -E[Ric] * E[α]       (negated mean curvature * clause density)
    ω*(G) = ω(G) / Var[Ric]      (normalised by curvature variance)

    Higher ω* → harder for a GNN-based solver.
    Returns NaN when variance is zero (perfectly uniform curvature).
    """
    omega = (-mean_ric) * clause_density
    if var_ric == 0.0:
        return float("nan")
    return omega / var_ric


def _compute_curvature(G, curvature: str, alpha: float, clause_density: float = 0.0):
    """Run the requested curvature computation and return stats dict."""
    stats: Dict[str, Any] = {}
    if curvature in ("orc", "both"):
        result = compute_ollivier_ricci(G, alpha=alpha)
        s = ricci_curvature_stats(result, attr="ricciCurvature")
        for k, v in s.items():
            stats[f"orc_{k}"] = v
        stats["orc_omega_star"] = _omega_star(
            mean_ric=stats.get("orc_mean", float("nan")),
            var_ric=stats.get("orc_std", 0.0) ** 2,
            clause_density=clause_density,
        )
    if curvature in ("bfc", "both"):
        result = compute_balanced_forman(G)
        s = ricci_curvature_stats(result, attr="bfc")
        for k, v in s.items():
            stats[f"bfc_{k}"] = v
        stats["bfc_omega_star"] = _omega_star(
            mean_ric=stats.get("bfc_mean", float("nan")),
            var_ric=stats.get("bfc_std", 0.0) ** 2,
            clause_density=clause_density,
        )
    return stats


def process_file(path: str, alpha: float, curvature: str) -> Dict[str, Any]:
    """Return a result dict with per-encoding stats for one CNF file."""
    name = os.path.basename(path)
    try:
        n_vars, clauses = parse_cnf_file(path)
        if not clauses:
            return {"file": name, "error": "empty_cnf"}

        row: Dict[str, Any] = {
            "file": name,
            "n_vars": n_vars,
            "n_clauses": len(clauses),
        }

        clause_density = len(clauses) / n_vars if n_vars > 0 else 0.0

        for enc_name, builder in _BUILDERS.items():
            G = builder(n_vars, clauses)

            if G.number_of_edges() == 0:
                row[enc_name] = {
                    "error": "no_edges",
                    "n_nodes": G.number_of_nodes(),
                    "n_edges": 0,
                }
                continue

            stats = _compute_curvature(G, curvature, alpha, clause_density=clause_density)
            stats["n_nodes"] = G.number_of_nodes()
            stats["n_edges"] = G.number_of_edges()

            mw = _mean_edge_weight(G)
            if mw is not None:
                stats["mean_edge_weight"] = mw

            row[enc_name] = stats

        return row

    except Exception as exc:  # noqa: BLE001
        return {
            "file": name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Aggregate statistics (across files, per encoding)
# ---------------------------------------------------------------------------

def _agg_key(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def aggregate_per_encoding(ok_results: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics for each encoding over all OK results."""
    agg: Dict[str, Any] = {}
    # Collect all prefixed scalar keys that appear in at least one result
    all_enc_keys: Dict[str, set] = {enc: set() for enc in ENCODINGS}
    for r in ok_results:
        for enc in ENCODINGS:
            if enc in r and isinstance(r[enc], dict) and "error" not in r[enc]:
                all_enc_keys[enc].update(r[enc].keys())

    for enc in ENCODINGS:
        enc_rows = [
            r[enc] for r in ok_results
            if enc in r and isinstance(r[enc], dict) and "error" not in r[enc]
        ]
        if not enc_rows:
            agg[enc] = {}
            continue
        agg[enc] = {
            k: _agg_key([row[k] for row in enc_rows])
            for k in all_enc_keys[enc]
            if all(k in row for row in enc_rows) and isinstance(enc_rows[0].get(k), (int, float))
        }
    return agg


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def _fmt(val: Any, width: int = 11, decimals: int = 5) -> str:
    if isinstance(val, float) and not (val != val):  # not NaN
        return f"{val:{width}.{decimals}f}"
    return f"{'—':>{width}}"


def print_comparison_table(agg: Dict[str, Any], n_ok: int, curvature: str) -> None:
    # Determine which curvature prefixes to show
    curv_prefixes = ["orc"] if curvature == "orc" else ["bfc"] if curvature == "bfc" else ["orc", "bfc"]
    curv_labels = {"orc": "ORC", "bfc": "BFC"}

    col_w = 12

    for prefix in curv_prefixes:
        header_parts = [
            f"{'Encoding':<8}",
            f"{'curv:mean':>{col_w}}",
            f"{'curv:std':>{col_w}}",
            f"{'curv:min':>{col_w}}",
            f"{'curv:max':>{col_w}}",
            f"{'omega*':>{col_w}}",
            f"{'n_nodes':>{col_w}}",
            f"{'n_edges':>{col_w}}",
            f"{'wt:mean':>{col_w}}",
        ]
        header = "  ".join(header_parts)
        label = curv_labels[prefix]

        print(f"\n{'='*len(header)}")
        print(f"{label} Curvature Comparison Across CNF Graph Encodings")
        print(f"(Averaged over {n_ok} instances, hop-distance metric)")
        print(f"{'='*len(header)}")
        print(header)
        print("-" * len(header))

        for enc in ENCODINGS:
            d = agg.get(enc, {})
            if not d:
                print(f"{enc:<8}  (no data)")
                continue

            def m(stat_key: str) -> float:
                entry = d.get(f"{prefix}_{stat_key}", {})
                return entry.get("mean", float("nan"))

            def mn(stat_key: str) -> float:
                entry = d.get(stat_key, {})
                return entry.get("mean", float("nan"))

            parts = [
                f"{enc:<8}",
                _fmt(m("mean"),        col_w),
                _fmt(m("std"),         col_w),
                _fmt(m("min"),         col_w),
                _fmt(m("max"),         col_w),
                _fmt(m("omega_star"),   col_w),
                _fmt(mn("n_nodes"),     col_w, 1),
                _fmt(mn("n_edges"),     col_w, 1),
                _fmt(mn("mean_edge_weight"), col_w, 3),
            ]
            print("  ".join(parts))

        print()
        print(f"  {label} distribution breakdown (mean across instances):")
        enc_header = "  ".join(f"{enc:>12}" for enc in ENCODINGS)
        print(f"  {'':8}  {'stat':>8}  {enc_header}")
        print(f"  {'':-<8}  {'':-<8}  " + "  ".join(f"{'':-<12}" for _ in ENCODINGS))
        for stat_key in ("mean", "std", "min", "max", "median", "omega_star"):
            vals = {
                enc: agg.get(enc, {}).get(f"{prefix}_{stat_key}", {}).get("mean", float("nan"))
                for enc in ENCODINGS
            }
            row = (f"  {'':8}  {stat_key:>8}  "
                   + "  ".join(_fmt(vals[enc], 12) for enc in ENCODINGS))
            print(row)

    print()


# ---------------------------------------------------------------------------
# CSV / JSON output helpers
# ---------------------------------------------------------------------------

_CSV_SCALAR_KEYS = (
    "orc_mean", "orc_std", "orc_min", "orc_max", "orc_median", "orc_omega_star",
    "bfc_mean", "bfc_std", "bfc_min", "bfc_max", "bfc_median", "bfc_omega_star",
    "n_nodes", "n_edges", "mean_edge_weight",
)

def _csv_fields() -> List[str]:
    return ["file", "n_vars", "n_clauses", "error"] + [
        f"{enc}_{k}" for enc in ENCODINGS for k in _CSV_SCALAR_KEYS
    ]


def _flatten_row(row: Dict) -> Dict:
    flat: Dict[str, Any] = {
        "file":      row.get("file", ""),
        "n_vars":    row.get("n_vars", ""),
        "n_clauses": row.get("n_clauses", ""),
        "error":     row.get("error", ""),
    }
    for enc in ENCODINGS:
        enc_data = row.get(enc, {})
        for k in _CSV_SCALAR_KEYS:
            key = f"{enc}_{k}"
            flat[key] = enc_data.get(k, "") if isinstance(enc_data, dict) else ""
    return flat


def write_csv(results: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_flatten_row(r) for r in results)
    print(f"Per-file CSV written to {path}")


def write_json(results: List[Dict], agg: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"aggregate": agg, "per_file": results}, f, indent=2)
    print(f"Full JSON written to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global ENCODINGS
    global _BUILDERS

    parser = argparse.ArgumentParser(
        description=(
            "Compare graph curvature across LCG, VCG, and WLIG "
            "encodings of CNF formulas."
        )
    )
    parser.add_argument(
        "folder",
        help="Folder containing *.cnf files (searched recursively)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="ORC laziness parameter in [0, 1]  (default: 0.5; only used for orc/both)",
    )
    parser.add_argument(
        "--curvature", choices=["orc", "bfc", "both"], default="orc",
        help="Curvature to compute: orc (Ollivier-Ricci, default), "
             "bfc (Balanced Forman), or both",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of CNF files to process  (default: all)",
    )
    parser.add_argument("--csv",  default=None, help="Save per-file results to CSV")
    parser.add_argument("--json", default=None, help="Save full results to JSON")
    parser.add_argument(
        "--vcg_aux_edge_budgets",
        default="",
        help=(
            "Comma-separated iteration counts for BFC-oriented VCG rewiring, "
            "e.g. '20,100,200'."
        ),
    )
    parser.add_argument(
        "--lcg_aux_edge_budgets",
        default="",
        help=(
            "Comma-separated iteration counts for BFC-oriented LCG rewiring "
            "(the actual NSNet encoding), e.g. '20,100,200'."
        ),
    )
    parser.add_argument(
        "--lcg_split_budgets",
        default="",
        help=(
            "Comma-separated split counts for clause-split LCG encoding, "
            "e.g. '5,10,20'.  Each value n_splits targets the n most negatively "
            "curved literal-clause edges and splits their clauses via a dummy variable."
        ),
    )
    args = parser.parse_args()

    vcg_budgets = _parse_int_csv(args.vcg_aux_edge_budgets)
    vcg_budgets = sorted(set(b for b in vcg_budgets if b >= 0))
    lcg_budgets = _parse_int_csv(args.lcg_aux_edge_budgets)
    lcg_budgets = sorted(set(b for b in lcg_budgets if b >= 0))
    split_budgets = _parse_int_csv(args.lcg_split_budgets)
    split_budgets = sorted(set(b for b in split_budgets if b >= 0))

    _BUILDERS = dict(_BASE_BUILDERS)
    for b in vcg_budgets:
        label = f"VCG_AUX_e{b}"
        _BUILDERS[label] = (lambda n_vars, clauses, b=b: cnf_to_vcg_bfc_augmented(n_vars, clauses, n_iterations=b))
    for b in lcg_budgets:
        label = f"LCG_AUX_e{b}"
        _BUILDERS[label] = (lambda n_vars, clauses, b=b: cnf_to_lcg_bfc_augmented(n_vars, clauses, n_iterations=b))
    for b in split_budgets:
        label = f"LCG_SPLIT_s{b}"
        _BUILDERS[label] = (lambda n_vars, clauses, b=b: cnf_to_lcg_clause_split(n_vars, clauses, n_splits=b))

    ENCODINGS = list(_BUILDERS.keys())

    files = _discover_cnf_files(args.folder, args.limit)
    print(f"Found {len(files)} CNF files in {args.folder!r}  "
          f"(curvature={args.curvature}, alpha={args.alpha})")
    if vcg_budgets:
        print(f"VCG auxiliary-edge budgets: {vcg_budgets}")
    if lcg_budgets:
        print(f"LCG auxiliary-edge budgets: {lcg_budgets}")
    if split_budgets:
        print(f"LCG clause-split budgets: {split_budgets}")

    results: List[Dict] = []
    iterable = tqdm(files, desc="Encoding + Curvature") if _HAS_TQDM else files
    for path in iterable:
        results.append(process_file(path, args.alpha, args.curvature))

    ok   = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]

    if errs:
        print(f"\n{len(errs)} file(s) failed:")
        for r in errs:
            print(f"  [ERROR] {r['file']}: {r['error']}")

    agg = aggregate_per_encoding(ok)
    print_comparison_table(agg, n_ok=len(ok), curvature=args.curvature)

    if args.csv:
        write_csv(results, args.csv)
    if args.json:
        write_json(results, agg, args.json)


if __name__ == "__main__":
    main()

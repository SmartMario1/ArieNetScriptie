"""
Compute graph curvature for a BPG (Bipartite Problem Graph).

Two curvature measures are supported:

  ORC  – Ollivier-Ricci Curvature
         Lazy random-walk / Wasserstein-1 formulation.  Exact LP solver,
         no GraphRicciCurvature / networkit dependency required.

  BFC  – Balanced Forman Curvature   (Topping et al., 2022)
         Purely combinatorial: counts triangles (#△) and 4-cycles (#□).
         Defined in eq. (16) of "Understanding over-squashing and
         bottlenecks on graphs via curvature" (ICLR 2022).
         Zero when min(d_i, d_j) = 1.

The BPG is treated as an undirected graph built from a BPG .pt file.

Usage
-----
  python compute_ricci_curvature.py path/to/graph.pt
  python compute_ricci_curvature.py path/to/graph.pt --alpha 0.5 --curvature orc
  python compute_ricci_curvature.py path/to/graph.pt --curvature bfc
  python compute_ricci_curvature.py path/to/graph.pt --curvature both --output results.json
"""

import argparse
import json

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


# ---------------------------------------------------------------------------
# BPG → NetworkX conversion
# ---------------------------------------------------------------------------

def bpg_to_networkx(bpg) -> nx.Graph:
    """Convert a BPG object to an undirected NetworkX bipartite graph.

    Node attributes
    ~~~~~~~~~~~~~~~
    bipartite=0  →  literal nodes  (named "lit_<index>")
    bipartite=1  →  clause  nodes  (named "cls_<index>")

    Parameters
    ----------
    bpg : BPG
        A loaded BPG torch_geometric.data.Data object.

    Returns
    -------
    nx.Graph
        Undirected bipartite graph with string node labels.
    """
    G = nx.Graph()

    n_literals = int(bpg.n_literals)
    n_clauses = int(bpg.n_clauses)

    # Add typed nodes
    for i in range(n_literals):
        G.add_node("lit_{}".format(i), bipartite=0)
    for j in range(n_clauses):
        G.add_node("cls_{}".format(j), bipartite=1)

    # Add edges – one per literal occurrence in a clause (no duplication)
    lit_indices = bpg.literal_indices_per_occurence.tolist()
    cls_indices = bpg.clause_indices_per_occurence.tolist()

    for l_idx, c_idx in zip(lit_indices, cls_indices):
        G.add_edge("lit_{}".format(l_idx), "cls_{}".format(c_idx))

    return G


# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature (self-contained, no GraphRicciCurvature needed)
# ---------------------------------------------------------------------------

def _node_measure(G, node, alpha):
    """Return the lazy random walk measure mu_x^alpha.

    With probability alpha the walker stays at *node*;
    with probability (1-alpha) it moves uniformly to a neighbor.

    Returns
    -------
    dict  {node: probability}
    """
    neighbors = list(G.neighbors(node))
    deg = len(neighbors)
    if deg == 0:
        return {node: 1.0}
    mu = {node: alpha}
    w = (1.0 - alpha) / deg
    for nb in neighbors:
        mu[nb] = mu.get(nb, 0.0) + w
    return mu


def _wasserstein1(G, mu_x, mu_y, dist_cache):
    """Compute the Wasserstein-1 (earth-mover) distance between two
    discrete measures on *G* using the Hungarian algorithm.

    For small support sets this is fast and exact.
    """
    support = sorted(set(mu_x) | set(mu_y))
    n = len(support)
    idx = {s: i for i, s in enumerate(support)}

    # Build cost matrix (shortest-path distances)
    cost = np.zeros((n, n))
    for i, si in enumerate(support):
        for j, sj in enumerate(support):
            if i <= j:
                key = (si, sj) if si <= sj else (sj, si)
                if key not in dist_cache:
                    try:
                        d = nx.shortest_path_length(G, si, sj)
                    except nx.NetworkXNoPath:
                        d = 1e9
                    dist_cache[key] = d
                cost[i][j] = dist_cache[key]
                cost[j][i] = cost[i][j]

    # Expand into supply/demand vectors
    supply = np.array([mu_x.get(s, 0.0) for s in support])
    demand = np.array([mu_y.get(s, 0.0) for s in support])

    # Re-formulate as balanced transport via replication
    # Each unit of supply/demand becomes one row/column
    # For efficiency, quantise to integers (precision 1e-9)
    SCALE = 10**9
    s_int = np.round(supply * SCALE).astype(np.int64)
    d_int = np.round(demand * SCALE).astype(np.int64)

    # Fix rounding so totals match
    diff = s_int.sum() - d_int.sum()
    if diff > 0:
        d_int[np.argmax(d_int)] += diff
    elif diff < 0:
        s_int[np.argmax(s_int)] += (-diff)

    # For the typical BPG case the supports are small so we can use
    # the standard LP formulation via scipy linprog, but it's even
    # simpler (and sufficient) to compute W1 with the network simplex
    # or, for very small supports, expand the Hungarian approach.
    # Use the direct formula via the POT-like approach:
    #   W1 = min_{T >= 0, T 1 = supply, T^T 1 = demand} <T, cost>
    # scipy's linear_sum_assignment works on *square* cost matrices for
    # the assignment problem.  For general OT with arbitrary masses we
    # use scipy.optimize.linprog.

    from scipy.optimize import linprog

    # Variables: T[i,j] for i in range(n), j in range(n)  flattened
    c = cost.flatten()
    n_vars = n * n

    # Equality constraints: row sums = supply, col sums = demand
    A_eq_rows = []
    b_eq = []
    for i in range(n):
        row = np.zeros(n_vars)
        for j in range(n):
            row[i * n + j] = 1.0
        A_eq_rows.append(row)
        b_eq.append(supply[i])
    for j in range(n):
        row = np.zeros(n_vars)
        for i in range(n):
            row[i * n + j] = 1.0
        A_eq_rows.append(row)
        b_eq.append(demand[j])

    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq)

    bounds = [(0, None)] * n_vars
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        return res.fun
    # Fallback: return a rough upper bound
    return float(np.sum(cost * np.outer(supply, demand)))


def compute_ollivier_ricci(G, alpha=0.5, **kwargs):
    """Compute Ollivier-Ricci curvature for every edge in *G*.

    Curvature values are stored on the edge attribute ``"ricciCurvature"``.

    Parameters
    ----------
    G     : nx.Graph
    alpha : float  –  laziness parameter in [0, 1]

    Returns
    -------
    A simple namespace with attribute ``.G`` carrying the annotated graph
    (compatible with ``ricci_curvature_stats``).
    """
    dist_cache = {}
    for u, v in G.edges():
        mu_u = _node_measure(G, u, alpha)
        mu_v = _node_measure(G, v, alpha)
        d_uv_key = (u, v) if u <= v else (v, u)
        if d_uv_key not in dist_cache:
            dist_cache[d_uv_key] = 1  # adjacent → distance 1
        w1 = _wasserstein1(G, mu_u, mu_v, dist_cache)
        kappa = 1.0 - w1 / dist_cache[d_uv_key]
        G.edges[u, v]["ricciCurvature"] = kappa

    class _Result:
        pass
    result = _Result()
    result.G = G
    return result


def ricci_curvature_stats(orc, attr="ricciCurvature"):
    """Return summary statistics for edge-level curvature values.

    Parameters
    ----------
    orc  : object with ``.G`` attribute  (result of any compute_* function)
    attr : str – edge attribute name to read  (default: "ricciCurvature")

    Returns
    -------
    dict with keys: n_edges, mean, std, min, max, median
    """
    curvatures = [
        data[attr]
        for _, _, data in orc.G.edges(data=True)
        if attr in data
    ]

    if not curvatures:
        return {}

    arr = np.array(curvatures, dtype=float)
    return {
        "n_edges": len(arr),
        "mean":    float(np.mean(arr)),
        "std":     float(np.std(arr)),
        "min":     float(np.min(arr)),
        "max":     float(np.max(arr)),
        "median":  float(np.median(arr)),
    }


# ---------------------------------------------------------------------------
# Balanced Forman Curvature (BFC)
# ---------------------------------------------------------------------------

def compute_balanced_forman(G: nx.Graph):
    """Compute Balanced Forman Curvature (BFC) for every edge in *G*.

    Implements equation (16) from:
      Topping et al. (2022) "Understanding over-squashing and bottlenecks
      on graphs via curvature"  (ICLR 2022, Appendix A.2).

    For a simple unweighted graph, the BFC of edge (i, j) is::

        Ric(i,j) = 2/d_i + 2/d_j − 2
                   + 2·|#△(i,j)| / max(d_i, d_j)
                   + |#△(i,j)| / min(d_i, d_j)
                   + γ_max(i,j)⁻¹ / max(d_i, d_j) · (|#□_i| + |#□_j|)

    Special cases
    -------------
    - Ric(i,j) = 0 when min(d_i, d_j) = 1  (leaf node).
    - When the 4-cycle sets #□_i and #□_j are both empty the last term is 0.
    - γ_max is clamped to max(γ_max, 1) to avoid division by zero when every
      4-cycle node has exactly one closing path  (standard convention).

    Definitions
    -----------
    #tri(i,j) = N(i) ∩ N(j)                                   (common neighbours)
    #sq_i(i,j) = { k ∈ N(i) \\ N(j) \\ {j} : (N(k) ∩ N(j)) \\ N(i) != {} }
    #sq_j(i,j) = { k ∈ N(j) \\ N(i) \\ {i} : (N(k) ∩ N(i)) \\ N(j) != {} }
    gamma_max(i,j) = max over k in #sq_i of |(N(k) ∩ N(j)) \\ N(i)| - 1
                 and over k in #sq_j of |(N(k) ∩ N(i)) \\ N(j)| - 1

    Curvature values are stored in edge attribute ``"bfc"``.

    Parameters
    ----------
    G : nx.Graph

    Returns
    -------
    An object with attribute ``.G`` (annotated graph), compatible with
    :func:`ricci_curvature_stats` when called with ``attr="bfc"``.
    """
    # Pre-build adjacency sets for O(1) membership tests
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}

    for u, v in G.edges():
        du = len(adj[u])
        dv = len(adj[v])

        # Leaf node: BFC is defined as zero
        if min(du, dv) <= 1:
            G.edges[u, v]["bfc"] = 0.0
            continue

        # --- triangle neighbours -----------------------------------------
        tri = adj[u] & adj[v]
        n_tri = len(tri)

        # --- 4-cycle sets ------------------------------------------------
        # sq_u: neighbours of u (not of v, not v itself) that have at least
        #       one neighbour which is also a neighbour of v but not of u
        #       and not u itself (u is always in N(k)∩N(v) \ N(u) since
        #       k~u and v~u but u∉N(u); excluding u avoids the degenerate
        #       path u-k-u-v).
        sq_u = {
            k for k in adj[u] - adj[v] - {v}
            if (adj[k] & adj[v]) - adj[u] - {u}
        }
        # sq_v: symmetric (exclude v for the same reason)
        sq_v = {
            k for k in adj[v] - adj[u] - {u}
            if (adj[k] & adj[u]) - adj[v] - {v}
        }
        n_sq = len(sq_u) + len(sq_v)

        # --- γ_max -------------------------------------------------------
        if n_sq == 0:
            gamma_max = 1  # term vanishes anyway; avoid unused division
        else:
            gamma_vals = []
            for k in sq_u:
                # |{w : w ∈ N(k) ∩ N(v), w ∉ N(u), w ≠ u}| - 1
                gamma_vals.append(len((adj[k] & adj[v]) - adj[u] - {u}) - 1)
            for k in sq_v:
                # |{w : w ∈ N(k) ∩ N(u), w ∉ N(v), w ≠ v}| - 1
                gamma_vals.append(len((adj[k] & adj[u]) - adj[v] - {v}) - 1)
            gamma_max = max(gamma_vals)
            # Clamp to 1: when every 4-cycle node has exactly one closing
            # path, gamma_max = 0, so gamma_max^{-1} would be undefined;
            # by convention we treat it as 1 (each node contributes 1).
            gamma_max = max(gamma_max, 1)

        # --- BFC formula (eq. 16) ----------------------------------------
        bfc = (
            2.0 / du
            + 2.0 / dv
            - 2.0
            + 2.0 * n_tri / max(du, dv)
            + n_tri / min(du, dv)
            + (1.0 / gamma_max) / max(du, dv) * n_sq
        )
        G.edges[u, v]["bfc"] = bfc

    class _Result:
        pass
    result = _Result()
    result.G = G
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute graph curvature of a BPG graph (.pt file)."
    )
    parser.add_argument("bpg_path", help="Path to a saved BPG .pt file")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="ORC laziness parameter in [0, 1]  (default: 0.5; only used for orc/both)"
    )
    parser.add_argument(
        "--curvature", choices=["orc", "bfc", "both"], default="orc",
        help="Which curvature to compute: orc (Ollivier-Ricci, default), "
             "bfc (Balanced Forman), or both"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save per-edge curvature data as JSON"
    )
    args = parser.parse_args()

    want_orc = args.curvature in ("orc", "both")
    want_bfc = args.curvature in ("bfc", "both")

    print("Loading BPG from {} ...".format(args.bpg_path))
    bpg = torch.load(args.bpg_path, weights_only=False)
    print("  n_literals = {},  n_clauses = {}".format(bpg.n_literals, bpg.n_clauses))

    print("Converting to NetworkX bipartite graph ...")
    G = bpg_to_networkx(bpg)
    print("  nodes = {},  edges = {}".format(G.number_of_nodes(), G.number_of_edges()))

    all_stats = {}
    result_obj = None  # last computed result (carries annotated G)

    if want_orc:
        print("\nComputing Ollivier-Ricci curvature  (alpha={}) ...".format(args.alpha))
        result_obj = compute_ollivier_ricci(G, alpha=args.alpha)
        orc_stats = ricci_curvature_stats(result_obj, attr="ricciCurvature")
        all_stats["orc"] = orc_stats
        print("  ORC statistics:")
        for k, v in orc_stats.items():
            print("    {}: {:.6f}".format(k, v) if isinstance(v, float) else "    {}: {}".format(k, v))

    if want_bfc:
        print("\nComputing Balanced Forman Curvature ...")
        result_obj = compute_balanced_forman(G)
        bfc_stats = ricci_curvature_stats(result_obj, attr="bfc")
        all_stats["bfc"] = bfc_stats
        print("  BFC statistics:")
        for k, v in bfc_stats.items():
            print("    {}: {:.6f}".format(k, v) if isinstance(v, float) else "    {}: {}".format(k, v))

    if args.output and result_obj is not None:
        edge_data = [
            {"u": u, "v": v, **{attr: data[attr] for attr in ("ricciCurvature", "bfc") if attr in data}}
            for u, v, data in result_obj.G.edges(data=True)
        ]
        with open(args.output, "w") as f:
            json.dump({"stats": all_stats, "edges": edge_data}, f, indent=2)
        print("\nCurvature data saved to {}".format(args.output))


if __name__ == "__main__":
    main()

"""
CNF formula → NetworkX graph encoders.

Three encodings are implemented, following the taxonomy in:
  Chen & Wang (2025) "Enhancing Modern SAT Solver With Machine Learning Method"
  GLSVLSI '25, https://doi.org/10.1145/3716368.3735251

  1. LCG  – Literal-Clause Graph (bipartite, literals ↔ clauses)
             + polarity edges (lit_{2k} ↔ lit_{2k+1} for each variable k).
             This matches the current BPG / NSNet encoding used in this project.

  2. VCG  – Variable-Clause Graph (bipartite, variables ↔ clauses).
             Positive and negative literals of the same variable are merged
             into a single variable node.

  3. WLIG – Weighted Literal-Incidence Graph.
             Nodes are literals; edges connect literals that co-occur in at
             least one clause.  Edge attribute ``weight`` = number of clauses
             in which the two literals co-occur together.

All encoders take the same inputs:
    n_vars  (int)               – number of variables (from the CNF header)
    clauses (List[List[int]])   – clauses as signed DIMACS integers

and return an ``nx.Graph`` with nodes labelled as strings for easy
interpretation.

Literal indexing (shared by LCG and WLIG)
------------------------------------------
  positive literal  v  → index 2*(v-1)       → node name "lit_{2*(v-1)}"
  negative literal -v  → index 2*(v-1)+1     → node name "lit_{2*(v-1)+1}"

This matches the indexing used in BPGParamBuilder.
"""

from __future__ import annotations

from collections import defaultdict
import random
from typing import List

import networkx as nx


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _literal_index(literal: int) -> int:
    """Map a signed DIMACS literal to a 0-based literal index."""
    v = abs(literal)
    return 2 * (v - 1) if literal > 0 else 2 * (v - 1) + 1


# ---------------------------------------------------------------------------
# 1. LCG – Literal-Clause Graph  (our current encoding)
# ---------------------------------------------------------------------------

def cnf_to_lcg(n_vars: int, clauses: List[List[int]]) -> nx.Graph:
    """Literal-Clause Graph (bipartite) with added polarity edges.

    Nodes
    -----
    - ``lit_<i>``  (i = 0 … 2*n_vars-1) – literal nodes
        even index 2k   → positive literal of variable k+1
        odd  index 2k+1 → negative literal of variable k+1
    - ``cls_<j>``  (j = 0 … n_clauses-1) – clause nodes

    Edges
    -----
    - (lit_<i>, cls_<j>) whenever literal i appears in clause j
    - (lit_<2k>, lit_<2k+1>) for each variable k  (polarity bridge)

    Parameters
    ----------
    n_vars  : int
    clauses : list of lists of signed ints (DIMACS)

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()

    # Literal nodes
    for k in range(n_vars):
        G.add_node(f"lit_{2*k}",     bipartite=0, kind="pos_literal", var=k)
        G.add_node(f"lit_{2*k+1}",   bipartite=0, kind="neg_literal", var=k)

    # Clause nodes
    for j in range(len(clauses)):
        G.add_node(f"cls_{j}", bipartite=1, kind="clause")

    # Literal–clause edges  (deduplicate within each clause)
    for j, clause in enumerate(clauses):
        seen = set()
        for lit in clause:
            l_idx = _literal_index(lit)
            if l_idx not in seen:
                G.add_edge(f"lit_{l_idx}", f"cls_{j}")
                seen.add(l_idx)

    # Polarity bridge edges
    for k in range(n_vars):
        G.add_edge(f"lit_{2*k}", f"lit_{2*k+1}", kind="polarity")

    return G


# ---------------------------------------------------------------------------
# 2. VCG – Variable-Clause Graph
# ---------------------------------------------------------------------------

def cnf_to_vcg(n_vars: int, clauses: List[List[int]]) -> nx.Graph:
    """Variable-Clause Graph (bipartite).

    Positive and negative literals of each variable are collapsed into a
    single variable node.  A variable is connected to every clause it
    appears in (regardless of polarity); duplicate appearances within the
    same clause collapse to a single edge.

    Nodes
    -----
    - ``var_<k>``  (k = 0 … n_vars-1) – variable nodes
    - ``cls_<j>``  (j = 0 … n_clauses-1) – clause nodes

    Edges
    -----
    - (var_<k>, cls_<j>) whenever variable k+1 appears in clause j

    Parameters
    ----------
    n_vars  : int
    clauses : list of lists of signed ints (DIMACS)

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()

    for k in range(n_vars):
        G.add_node(f"var_{k}", bipartite=0, kind="variable")

    for j in range(len(clauses)):
        G.add_node(f"cls_{j}", bipartite=1, kind="clause")

    for j, clause in enumerate(clauses):
        seen_vars: set = set()
        for lit in clause:
            k = abs(lit) - 1
            if k not in seen_vars:
                G.add_edge(f"var_{k}", f"cls_{j}")
                seen_vars.add(k)

    return G


def _bfc_general(u: str, v: str, adj: dict) -> float:
    """Balanced Forman Curvature for any edge (u, v).

    Exact implementation of eq. (16) from Topping et al. (2022), matching
    ``compute_ricci_curvature.compute_balanced_forman``.  Works for both
    bipartite and non-bipartite graphs (e.g. LCG with polarity bridge edges).

    Parameters
    ----------
    u, v : str   – the two endpoint node names
    adj  : dict  – full adjacency sets {node: set(neighbours)}
    """
    du = len(adj[u])
    dv = len(adj[v])

    if min(du, dv) <= 1:
        return 0.0

    n_tri = len(adj[u] & adj[v])

    sq_u = {k for k in adj[u] - adj[v] - {v} if (adj[k] & adj[v]) - adj[u] - {u}}
    sq_v = {k for k in adj[v] - adj[u] - {u} if (adj[k] & adj[u]) - adj[v] - {v}}
    n_sq = len(sq_u) + len(sq_v)

    if n_sq == 0:
        gamma_max = 1
    else:
        gamma_vals = (
            [len((adj[k] & adj[v]) - adj[u] - {u}) - 1 for k in sq_u]
            + [len((adj[k] & adj[u]) - adj[v] - {v}) - 1 for k in sq_v]
        )
        gamma_max = max(max(gamma_vals), 1)

    return (
        2.0 / du + 2.0 / dv - 2.0
        + 2.0 * n_tri / max(du, dv)
        + n_tri / min(du, dv)
        + n_sq / (gamma_max * max(du, dv))
    )


def _bfc_bipartite(
    var_node: str,
    cls_node: str,
    var_to_cls: dict,
    cls_to_var: dict,
) -> float:
    """Balanced Forman Curvature for a bipartite edge (var_node, cls_node).

    Specialised, faster version of ``_bfc_general`` for pure bipartite graphs
    (no triangles).  Matches ``compute_ricci_curvature.compute_balanced_forman``
    via the formula:

        Ric(u, v) = 2/d_u + 2/d_v - 2 + n_sq / (γ_max · max(d_u, d_v))
    """
    d_u = len(var_to_cls[var_node])
    d_v = len(cls_to_var[cls_node])

    if min(d_u, d_v) <= 1:
        return 0.0

    s_u = var_to_cls[var_node]
    s_v = cls_to_var[cls_node]

    sq_u: list[int] = []
    for cls_k in s_u:
        if cls_k == cls_node:
            continue
        shared_count = len((cls_to_var[cls_k] & s_v) - {var_node})
        if shared_count > 0:
            sq_u.append(shared_count - 1)

    sq_v: list[int] = []
    for var_l in s_v:
        if var_l == var_node:
            continue
        shared_count = len((var_to_cls[var_l] & s_u) - {cls_node})
        if shared_count > 0:
            sq_v.append(shared_count - 1)

    n_sq = len(sq_u) + len(sq_v)
    if n_sq == 0:
        return 2.0 / d_u + 2.0 / d_v - 2.0

    gamma_max = max(max(sq_u, default=0), max(sq_v, default=0))
    gamma_max = max(gamma_max, 1)

    return 2.0 / d_u + 2.0 / d_v - 2.0 + n_sq / (gamma_max * max(d_u, d_v))


def cnf_to_vcg_bfc_augmented(
    n_vars: int,
    clauses: List[List[int]],
    n_iterations: int = 50,
    p_random: float = 0.1,
    max_greedy_candidates: int = 64,
    seed: int = 0,
) -> nx.Graph:
    """Variable-Clause Graph rewired via stochastic discrete Ricci flow.

    Implements Algorithm 1 of Skenderi (2025) "On the Hardness of Learning
    GNN-based SAT Solvers: The Role of Graph Ricci Curvature".

    At each iteration the edge with the most negative Balanced Forman Curvature
    (BFC) is targeted.  Candidate new edges are formed from the cross-product
    S1(var_node) × S1(cls_node) (neighbors of each endpoint) that are not yet
    present.  Adding such an edge completes 4-cycles through the target edge,
    directly increasing its BFC and reducing the oversquashing bottleneck.

    Unlike degree-preserving edge swaps, this procedure **adds** edges; the
    graph gains connectivity in exactly the regions that are most bottlenecked.

    Parameters
    ----------
    n_vars : int
    clauses : list of lists of signed ints (DIMACS)
    n_iterations : int
        Number of rewiring steps (N in Algorithm 1).
    p_random : float
        Probability of picking a candidate edge uniformly at random instead of
        greedily (p in Algorithm 1).  p_random=1.0 is fully random; 0.0 is
        fully greedy.
    max_greedy_candidates : int
        When choosing greedily, evaluate at most this many candidates to bound
        runtime on dense graphs.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    nx.Graph
        VCG with additional edges that improve mean BFC.
    """
    G = cnf_to_vcg(n_vars, clauses)

    if n_iterations <= 0:
        return G

    rng = random.Random(seed)

    var_nodes = [f"var_{k}" for k in range(n_vars)]
    cls_nodes = [n for n in G.nodes() if str(n).startswith("cls_")]
    if not var_nodes or not cls_nodes:
        return G

    # Maintain adjacency as sets for O(1) membership tests.
    var_to_cls: dict[str, set] = {v: set(G.neighbors(v)) for v in var_nodes}
    cls_to_var: dict[str, set] = {c: set(G.neighbors(c)) for c in cls_nodes}

    # Compute initial BFC for every edge.
    edge_bfc: dict[tuple, float] = {}
    for u, v in G.edges():
        var_n, cls_n = (u, v) if str(u).startswith("var_") else (v, u)
        edge_bfc[(var_n, cls_n)] = _bfc_bipartite(var_n, cls_n, var_to_cls, cls_to_var)

    # Track which edges have exhausted their candidate pool so we skip them.
    exhausted: set[tuple] = set()

    for _ in range(n_iterations):
        # Select the most negatively curved non-exhausted edge.
        active = {e: b for e, b in edge_bfc.items() if e not in exhausted}
        if not active:
            break
        target_var, target_cls = min(active, key=active.get)

        s_u = var_to_cls[target_var]
        s_v = cls_to_var[target_cls]

        # Candidate new edges: (var_l, cls_k) with cls_k ∈ S1(target_var),
        # var_l ∈ S1(target_cls), edge not yet present.
        candidates: list[tuple[str, str]] = []
        for cls_k in s_u:
            if cls_k == target_cls:
                continue
            for var_l in s_v:
                if var_l == target_var:
                    continue
                if var_l not in cls_to_var[cls_k]:
                    candidates.append((var_l, cls_k))

        if not candidates:
            exhausted.add((target_var, target_cls))
            continue

        if len(candidates) > max_greedy_candidates:
            rng.shuffle(candidates)
            candidates = candidates[:max_greedy_candidates]

        if rng.random() < p_random:
            new_var, new_cls = rng.choice(candidates)
        else:
            # Greedy: pick the candidate that maximally improves the target
            # edge's own BFC (Algorithm 1 of Skenderi 2025).
            best_var, best_cls = candidates[0]
            best_improvement = -float("inf")
            old_bfc = edge_bfc[(target_var, target_cls)]
            for var_l, cls_k in candidates:
                cls_to_var[cls_k].add(var_l)
                var_to_cls[var_l].add(cls_k)
                improvement = _bfc_bipartite(target_var, target_cls, var_to_cls, cls_to_var) - old_bfc
                cls_to_var[cls_k].remove(var_l)
                var_to_cls[var_l].remove(cls_k)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_var, best_cls = var_l, cls_k
            new_var, new_cls = best_var, best_cls

        # Commit the new edge.
        G.add_edge(new_var, new_cls)
        var_to_cls[new_var].add(new_cls)
        cls_to_var[new_cls].add(new_var)

        # Recompute BFC for all edges that share an endpoint with the new edge,
        # since their degree and 4-cycle counts have changed.
        affected: set[tuple] = set()
        for cls_n in var_to_cls[new_var]:
            affected.add((new_var, cls_n))
        for var_n in cls_to_var[new_cls]:
            affected.add((var_n, new_cls))

        for var_n, cls_n in affected:
            edge_bfc[(var_n, cls_n)] = _bfc_bipartite(var_n, cls_n, var_to_cls, cls_to_var)
        # Also remove the new edge from exhausted (it's now in the graph).
        exhausted.discard((new_var, new_cls))

    return G


def cnf_to_lcg_bfc_augmented(
    n_vars: int,
    clauses: List[List[int]],
    n_iterations: int = 50,
    p_random: float = 0.5,
    max_greedy_candidates: int = 64,
    seed: int = 0,
) -> nx.Graph:
    """Literal-Clause Graph rewired via stochastic discrete Ricci flow.

    Same algorithm as ``cnf_to_vcg_bfc_augmented`` but operates on the LCG
    (the actual NSNet / AriENet encoding) instead of the VCG.

    LCG differences relevant to rewiring
    -------------------------------------
    - Literal nodes have degree = clause_occurrences + 1 (polarity bridge),
      so their ``2/d`` term is smaller and degree-increase penalties are lower.
    - The polarity bridge ``lit_{2k} – lit_{2k+1}`` is a neighbour of each
      literal, so the candidate set includes **literal–literal** edges in
      addition to literal–clause edges.  Adding a literal–literal edge creates
      a 4-cycle without touching any clause's degree, avoiding the dominant
      penalty that makes VCG rewiring unproductive on small 3-SAT instances.
    - Clause–clause pairs are excluded from candidates (they don't appear in
      the original graph and carry no structural meaning).

    Uses ``_bfc_general`` which handles the non-bipartite polarity bridge edges
    correctly (same formula as ``compute_ricci_curvature.compute_balanced_forman``).

    Parameters
    ----------
    n_vars : int
    clauses : list of lists of signed ints (DIMACS)
    n_iterations : int
        Number of rewiring steps.
    p_random : float
        Probability of random vs greedy candidate selection.
    max_greedy_candidates : int
        Max candidates evaluated per greedy step.
    seed : int

    Returns
    -------
    nx.Graph  – LCG with additional edges improving BFC bottlenecks.
    """
    G = cnf_to_lcg(n_vars, clauses)

    if n_iterations <= 0:
        return G

    rng = random.Random(seed)

    # Single adjacency dict for all nodes (handles non-bipartite polarity edges).
    adj: dict[str, set] = {v: set(G.neighbors(v)) for v in G.nodes()}

    # Canonical edge key: always (smaller, larger) string for deduplication.
    def _key(a: str, b: str) -> tuple:
        return (a, b) if a < b else (b, a)

    edge_bfc: dict[tuple, float] = {
        _key(u, v): _bfc_general(u, v, adj)
        for u, v in G.edges()
    }

    exhausted: set[tuple] = set()

    for _ in range(n_iterations):
        active = {e: b for e, b in edge_bfc.items() if e not in exhausted}
        if not active:
            break
        e_target = min(active, key=active.get)
        u, v = e_target

        # Candidates: (k, l) where k ∈ adj[u], l ∈ adj[v], k≠l, (k,l) ∉ E,
        # excluding clause–clause pairs.
        candidates: list[tuple[str, str]] = []
        for k in adj[u]:
            if k == v:
                continue
            for l in adj[v]:
                if l == u or l == k:
                    continue
                if str(k).startswith("cls_") and str(l).startswith("cls_"):
                    continue   # no clause–clause edges
                if l not in adj[k]:
                    candidates.append((k, l))

        if not candidates:
            exhausted.add(e_target)
            continue

        if len(candidates) > max_greedy_candidates:
            rng.shuffle(candidates)
            candidates = candidates[:max_greedy_candidates]

        if rng.random() < p_random:
            new_u, new_v = rng.choice(candidates)
        else:
            best_u, best_v = candidates[0]
            best_improvement = -float("inf")
            old_bfc = edge_bfc[e_target]
            for k, l in candidates:
                adj[k].add(l)
                adj[l].add(k)
                improvement = _bfc_general(u, v, adj) - old_bfc
                adj[k].remove(l)
                adj[l].remove(k)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_u, best_v = k, l
            new_u, new_v = best_u, best_v

        # Commit the new edge.
        G.add_edge(new_u, new_v)
        adj[new_u].add(new_v)
        adj[new_v].add(new_u)

        # Recompute BFC for all edges incident to either endpoint.
        for w in adj[new_u]:
            k = _key(new_u, w)
            edge_bfc[k] = _bfc_general(new_u, w, adj)
        for w in adj[new_v]:
            k = _key(new_v, w)
            edge_bfc[k] = _bfc_general(new_v, w, adj)
        exhausted.discard(_key(new_u, new_v))

    return G


# ---------------------------------------------------------------------------
# 3. LCG + Clause-Bridge Auxiliary Nodes
# ---------------------------------------------------------------------------

def cnf_to_lcg_clause_bridge(
    n_vars: int,
    clauses: List[List[int]],
    max_aux_nodes: int | None = None,
) -> nx.Graph:
    """LCG augmented with auxiliary nodes linking clauses at one-hop variable distance.

    Two clauses are at *one-hop variable distance* if they share no variable
    directly (not adjacent in the VCG), but there exists a *bridge clause*
    that shares at least one variable with each of them.  For every such pair
    an auxiliary node is inserted and connected to both clauses.

    Example
    -------
    C1 = {2, ¬3},  C2 = {3, 4},  C3 = {¬4, 5}
    - C1 ∩vars C2 = {3}  → adjacent in VCG → no aux node.
    - C2 ∩vars C3 = {4}  → adjacent in VCG → no aux node.
    - C1 ∩vars C3 = {}   → not adjacent, but C2 shares var 3 with C1 and var 4
                            with C3 → C2 is a bridge.
      ⟹  aux_0 is added, connected to cls_0 (C1) and cls_2 (C3).  C2 is NOT
          connected to aux_0.

    When there are more qualifying pairs than ``max_aux_nodes``, the pairs with
    the fewest bridge clauses are selected first (most bottlenecked connections).

    Parameters
    ----------
    n_vars  : int
    clauses : list of lists of signed ints (DIMACS)
    max_aux_nodes : int or None
        Maximum number of auxiliary nodes to add.  Defaults to ``len(clauses)``.
        Set to ``None`` for unlimited (can create very large graphs).

    Returns
    -------
    nx.Graph
    """
    G = cnf_to_lcg(n_vars, clauses)

    n_clauses = len(clauses)
    if n_clauses < 3:
        return G

    if max_aux_nodes is None:
        max_aux_nodes = n_clauses  # unlimited would create O(n²) nodes

    # Variable set per clause (polarity-independent).
    clause_vars: list[frozenset] = [
        frozenset(abs(lit) for lit in clause) for clause in clauses
    ]

    # Variable → clause indices.
    var_to_cls_idx: dict[int, set] = defaultdict(set)
    for j, vs in enumerate(clause_vars):
        for v in vs:
            var_to_cls_idx[v].add(j)

    # VCG-adjacent pairs as a set for O(1) lookup.
    # Pairs stored as (smaller_idx, larger_idx).
    vcg_adj: set[tuple[int, int]] = set()
    for j in range(n_clauses):
        for v in clause_vars[j]:
            for k in var_to_cls_idx[v]:
                if k > j:
                    vcg_adj.add((j, k))

    # VCG-neighbours of each clause (sorted for deterministic pair enumeration).
    vcg_nbrs: list[list[int]] = []
    for j in range(n_clauses):
        nbr: set[int] = set()
        for v in clause_vars[j]:
            nbr.update(var_to_cls_idx[v])
        nbr.discard(j)
        vcg_nbrs.append(sorted(nbr))

    # Count bridge clauses per qualifying pair.
    # A pair (ci, cj) qualifies if not VCG-adjacent but share ≥1 bridge.
    # ci < cj always because vcg_nbrs[b] is sorted and we iterate a < c.
    bridge_count: dict[tuple[int, int], int] = defaultdict(int)
    for b in range(n_clauses):
        nbrs = vcg_nbrs[b]
        m = len(nbrs)
        for a in range(m):
            ci = nbrs[a]
            for c in range(a + 1, m):
                cj = nbrs[c]
                if (ci, cj) not in vcg_adj:
                    bridge_count[(ci, cj)] += 1

    # Select the most bottlenecked pairs (fewest bridges = weakest connectivity).
    qualifying = sorted(bridge_count, key=bridge_count.__getitem__)
    if max_aux_nodes is not None:
        qualifying = qualifying[:max_aux_nodes]

    # Add one auxiliary node per selected pair.
    for idx, (ci, cj) in enumerate(qualifying):
        aux = f"aux_{idx}"
        G.add_node(aux, kind="clause_bridge")
        G.add_edge(aux, f"cls_{ci}")
        G.add_edge(aux, f"cls_{cj}")

    return G


# ---------------------------------------------------------------------------
# 4. LCG + Intra-clause literal co-occurrence edges
# ---------------------------------------------------------------------------

def cnf_to_lcg_cooccurrence(n_vars: int, clauses: List[List[int]]) -> nx.Graph:
    """LCG with added co-occurrence edges between literals in the same clause.

    For each clause, an edge is added between every pair of **actual** literals
    that appear in it (not their complements).  Together with the existing
    literal–clause edges, each clause of size k contributes k triangles of the
    form ``lit_a – lit_b – cls_j – lit_a``.  Triangles directly improve
    Balanced Forman Curvature via the ``n_tri`` terms in the Topping formula.

    The polarity bridge (``lit_{2k} – lit_{2k+1}``) already links each literal
    to its complement; co-occurrence edges only connect literals that co-occur
    in a clause, so they carry distinct structural information.

    Parameters
    ----------
    n_vars  : int
    clauses : list of lists of signed ints (DIMACS)

    Returns
    -------
    nx.Graph
    """
    G = cnf_to_lcg(n_vars, clauses)

    for clause in clauses:
        lits = [f"lit_{_literal_index(l)}" for l in clause]
        for i in range(len(lits)):
            for j in range(i + 1, len(lits)):
                if not G.has_edge(lits[i], lits[j]):
                    G.add_edge(lits[i], lits[j], kind="cooccurrence")

    return G


# ---------------------------------------------------------------------------
# 5. LCG + Clause Splitting via Dummy Variables (BFC-targeted)
# ---------------------------------------------------------------------------

def cnf_to_lcg_clause_split(
    n_vars: int,
    clauses: List[List[int]],
    n_splits: int = 10,
    seed: int = 0,
) -> nx.Graph:
    """LCG with BFC-targeted clause splitting via dummy variables.

    Approach
    --------
    1. Build the standard LCG.
    2. Compute the Balanced Forman Curvature (BFC) for every edge.
    3. Rank all literal–clause edges by BFC (most negative first).
    4. For each of the ``n_splits`` most negatively curved edges whose clause
       has not yet been split, split that clause by introducing a fresh dummy
       variable ``z``.  The original clause ``(l₁ ∨ … ∨lₖ)`` is replaced by
       two logically equivalent copies::

           (l₁ ∨ … ∨ lₖ ∨  z)   →  cls_<j>_a
           (l₁ ∨ … ∨ lₖ ∨ ¬z)   →  cls_<j>_b

       Both new clause nodes inherit **all** original literal–clause edges.
       A polarity bridge ``lit_z_pos – lit_z_neg`` is added for the dummy
       variable, which is standard in the LCG encoding.

    Why this reduces oversquashing
    --------------------------------
    The two new clause nodes ``cls_j_a`` and ``cls_j_b`` share every original
    literal neighbour, while each is exclusively connected to one polarity of
    the dummy variable.  Together with the polarity bridge they create new
    short cycles through the previously bottlenecked literal–clause edge,
    increasing its BFC and reducing the information bottleneck identified by
    Topping et al. (2022).

    Logical equivalence
    -------------------
    ``(C ∨ z) ∧ (C ∨ ¬z)`` is a tautology when z is a fresh variable, so
    when intersected with the rest of the formula the two clauses together
    are satisfiable iff the original clause ``C`` is satisfiable.  No
    spurious constraints are introduced.

    Parameters
    ----------
    n_vars : int
        Number of variables in the original CNF formula.
    clauses : list of lists of signed ints (DIMACS)
        Clause list of the original formula.
    n_splits : int
        Maximum number of clause splits to perform.  Each split targets the
        literal–clause edge with the most negative BFC whose clause has not
        yet been processed.
    seed : int
        Unused; reserved for future stochastic variants.

    Returns
    -------
    nx.Graph
        LCG with split clauses.  Original ``cls_<j>`` nodes are **replaced**
        by ``cls_<j>_a`` and ``cls_<j>_b`` for each split clause; unsplit
        clauses retain their ``cls_<j>`` name.  Dummy literal nodes are named
        ``lit_dummy_<d>_pos`` and ``lit_dummy_<d>_neg`` for split index ``d``.
    """
    G = cnf_to_lcg(n_vars, clauses)

    if n_splits <= 0:
        return G

    # ------------------------------------------------------------------
    # Step 1: Compute BFC for every edge using the full adjacency dict.
    # ------------------------------------------------------------------
    adj: dict[str, set] = {v: set(G.neighbors(v)) for v in G.nodes()}

    def _key(a: str, b: str) -> tuple:
        return (a, b) if a < b else (b, a)

    edge_bfc: dict[tuple, float] = {
        _key(u, v): _bfc_general(u, v, adj)
        for u, v in G.edges()
    }

    # ------------------------------------------------------------------
    # Step 2: Rank literal–clause edges by BFC (most negative first).
    # ------------------------------------------------------------------
    lit_cls_edges: list[tuple[tuple, float]] = []
    for (a, b), bfc_val in edge_bfc.items():
        a_is_lit = a.startswith("lit_") and not a.startswith("lit_dummy")
        b_is_lit = b.startswith("lit_") and not b.startswith("lit_dummy")
        a_is_cls = a.startswith("cls_")
        b_is_cls = b.startswith("cls_")
        if (a_is_lit and b_is_cls) or (a_is_cls and b_is_lit):
            lit_cls_edges.append(((a, b), bfc_val))

    lit_cls_edges.sort(key=lambda x: x[1])  # ascending = most negative first

    # ------------------------------------------------------------------
    # Step 3: Split up to n_splits unique clauses.
    # ------------------------------------------------------------------
    split_clauses: set[str] = set()
    dummy_index = 0
    # Map original clause node name → (clause_a_name, clause_b_name)
    split_map: dict[str, tuple[str, str]] = {}

    for (a, b), _ in lit_cls_edges:
        if dummy_index >= n_splits:
            break

        # Identify which endpoint is the clause node.
        cls_node = a if a.startswith("cls_") else b

        if cls_node in split_clauses:
            continue  # already split this clause

        # ------------------------------------------------------------------
        # Perform the split.
        # ------------------------------------------------------------------
        original_lits = list(adj[cls_node])  # all literal neighbours

        cls_a = f"{cls_node}_a"
        cls_b = f"{cls_node}_b"
        lit_pos = f"lit_dummy_{dummy_index}_pos"
        lit_neg = f"lit_dummy_{dummy_index}_neg"

        # Add the two replacement clause nodes.
        G.add_node(cls_a, kind="clause_split_a", original=cls_node)
        G.add_node(cls_b, kind="clause_split_b", original=cls_node)

        # Add dummy variable literal nodes + polarity bridge.
        G.add_node(lit_pos, kind="dummy_pos_literal", var=f"dummy_{dummy_index}")
        G.add_node(lit_neg, kind="dummy_neg_literal", var=f"dummy_{dummy_index}")
        G.add_edge(lit_pos, lit_neg, kind="polarity")

        # Connect both clause copies to all original literals.
        for lit_node in original_lits:
            G.add_edge(lit_node, cls_a)
            G.add_edge(lit_node, cls_b)

        # Connect each clause copy to one polarity of the dummy variable.
        G.add_edge(lit_pos, cls_a)
        G.add_edge(lit_neg, cls_b)

        # Remove the original clause node (it has been replaced by _a and _b).
        G.remove_node(cls_node)

        split_clauses.add(cls_node)
        split_map[cls_node] = (cls_a, cls_b)
        dummy_index += 1

    return G


# ---------------------------------------------------------------------------
# 6. WLIG – Weighted Literal-Incidence Graph
# ---------------------------------------------------------------------------

def cnf_to_wlig(n_vars: int, clauses: List[List[int]]) -> nx.Graph:
    """Weighted Literal-Incidence Graph.

    Two literals are connected by an edge if they co-occur in at least one
    clause.  The edge attribute ``weight`` counts how many clauses the pair
    appears in together.

    Nodes
    -----
    - ``lit_<i>``  (i = 0 … 2*n_vars-1) – same indexing as LCG

    Edges
    -----
    - (lit_<i>, lit_<j>) if literals i and j co-occur in ≥1 clause
    - edge attribute ``weight`` = co-occurrence count

    Note on self-loops
    ------------------
    A literal cannot co-occur with itself in the same position, so no
    self-loops are added.  If a literal appears twice in a clause (which
    is technically ill-formed DIMACS) the duplicate is ignored.

    Parameters
    ----------
    n_vars  : int
    clauses : list of lists of signed ints (DIMACS)

    Returns
    -------
    nx.Graph  (weighted)
    """
    G = nx.Graph()

    for k in range(n_vars):
        G.add_node(f"lit_{2*k}",   kind="pos_literal", var=k)
        G.add_node(f"lit_{2*k+1}", kind="neg_literal", var=k)

    # Count co-occurrences of literal pairs across clauses
    cooccur: defaultdict = defaultdict(int)
    for clause in clauses:
        # Deduplicate literals within the clause (preserving first appearance)
        lit_indices = list(dict.fromkeys(_literal_index(l) for l in clause))
        n = len(lit_indices)
        for a in range(n):
            for b in range(a + 1, n):
                i, j = lit_indices[a], lit_indices[b]
                key = (min(i, j), max(i, j))
                cooccur[key] += 1

    for (i, j), w in cooccur.items():
        G.add_edge(f"lit_{i}", f"lit_{j}", weight=float(w))

    return G

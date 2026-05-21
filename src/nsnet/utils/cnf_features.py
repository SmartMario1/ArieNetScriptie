"""CNF-level feature precomputation.

Computes per-literal features from raw CNF **before** graph construction.
All functions operate directly on (n_vars, clauses) and return torch tensors
that can be cached alongside the processed BPG files.

Literal indexing (consistent with BPGParamBuilder throughout this project)
--------------------------------------------------------------------------
    positive literal  v  →  index 2*(v-1)       (even)
    negative literal -v  →  index 2*(v-1)+1     (odd)

Available features
------------------
UP-based (Tier 1)  — compute_up_features()
    For each literal (variable v, polarity ±), runs unit propagation (UP)
    under the corresponding assumption via pysat and records four scalars:
        0  up_reach_pct      fraction of clauses satisfied by UP closure
        1  forced_count_pct  fraction of other variables forced by UP
        2  conflict_flag     1.0 iff UP derives a contradiction (failed literal)
        3  log_up_reach      log(1 + #clauses satisfied by UP closure)

Satisfaction-ratio (CNF-level refactor)  — compute_satisfaction_ratio_per_literal()
    For each literal L, the fraction of all clauses that *contain* L
    (immediate satisfaction count).  This is a fast, CNF-level approximation
    of the per-edge 2-hop neighbourhood ratio computed by BPGParamBuilder.
"""

from __future__ import annotations

import math
from typing import List

import torch


# ---------------------------------------------------------------------------
# Unit-propagation features
# ---------------------------------------------------------------------------

def compute_up_features(
    n_vars: int,
    clauses: List[List[int]],
) -> torch.Tensor:
    """Compute unit-propagation-based features for every literal.

    For each variable v ∈ [1..n_vars] and each polarity the SAT solver runs
    unit propagation under the corresponding literal assumption and records
    four scalar outcomes.

    Parameters
    ----------
    n_vars : int
        Number of variables (taken from the CNF header ``p cnf N M``).
    clauses : list[list[int]]
        Clauses as signed DIMACS integers, e.g. ``[[1, -3], [-2, 3]]``.

    Returns
    -------
    torch.Tensor of shape ``(2 * n_vars, 4)``, dtype ``float32``.

    Row layout
        ``2*(v-1)``   – features for the **positive** literal of variable v
                        (assumption: v = True, DIMACS literal ``+v``)
        ``2*(v-1)+1`` – features for the **negative** literal of variable v
                        (assumption: v = False, DIMACS literal ``-v``)

    Column layout
        0  ``up_reach_pct``     fraction of clauses satisfied by the UP closure
        1  ``forced_count_pct`` fraction of other variables assigned by UP
        2  ``conflict_flag``    1.0 iff UP derives a contradiction (failed literal)
        3  ``log_up_reach``     log(1 + number of clauses satisfied by UP closure)

    Notes
    -----
    Uses PySAT's Glucose3 C extension for fast propagation.  Requires
    ``python-sat`` (``pip install python-sat``).  All four features are zero
    for a variable that does not appear in any clause and is not forced by UP.
    """
    try:
        from pysat.solvers import Glucose3
    except ImportError as exc:
        raise ImportError(
            "pysat is required for UP features. "
            "Install with:  pip install python-sat"
        ) from exc

    n_clauses = len(clauses)
    features = torch.zeros(2 * n_vars, 4, dtype=torch.float32)

    if n_clauses == 0 or n_vars == 0:
        return features

    solver = Glucose3(bootstrap_with=clauses)
    try:
        for v in range(1, n_vars + 1):
            # offset 0 → positive literal (assume v=True,  DIMACS +v)
            # offset 1 → negative literal (assume v=False, DIMACS -v)
            for offset, dimacs_lit in enumerate([v, -v]):
                lit_idx = 2 * (v - 1) + offset

                status, propagated = solver.propagate(assumptions=[dimacs_lit])

                if not status:
                    # Failed literal: UP derives a contradiction under this assumption.
                    # The variable must take the *opposite* polarity.
                    features[lit_idx, 2] = 1.0
                else:
                    # `propagated` is the list of all literals that are forced True
                    # under the assumption (including the assumption itself).
                    true_lits: set[int] = set(propagated)

                    # Count clauses satisfied by the UP closure (at least one
                    # literal in the clause is in the forced-true set).
                    n_sat = sum(
                        1 for clause in clauses
                        if any(lit in true_lits for lit in clause)
                    )

                    up_reach_pct = n_sat / n_clauses
                    log_up_reach = math.log(1.0 + n_sat)

                    # Variables forced by UP, excluding the assumed variable itself.
                    forced_vars = {abs(l) for l in propagated if abs(l) != v}
                    forced_count_pct = len(forced_vars) / max(1, n_vars - 1)

                    features[lit_idx, 0] = up_reach_pct
                    features[lit_idx, 1] = forced_count_pct
                    # features[lit_idx, 2] stays 0.0 — no conflict
                    features[lit_idx, 3] = log_up_reach
    finally:
        solver.delete()

    return features


# ---------------------------------------------------------------------------
# CNF-level satisfaction ratio (refactored from BPGParamBuilder)
# ---------------------------------------------------------------------------

def compute_satisfaction_ratio_per_literal(
    n_vars: int,
    clauses: List[List[int]],
) -> torch.Tensor:
    """Compute a simple per-literal satisfaction ratio from the raw CNF.

    For each literal L this is the fraction of all clauses that *contain* L,
    i.e. the fraction of clauses that would be immediately satisfied if L were
    set to True.  This is a fast, CNF-level refactoring of the per-edge
    2-hop neighbourhood ratio computed by
    ``BPGParamBuilder.local_satisfaction_percentage()``.

    Parameters
    ----------
    n_vars : int
    clauses : list[list[int]]
        Clauses as signed DIMACS integers.

    Returns
    -------
    torch.Tensor of shape ``(2 * n_vars,)``, dtype ``float32``.
        Index ``2*(v-1)``   → positive literal v
        Index ``2*(v-1)+1`` → negative literal -v
    """
    n_clauses = len(clauses)
    counts = torch.zeros(2 * n_vars, dtype=torch.float32)

    if n_clauses == 0:
        return counts

    for clause in clauses:
        for lit in clause:
            v = abs(lit)
            if 1 <= v <= n_vars:
                idx = 2 * (v - 1) if lit > 0 else 2 * (v - 1) + 1
                counts[idx] += 1.0

    return counts / n_clauses

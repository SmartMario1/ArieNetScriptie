"""
Generate out-of-distribution (OOD) benchmark for k-SAT instances (k = 3, 4, 5).

This is a generalisation of generate_ood_benchmark.py that supports clause
widths beyond 3.  All three sweep modes from the original 3-SAT generator are
preserved:

  size   — Vary n_vars at the phase-transition clause ratio for the chosen k.
            Buckets are labelled  n<vars>/

  ratio  — Fix n_vars and sweep the clause-to-variable ratio α.
            Buckets are labelled  alpha<value>/

  community — Fix n_vars and α but generate instances with planted community
              structure in the Variable Incidence Graph.
              Buckets are labelled  comm<n_communities>_pin<p_in>/

Phase-transition ratios (empirical / belief-propagation):
  k=3  →  α* ≈ 4.267   (Mézard-Montanari + finite-size correction)
  k=4  →  α* ≈ 9.931
  k=5  →  α* ≈ 20.80

Usage examples:
    # 4-SAT, size sweep (default)
    python generate_ksat_ood_benchmark.py --k 4 --mode size

    # 5-SAT, ratio sweep: n=60 vars, α from 15 to 25
    python generate_ksat_ood_benchmark.py --k 5 --mode ratio \\
        --n_vars 60 --alpha_start 15.0 --alpha_step 1.0 --n_steps 11

    # 4-SAT, community sweep: n=80 vars, 4 communities
    python generate_ksat_ood_benchmark.py --k 4 --mode community \\
        --n_vars 80 --n_communities 4 --pin_start 0.4 --pin_step 0.1 --n_steps 6
"""

import os
import sys
import argparse
import random
import subprocess
import tempfile
import networkx as nx

from concurrent.futures import ProcessPoolExecutor
from cnfgen import RandomKCNF
from nsnet.utils.utils import write_dimacs_to

# ---------------------------------------------------------------------------
# Solver binary (march lookahead — fast on random k-SAT near phase transition)
# ---------------------------------------------------------------------------

MARCH_BINARY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'march_unmodified', 'march', 'march_nh',
)

# ---------------------------------------------------------------------------
# Phase-transition ratios for k-SAT
# k=3: Mézard-Montanari + Kirousis finite-size correction term
# k=4/5: best known empirical / cavity-method estimates
# ---------------------------------------------------------------------------

# Base ratio α*(k) at the phase transition
_PT_RATIO = {
    3: 4.267,
    4: 9.931,
    5: 20.80,
}

# Finite-size correction coefficient c(k) for the Mézard-Montanari formula
#   n_clauses = α*(k) * n_vars + c(k) * n_vars^(-2/3)
# Only the k=3 coefficient is well-established; for k=4/5 we use 0.
_PT_CORRECTION = {
    3: 58.26,
    4: 0.0,
    5: 0.0,
}


def _critical_n_clauses(k, n_vars):
    """Phase-transition clause count for k-SAT with n_vars variables."""
    alpha = _PT_RATIO[k]
    c = _PT_CORRECTION[k]
    return int(alpha * n_vars + c * pow(n_vars, -2 / 3.)) if c != 0.0 \
        else int(alpha * n_vars)


# ---------------------------------------------------------------------------
# SAT checking
# ---------------------------------------------------------------------------

def _solve_with_timeout(clauses, n_vars, timeout=10):
    """
    Run march on a temporary DIMACS file.  Returns True iff SAT within timeout.
    Falls back gracefully if the binary is missing (uses PySAT CaDiCaL).
    """
    if os.path.isfile(MARCH_BINARY):
        lines = [f'p cnf {n_vars} {len(clauses)}']
        for clause in clauses:
            lines.append(' '.join(map(str, clause)) + ' 0')
        dimacs = '\n'.join(lines)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.cnf', delete=False
        ) as f:
            f.write(dimacs)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [MARCH_BINARY, tmp_path],
                capture_output=True, text=True, timeout=timeout,
            )
            return (
                'SATISFIABLE' in result.stdout
                and 'UNSATISFIABLE' not in result.stdout
            )
        except subprocess.TimeoutExpired:
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        # Fallback: use PySAT
        try:
            from pysat.solvers import Cadical
            with Cadical(bootstrap_with=clauses) as s:
                return s.solve()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Clause samplers
# ---------------------------------------------------------------------------

def _random_sat_clauses(k, n_vars, n_clauses):
    """
    Return a list of satisfiable k-SAT clauses using cnfgen's uniform sampler.
    Retries until SAT (with a 10 s timeout for the SAT check).
    """
    while True:
        cnf = RandomKCNF(k, n_vars, n_clauses)
        clauses = [list(cnf._compress_clause(c)) for c in cnf.clauses()]
        if _solve_with_timeout(clauses, n_vars):
            return clauses


def _community_sat_clauses(k, n_vars, n_clauses, n_communities, p_in):
    """
    Return satisfiable k-SAT clauses whose VIG has planted community structure.

    Strategy:
      1. Build a planted-partition graph G via networkx with l=n_communities,
         k_per_comm = n_vars // n_communities nodes per group.
      2. Sample clauses from k-cliques rooted in graph edges:
           • Start with a random edge (u, v).
           • Greedily extend to a k-clique by adding variables from
             the combined neighbourhood of the current set (biases toward
             within-community groups when p_in is high).
           • If the neighbourhood is exhausted, fall back to random variables.
      3. Assign random polarities and retry until SAT + connected.
    """
    k_per_comm = n_vars // n_communities
    n_vars_used = k_per_comm * n_communities

    # p_out that preserves overall edge density at the given clause ratio
    total_pairs = n_vars_used * (n_vars_used - 1) / 2
    edges_per_clause = k * (k - 1) // 2          # C(k, 2)
    target_density = min(1.0, (n_clauses * edges_per_clause) / total_pairs)
    intra_frac = (k_per_comm - 1) / (n_vars_used - 1)
    inter_frac = (n_vars_used - k_per_comm) / (n_vars_used - 1)
    p_out = max(0.001, min(0.999,
        (target_density - p_in * intra_frac) / inter_frac))

    def _build_graph():
        return nx.generators.community.planted_partition_graph(
            n_communities, k_per_comm, p_in, p_out,
            seed=random.randint(0, 2 ** 31),
        )

    G = _build_graph()
    nodes = list(G.nodes())
    edges = list(G.edges())

    while True:
        clause_set = set()
        attempts = 0
        while len(clause_set) < n_clauses and attempts < n_clauses * 200:
            attempts += 1
            if not edges:
                break
            # Start with a random edge
            u, v = random.choice(edges)
            chosen = [u, v]
            # Extend to a k-tuple
            for _ in range(k - 2):
                nbrs = set()
                for node in chosen:
                    nbrs.update(G.neighbors(node))
                nbrs -= set(chosen)
                if not nbrs:
                    nbrs = set(nodes) - set(chosen)
                if not nbrs:
                    break
                chosen.append(random.choice(list(nbrs)))
            if len(chosen) < k:
                continue
            triple = tuple(sorted(chosen))
            if triple in clause_set:
                continue
            clause_set.add(triple)

        if len(clause_set) < n_clauses:
            G = _build_graph()
            edges = list(G.edges())
            clause_set = set()
            continue

        clauses = []
        for tpl in clause_set:
            clause = [((v + 1) if random.random() > 0.5 else -(v + 1))
                      for v in tpl]
            clauses.append(clause)

        if _solve_with_timeout(clauses, n_vars_used):
            return clauses, n_vars_used


# ---------------------------------------------------------------------------
# Per-instance worker functions (module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _generate_size_instance(args):
    out_path, k, n_vars = args
    n_clauses = _critical_n_clauses(k, n_vars)
    clauses = _random_sat_clauses(k, n_vars, n_clauses)
    write_dimacs_to(n_vars, clauses, out_path)
    return out_path


def _generate_ratio_instance(args):
    out_path, k, n_vars, alpha = args
    n_clauses = max(1, int(alpha * n_vars))
    clauses = _random_sat_clauses(k, n_vars, n_clauses)
    write_dimacs_to(n_vars, clauses, out_path)
    return out_path


def _generate_community_instance(args):
    out_path, k, n_vars, n_clauses, n_communities, p_in = args
    clauses, n_vars_actual = _community_sat_clauses(
        k, n_vars, n_clauses, n_communities, p_in
    )
    write_dimacs_to(n_vars_actual, clauses, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Bucket generators
# ---------------------------------------------------------------------------

def generate_size_bucket(size_dir, k, n_vars, n_problems, n_process):
    os.makedirs(size_dir, exist_ok=True)
    tasks = [
        (os.path.abspath(os.path.join(size_dir, '%.5d.cnf' % i)), k, n_vars)
        for i in range(n_problems)
    ]
    with ProcessPoolExecutor(max_workers=n_process) as pool:
        for i, _ in enumerate(pool.map(_generate_size_instance, tasks)):
            print(f'  [{i + 1}/{n_problems}]', end='\r')
    print(f'  Done: {n_problems} instances in {size_dir}')


def generate_ratio_bucket(size_dir, k, n_vars, alpha, n_problems, n_process):
    os.makedirs(size_dir, exist_ok=True)
    tasks = [
        (os.path.abspath(os.path.join(size_dir, '%.5d.cnf' % i)), k, n_vars, alpha)
        for i in range(n_problems)
    ]
    with ProcessPoolExecutor(max_workers=n_process) as pool:
        for i, _ in enumerate(pool.map(_generate_ratio_instance, tasks)):
            print(f'  [{i + 1}/{n_problems}]', end='\r')
    print(f'  Done: {n_problems} instances in {size_dir}')


def generate_community_bucket(size_dir, k, n_vars, n_clauses, n_communities,
                               p_in, n_problems, n_process):
    os.makedirs(size_dir, exist_ok=True)
    tasks = [
        (os.path.abspath(os.path.join(size_dir, '%.5d.cnf' % i)),
         k, n_vars, n_clauses, n_communities, p_in)
        for i in range(n_problems)
    ]
    with ProcessPoolExecutor(max_workers=n_process) as pool:
        for i, _ in enumerate(pool.map(_generate_community_instance, tasks)):
            print(f'  [{i + 1}/{n_problems}]', end='\r')
    print(f'  Done: {n_problems} instances in {size_dir}')


# ---------------------------------------------------------------------------
# Training / validation data generator (mirrors generate_3-sat_data.py)
# ---------------------------------------------------------------------------

def _generate_train_instance(args):
    """Worker: generate one training instance at a random n_vars in [min_n, max_n]."""
    out_path, k, min_n, max_n = args
    while True:
        n_vars = random.randint(min_n, max_n)
        n_clauses = _critical_n_clauses(k, n_vars)
        try:
            cnf = RandomKCNF(k, n_vars, n_clauses)
            clauses = [list(cnf._compress_clause(c)) for c in cnf.clauses()]
            vig = nx.Graph()
            vig.add_nodes_from(range(1, n_vars + 1))
            for clause in clauses:
                lits = [abs(l) for l in clause]
                for i in range(len(lits)):
                    for j in range(i + 1, len(lits)):
                        vig.add_edge(lits[i], lits[j])
            if not nx.is_connected(vig):
                continue
            if _solve_with_timeout(clauses, n_vars):
                write_dimacs_to(n_vars, clauses, out_path)
                return out_path
        except Exception:
            continue


def generate_train_split(out_dir, k, n_instances, min_n, max_n, n_process=8):
    """
    Generate a training (or validation) split of k-SAT instances at the
    phase-transition ratio, with n_vars drawn uniformly from [min_n, max_n].
    Equivalent to the 3-SAT generate_3-sat_data.py generator but for k-SAT.
    """
    os.makedirs(out_dir, exist_ok=True)
    tasks = [
        (os.path.abspath(os.path.join(out_dir, '%.5d.cnf' % i)), k, min_n, max_n)
        for i in range(n_instances)
    ]
    with ProcessPoolExecutor(max_workers=n_process) as pool:
        for i, _ in enumerate(pool.map(_generate_train_instance, tasks)):
            if (i + 1) % 100 == 0:
                print(f'  [{i + 1}/{n_instances}]', end='\r')
    print(f'\n  Done: {n_instances} training instances in {out_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            'Generate OOD k-SAT benchmark (k ∈ {3, 4, 5}), with size, ratio, '
            'or community sweep.  Also supports training-split generation.'
        )
    )
    parser.add_argument(
        '--k',
        type=int,
        default=4,
        choices=[3, 4, 5],
        help='Clause width k (default: 4)',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='size',
        choices=['size', 'ratio', 'community', 'train'],
        help='Sweep mode: size | ratio | community | train  (default: size)',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help=(
            'Root output directory.  '
            'Default: SATSolving/<k>-sat/ood_<mode>  (or train_first for train mode)'
        ),
    )
    parser.add_argument(
        '--n_problems',
        type=int,
        default=100,
        help='Number of problems per bucket (default: 100)',
    )
    parser.add_argument(
        '--n_process',
        type=int,
        default=8,
        help='Number of parallel worker processes (default: 8)',
    )

    # --- size mode ---
    parser.add_argument(
        '--start_n',
        type=int,
        default=30,
        help='[size] Starting number of variables (default: 30)',
    )
    parser.add_argument(
        '--step_size',
        type=int,
        default=20,
        help='[size] Variable increase per bucket (default: 20)',
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=10,
        help='Number of buckets / steps to generate (default: 10)',
    )

    # --- ratio mode ---
    parser.add_argument(
        '--n_vars',
        type=int,
        default=60,
        help='[ratio/community] Number of variables per instance (default: 60)',
    )
    parser.add_argument(
        '--alpha_start',
        type=float,
        default=None,
        help=(
            '[ratio] Starting α.  Default: 0.7 × α*(k)  '
            '(well below phase transition → easy SAT with large backbone)'
        ),
    )
    parser.add_argument(
        '--alpha_step',
        type=float,
        default=None,
        help='[ratio] α increment per bucket.  Default: 0.05 × α*(k)',
    )

    # --- community mode ---
    parser.add_argument(
        '--n_communities',
        type=int,
        default=4,
        help='[community] Number of planted communities (default: 4)',
    )
    parser.add_argument(
        '--pin_start',
        type=float,
        default=0.4,
        help='[community] Starting intra-community edge probability p_in (default: 0.4)',
    )
    parser.add_argument(
        '--pin_step',
        type=float,
        default=0.1,
        help='[community] p_in increment per bucket (default: 0.1)',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='[community] Clause-to-variable ratio α.  Default: α*(k)',
    )

    # --- train mode ---
    parser.add_argument(
        '--n_instances',
        type=int,
        default=5000,
        help='[train] Total number of training instances to generate (default: 5000)',
    )
    parser.add_argument(
        '--min_n',
        type=int,
        default=10,
        help='[train] Minimum number of variables (default: 10)',
    )
    parser.add_argument(
        '--max_n',
        type=int,
        default=60,
        help='[train] Maximum number of variables (default: 60)',
    )

    opts = parser.parse_args()

    k = opts.k
    alpha_star = _PT_RATIO[k]

    # Fill defaults that depend on k
    if opts.alpha_start is None:
        opts.alpha_start = round(0.7 * alpha_star, 3)
    if opts.alpha_step is None:
        opts.alpha_step = round(0.05 * alpha_star, 3)
    if opts.alpha is None:
        opts.alpha = alpha_star

    # Default output directory
    if opts.out_dir is None:
        if opts.mode == 'train':
            opts.out_dir = f'SATSolving/{k}-sat/train_first'
        else:
            opts.out_dir = f'SATSolving/{k}-sat/ood_{opts.mode}'

    print(f'{k}-SAT OOD Benchmark  (mode={opts.mode})')
    print(f'  α*(k={k})   : {alpha_star}')
    print(f'  Output dir  : {opts.out_dir}')
    print(f'  Workers     : {opts.n_process}')
    print()

    if opts.mode == 'size':
        sizes = [opts.start_n + i * opts.step_size for i in range(opts.n_steps)]
        print(f'  Sizes       : {sizes}')
        for n_vars in sizes:
            folder = os.path.join(opts.out_dir, f'n{n_vars}')
            n_cl = _critical_n_clauses(k, n_vars)
            print(f'Bucket n{n_vars}  (n_vars={n_vars}, n_clauses≈{n_cl})')
            generate_size_bucket(folder, k, n_vars, opts.n_problems, opts.n_process)

    elif opts.mode == 'ratio':
        alphas = [round(opts.alpha_start + i * opts.alpha_step, 4)
                  for i in range(opts.n_steps)]
        print(f'  n_vars      : {opts.n_vars}')
        print(f'  α values    : {alphas}')
        for alpha in alphas:
            folder = os.path.join(opts.out_dir, f'alpha{alpha:.4f}')
            print(f'Bucket alpha{alpha:.4f}  (n_vars={opts.n_vars}, α={alpha})')
            generate_ratio_bucket(folder, k, opts.n_vars, alpha,
                                  opts.n_problems, opts.n_process)

    elif opts.mode == 'community':
        pin_values = [round(opts.pin_start + i * opts.pin_step, 4)
                      for i in range(opts.n_steps)]
        n_clauses = max(1, int(opts.alpha * opts.n_vars))
        print(f'  n_vars      : {opts.n_vars}')
        print(f'  n_communities: {opts.n_communities}')
        print(f'  α           : {opts.alpha}  →  n_clauses≈{n_clauses}')
        print(f'  p_in values : {pin_values}')
        for p_in in pin_values:
            folder = os.path.join(
                opts.out_dir,
                f'comm{opts.n_communities}_pin{p_in:.4f}'
            )
            print(f'Bucket comm{opts.n_communities}_pin{p_in:.4f}')
            generate_community_bucket(
                folder, k, opts.n_vars, n_clauses,
                opts.n_communities, p_in, opts.n_problems, opts.n_process,
            )

    elif opts.mode == 'train':
        print(f'  n_instances : {opts.n_instances}')
        print(f'  n_vars range: [{opts.min_n}, {opts.max_n}]')
        generate_train_split(
            opts.out_dir, k, opts.n_instances,
            opts.min_n, opts.max_n, opts.n_process,
        )

    print('\nAll buckets generated.')


if __name__ == '__main__':
    main()

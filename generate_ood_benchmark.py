"""
Generate out-of-distribution (OOD) benchmark for 3-SAT instances.

Supports three sweep modes (selectable via --mode):

  size   — Vary the number of variables while keeping the clause ratio at the
            Mézard-Montanari phase-transition value:
                n_clauses = 4.258 * n_vars + 58.26 * n_vars^(-2/3)
            Buckets are labelled  n<vars>/

  ratio  — Fix the number of variables and sweep the clause-to-variable ratio α
            from --alpha_start to --alpha_start + (--n_steps-1)*--alpha_step.
            Instances near the phase transition (α ≈ 4.267) are hardest; going
            below (α ≈ 3.5) produces easy SAT with large backbones.
            Buckets are labelled  alpha<value>/

  community — Fix n_vars and α but generate instances whose Variable Incidence
              Graph (VIG) has planted community structure.  Variables are
              divided into --n_communities groups; clauses are sampled so that
              a fraction --p_in of variable-pairs within each clause come from
              the same community (controls intra-community edge density).
              High --p_in → high VIG modularity → low Balanced Forman Curvature
              → stress-test for oversquashing.
              Buckets are labelled  comm<n_communities>_pin<p_in>/

Usage examples:
    # size sweep (default)
    python generate_ood_benchmark.py --mode size

    # ratio sweep: n=100 vars, α from 3.0 to 4.5 in 0.25 steps
    python generate_ood_benchmark.py --mode ratio --n_vars 100 \\
        --alpha_start 3.0 --alpha_step 0.25 --n_steps 7

    # community sweep: n=100 vars, 5 communities, p_in from 0.4 to 0.9
    python generate_ood_benchmark.py --mode community --n_vars 100 \\
        --n_communities 5 --pin_start 0.4 --pin_step 0.1 --n_steps 6
"""

import os
import sys
import argparse
import random
import itertools
import subprocess
import tempfile
import networkx as nx

from concurrent.futures import ProcessPoolExecutor
from cnfgen import RandomKCNF
from nsnet.utils.utils import write_dimacs_to

# Path to the unmodified march lookahead solver (vastly better than CDCL on
# random 3-SAT — lookahead propagation dominates CDCL at these densities).
MARCH_BINARY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'march_unmodified', 'march', 'march_nh',
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _critical_n_clauses(n_vars):
    """Mézard-Montanari phase-transition clause count."""
    return int(4.258 * n_vars + 58.26 * pow(n_vars, -2 / 3.))


def _solve_with_timeout(clauses, n_vars, timeout=10):
    """
    Run the march lookahead solver on a temporary DIMACS file with a hard
    wall-clock timeout.  Returns True if SAT, False otherwise (UNSAT/timeout).
    Using march instead of a CDCL solver (Glucose/CaDiCaL) because march's
    lookahead propagation is far faster on random 3-SAT near the phase
    transition — the domain where we need to filter instances most often.
    """
    # Write a minimal DIMACS file
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
            capture_output=True, text=True, timeout=timeout
        )
        return 'SATISFIABLE' in result.stdout and 'UNSATISFIABLE' not in result.stdout
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _random_sat_clauses(n_vars, n_clauses):
    """
    Return a list of satisfiable 3-SAT clauses using cnfgen's uniform random
    sampler.  Retries until SAT.  march is used for the SAT check with a 10s
    timeout so hard UNSAT instances are abandoned quickly.
    """
    while True:
        cnf = RandomKCNF(3, n_vars, n_clauses)
        clauses = [list(cnf._compress_clause(c)) for c in cnf.clauses()]
        if _solve_with_timeout(clauses, n_vars):
            return clauses


def _community_sat_clauses(n_vars, n_clauses, n_communities, p_in):
    """
    Return a list of satisfiable, connected 3-SAT clauses whose VIG has
    planted community structure.

    Strategy (library-based):
      1. Build a planted-partition graph G via
         networkx.generators.community.planted_partition_graph with l=n_communities,
         k=n_vars//n_communities, p_in=p_in, p_out=p_out.  p_out is chosen so
         that the overall expected edge density matches a random graph at the
         given clause-to-variable ratio.
      2. Sample clauses by:
           a. Pick a random edge (u, v) from G.
           b. Pick a third variable w uniformly at random from G's neighbours of
              u or v (biases toward within-community triples when p_in is high).
           c. Assign random polarities.
      3. Retry until SAT + connected.
    """
    # Partition variables 0..n_vars-1 into n_communities groups
    k = n_vars // n_communities          # variables per community
    n_vars_used = k * n_communities      # may be slightly less than n_vars

    # Estimate p_out to preserve rough overall edge density.
    # In a random 3-SAT VIG with n_clauses clauses each contributing C(3,2)=3
    # edges (with duplicates), the expected number of distinct edges is roughly
    # min(n_clauses * 3, C(n_vars, 2)).  We target the same density.
    total_pairs = n_vars_used * (n_vars_used - 1) / 2
    target_density = min(1.0, (n_clauses * 3) / total_pairs)
    # p_out such that: p_in * (k-1)/(n_vars_used-1) + p_out * (n_vars_used-k)/(n_vars_used-1) = target_density
    intra_frac = (k - 1) / (n_vars_used - 1)
    inter_frac = (n_vars_used - k) / (n_vars_used - 1)
    p_out = max(0.001, min(0.999, (target_density - p_in * intra_frac) / inter_frac))

    # Build planted-partition graph (variables are 0-indexed nodes)
    G = nx.generators.community.planted_partition_graph(
        n_communities, k, p_in, p_out, seed=random.randint(0, 2**31)
    )
    # Map node IDs to 1-indexed variable numbers
    nodes = list(G.nodes())
    edges = list(G.edges())

    while True:
        # Sample clauses from the graph structure
        clause_set = set()
        attempts = 0
        while len(clause_set) < n_clauses and attempts < n_clauses * 200:
            attempts += 1
            if not edges:
                break
            u, v = random.choice(edges)
            # Pick third variable from neighbours of u or v for locality
            candidates = list(set(G.neighbors(u)) | set(G.neighbors(v)))
            candidates = [w for w in candidates if w != u and w != v]
            if not candidates:
                # fall back to any variable
                candidates = [w for w in nodes if w != u and w != v]
            if not candidates:
                continue
            w = random.choice(candidates)
            triple = tuple(sorted([u, v, w]))
            if triple in clause_set:
                continue
            clause_set.add(triple)

        if len(clause_set) < n_clauses:
            # Could not fill; regenerate graph and retry
            G = nx.generators.community.planted_partition_graph(
                n_communities, k, p_in, p_out, seed=random.randint(0, 2**31)
            )
            edges = list(G.edges())
            clause_set = set()
            continue

        # Assign random polarities; variables are 1-indexed in DIMACS
        clauses = []
        for triple in clause_set:
            clause = [((v + 1) if random.random() > 0.5 else -(v + 1)) for v in triple]
            clauses.append(clause)

        # Ensure SAT (10s cap so hard UNSAT instances are abandoned quickly)
        if _solve_with_timeout(clauses, n_vars_used):
            return clauses, n_vars_used


# ---------------------------------------------------------------------------
# Per-instance worker functions (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _generate_size_instance(args):
    """Worker: generate one size-sweep instance."""
    out_path, n_vars, alpha = args
    n_clauses = int(alpha * n_vars + 58.26 * pow(n_vars, -2 / 3.)) if alpha is None \
        else int(alpha * n_vars)
    # Use the Mézard-Montanari formula when alpha is None (size mode)
    if alpha is None:
        n_clauses = _critical_n_clauses(n_vars)
    clauses = _random_sat_clauses(n_vars, n_clauses)
    write_dimacs_to(n_vars, clauses, out_path)
    return out_path


def generate_instance(args):
    """Generate a single satisfiable 3-SAT instance and write it to disk.
    Kept for backward compatibility (size sweep at phase-transition ratio)."""
    out_path, n_vars = args
    n_clauses = _critical_n_clauses(n_vars)
    clauses = _random_sat_clauses(n_vars, n_clauses)
    write_dimacs_to(n_vars, clauses, out_path)
    return out_path


def _generate_ratio_instance(args):
    """Worker: generate one ratio-sweep instance."""
    out_path, n_vars, alpha = args
    n_clauses = max(1, int(alpha * n_vars))
    clauses = _random_sat_clauses(n_vars, n_clauses)
    write_dimacs_to(n_vars, clauses, out_path)
    return out_path


def _generate_community_instance(args):
    """Worker: generate one community-sweep instance."""
    out_path, n_vars, n_clauses, n_communities, p_in = args
    clauses, n_vars_actual = _community_sat_clauses(n_vars, n_clauses, n_communities, p_in)
    write_dimacs_to(n_vars_actual, clauses, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Bucket generators
# ---------------------------------------------------------------------------

def generate_size_bucket(size_dir, n_vars, n_problems, n_process):
    """Generate all problems for a single size bucket."""
    os.makedirs(size_dir, exist_ok=True)
    tasks = [
        (os.path.abspath(os.path.join(size_dir, '%.5d.cnf' % i)), n_vars)
        for i in range(n_problems)
    ]
    with ProcessPoolExecutor(max_workers=n_process) as pool:
        for i, _ in enumerate(pool.map(generate_instance, tasks)):
            print(f'  [{i + 1}/{n_problems}]', end='\r')
    print(f'  Done: {n_problems} instances in {size_dir}')


def generate_ratio_bucket(size_dir, n_vars, alpha, n_problems, n_process):
    """Generate all problems for a single ratio bucket."""
    os.makedirs(size_dir, exist_ok=True)
    tasks = [
        (os.path.abspath(os.path.join(size_dir, '%.5d.cnf' % i)), n_vars, alpha)
        for i in range(n_problems)
    ]
    with ProcessPoolExecutor(max_workers=n_process) as pool:
        for i, _ in enumerate(pool.map(_generate_ratio_instance, tasks)):
            print(f'  [{i + 1}/{n_problems}]', end='\r')
    print(f'  Done: {n_problems} instances in {size_dir}')


def generate_community_bucket(size_dir, n_vars, n_clauses, n_communities, p_in,
                               n_problems, n_process):
    """Generate all problems for a single community bucket."""
    os.makedirs(size_dir, exist_ok=True)
    tasks = [
        (os.path.abspath(os.path.join(size_dir, '%.5d.cnf' % i)),
         n_vars, n_clauses, n_communities, p_in)
        for i in range(n_problems)
    ]
    with ProcessPoolExecutor(max_workers=n_process) as pool:
        for i, _ in enumerate(pool.map(_generate_community_instance, tasks)):
            print(f'  [{i + 1}/{n_problems}]', end='\r')
    print(f'  Done: {n_problems} instances in {size_dir}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate OOD 3-SAT benchmark (size, ratio, or community sweep).'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='size',
        choices=['size', 'ratio', 'community'],
        help='Sweep mode: size | ratio | community  (default: size)',
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='SATSolving/3-sat/ood_test',
        help='Root output directory (default: SATSolving/3-sat/ood_test)',
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
        default=50,
        help='[size] Starting number of variables (default: 50)',
    )
    parser.add_argument(
        '--step_size',
        type=int,
        default=50,
        help='[size] Variable increase per bucket (default: 50)',
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=10,
        help='Number of buckets to generate (default: 10)',
    )

    # --- ratio mode ---
    parser.add_argument(
        '--n_vars',
        type=int,
        default=100,
        help='[ratio/community] Number of variables per instance (default: 100)',
    )
    parser.add_argument(
        '--alpha_start',
        type=float,
        default=3.0,
        help='[ratio] Starting clause-to-variable ratio α (default: 3.0)',
    )
    parser.add_argument(
        '--alpha_step',
        type=float,
        default=0.2,
        help='[ratio] α increment per bucket (default: 0.2)',
    )

    # --- community mode ---
    parser.add_argument(
        '--n_communities',
        type=int,
        default=5,
        help='[community] Number of planted communities (default: 5)',
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
        default=4.258,
        help='[community] Clause-to-variable ratio α (default: 4.258, phase transition)',
    )

    opts = parser.parse_args()

    print(f'OOD Benchmark generation  (mode={opts.mode})')
    print(f'  Output dir : {opts.out_dir}')
    print(f'  Problems   : {opts.n_problems} per bucket')
    print(f'  Workers    : {opts.n_process}')
    print()

    if opts.mode == 'size':
        sizes = [opts.start_n + i * opts.step_size for i in range(opts.n_steps)]
        print(f'  Sizes      : {sizes}')
        for n_vars in sizes:
            folder_name = f'n{n_vars}'
            size_dir = os.path.join(opts.out_dir, folder_name)
            print(f'Generating bucket: {folder_name}  (n_vars={n_vars})')
            generate_size_bucket(size_dir, n_vars, opts.n_problems, opts.n_process)

    elif opts.mode == 'ratio':
        alphas = [round(opts.alpha_start + i * opts.alpha_step, 4)
                  for i in range(opts.n_steps)]
        print(f'  n_vars     : {opts.n_vars}')
        print(f'  α values   : {alphas}')
        for alpha in alphas:
            folder_name = f'alpha{alpha:.4f}'
            size_dir = os.path.join(opts.out_dir, folder_name)
            print(f'Generating bucket: {folder_name}  (n_vars={opts.n_vars}, α={alpha})')
            generate_ratio_bucket(size_dir, opts.n_vars, alpha,
                                  opts.n_problems, opts.n_process)

    elif opts.mode == 'community':
        pin_values = [round(opts.pin_start + i * opts.pin_step, 4)
                      for i in range(opts.n_steps)]
        n_clauses = max(1, int(opts.alpha * (opts.n_vars // opts.n_communities) * opts.n_communities))
        print(f'  n_vars     : {opts.n_vars}')
        print(f'  n_communities : {opts.n_communities}')
        print(f'  α          : {opts.alpha}  →  n_clauses ≈ {n_clauses}')
        print(f'  p_in values: {pin_values}')
        for p_in in pin_values:
            folder_name = f'comm{opts.n_communities}_pin{p_in:.4f}'
            size_dir = os.path.join(opts.out_dir, folder_name)
            print(f'Generating bucket: {folder_name}  '
                  f'(n_communities={opts.n_communities}, p_in={p_in})')
            generate_community_bucket(size_dir, opts.n_vars, n_clauses,
                                       opts.n_communities, p_in,
                                       opts.n_problems, opts.n_process)

    print('\nAll buckets generated.')


if __name__ == '__main__':
    main()

"""
Generate backbone labels for SAT instances.

A literal is in the backbone if it must be true in ALL satisfying assignments.
This script computes backbone for CNF files and saves them as a pickle file.

Usage:
    python generate_backbone_labels.py <data_dir> [--out_dir OUTPUT_DIR] [--n_process N]
"""

import os
import sys
import argparse
import glob
import pickle
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from nsnet.utils.utils import parse_cnf_file


def compute_backbone_pysat(cnf_path, timeout=30):
    """
    Compute backbone using PySAT library (if available).
    
    Returns:
        backbone: dict mapping var -> True/False/None (None = not in backbone)
        success: bool indicating if computation succeeded
    """
    try:
        from pysat.solvers import Solver
        from pysat.formula import CNF
    except ImportError:
        print("[ERROR] PySAT not installed. Install with: pip install python-sat")
        return None, False
    
    try:
        # Parse CNF
        n_vars, clauses = parse_cnf_file(cnf_path)
        
        if n_vars == 0 or len(clauses) == 0:
            return {}, True
        
        # Check if satisfiable first
        with Solver(name='cadical', bootstrap_with=clauses) as solver:
            if not solver.solve():
                # UNSAT - no backbone
                return None, False
        
        # Compute backbone by testing each variable
        backbone = {}
        
        for var in range(1, n_vars + 1):
            # Test if var=True is forced
            with Solver(name='cadical', bootstrap_with=clauses) as solver:
                # Add constraint: var must be False
                solver.add_clause([-var])
                if not solver.solve():
                    # UNSAT when var=False, so var must be True in all solutions
                    backbone[var] = True
                    continue
            
            # Test if var=False is forced
            with Solver(name='cadical', bootstrap_with=clauses) as solver:
                # Add constraint: var must be True
                solver.add_clause([var])
                if not solver.solve():
                    # UNSAT when var=True, so var must be False in all solutions
                    backbone[var] = False
                    continue
            
            # Variable is free (not in backbone)
            backbone[var] = None
        
        return backbone, True
        
    except Exception as e:
        print(f"[ERROR] Failed to compute backbone for {cnf_path}: {e}")
        return None, False


def compute_backbone_external_solver(cnf_path, timeout=30):
    """
    Compute backbone using external solver (neuroback's kissat or march).
    
    This is a fallback if PySAT is not available.
    """
    import subprocess
    import tempfile
    
    # Check if kissat solver exists
    kissat_path = os.path.join(os.path.dirname(__file__), '..', 'neuroback', 'solver', 'build', 'kissat')
    
    if not os.path.exists(kissat_path):
        print(f"[WARNING] Kissat solver not found at {kissat_path}")
        return None, False
    
    try:
        # Parse CNF to get n_vars
        n_vars, clauses = parse_cnf_file(cnf_path)
        
        if n_vars == 0 or len(clauses) == 0:
            return {}, True
        
        # Run kissat with backbone option (if supported)
        # This is a simplified version - you may need to adapt based on solver capabilities
        result = subprocess.run(
            [kissat_path, cnf_path, '--backbone'],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Parse backbone from output
        backbone = {}
        for line in result.stdout.split('\n'):
            if line.startswith('v '):
                # Parse backbone literals
                parts = line.split()[1:]
                for lit_str in parts:
                    lit = int(lit_str)
                    if lit == 0:
                        break
                    var = abs(lit)
                    backbone[var] = (lit > 0)
        
        return backbone, True
        
    except subprocess.TimeoutExpired:
        print(f"[WARNING] Timeout computing backbone for {cnf_path}")
        return None, False
    except Exception as e:
        print(f"[ERROR] Failed to compute backbone for {cnf_path}: {e}")
        return None, False


def process_single_file(args):
    """Process a single CNF file and return its backbone."""
    cnf_path, use_pysat, timeout = args
    
    if use_pysat:
        backbone, success = compute_backbone_pysat(cnf_path, timeout)
    else:
        backbone, success = compute_backbone_external_solver(cnf_path, timeout)
    
    if not success:
        return cnf_path, None
    
    return cnf_path, backbone


def main():
    parser = argparse.ArgumentParser(description='Generate backbone labels for SAT instances')
    parser.add_argument('data_dir', type=str, help='Directory with CNF files')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: same as data_dir)')
    parser.add_argument('--n_process', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout per file in seconds')
    parser.add_argument('--use_pysat', action='store_true', help='Use PySAT library (default: try PySAT first)')
    
    args = parser.parse_args()
    
    # Find all CNF files
    all_files = sorted(glob.glob(os.path.join(args.data_dir, '**/*.cnf'), recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]
    
    print(f'Found {len(all_files)} CNF files in {args.data_dir}')
    
    if len(all_files) == 0:
        print('[ERROR] No CNF files found!')
        return
    
    # Determine output directory
    out_dir = args.out_dir if args.out_dir else args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    
    # Check if PySAT is available
    use_pysat = args.use_pysat
    if use_pysat:
        try:
            import pysat
            print("Using PySAT for backbone computation")
        except ImportError:
            print("[WARNING] PySAT not available, falling back to external solver")
            use_pysat = False
    
    # Process files in parallel
    print(f'Computing backbone for {len(all_files)} files using {args.n_process} processes...')
    
    task_args = [(f, use_pysat, args.timeout) for f in all_files]
    
    backbones = {}
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=args.n_process) as executor:
        results = list(tqdm(executor.map(process_single_file, task_args), total=len(all_files)))
    
    # Process results
    for cnf_path, backbone in results:
        if backbone is None:
            failed_files.append(cnf_path)
        else:
            backbones[cnf_path] = backbone
    
    print(f'\nSuccessfully computed backbone for {len(backbones)}/{len(all_files)} files')
    print(f'Failed: {len(failed_files)} files')
    
    if len(failed_files) > 0:
        print(f'Failed files: {failed_files[:10]}...')
    
    # Save backbones as pickle file
    output_file = os.path.join(out_dir, 'backbones.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(backbones, f)
    
    print(f'\nBackbone labels saved to: {output_file}')
    
    # Also save as separate .backbone files (neuroback format) for compatibility
    backbone_dir = os.path.join(out_dir, 'backbone_files')
    os.makedirs(backbone_dir, exist_ok=True)
    
    for cnf_path, backbone in backbones.items():
        basename = os.path.basename(cnf_path)
        backbone_file = os.path.join(backbone_dir, basename + '.backbone')
        
        with open(backbone_file, 'w') as f:
            f.write('c Backbone literals\n')
            for var, value in backbone.items():
                if value is not None:  # Only write backbone literals
                    lit = var if value else -var
                    f.write(f'{lit} 0\n')
    
    print(f'Individual backbone files saved to: {backbone_dir}')


if __name__ == '__main__':
    main()

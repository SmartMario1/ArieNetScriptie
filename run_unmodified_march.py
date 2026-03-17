"""
Run unmodified march solver on test instances.
This script runs the standard march solver without any neural network guidance
and produces the same output format as run_model_weighted.py for comparison.
"""
import os
import argparse
import glob
import time
import subprocess
import tempfile
from pathlib import Path


# Path to unmodified march solver (relative to nsnet directory)
MARCH_UNMODIFIED_PATH = "./march_unmodified/march/march_nh"


def load_dimacs_cnf(filepath):
    """Load a CNF formula from a DIMACS file."""
    clauses = []
    num_vars = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('c') or line.startswith('%') or line.startswith('0'):
                continue
            elif line.startswith('p'):
                parts = line.split()
                num_vars = int(parts[2])
            else:
                clause = [int(x) for x in line.split() if int(x) != 0]
                if clause:
                    clauses.append(clause)
    
    return clauses, num_vars


def solve_cnf_unmodified(filepath, seed=1, timeout=5000, debug=False):
    """
    Solve CNF with unmodified march solver.
    
    Args:
        filepath: Path to CNF file
        seed: Random seed (may not be supported by march)
        timeout: Timeout in seconds
        debug: If True, print solver stdout/stderr
        
    Returns:
        Dictionary with solver statistics
    """
    try:
        call = [MARCH_UNMODIFIED_PATH, filepath]
        # Note: unmodified march may not support seed parameter
        # If it does, uncomment:
        # if seed is not None:
        #     call.append(f"-seed={seed}")
        
        result = subprocess.run(
            call,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        return {'Result': 'TIMEOUT', 'decisions': -1, 'conflicts': -1, 
                'propagations': -1, 'restarts': -1, 'CPU time': timeout}
    
    # Debug output
    if debug:
        print(f"\n=== SOLVER OUTPUT ===")
        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")
        print(f"Return code: {result.returncode}")
        print(f"==================\n")
    
    # Parse solver output
    stats = parse_march_output(stdout)
    
    return stats


def parse_march_output(stdout):
    """Parse march solver output to extract statistics."""
    stats = {}
    
    for line in stdout.splitlines():
        line = line.strip()
        
        # Look for march output patterns
        if 'SATISFIABLE' in line.upper() and 'UNSATISFIABLE' not in line.upper():
            stats['Result'] = 'SATISFIABLE'
        elif 'UNSATISFIABLE' in line.upper():
            stats['Result'] = 'UNSATISFIABLE'
        
        # March output patterns - adapt based on actual output
        # Common patterns to look for:
        if 'decisions' in line.lower() and ':' in line:
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    # Extract number from the part after ':'
                    num_str = parts[1].strip().split()[0]
                    stats['decisions'] = int(num_str)
                except (ValueError, IndexError):
                    pass
        
        if 'conflicts' in line.lower() and ':' in line:
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    num_str = parts[1].strip().split()[0]
                    stats['conflicts'] = int(num_str)
                except (ValueError, IndexError):
                    pass
        
        if 'propagations' in line.lower() and ':' in line:
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    num_str = parts[1].strip().split()[0]
                    stats['propagations'] = int(num_str)
                except (ValueError, IndexError):
                    pass
        
        if 'cpu time' in line.lower() or 'time' in line.lower():
            if ':' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    try:
                        time_str = parts[1].strip().split()[0]
                        stats['CPU time'] = float(time_str)
                    except (ValueError, IndexError):
                        pass
    
    # Set default result if not found
    if 'Result' not in stats:
        stats['Result'] = 'UNKNOWN'
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Run unmodified march solver on test instances'
    )
    parser.add_argument('test_dir', type=str, 
                       help='Directory with testing data (CNF files)')
    parser.add_argument('--timeout', type=int, default=5000,
                       help='Solver timeout in seconds (default: 5000)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (may not be used by unmodified march)')
    parser.add_argument('--debug', action='store_true',
                       help='Print solver output for first 3 instances')
    parser.add_argument('--output_file', type=str, default=None,
                       help='File to save detailed results (optional)')
    
    args = parser.parse_args()
    
    # Find all CNF files
    cnf_patterns = ['*.cnf', '*.cnf.gz', '*.dimacs']
    all_files = []
    for pattern in cnf_patterns:
        all_files.extend(glob.glob(os.path.join(args.test_dir, pattern)))
    
    all_files = sorted(all_files)
    
    if len(all_files) == 0:
        print(f"No CNF files found in {args.test_dir}")
        return 1
    
    print('='*80)
    print('RUNNING UNMODIFIED MARCH SOLVER')
    print('='*80)
    print(f'Test directory: {args.test_dir}')
    print(f'Timeout: {args.timeout}s')
    print(f'Total files to process: {len(all_files)}')
    print('='*80)
    
    # Solve all instances
    print('\nSolving instances...')
    t0 = time.time()
    
    all_results = []
    solved_count = 0
    unsat_count = 0
    timeout_count = 0
    unknown_count = 0
    total_decisions = 0
    total_conflicts = 0
    valid_decision_count = 0
    valid_conflict_count = 0
    
    for idx, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        
        # Load CNF to get metadata
        try:
            clauses, num_vars = load_dimacs_cnf(filepath)
        except Exception as e:
            print(f"  Warning: Could not parse {filename}: {e}")
            continue
        
        # Solve with unmodified march
        debug_this = args.debug and idx < 3
        stats = solve_cnf_unmodified(filepath, seed=args.seed, 
                                     timeout=args.timeout, debug=debug_this)
        
        stats['filename'] = filename
        stats['filepath'] = filepath
        stats['num_vars'] = num_vars
        stats['num_clauses'] = len(clauses)
        
        all_results.append(stats)
        
        result = stats.get('Result', 'UNKNOWN')
        if result == 'SATISFIABLE':
            solved_count += 1
        elif result == 'UNSATISFIABLE':
            unsat_count += 1
        elif result == 'TIMEOUT':
            timeout_count += 1
        else:
            unknown_count += 1
        
        if 'decisions' in stats and stats['decisions'] > 0:
            total_decisions += stats['decisions']
            valid_decision_count += 1
        if 'conflicts' in stats and stats['conflicts'] > 0:
            total_conflicts += stats['conflicts']
            valid_conflict_count += 1
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f'  Processed {idx + 1}/{len(all_files)} files...')
    
    t = time.time() - t0
    
    # Print summary statistics (same format as run_model_weighted.py)
    print('\n' + '='*60)
    print('SUMMARY STATISTICS')
    print('='*60)
    print(f'Total files: {len(all_files)}')
    print(f'SATISFIABLE: {solved_count} ({100*solved_count/len(all_files):.1f}%)')
    print(f'UNSATISFIABLE: {unsat_count} ({100*unsat_count/len(all_files):.1f}%)')
    print(f'TIMEOUT: {timeout_count} ({100*timeout_count/len(all_files):.1f}%)')
    print(f'UNKNOWN: {unknown_count} ({100*unknown_count/len(all_files):.1f}%)')
    print(f'\nInstances with valid decision count: {valid_decision_count}')
    if valid_decision_count > 0:
        print(f'Average decisions (valid only): {total_decisions/valid_decision_count:.1f}')
    else:
        print(f'Average decisions (valid only): N/A')
    print(f'\nInstances with valid conflict count: {valid_conflict_count}')
    if valid_conflict_count > 0:
        print(f'Average conflicts (valid only): {total_conflicts/valid_conflict_count:.1f}')
    else:
        print(f'Average conflicts (valid only): N/A')
    print(f'\nTotal time: {t:.2f}s')
    print(f'Time per instance: {t/len(all_files):.2f}s')
    print('='*60)
    
    # Save detailed results if requested
    if args.output_file:
        import json
        output_data = {
            'test_dir': args.test_dir,
            'timeout': args.timeout,
            'total_files': len(all_files),
            'total_time': t,
            'summary': {
                'satisfiable': solved_count,
                'unsatisfiable': unsat_count,
                'timeout': timeout_count,
                'unknown': unknown_count,
                'valid_decision_count': valid_decision_count,
                'valid_conflict_count': valid_conflict_count,
                'avg_decisions': total_decisions/valid_decision_count if valid_decision_count > 0 else None,
                'avg_conflicts': total_conflicts/valid_conflict_count if valid_conflict_count > 0 else None,
            },
            'results': all_results
        }
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f'\nDetailed results saved to: {args.output_file}')
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

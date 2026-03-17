"""
Run nsnet model with weighted predictions for guided SAT solving.
This script generates predictions with confidence-based weights and uses
the modified glucose solver from RLAF for guided solving.
"""
import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
import random
import glob
import time
import subprocess
import tempfile
from pathlib import Path

from src.nsnet.utils.options import add_model_options
from src.nsnet.utils.dataloader import get_dataloader
from src.nsnet.models.bp import BP
from src.nsnet.models.nsnet import NSNet
from src.nsnet.models.neurosat import NeuroSAT
from src.nsnet.models.arienet import ArieNet
from torch_geometric.loader import DataLoader
from itertools import accumulate
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))


# Paths to modified solver binaries (relative to nsnet directory)
GLUCOSE_WEIGHTED_PATH = "../RLAF/solvers/glucose_weighted/simp/glucose_static"
MARCH_WEIGHTED_PATH = "./march_weighted/march_nh"


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


def cnf_to_dimacs(clauses, num_vars, var_params=None):
    """
    Convert CNF formula to DIMACS string with optional variable parameterization.
    
    Args:
        clauses: List of clauses (each clause is a list of signed integers)
        num_vars: Number of variables
        var_params: np.ndarray of shape [num_vars, 2] where:
                    var_params[i, 0] = polarity (phase)
                    var_params[i, 1] = weight
    
    Returns:
        DIMACS format string
    """
    num_clauses = len(clauses)
    lines = [f"p cnf {num_vars} {num_clauses}"]
    
    if var_params is not None:
        assert num_vars == var_params.shape[0], f"{num_vars} != {var_params.shape[0]}"
        
        params = ["c weight"]
        for i in range(num_vars):
            weight = float(var_params[i, 1])
            sgn = 1 if var_params[i, 0] > 0 else -1
            params.append(f"{sgn * weight:.4f}")
        lines.append(" ".join(params))
    
    for clause in clauses:
        clause_line = " ".join(map(str, clause)) + " 0"
        lines.append(clause_line)
    
    return "\n".join(lines)


def solve_cnf_weighted(clauses, num_vars, var_params, seed=1, timeout=5000, solver='glucose', debug=False):
    """
    Solve CNF with weighted variable guidance using modified solver.
    
    Args:
        clauses: List of clauses
        num_vars: Number of variables
        var_params: Variable parameterization [num_vars, 2]
        seed: Random seed
        timeout: Timeout in seconds
        solver: Solver to use ('glucose' or 'march')
        debug: If True, print solver stdout/stderr
        
    Returns:
        Dictionary with solver statistics
    """
    dimacs_str = cnf_to_dimacs(clauses, num_vars, var_params)
    
    # Build command line based on solver type
    if solver == 'glucose':
        call = [GLUCOSE_WEIGHTED_PATH]
        if seed is not None:
            assert seed > 0
            call.append(f"-rnd-seed={seed}")
        
        # Glucose reads from stdin
        with tempfile.TemporaryFile(mode='w+') as tmp_file:
            tmp_file.write(dimacs_str)
            tmp_file.seek(0)
            
            try:
                result = subprocess.run(
                    call,
                    capture_output=True,
                    text=True,
                    stdin=tmp_file,
                    timeout=timeout
                )
                stdout = result.stdout
                stderr = result.stderr
            except subprocess.TimeoutExpired:
                return {'Result': 'TIMEOUT', 'decisions': -1, 'conflicts': -1, 
                        'propagations': -1, 'restarts': -1, 'CPU time': timeout}
                
    elif solver == 'march':
        # March needs file as command-line argument
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as tmp_file:
            tmp_file.write(dimacs_str)
            tmp_file_path = tmp_file.name
        
        try:
            call = [MARCH_WEIGHTED_PATH, tmp_file_path]
            if seed is not None:
                assert seed > 0
                call.append(f"-seed={seed}")
            
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
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    # Debug output
    if debug:
        print(f"\n=== SOLVER OUTPUT ===")
        print(f"STDOUT:\n{stdout}")
        print(f"STDERR:\n{stderr}")
        print(f"Return code: {result.returncode}")
        print(f"==================\n")
    
    # Parse solver output based on solver type
    if solver == 'glucose':
        stats = parse_glucose_output(stdout)
    elif solver == 'march':
        stats = parse_march_output(stdout)
    else:
        stats = {}
    
    return stats


def parse_glucose_output(stdout):
    """Parse glucose solver output to extract statistics."""
    stats = {}
    
    for line in stdout.splitlines():
        line = line.strip()
        
        # Look for standard glucose output patterns
        if line.startswith('c decisions'):
            parts = line.split(':')
            if len(parts) > 1:
                stats['decisions'] = int(parts[1].strip().split()[0])
        elif line.startswith('c conflicts'):
            parts = line.split(':')
            if len(parts) > 1:
                stats['conflicts'] = int(parts[1].strip().split()[0])
        elif line.startswith('c propagations'):
            parts = line.split(':')
            if len(parts) > 1:
                stats['propagations'] = int(parts[1].strip().split()[0])
        elif line.startswith('c restarts'):
            parts = line.split(':')
            if len(parts) > 1:
                stats['restarts'] = int(parts[1].strip().split()[0])
        elif line.startswith('c CPU time'):
            parts = line.split(':')
            if len(parts) > 1:
                stats['CPU time'] = float(parts[1].strip().split()[0])
        elif line.startswith('s'):
            stats['Result'] = line.split()[1]
    
    return stats


def parse_march_output(stdout):
    """Parse march solver output to extract statistics."""
    stats = {}
    
    for line in stdout.splitlines():
        line = line.strip()
        
        # Look for march output patterns
        if 'SATISFIABLE' in line.upper():
            stats['Result'] = 'SATISFIABLE'
        elif 'UNSATISFIABLE' in line.upper():
            stats['Result'] = 'UNSATISFIABLE'
        
        # March may report statistics differently
        # Adapt based on actual march_weighted output format
        if line.startswith('c decisions'):
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    stats['decisions'] = int(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        elif line.startswith('c conflicts'):
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    stats['conflicts'] = int(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        elif line.startswith('c propagations'):
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    stats['propagations'] = int(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        elif 'time' in line.lower() and ':' in line:
            parts = line.split(':')
            if len(parts) > 1:
                try:
                    # Try to extract time value
                    time_str = parts[1].strip().split()[0]
                    stats['CPU time'] = float(time_str)
                except (ValueError, IndexError):
                    pass
    
    # Set defaults if not found
    if 'Result' not in stats:
        stats['Result'] = 'UNKNOWN'
    if 'decisions' not in stats:
        stats['decisions'] = -1
    if 'conflicts' not in stats:
        stats['conflicts'] = -1
    if 'propagations' not in stats:
        stats['propagations'] = -1
    if 'CPU time' not in stats:
        stats['CPU time'] = -1
    
    return stats


def predictions_to_var_params(v_probs, weight_scale=1.0, weight_range=(1.0, 2.0), constant_weight=None, random_guidance=False):
    """
    Convert model predictions to variable parameterization with confidence-based weights.
    
    Args:
        v_probs: Model output [num_vars, 2] with probabilities for negative and positive literals
        weight_scale: Scale factor for weights
        weight_range: Tuple (min_weight, max_weight) for weight clipping
        constant_weight: If not None, use this constant value for all weights
        random_guidance: If True, use random phase and weights (ignores v_probs)
        
    Returns:
        var_params: np.ndarray [num_vars, 2] where:
                    var_params[:, 0] = phase (0 or 1)
                    var_params[:, 1] = weight (confidence)
    """
    # v_probs shape: [num_vars, 2] where [:, 0] is negative literal, [:, 1] is positive literal
    v_probs_np = v_probs.cpu().numpy()
    num_vars = v_probs_np.shape[0]
    
    if random_guidance:
        # Random phase and random weights
        phase = np.random.randint(0, 2, size=num_vars).astype(float)
        weight = np.random.uniform(weight_range[0], weight_range[1], size=num_vars)
    else:
        neg_probs = v_probs_np[:, 0]  # P(x_i = 0)
        pos_probs = v_probs_np[:, 1]  # P(x_i = 1)
        
        # Phase: choose the more likely polarity
        phase = (pos_probs > neg_probs).astype(float)
        
        if constant_weight is not None:
            # Use constant weight for all variables
            weight = np.full(num_vars, constant_weight)
        else:
            # Weight: based on confidence (how far from 0.5 the prediction is)
            # Higher confidence (closer to 0 or 1) gets higher weight
            max_prob = np.maximum(neg_probs, pos_probs)
            
            # Map confidence to weight range
            # max_prob in [0.5, 1.0] maps to [min_weight, max_weight]
            confidence = (max_prob - 0.5) * 2.0  # Map [0.5, 1.0] to [0, 1]
            weight = weight_range[0] + confidence * (weight_range[1] - weight_range[0])
            weight = weight * weight_scale
            
            # Clip to ensure within range
            weight = np.clip(weight, weight_range[0], weight_range[1])
    
    var_params = np.stack([phase, weight], axis=1)
    return var_params


def main():
    parser = argparse.ArgumentParser(
        description='Run nsnet model with weighted predictions for guided SAT solving'
    )
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint to load')
    parser.add_argument('--solver', type=str, choices=['glucose', 'march'], default='glucose',
                       help='Solver to use (glucose or march)')
    parser.add_argument('--glucose_path', type=str, default=None, 
                       help='Path to modified glucose_weighted binary')
    parser.add_argument('--march_path', type=str, default=None,
                       help='Path to modified march_weighted binary')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (reduce for large instances)')
    parser.add_argument('--no_pin_memory', action='store_true',
                       help='Disable pin_memory (helps with CUDA OOM on large instances)')
    parser.add_argument('--weight_scale', type=float, default=1.0, help='Scale factor for weights')
    parser.add_argument('--min_weight', type=float, default=1.0, help='Minimum weight value')
    parser.add_argument('--max_weight', type=float, default=2.0, help='Maximum weight value')
    parser.add_argument('--constant_weight', type=float, default=None,
                       help='If set, use this constant weight for all variables (ignores other weight params)')
    parser.add_argument('--random_guidance', action='store_true',
                       help='Use random phase and weights instead of model predictions')
    parser.add_argument('--timeout', type=int, default=5000, help='Solver timeout in seconds')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--debug', action='store_true',
                       help='Print solver output for first 3 instances')
    add_model_options(parser)
    
    opts = parser.parse_args()
    opts.task = 'sat-solving'
    
    # Update solver paths if provided
    global GLUCOSE_WEIGHTED_PATH, MARCH_WEIGHTED_PATH
    if opts.glucose_path is not None:
        GLUCOSE_WEIGHTED_PATH = opts.glucose_path
    if opts.march_path is not None:
        MARCH_WEIGHTED_PATH = opts.march_path
    
    # Set random seeds
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)
    
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)
    
    # Adjust for memory constraints
    opts.pin_memory = not opts.no_pin_memory
    if opts.no_pin_memory:
        print("\n[WARNING] pin_memory disabled - may be slower but uses less memory")
    
    # Load model
    models = {
        'BP': BP,
        'NSNet': NSNet,
        'NeuroSAT': NeuroSAT,
        'ArieNet': ArieNet,
    }
    
    # Import backbone model if needed
    if opts.model == 'Backbone':
        from train_arienet_backbone import ArieNetBackbone
        models['Backbone'] = ArieNetBackbone
    elif opts.model == 'BackboneCanonical':
        from train_arienet_backbone import ArieNetBackbone
        models['BackboneCanonical'] = ArieNetBackbone
    
    # Handle different model initialization
    if opts.model in ['Backbone', 'BackboneCanonical']:
        # Backbone model doesn't use opts for initialization
        use_subgraph = (opts.model == 'BackboneCanonical')
        model = models[opts.model](device=opts.device, use_subgraph_features=use_subgraph, subgraph_dim=32)
        model.to(opts.device)
        
        print(f'Loading model checkpoint from {opts.checkpoint}...')
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        else:
            checkpoint = torch.load(opts.checkpoint)
        
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
    else:
        # Standard models use opts
        model = models[opts.model](opts)
        model.to(opts.device)
        
        print(f'Loading model checkpoint from {opts.checkpoint}...')
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        else:
            checkpoint = torch.load(opts.checkpoint)
        
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()
    
    # Get test data
    if opts.model in ['Backbone', 'BackboneCanonical']:
        # Backbone model uses BPG format
        if opts.model == 'BackboneCanonical':
            from train_arienet_backbone_canonical import CanonicalBackboneBPGDataset
            test_dataset = CanonicalBackboneBPGDataset(opts.test_dir, compute_subgraphs=True)
        else:
            from train_arienet_backbone import BackboneBPGDataset
            test_dataset = BackboneBPGDataset(opts.test_dir)
        test_loader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False, 
                                num_workers=opts.num_workers, pin_memory=opts.pin_memory)
        all_files = test_dataset.cnf_files
    else:
        test_loader = get_dataloader(opts.test_dir, opts, 'test')
        all_files = sorted(glob.glob(opts.test_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]
    
    print(f'\nProcessing {len(all_files)} CNF files...')
    print(f'Solver: {opts.solver}')
    print(f'Model: {opts.model}')
    if opts.random_guidance:
        print(f'Using RANDOM guidance')
    elif opts.constant_weight is not None:
        print(f'Using constant weight: {opts.constant_weight}')
    else:
        print(f'Weight range: [{opts.min_weight}, {opts.max_weight}]')
        print(f'Weight scale: {opts.weight_scale}')
    
    t0 = time.time()
    all_results = []
    
    # Generate predictions
    print('\n1. Generating predictions with GNN...')
    predictions_by_file = []
    
    i = 0
    for data in test_loader:
        data = data.to(opts.device)
        with torch.no_grad():
            if opts.model in ['Backbone', 'BackboneCanonical']:
                # Backbone model outputs raw logits, need to apply softmax
                v_logits = model(data)  # [total_vars_in_batch, 2]
                v_probs = F.softmax(v_logits, dim=1)
            else:
                v_probs = model(data)  # [total_vars_in_batch, 2]
            
            # Get the number of variables per graph in the batch
            if opts.model in ['Backbone', 'BackboneCanonical']:
                # Backbone model uses BPG format with n_literals
                if isinstance(data.n_literals, torch.Tensor):
                    v_sizes = (data.n_literals / 2).int().tolist()
                elif isinstance(data.n_literals, list):
                    v_sizes = [int(n / 2) for n in data.n_literals]
                else:
                    v_sizes = [int(data.n_literals / 2)]
            else:
                # l_size is the number of literals (2 * num_vars), so divide by 2 to get num_vars
                if hasattr(data, 'l_size'):
                    # l_size should be a tensor with one value per graph in batch
                    if isinstance(data.l_size, torch.Tensor):
                        v_sizes = (data.l_size / 2).int().tolist()
                    else:
                        # Single graph case
                        v_sizes = [int(data.l_size / 2)]
                elif hasattr(data, 'n_literals'):
                    # Alternative attribute name used by some models
                    if isinstance(data.n_literals, torch.Tensor):
                        v_sizes = (data.n_literals / 2).int().tolist()
                    elif isinstance(data.n_literals, list):
                        v_sizes = [int(n / 2) for n in data.n_literals]
                    else:
                        v_sizes = [int(data.n_literals / 2)]
                else:
                    raise AttributeError(f"Data object has no l_size or n_literals attribute. Available attributes: {dir(data)}")
            
            # Split predictions by file using cumulative offsets
            offset = 0
            for v_size in v_sizes:
                file_probs = v_probs[offset:offset+v_size]
                predictions_by_file.append(file_probs)
                offset += v_size
                i += 1
    
    assert i == len(all_files), f"Mismatch: {i} predictions vs {len(all_files)} files"
    
    # Solve with weighted guidance
    print('\n2. Solving with weighted guidance...')
    solved_count = 0
    unsat_count = 0
    timeout_count = 0
    unknown_count = 0
    total_decisions = 0
    total_conflicts = 0
    valid_decision_count = 0
    valid_conflict_count = 0
    
    for idx, (filepath, v_probs) in enumerate(zip(all_files, predictions_by_file)):
        filename = os.path.basename(filepath)
        
        # Load original CNF
        clauses, num_vars = load_dimacs_cnf(filepath)
        
        # Convert predictions to var_params
        var_params = predictions_to_var_params(
            v_probs,
            weight_scale=opts.weight_scale,
            weight_range=(opts.min_weight, opts.max_weight),
            constant_weight=opts.constant_weight,
            random_guidance=opts.random_guidance
        )
        
        # Solve with weighted guidance
        debug_this = opts.debug and idx < 3
        stats = solve_cnf_weighted(clauses, num_vars, var_params, 
                                   seed=opts.seed, timeout=opts.timeout, 
                                   solver=opts.solver, debug=debug_this)
        
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
    
    # Print summary statistics
    print('\n' + '='*60)
    print('SUMMARY STATISTICS')
    print('='*60)
    print(f'Total files: {len(all_files)}')
    print(f'SATISFIABLE: {solved_count} ({100*solved_count/len(all_files):.1f}%)')
    print(f'UNSATISFIABLE: {unsat_count} ({100*unsat_count/len(all_files):.1f}%)')
    print(f'TIMEOUT: {timeout_count} ({100*timeout_count/len(all_files):.1f}%)')
    print(f'UNKNOWN: {unknown_count} ({100*unknown_count/len(all_files):.1f}%)')
    print(f'\\nInstances with valid decision count: {valid_decision_count}')
    if valid_decision_count > 0:
        print(f'Average decisions (valid only): {total_decisions/valid_decision_count:.1f}')
    else:
        print(f'Average decisions: N/A (no valid data)')
    print(f'\\nInstances with valid conflict count: {valid_conflict_count}')
    if valid_conflict_count > 0:
        print(f'Average conflicts (valid only): {total_conflicts/valid_conflict_count:.1f}')
    else:
        print(f'Average conflicts: N/A (no valid data)')
    print(f'\\nTotal time: {t:.2f}s')
    print(f'Time per instance: {t/len(all_files):.2f}s')
    print('='*60)



if __name__ == '__main__':
    main()

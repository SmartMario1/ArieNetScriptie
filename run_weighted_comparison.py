"""
Run comparison experiments with ArieNet, NSNet, and Backbone models.
This script executes various runs testing weighted and constant weight modes,
as well as random guidance as a baseline.
"""
import subprocess
import os
import sys
import json
from datetime import datetime
from pathlib import Path


def run_experiment(test_dir, checkpoint, model_name, use_weights, solver='glucose',
                   timeout=5000, seed=0, debug=False, batch_size=32, num_workers=4,
                   no_pin_memory=False, random_guidance=False, additional_args=None):
    """
    Run a single experiment configuration.
    
    Args:
        test_dir: Directory with test data
        checkpoint: Path to model checkpoint (can be None for random guidance)
        model_name: Name of the model (ArieNet, NSNet, or Backbone)
        use_weights: If True, use confidence-based weights; if False, use constant weight 1.0
        solver: Solver to use (glucose or march)
        timeout: Solver timeout in seconds
        seed: Random seed
        random_guidance: If True, use random phase and weights
        additional_args: List of additional command-line arguments
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if random_guidance:
        weight_mode = "random"
    else:
        weight_mode = "weighted" if use_weights else "constant"
    
    cmd = [
        sys.executable,  # Use current Python interpreter
        "run_model_weighted.py",
        test_dir,
        "--checkpoint", checkpoint if checkpoint else "dummy",  # Dummy for random guidance
        "--model", model_name,
        "--solver", solver,
        "--timeout", str(timeout),
        "--seed", str(seed),
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers)
    ]
    
    if random_guidance:
        cmd.append("--random_guidance")
    elif not use_weights:
        cmd.extend(["--constant_weight", "1.0"])
    
    if debug:
        cmd.append("--debug")
    
    if no_pin_memory:
        cmd.append("--no_pin_memory")
    
    if additional_args:
        cmd.extend(additional_args)
    
    print("\n" + "="*80)
    if random_guidance:
        print(f"Running: {model_name} - Random Guidance")
    else:
        print(f"Running: {model_name} - {'Weighted' if use_weights else 'Constant Weight (1.0)'}")
    print(f"Command: {' '.join(cmd)}")
    print("="*80 + "\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.returncode, result.stdout, result.stderr


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run comparison experiments with weighted and constant weight modes'
    )
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('--arienet_checkpoint', type=str, required=True,
                       help='Path to ArieNet checkpoint')
    parser.add_argument('--nsnet_checkpoint', type=str, required=True,
                       help='Path to NSNet checkpoint')
    parser.add_argument('--backbone_checkpoint', type=str, default=None,
                       help='Path to Backbone checkpoint (optional)')
    parser.add_argument('--backbone_canonical_checkpoint', type=str, default=None,
                       help='Path to Backbone with Canonical Subgraph Features checkpoint (optional)')
    parser.add_argument('--solver', type=str, choices=['glucose', 'march'], default='glucose',
                       help='Solver to use')
    parser.add_argument('--timeout', type=int, default=5000,
                       help='Solver timeout in seconds')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--log_file', type=str, default=None,
                       help='File to save experiment logs')
    parser.add_argument('--debug', action='store_true',
                       help='Print solver output for first 3 instances of each experiment')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (reduce for large instances)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--no_pin_memory', action='store_true',
                       help='Disable pin_memory (helps with CUDA OOM on large instances)')
    parser.add_argument('--additional_args', nargs='*', default=[],
                       help='Additional arguments to pass to run_model_weighted.py')
    
    args = parser.parse_args()
    
    # Prepare log file
    if args.log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"comparison_log_{timestamp}.json"
    
    # List of experiments to run (order: Backbone -> Backbone Canonical -> ArieNet -> NSNet)
    experiments = []
    
    # Add backbone experiments if checkpoint provided
    if args.backbone_checkpoint:
        experiments.extend([
            ('Backbone', args.backbone_checkpoint, True, False),   # Backbone with weights
            ('Backbone', args.backbone_checkpoint, False, False),  # Backbone constant weight
            ('Backbone', args.backbone_checkpoint, False, True),   # Backbone random guidance
        ])
    
    # Add canonical backbone experiments if checkpoint provided
    if args.backbone_canonical_checkpoint:
        experiments.extend([
            ('BackboneCanonical', args.backbone_canonical_checkpoint, True, False),   # Backbone Canonical with weights
            ('BackboneCanonical', args.backbone_canonical_checkpoint, False, False),  # Backbone Canonical constant weight
            ('BackboneCanonical', args.backbone_canonical_checkpoint, False, True),   # Backbone Canonical random guidance
        ])
    
    # Add ArieNet experiments
    experiments.extend([
        ('ArieNet', args.arienet_checkpoint, True, False),   # ArieNet with weights
        ('ArieNet', args.arienet_checkpoint, False, False),  # ArieNet constant weight
        ('ArieNet', args.arienet_checkpoint, False, True),   # ArieNet random guidance
    ])
    
    # Add NSNet experiments
    experiments.extend([
        ('NSNet', args.nsnet_checkpoint, True, False),       # NSNet with weights
        ('NSNet', args.nsnet_checkpoint, False, False),      # NSNet constant weight
    ])
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'test_dir': args.test_dir,
        'solver': args.solver,
        'timeout': args.timeout,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': not args.no_pin_memory,
        'experiments': []
    }
    
    print("\n" + "="*80)
    print("WEIGHTED SOLVER COMPARISON EXPERIMENTS")
    print("="*80)
    print(f"Test directory: {args.test_dir}")
    print(f"Solver: {args.solver}")
    print(f"Timeout: {args.timeout}s")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Pin memory: {not args.no_pin_memory}")
    print(f"ArieNet checkpoint: {args.arienet_checkpoint}")
    print(f"NSNet checkpoint: {args.nsnet_checkpoint}")
    if args.backbone_checkpoint:
        print(f"Backbone checkpoint: {args.backbone_checkpoint}")
    if args.backbone_canonical_checkpoint:
        print(f"Backbone Canonical checkpoint: {args.backbone_canonical_checkpoint}")
    print(f"Total experiments: {len(experiments)}")
    print("="*80)
    
    # Run all experiments
    for i, (model_name, checkpoint, use_weights, random_guidance) in enumerate(experiments, 1):
        print(f"\n\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(experiments)}")
        print(f"{'='*80}")
        
        return_code, stdout, stderr = run_experiment(
            test_dir=args.test_dir,
            checkpoint=checkpoint,
            model_name=model_name,
            use_weights=use_weights,
            random_guidance=random_guidance,
            solver=args.solver,
            timeout=args.timeout,
            seed=args.seed,
            debug=args.debug,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            no_pin_memory=args.no_pin_memory,
            additional_args=args.additional_args
        )
        
        # Print output
        print(stdout)
        if stderr:
            print("STDERR:", stderr, file=sys.stderr)
        
        # Record results
        experiment_result = {
            'experiment_number': i,
            'model': model_name,
            'weight_mode': 'random' if random_guidance else ('weighted' if use_weights else 'constant'),
            'checkpoint': checkpoint,
            'return_code': return_code,
            'success': return_code == 0
        }
        
        # Try to extract summary statistics from stdout
        if 'SUMMARY STATISTICS' in stdout:
            lines = stdout.split('\n')
            for j, line in enumerate(lines):
                if 'Total files:' in line:
                    try:
                        experiment_result['total_files'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('SATISFIABLE:'):
                    try:
                        parts = line.split(':')[1].strip().split('(')
                        experiment_result['satisfiable'] = int(parts[0].strip())
                        experiment_result['satisfiable_pct'] = parts[1].strip('%)').strip()
                    except:
                        pass
                elif line.startswith('UNSATISFIABLE:'):
                    try:
                        parts = line.split(':')[1].strip().split('(')
                        experiment_result['unsatisfiable'] = int(parts[0].strip())
                        experiment_result['unsatisfiable_pct'] = parts[1].strip('%)').strip()
                    except:
                        pass
                elif line.startswith('TIMEOUT:'):
                    try:
                        parts = line.split(':')[1].strip().split('(')
                        experiment_result['timeout'] = int(parts[0].strip())
                        experiment_result['timeout_pct'] = parts[1].strip('%)').strip()
                    except:
                        pass
                elif line.startswith('UNKNOWN:'):
                    try:
                        parts = line.split(':')[1].strip().split('(')
                        experiment_result['unknown'] = int(parts[0].strip())
                        experiment_result['unknown_pct'] = parts[1].strip('%)').strip()
                    except:
                        pass
                elif 'Instances with valid decision count:' in line:
                    try:
                        experiment_result['valid_decision_count'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Average decisions (valid only):' in line:
                    try:
                        avg_str = line.split(':')[1].strip()
                        if 'N/A' not in avg_str:
                            avg_decisions = float(avg_str)
                            experiment_result['avg_decisions'] = avg_decisions
                            # Calculate total decisions
                            if 'valid_decision_count' in experiment_result:
                                experiment_result['total_decisions'] = int(avg_decisions * experiment_result['valid_decision_count'])
                    except:
                        pass
                elif 'Instances with valid conflict count:' in line:
                    try:
                        experiment_result['valid_conflict_count'] = int(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Average conflicts (valid only):' in line:
                    try:
                        avg_str = line.split(':')[1].strip()
                        if 'N/A' not in avg_str:
                            avg_conflicts = float(avg_str)
                            experiment_result['avg_conflicts'] = avg_conflicts
                            # Calculate total conflicts
                            if 'valid_conflict_count' in experiment_result:
                                experiment_result['total_conflicts'] = int(avg_conflicts * experiment_result['valid_conflict_count'])
                    except:
                        pass
                elif 'Total time:' in line:
                    try:
                        experiment_result['total_time'] = float(line.split(':')[1].strip().rstrip('s'))
                    except:
                        pass
        
        results_summary['experiments'].append(experiment_result)
        
        # Save intermediate results
        with open(args.log_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nExperiment {i}/{len(experiments)} completed. Status: {'SUCCESS' if return_code == 0 else 'FAILED'}")
    
    # Print final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for exp in results_summary['experiments']:
        print(f"\n{exp['experiment_number']}. {exp['model']} - {exp['weight_mode'].upper()}")
        print(f"   Status: {'✓ SUCCESS' if exp['success'] else '✗ FAILED'}")
        if 'satisfiable' in exp:
            print(f"   SATISFIABLE: {exp['satisfiable']}/{exp.get('total_files', '?')} ({exp.get('satisfiable_pct', '?')}%)")
        if 'unsatisfiable' in exp:
            print(f"   UNSATISFIABLE: {exp['unsatisfiable']}/{exp.get('total_files', '?')} ({exp.get('unsatisfiable_pct', '?')}%)")
        if 'timeout' in exp:
            print(f"   TIMEOUT: {exp['timeout']}/{exp.get('total_files', '?')} ({exp.get('timeout_pct', '?')}%)")
        if 'avg_decisions' in exp:
            print(f"   Avg Decisions: {exp['avg_decisions']:.1f} (valid instances: {exp.get('valid_decision_count', '?')})")
        if 'total_decisions' in exp:
            print(f"   Total Decisions: {exp['total_decisions']}")
        if 'avg_conflicts' in exp:
            print(f"   Avg Conflicts: {exp['avg_conflicts']:.1f} (valid instances: {exp.get('valid_conflict_count', '?')})")
        if 'total_conflicts' in exp:
            print(f"   Total Conflicts: {exp['total_conflicts']}")
        if 'total_time' in exp:
            print(f"   Total Time: {exp['total_time']:.2f}s")

    
    print("\n" + "="*80)
    print(f"Complete log saved to: {args.log_file}")
    print("="*80 + "\n")
    
    # Return success if all experiments succeeded
    all_success = all(exp['success'] for exp in results_summary['experiments'])
    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

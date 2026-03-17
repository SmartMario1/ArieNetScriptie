import argparse
import sys

from nsnet import train_model

# [tool.setuptools.package-dir]
# "nsnet" = "src/"
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, choices=['ArieNet', 'NSNet', 'MatrixNet'])
    parser.add_argument('--n_formulas', type=int, default=5000)
    parser.add_argument('--dummy', action='store_true', default=False)
    parser.add_argument('--restore', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)

    args = parser.parse_args()
    sys.argv = ['train_model.py', 'sat-solving']

    if args.dummy:
        id = 'dummy'
        sys.argv += [id] # exp id
        sys.argv += ['/home/sander/Thesis2/nsnet/SATSolving/3-sat/train_first/']
        sys.argv += ['--valid_dir', '/home/sander/Thesis2/nsnet/SATSolving/3-sat/valid_first/']
    elif args.model == 'ArieNet':
        id = 'sat_arienet_3sat_marginal_on_5000'
        sys.argv +=[id]
        sys.argv += ['/home/sander/Thesis2/nsnet/SATSolving/3-sat/train_first_4000_ArieNet']
        sys.argv += ['--valid_dir', '/home/sander/Thesis2/nsnet/SATSolving/3-sat/valid_first_1000_ArieNet']

    elif args.model == 'MatrixNet':
        id = 'sat_matrixnet_3-sat_marginal_on_4000'
        sys.argv +=[id]
        sys.argv += ['/home/sander/Thesis2/nsnet/SATSolving/3-sat/train_first_4000_MatrixNet/']
        sys.argv += ['--valid_dir', '/home/sander/Thesis2/nsnet/SATSolving/3-sat/valid_first_1000_MatrixNet/']
    elif args.model == 'NSNet':
        id = 'sat_nsnet_3-sat_marginal_on_5000'
        sys.argv +=[id]
        sys.argv += ['/home/sander/Thesis2/nsnet/SATSolving/3-sat/train_first_4000_ArieNet']
        sys.argv += ['--valid_dir', '/home/sander/Thesis2/nsnet/SATSolving/3-sat/valid_first_1000_ArieNet']
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.restore > 0:
        sys.argv += ['--restore', f'/home/sander/Thesis2/nsnet/runs/{id}/checkpoints/model_{args.restore}.pt']
    
    sys.argv += ['--epochs', str(args.epochs), '--scheduler', 'ReduceLROnPlateau', '--lr_step_size', '20', '--loss', 'marginal']
    sys.argv += ['--model', args.model]

    print("Running with the following arguments:")
    print(sys.argv)

    train_model.main()
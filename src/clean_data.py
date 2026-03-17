import os
import argparse
import glob

from nsnet.utils.utils import write_dimacs_to, preprocess
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    opts = parser.parse_args()
    print(opts)
    
    print('Cleaning files...')
    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]

    total_files = len(all_files)
    for i, f in enumerate(all_files):
        n_vars, clauses = preprocess(f)
        write_dimacs_to(n_vars, clauses, f)
        if (i + 1) % 10 == 0:
            print(f'Cleaned {i + 1}/{total_files}', end='\r')
    

if __name__ == '__main__':
    main()
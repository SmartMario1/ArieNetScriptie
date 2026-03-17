import os
import argparse
import glob
import pickle
import shutil

from nsnet.utils.solvers import MCSolver, SATSolver, MESolver
from concurrent.futures.process import ProcessPoolExecutor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['model-counting', 'assignment', 'marginal'], help='Task')
    parser.add_argument('data_dir', type=str, help='Directory with sat data')
    parser.add_argument('--out_dir', type=str, default=None, help='Output Directory with sat data')
    parser.add_argument('--n_process', type=int, default=8, help='Number of processes')
    parser.add_argument('--timeout', type=int, default=5000, help='Timeout')

    opts = parser.parse_args()

    ##########
    # opts.timeout = 50
    # Use sequential processing for memory-intensive marginal computation (BDDs)
    # Other tasks can use more parallelism
    opts.n_process = 3 if opts.task == 'marginal' else 6
    ########

    if opts.task == 'model-counting':
        opts.solver = 'DSHARP'
    elif opts.task == 'assignment':
        opts.solver = 'CaDiCaL'
    else:
        opts.solver = 'bdd_minisat_all'

    if opts.out_dir is not None:
        os.makedirs(opts.out_dir, exist_ok=True)

    if opts.task == 'model-counting':
        solver = MCSolver(opts)
    elif opts.task == 'assignment':
        solver = SATSolver(opts)
    else:
        solver = MESolver(opts) # for marginals
        # This produces a list of marginal numbers (what is the order?)

    labels = []
    
    print('Generating labels...')

    ##########
    # all_files = [opts.data_dir + '/09459.cnf']
    ###########

    all_files = sorted(glob.glob(opts.data_dir + '/**/*.cnf', recursive=True))
    all_files = [os.path.abspath(f) for f in all_files]
    
    print(f'Found {len(all_files)} CNF files to process')
    
    if len(all_files) == 0:
        print('[ERROR] No CNF files found in directory!')
        return
    
    # Run the solver (in parallel?)
    # Dit loopt volgens mij soms vast... waarom?
    # Wat genereert alle output van de solver als print, is dat in solver.run?
    # Timeout staat op 5000
    # Stored het ergens hoe veel formules falen?
    print(f'Starting parallel processing with {opts.n_process} workers...')
    
    try:
        with ProcessPoolExecutor(max_workers=opts.n_process) as pool: #default n_process=8
            results = list(pool.map(solver.run, all_files))
        print(f'Parallel processing completed, got {len(results)} results')
    except Exception as e:
        print(f'[ERROR] ProcessPoolExecutor failed: {e}')
        import traceback
        traceback.print_exc()
        return
    
    tot = len(all_files)
    cnt = 0
    failed_files = []

    print('Processing results...')
    for i, result in enumerate(results):
        try:
            if opts.task == 'model-counting':
                complete, counting, t = result
            elif opts.task == 'assignment':
                complete, assignment, _, t = result
            else:
                # Unpack additional debug info for marginal task
                complete, marginal, t, error_info, stdout_output, stderr_output = result
        except Exception as e:
            print(f"\n[ERROR] Failed to unpack result for file {i}: {all_files[i]}")
            print(f"  Exception: {e}")
            print(f"  Result: {result}")
            failed_files.append((all_files[i], 0, f'UNPACK_ERROR: {e}', '', ''))
            continue
        
        if complete:
            cnt += 1
            if opts.task == 'model-counting':
                ln_counting = float(counting.ln())
                labels.append(ln_counting)
            elif opts.task == 'assignment':
                labels.append(assignment)
            else:
                labels.append(marginal)
            
            if opts.out_dir is not None:
                shutil.copyfile(all_files[i], os.path.join(opts.out_dir, '%.5d.cnf' % (cnt)))
        else:
            # Log failed files instead of deleting them
            failed_files.append((all_files[i], t, error_info if opts.task == 'marginal' else '', 
                               stdout_output if opts.task == 'marginal' else '', 
                               stderr_output if opts.task == 'marginal' else ''))
            print(f"\n[FAILED] Could not generate label for: {os.path.basename(all_files[i])} (time: {t:.2f}s)")
            if opts.task == 'marginal':
                print(f"  Error: {error_info}")
                if stderr_output:
                    print(f"  STDERR: {stderr_output[:500]}")
                if stdout_output:
                    print(f"  STDOUT: {stdout_output[:500]}")

        if (i + 1) % 10 == 0:
            print(f'Processed {i + 1}/{tot} files', end='\r')

    r = cnt / tot
    print('\n' + '='*60)
    print('Total: %d, Labeled: %d, Failed: %d, Ratio: %.4f.' % (tot, cnt, len(failed_files), r))
    
    if failed_files:
        print('\nFailed files:')
        for fail_data in failed_files:
            filepath, fail_time = fail_data[0], fail_data[1]
            print(f'  - {os.path.basename(filepath)} (time: {fail_time:.2f}s)')
            if len(fail_data) > 2 and fail_data[2]:  # has error_info
                print(f'    Error: {fail_data[2]}')
    print('='*60)

    # Dump all labels to a single file, i.e. 'marginals.pkl
    if opts.out_dir is not None:
        if opts.task == 'model-counting':
            labels_file = os.path.join(opts.out_dir, 'countings.pkl')
        elif opts.task == 'assignment':
            labels_file = os.path.join(opts.out_dir, 'assignments.pkl')
        else:
            labels_file = os.path.join(opts.out_dir, 'marginals.pkl')
    else:
        if opts.task == 'model-counting':
            labels_file = os.path.join(opts.data_dir, 'countings.pkl')
        elif opts.task == 'assignment':
            labels_file = os.path.join(opts.data_dir, 'assignments.pkl')
        else:
            labels_file = os.path.join(opts.data_dir, 'marginals.pkl')
    
    print(f'\nSaving {len(labels)} labels to: {labels_file}')
    
    if len(labels) == 0:
        print('[WARNING] No labels were successfully generated!')
    
    try:
        with open(labels_file, 'wb') as f:
            pickle.dump(labels, f)
        print(f'[SUCCESS] Labels saved successfully to {labels_file}')
    except Exception as e:
        print(f'[ERROR] Failed to save labels file: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

import torch
import torch.nn.functional as F
import os
import sys
import argparse
import numpy as np
import random
import pickle
import math
import time

from nsnet.utils.options import add_model_options
from nsnet.utils.logger import Logger
from nsnet.utils.dataloader import get_dataloader
from nsnet.models.bp import BP
from nsnet.models.nsnet import NSNet
from nsnet.models.neurosat import NeuroSAT
# from torch_scatter import scatter_sum

import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['model-counting', 'sat-solving'], help='Experiment task')
    parser.add_argument('test_dir', type=str, help='Directory with testing data')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to be tested') # que is this?
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    add_model_options(parser)
    
    opts = parser.parse_args()

    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(opts.seed)
    random.seed(opts.seed)

    t0 = time.time()
    
    if opts.checkpoint is not None:
        opts.log_dir = os.path.abspath(os.path.join(opts.checkpoint,  '..', '..'))
        opts.eval_dir = os.path.join(opts.log_dir, 'evaluations')
        dataset_name = os.path.abspath(opts.test_dir).split(os.path.sep)[-2]
        checkpoint_name = os.path.splitext(os.path.basename(opts.checkpoint))[0]
    else:
        opts.log_dir = os.path.abspath(os.path.join('runs', opts.model))
        opts.eval_dir = os.path.join(opts.log_dir, 'evaluations')
        dataset_name = os.path.abspath(opts.test_dir).split(os.path.sep)[-2]
        checkpoint_name = 'no_training'

    os.makedirs(opts.eval_dir, exist_ok=True)

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)


    # WAT IN DEZE CODE DOET 'Processing...' 'Done!'
    models = {
        'BP': BP,
        'NSNet': NSNet,
        'NeuroSAT': NeuroSAT,
    }

    model = models[opts.model](opts)
    model.to(opts.device)
    test_loader = get_dataloader(opts.test_dir, opts, 'test')

    if opts.checkpoint is not None:
        print('Loading model checkpoint from %s..' % opts.checkpoint)
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        else:
            checkpoint = torch.load(opts.checkpoint)

        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    model.to(opts.device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    all_results = []
    tot = 0
    rmse = 0
    solved = 0

    print('Testing...')
    model.eval()
    n_batches = len(test_loader)
    for i, data in enumerate(test_loader):
        data = data.to(opts.device)
        batch_size = data.c_size.shape[0]
        with torch.no_grad():
            if opts.task == 'model-counting':
                preds = model(data)
                labels = data.y
                all_results.extend(preds.tolist())
                mse = F.mse_loss(preds, labels).item()
                rmse += mse * batch_size
            else:
                c_size = data.c_size.sum().item()
                c_batch = data.c_batch
                l_edge_index = data.l_edge_index
                c_edge_index = data.c_edge_index

                # Ook hier, model(data) geeft probabilities, dat zijn de predictions
                v_prob = model(data)
                v_assign = (v_prob > 0.5).float()
                preds = v_assign[:, 0]
                l_assign = v_assign.reshape(-1)

                # This first 'scatter_sums', and then sets the max of any value in the tensor to 1
                # index in scatter sum is a tensor, so that each cell in the source tensor is summed to the value in its matching cell in 'index'

                ##
                # Voor zover cellen in 'src' ook bestaan in 'index' worden ze geplaatst in 'target'
                # Daarbij wordt de 'dim' dimensie bepaald door de waarde in 'index'
                # En de andere dimensies matchen met die van de cell in 'src' 

                # New Pytorch
                c_sat = torch.clamp(torch.zeros(c_size, dtype=l_assign.dtype).scatter_reduce(0, c_edge_index, l_assign[l_edge_index], reduce="sum"), max=1)
                sat_batch = (torch.zeros(batch_size, dtype=c_sat.dtype).scatter_reduce(0, c_batch, c_sat, reduce="sum") == data.c_size).float()
                ####            
                # c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                # sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()
                ##
                all_results.extend(sat_batch.tolist())
                solved += sat_batch.sum().item()
            
        tot += batch_size

        if (i + 1) % 10 == 0:
            print(f'Testing batch {i + 1}/{n_batches}', end='\r')
    
    if opts.task == 'model-counting':
        rmse = math.sqrt(rmse / tot)
        print('Total: %d, RMSE: %f' % (tot, rmse))
    else:
        r = solved / tot
        print('Total: %d, Solved: %f, Ratio: %f' % (tot, solved, r))

    t = time.time() - t0
    print('Solving Time: %f' % t)

    with open('%s/task=%s_dataset=%s_checkpoint=%s_n_rounds=%d.pkl' % \
        (opts.eval_dir, opts.task, dataset_name, checkpoint_name, opts.n_rounds), 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    main()

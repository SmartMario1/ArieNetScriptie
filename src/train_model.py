import argparse
import math
import os
import pdb
import random
import sys
import time

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
# Functional has functions that operate on data, but don't
# maintain internal states like nn.Module
import torch.nn.functional as F
import torch.optim as optim
from nsnet.models.arienet import ArieNet, GNNSquared, MatrixNet
from nsnet.models.neurosat import NeuroSAT
from nsnet.models.nsnet import NSNet
from nsnet.utils.dataloader import get_dataloader
# from torch_scatter import scatter_sum
from nsnet.utils.dataset import BPG
from nsnet.utils.logger import Logger
from nsnet.utils.options import add_model_options
from nsnet.utils.torch_utils import scatter_sum, split_ind_2_src_matrix
from nsnet.utils.utils import safe_log
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR


def n_true_formulas(v_prob, data, device):
    """
    From a list of list of proabilities (one sublist of size 2 for each variable)
    And from a batched data structure, calculate the number of true CNF formulas
    """
    # Changed this to fix every variable to 0 or 1
    # Assignment in the order of variable number
    v_assign = torch.tensor([[1,0] if v[0] > v[1] else [0,1] for v in v_prob], device=device)

    # This happens all the time!
    # for i,v in enumerate(v_prob):
    #     if (v[0] >0.5) != (v[1] < 0.5):
    #         print(i)
    #         print(v)
    #         break
            # input()

    # if any(v[0]+v[1] != 1 for v in v_assign):
    #     raise ValueError('Invalid assignment')
    
    # Assignment for each literal
    l_assign = v_assign.reshape(-1)
    # l_assign = [x for xs in v_assign for x in xs]

    model = None
    if isinstance(data[0], BPG):
        model = "arienet"
    elif isinstance(data[0], MatrixNet):
        model = "matrixnet"
    elif isinstance(data[0], NSNet):
        model = "nsnet"
 
    if model=='arienet':
        # For summing to clause satisfaction
        l_index_per_occurence = data.literal_indices_per_occurence
        c_index_per_occurence = data.clause_indices_per_occurence
        n_clauses = data.n_clauses.sum().item()

        # For summing to formula (batch element) satisfaction
        n_clauses_per_batch = data.n_clauses
        form_index_per_clause = torch.tensor([i for i, batch_size in enumerate(data.n_clauses) for _ in range(batch_size)], dtype=torch.long, device=device)
        batch_size = data.n_clauses.shape[0]
    
    elif model=='matrixnet':

        c_index_per_occurence, l_index_per_occurence = split_ind_2_src_matrix(data.clause_to_literals_matrix)
        n_clauses = data.n_clauses.sum().item()

        n_clauses_per_batch = data.n_clauses
        form_index_per_clause = torch.tensor([i for i, batch_size in enumerate(data.n_clauses) for _ in range(batch_size)], dtype=torch.long, device=device)
        batch_size = data.n_clauses.shape[0]

    elif model=='nsnet':
        # For summing to clause satisfaction; this is without repeats
        # so size is the total sum of clause sizes, not the number of edges (which is double)
        l_index_per_occurence = data.l_edge_index
        c_index_per_occurence = data.c_edge_index                        
        n_clauses = data.c_size.sum().item() 

        # For summing to formula (batch element) satisfaction
        n_clauses_per_batch = data.c_size
        form_index_per_clause = torch.tensor([i for i, batch_size in enumerate(data.c_size) for _ in range(batch_size)], dtype=torch.long, device=device)
        batch_size = data.c_size.shape[0]

    # New Pytorch
    c_sat = torch.clamp(torch.zeros(n_clauses, dtype=l_assign.dtype, device=device).scatter_reduce(0, c_index_per_occurence, l_assign[l_index_per_occurence], reduce="sum"), max=1)
    formula_sat = (torch.zeros(batch_size, dtype=c_sat.dtype, device=device).scatter_reduce(0, form_index_per_clause, c_sat, reduce="sum") == n_clauses_per_batch).float()
    #####
    # c_sat = torch.clamp(scatter_sum(l_assign[l_index_per_occurence], c_index_per_occurence, dim=0, dim_size=n_clauses), max=1)
    # formula_sat = (scatter_sum(c_sat, form_index_per_clause, dim=0, dim_size=batch_size) == n_clauses_per_batch).float()
    return formula_sat.sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['model-counting', 'sat-solving'], help='Experiment task')
    parser.add_argument('exp_id', type=str, help='Experiment id')
    parser.add_argument('train_dir', type=str, help='Directory with training data')
    parser.add_argument('--train_size', type=int, default=None, help='Number of training data')
    parser.add_argument('--valid_dir', type=str, default=None, help='Directory with validating data')
    parser.add_argument('--loss', type=str, choices=['assignment', 'marginal'], default='marginal', help='Loss type for SAT solving')
    parser.add_argument('--restore', type=str, default=None, help='Continue training from a checkpoint')
    parser.add_argument('--save_model_epochs', type=int, default=1, help='Number of epochs between model savings')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs during training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_dacay', type=float, default=1e-10, help='L2 regularization weight')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')
    parser.add_argument('--lr_step_size', type=int, default=20, help='Learning rate step size')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Learning rate factor')
    parser.add_argument('--lr_patience', type=int, default=20, help='Learning rate patience')
    parser.add_argument('--clip_norm', type=float, default=0.65, help='Clipping norm')
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

    opts.log_dir = os.path.join('runs', opts.exp_id)
    opts.checkpoint_dir = os.path.join(opts.log_dir, 'checkpoints')

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ########### ARIE ADD
    opts.batch_size = 24
    # if opts.model == 'ArieNet':
    #     opts.model = 'GNNSquared'
    #     opts.base_GNN = ArieNet
    #     opts.feature_GNN = ArieNet
    # opts.model='ArieNet'
    # opts.device='cpu'
    # opts.model='NSNet'
    ###################
    models = {
        'NSNet': NSNet,
        'NeuroSAT': NeuroSAT,
        'ArieNet': ArieNet,
        'GNNSquared': GNNSquared,
        'MatrixNet': MatrixNet
    }

    # Here the nn.Module is initialised
    model = models[opts.model](opts)
    model.to(opts.device)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_dacay)

    start_processing_time = time.time()
    # Builds all the graphs
    train_loader = get_dataloader(opts.train_dir, opts, 'train', opts.train_size)
    
    if opts.valid_dir is not None:
        valid_loader = get_dataloader(opts.valid_dir, opts, 'valid')
    else:
        valid_loader = None

    print('Data processing time: %f' % (time.time() - start_processing_time))
    
    if opts.scheduler is not None:
        if opts.scheduler == 'ReduceLROnPlateau':
            assert opts.valid_dir is not None
            scheduler = ReduceLROnPlateau(optimizer, factor=opts.lr_factor, patience=opts.lr_patience)
        else:
            assert opts.scheduler == 'StepLR'
            scheduler = StepLR(optimizer, step_size=opts.lr_step_size, gamma=opts.lr_factor)

    best_loss = float('inf')


    # with profiler.profile(use_cuda=False, profile_memory=True) as prof:
    start_epoch = 0
    if opts.restore != None:
        print('Loading model checkpoint from %s..' % opts.restore)
        if opts.device == 'cpu':
            checkpoint = torch.load(opts.restore, map_location='cpu')
        else:
            checkpoint = torch.load(opts.restore)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        model.to(opts.device)

    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + opts.epochs):
        if epoch==1:
            print("Time for first epoch: ", time.time()-start_time)
        print('EPOCH #%d' % epoch)
        print('Training...')
        # print(time.time())
        train_loss = 0
        train_tot = 0
        train_rmse = 0
        train_cnt = 0

        print("Using device:", opts.device)
        print("All settings:", opts)

        model.train()
        n_batches_train = len(train_loader)
        for i, data in enumerate(train_loader):
            # # SHOULDN't THIS BE JUST BEFORE THE BACKWARD PASS?
            optimizer.zero_grad()
            # print("A", opts.device)
            data = data.to(opts.device)
            # import pdb; pdb.set_trace()

            if model.__class__.__name__ in ['ArieNet', 'GNNSquared', 'MatrixNet']:
                # pdb.set_trace()
                batch_size = len(data.n_clauses)
            else:
                # pdb.set_trace()
                batch_size = data.c_size.shape[0]
            
            if opts.task == 'model-counting':
                # This is a forward pass
                # print("Going forward")
                preds = model(data)
                # print("Finished forward")
                # Apparently, the labels are in order of literal occurence
                labels = data.y
                loss = F.mse_loss(preds, labels)
                mse = loss.item()
                train_rmse += mse * batch_size
            else:
                # print("Going forward")
                v_prob = model(data)
                # print("Finished forward")
                # c_size is the number of clauses
                # l_edge_index is literal_index per occurence (niet + EN -)
                # c_edge_index is de clause index per occurence

                # som want dit maakt een lijst van c_size per batch item
                # if model.__class__.__name__ != 'ArieNet':
                #     n_clauses_per_batch = data.c_size
                #     n_clauses = data.c_size.sum().item() 
                #     form_index_per_clause = torch.tensor([i for i, batch_size in enumerate(data.c_size) for _ in range(batch_size)], dtype=torch.long, device=opts.device)
                #     l_index_per_edge = data.l_edge_index
                #     c_index_per_edge = data.c_edge_index


                # if model.__class__.__name__ == 'ArieNet':
                #     n_clauses_per_batch = data.n_clauses
                #     n_clauses = data.n_clauses.sum().item()
                #     form_index_per_clause = torch.tensor([i for i, batch_size in enumerate(data.n_clauses) for _ in range(batch_size)], dtype=torch.long, device=opts.device)
                #     l_index_per_edge = data.l_index_per_edge
                #     c_index_per_edge = data.c_index_per_edge


                if opts.loss == 'assignment':
                    preds = v_prob[:, 0]
                    labels = data.y
                    loss = F.binary_cross_entropy(preds, labels)
                else:
                    preds = v_prob
                    labels = data.y
                    labels = torch.stack([labels, 1-labels], dim=1)
                    loss = F.kl_div(safe_log(preds), labels)

                train_cnt += n_true_formulas(v_prob, data, opts.device)

            # Weighted average voor batch_size? 
            train_loss += loss.item() * batch_size
            train_tot += batch_size

            # Output of .backward() is stored in param.grad (for param in model.parameters())
            # print("Going backward")

            loss.backward()
            
            # print("Loss requires grad:", loss.requires_grad)
            # def print_graph_stats(tensor):
            #     if hasattr(tensor, 'grad_fn'):
            #         print(f"Tensor grad_fn: {tensor.grad_fn}")
            #         print(f"Tensor requires_grad: {tensor.requires_grad}")

            # print_graph_stats(loss)

            # input()

            # Print gradients for specific tensors
            # print("\n--- Gradient Information ---")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name} grad:")
            #         print(f"  Shape: {param.grad.shape}")
            #         print(f"  Mean: {param.grad.mean().item()}")
            #         print(f"  Max: {param.grad.abs().max().item()}")
            #         print(f"  Is NaN: {torch.isnan(param.grad).any().item()}")
            #         print(f"  Is Inf: {torch.isinf(param.grad).any().item()}")
            # input()

            # print("Finished backward")
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)

            # This does update of the parameters with the gradients
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Training batch {i + 1}/{n_batches_train}, Loss: {loss.item():.4f}', end='\r')
            
        # Ja weighted average, want hier delen door train_tot
        # Maar dat klopt alleen als 'loss' een gemiddelde is
        # Als los een som is, gaat dit fout
        train_loss /= train_tot
        print('Training LR: %f, Training loss: %f' % (optimizer.param_groups[0]['lr'], train_loss))

        if opts.task == 'model-counting':
            train_rmse = math.sqrt(train_rmse / train_tot)
            print('Training RMSE: %f' % train_rmse)
        else:
            train_acc = train_cnt / train_tot
            print('Training accuracy: %f' % train_acc)

        if epoch % opts.save_model_epochs == 0:
            torch.save({
                'state_dict': model.state_dict(), 
                'epoch': epoch,
                'optimizer': optimizer.state_dict()}, 
                os.path.join(opts.checkpoint_dir, 'model_%d.pt' % epoch)
            )
        
        if opts.valid_dir is not None:
            with torch.no_grad(): # Deactivate gradients for the following code
                print('Validating...')

                valid_loss = 0
                valid_tot = 0
                valid_rmse = 0
                valid_cnt = 0

                model.eval()
                n_batches_valid = len(valid_loader)
                for i, data in enumerate(valid_loader):
                    data = data.to(opts.device)

                    if model.__class__.__name__ in ['ArieNet', 'GNNSquared', 'MatrixNet']:
                        batch_size = data.n_clauses.shape[0]
                    else:
                        batch_size = data.c_size.shape[0]

                    with torch.no_grad():
                        if opts.task == 'model-counting':
                            preds = model(data)
                            labels = data.y
                            loss = F.mse_loss(preds, labels)
                            mse = loss.item()
                            valid_rmse += mse * batch_size
                        else:
                            v_prob = model(data)
                            # c_size = data.c_size.sum().item()
                            # c_batch = data.c_batch
                            # l_edge_index = data.l_edge_index
                            # c_edge_index = data.c_edge_index

                            if opts.loss == 'assignment':
                                preds = v_prob[:, 0]
                                labels = data.y
                                loss = F.binary_cross_entropy(preds, labels)
                            else:
                                preds = v_prob
                                labels = data.y
                                labels = torch.stack([labels, 1-labels], dim=1)
                                loss = F.kl_div(safe_log(preds), labels)
                            
                            # v_assign = (v_prob > 0.5).float()
                            # preds = v_assign[:, 0]
                            # l_assign = v_assign.reshape(-1)
                            # c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                            # sat_batch = (scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size) == data.c_size).float()

                            valid_cnt += n_true_formulas(v_prob, data, opts.device)
                            # valid_cnt += sat_batch.sum().item()
                                    
                    valid_loss += loss.item() * batch_size
                    valid_tot += batch_size

                    if (i + 1) % 10 == 0:
                        print(f'Validating batch {i + 1}/{n_batches_valid}, Loss: {loss.item():.4f}', end='\r')
            
            valid_loss /= valid_tot
            print('Validating loss: %f' % valid_loss)

            if opts.task == 'model-counting':
                valid_rmse = math.sqrt(valid_rmse / valid_tot)
                print('Validating RMSE: %f' % valid_rmse)
            else:
                valid_acc = valid_cnt / valid_tot
                print('Validating accuracy: %f' % valid_acc)

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch, 
                    'optimizer': optimizer.state_dict()}, 
                    os.path.join(opts.checkpoint_dir, 'model_best.pt')
                )

            if opts.scheduler is not None:
                if opts.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()
        else:
            if opts.scheduler is not None:
                scheduler.step()

    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == '__main__':
    print("running main")
    main()

import math
import pdb
import time

import torch
import torch.nn as nn
from nsnet.models.mlp import MLP
from nsnet.utils.torch_utils import (scatter_logsumexp, scatter_sum,
                                     split_ind_2_src_matrix, swap_even_odd)

# from torch_scatter import scatter_sum, scatter_logsumexp

class NeighborhoodFeatureNet(nn.Module):
    def __init__(self, opts):
        super(ArieNet, self).__init__()
        self.opts = opts
        self.c2l_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.l2c_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.denom = math.sqrt(self.opts.dim)

        self.c2l_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2c_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)

        # A3                                       # input dim     # hidden dim?   # output dim?
        self.l2c_msg_norm = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.softmax = nn.Softmax(dim=1)

    def sum_c2l_by_literal_occurence(self, data, c2l_edges_feat):
        return scatter_sum(data.c2l_msg_receiver_indices, c2l_edges_feat[data.c2l_msg_sender_indices], data.n_edges, self.opts.device)

    def sum_l2c_per_assignment(self, data, l2c_edges_feat):
        return scatter_sum(data.l2c_assignment_indices, l2c_edges_feat[data.l2c_assignment_neighborhoods], len(data.l2c_msg_receiver_indices), self.opts.device)

    def max_l2c_satisfying_assignments(self, data, l2c_msgs_per_assignment):
        return scatter_logsumexp(data.l2c_msg_receiver_indices, l2c_msgs_per_assignment, data.n_edges, self.opts.device)
    
    def forward(self, data):
        n_edges = data.n_edges

        # Init, all edge features
        c2l_edges_feat = (self.c2l_edges_init / self.denom).repeat(n_edges, 1)
        l2c_edges_feat = (self.l2c_edges_init / self.denom).repeat(n_edges, 1)

        for _ in range(self.opts.n_rounds):
            ##### First update ##########
            l2c_msg_argument = self.sum_c2l_by_literal_occurence(data, c2l_edges_feat)

            # A1
            l2c_msg = self.l2c_msg_update(l2c_msg_argument)

            # A2
            l2c_negated_msg = swap_even_odd(l2c_msg)
            l2c_edges_feat = self.l2c_msg_norm(torch.cat([l2c_msg, l2c_negated_msg], dim=1))

            ##### Second update #########
            # Softmax over all satisfying assignments
            l2c_msgs_per_assignment = self.sum_l2c_per_assignment(data, l2c_edges_feat)
            c2l_msg_argument = self.max_l2c_satisfying_assignments(data, l2c_msgs_per_assignment)

            # A3edge_index
            c2l_edges_feat = self.c2l_msg_update(torch.cat([c2l_msg_argument], dim=1))
 

        # Feature extraction as global pooling, or as model counting
        l_features = scatter_sum(data.literal_indices_per_edge, c2l_edges_feat, data.n_literals.sum().item(), self.opts.device)
        l_features = self.l_readout(l_features)
        v_features = l_features.reshape(-1, 2)
        return self.softmax(v_features)

class GNNSquared(nn.Module):
    def __init__(self, opts):
        super(GNNSquared, self).__init__()
        self.base_GNN = opts.base_GNN(opts)
        self.feature_GNN = opts.feature_GNN(opts)

    def forward(self, data):
        return self.base_GNN(data)

class MatrixNet(nn.Module):
    def __init__(self, opts):
        """
        opts needs:
        - dim
        - n_mlp_layers
        - activation

        Waar komen deze vandaan?

        c2l: clause to literal
        """
        super(MatrixNet, self).__init__()
        self.opts = opts
        self.c2l_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.l2c_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.denom = math.sqrt(self.opts.dim)

        self.c2l_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2c_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)

        # A3                                       # input dim     # hidden dim?   # output dim?
        self.l2c_msg_norm = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.softmax = nn.Softmax(dim=1)

    def sum_c2l_by_literal_occurence(self, data, c2l_edges_feat):
        edge_indices, literal_neighborhood_indices = split_ind_2_src_matrix(data.edge_to_literal_neighborhood_matrix)
        return scatter_sum(edge_indices, c2l_edges_feat[literal_neighborhood_indices], data.n_edges, self.opts.device)

    def sum_l2c_per_assignment(self, data, l2c_edges_feat):
        assignment_indices, assignment_neighborhood_indices = split_ind_2_src_matrix(data.clause_assignment_to_neighborhood_matrix)
        return scatter_sum(assignment_indices, l2c_edges_feat[assignment_neighborhood_indices], len(data.clause_assignment_to_neighborhood_matrix), self.opts.device)

    def max_l2c_satisfying_assignments(self, data, l2c_msgs_per_assignment):
        edge_indices, assignment_indices = split_ind_2_src_matrix(data.edge_to_clause_assignment_matrix)
        return scatter_logsumexp(edge_indices, l2c_msgs_per_assignment[assignment_indices], data.n_edges, self.opts.device)


    def forward(self, data):
        n_edges = data.n_edges

        if self.opts.task == 'model-counting':
            raise NotImplementedError
        
        # Init, all edge features
        c2l_edges_feat = (self.c2l_edges_init / self.denom).repeat(n_edges, 1)
        l2c_edges_feat = (self.l2c_edges_init / self.denom).repeat(n_edges, 1)

        for _ in range(self.opts.n_rounds):

            ##### First update ##########
            l2c_msg_argument = self.sum_c2l_by_literal_occurence(data, c2l_edges_feat)

            # A1
            l2c_msg = self.l2c_msg_update(l2c_msg_argument)

            # Should be 2D, N x 1
            # local_satisfaction_percentages = data.local_satisfaction_percentage_per_edge.unsqueeze(1)

            # For the l->c features and the sat percentages, we swap the + and - occurrences
            # this relies on the - to always be right next to the +, which is not so nice.
            # negated_local_satisfaction_percentages = swap_even_odd(local_satisfaction_percentages)
            l2c_negated_msg = swap_even_odd(l2c_msg)

            # A2
            # l2c_edges_feat = self.l2c_msg_norm(torch.cat([l2c_msg, l2c_negated_msg, local_satisfaction_percentages, negated_local_satisfaction_percentages], dim=1))
            l2c_edges_feat = self.l2c_msg_norm(torch.cat([l2c_msg, l2c_negated_msg], dim=1))

            ##### Second update #########
            # Softmax over all satisfying assignments
            l2c_msgs_per_assignment = self.sum_l2c_per_assignment(data, l2c_edges_feat)
            c2l_msg_argument = self.max_l2c_satisfying_assignments(data, l2c_msgs_per_assignment)

            # A3
            # c2l_edges_feat = self.c2l_msg_update(torch.cat([c2l_msg_argument, local_satisfaction_percentages], dim=1))
            c2l_edges_feat = self.c2l_msg_update(c2l_msg_argument)


        if self.opts.task == 'model-counting':
            raise NotImplementedError
        else:
            literal_indices_per_edge, edge_indices = split_ind_2_src_matrix(data.literal_to_edges_matrix)

            # Readout of literal features as an nvar x 2 tensor
            l_features = scatter_sum(literal_indices_per_edge, c2l_edges_feat[edge_indices], data.n_literals.sum().item(), self.opts.device)
            l_features = self.l_readout(l_features)
            v_features = l_features.reshape(-1, 2)
            return self.softmax(v_features)


class ArieNet(nn.Module):
    def __init__(self, opts):
        """
        opts needs:
        - dim
        - n_mlp_layers
        - activation

        Waar komen deze vandaan?

        c2l: clause to literal
        """
        super(ArieNet, self).__init__()
        self.opts = opts
        self.c2l_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.l2c_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.denom = math.sqrt(self.opts.dim)

        self.c2l_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim + 1, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2c_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)

        # A3                                       # input dim     # hidden dim?   # output dim?
        self.l2c_msg_norm = MLP(self.opts.n_mlp_layers, self.opts.dim * 2 + 2, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.softmax = nn.Softmax(dim=1)
    
    def sum_c2l_by_literal_occurence(self, data, c2l_edges_feat):
        # Fails if there are more edges than groups to sum over because there are unconnected edges I think
        return scatter_sum(data.c2l_msg_receiver_indices, c2l_edges_feat[data.c2l_msg_sender_indices], data.n_edges, self.opts.device)

    def sum_l2c_per_assignment(self, data, l2c_edges_feat):
        return scatter_sum(data.l2c_assignment_indices, l2c_edges_feat[data.l2c_assignment_neighborhoods], len(data.l2c_msg_receiver_indices), self.opts.device)

    def max_l2c_satisfying_assignments(self, data, l2c_msgs_per_assignment):
        return scatter_logsumexp(data.l2c_msg_receiver_indices, l2c_msgs_per_assignment, len(torch.unique(data.l2c_msg_receiver_indices)), self.opts.device)

    def forward(self, data):        
        n_edges = data.n_edges

        if self.opts.task == 'model-counting':
            raise NotImplementedError
        
        # Init, all edge features
        c2l_edges_feat = (self.c2l_edges_init / self.denom).repeat(n_edges, 1)
        l2c_edges_feat = (self.l2c_edges_init / self.denom).repeat(n_edges, 1)

        for _ in range(self.opts.n_rounds):

            ##### First update ##########
            l2c_msg_argument = self.sum_c2l_by_literal_occurence(data, c2l_edges_feat)

            # A1
            l2c_msg = self.l2c_msg_update(l2c_msg_argument)

            # Should be 2D, N x 1
            local_satisfaction_percentages = data.local_satisfaction_percentage_per_edge.unsqueeze(1)

            # For the l->c features and the sat percentages, we swap the + and - occurrences
            # this relies on the - to always be right next to the +, which is not so nice.
            negated_local_satisfaction_percentages = swap_even_odd(local_satisfaction_percentages)
            l2c_negated_msg = swap_even_odd(l2c_msg)

            # A2
            l2c_edges_feat = self.l2c_msg_norm(torch.cat([l2c_msg, l2c_negated_msg, local_satisfaction_percentages, negated_local_satisfaction_percentages], dim=1))
            # l2c_edges_feat = self.l2c_msg_norm(torch.cat([l2c_msg, l2c_negated_msg], dim=1))

            ##### Second update #########
            # Softmax over all satisfying assignments
            l2c_msgs_per_assignment = self.sum_l2c_per_assignment(data, l2c_edges_feat)
            c2l_msg_argument = self.max_l2c_satisfying_assignments(data, l2c_msgs_per_assignment)

            # A3
            c2l_edges_feat = self.c2l_msg_update(torch.cat([c2l_msg_argument, local_satisfaction_percentages], dim=1))
            # c2l_edges_feat = self.c2l_msg_update(c2l_msg_argument)

        if self.opts.task == 'model-counting':
            raise NotImplementedError
        else:
            # Readout of literal features as an nvar x 2 tensor
            l_features = scatter_sum(data.literal_indices_per_edge, c2l_edges_feat, data.n_literals.sum().item(), self.opts.device)
            l_features = self.l_readout(l_features)
            v_features = l_features.reshape(-1, 2)
            return self.softmax(v_features)
            v_features = l_features.reshape(-1, 2)
            return self.softmax(v_features)

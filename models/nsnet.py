import math
import pdb

import torch
import torch.nn as nn
from nsnet.models.mlp import MLP


class NSNet(nn.Module):
    def __init__(self, opts):
        """
        opts needs:
        - dim
        - n_mlp_layers
        - activation

        Waar komen deze vandaan?

        c2l: clause to literal
        """
        super(NSNet, self).__init__()
        self.opts = opts
        self.c2l_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.l2c_edges_init = nn.Parameter(torch.randn(1, self.opts.dim))
        self.denom = math.sqrt(self.opts.dim)

        self.c2l_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2c_msg_update = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)

        # Wat is dit?                                       # input dim     # hidden dim?   # output dim?
        self.l2c_msg_norm = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, data):

        # import pdb; pdb.set_trace()
        #n_vars * 2
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()

        # Totaal aantal literals occurences in de CNF keer 2 (+ en -)
        # Dit is 924 met batch 1, terwijl eerste cnf er 414 heeft..
        # wss is dit een random CNF, niet 00000
        num_edges = data.num_edges

        # Map van 'edge index' (genummerd op volgorde van literal occurence, +,- na elkaar)
        # Naar de literal index in [0, 2N-1] format
        sign_l_edge_index = data.sign_l_edge_index
 
        # Per edge c-l, de 'volgorde index' van l in alle andere clauses waar l of -l in zit,
        # eerst alsof l in elke clause zit, dan alsof -l in elke clause zit
        c2l_msg_repeat_index = data.c2l_msg_repeat_index

        # Even lang als vorige lijst, maar per edge c1-l, de index van c1-(+)-l
        # keer het aantal edges l-(+)-c2
        # En dan de index van c1-(-)-l, keer het aantal l-(+)-c2
        c2l_msg_scatter_index = data.c2l_msg_scatter_index

        l2c_msg_aggr_repeat_index = data.l2c_msg_aggr_repeat_index
        l2c_msg_aggr_scatter_index = data.l2c_msg_aggr_scatter_index
        l2c_msg_scatter_index = data.l2c_msg_scatter_index

        if self.opts.task == 'model-counting':
            c_blf_repeat_index = data.c_blf_repeat_index
            c_blf_scatter_index = data.c_blf_scatter_index
            c_blf_norm_index = data.c_blf_norm_index
            v_degrees = data.v_degrees
            c_batch = data.c_batch
            v_batch = data.v_batch
            c_bethes = []
            v_bethes = []
        
        # Init, all edge features
        c2l_edges_feat = (self.c2l_edges_init / self.denom).repeat(num_edges, 1)
        l2c_edges_feat = (self.l2c_edges_init / self.denom).repeat(num_edges, 1)

        # print(c2l_edges_feat.requires_grad)
        # print(l2c_edges_feat.requires_grad)

        for _ in range(self.opts.n_rounds):
            # Wat is shape van c2l_msg_repeat_index, c2l_msg_scatter_index? hoe is dit index?
            # c2l_edges_feat[c2l_msg_repeat_index]
            # is, per edge c1-(+)-l, lijst van alle features van l-(+)-c2
            # en, per edge c1-(-)-l, lijst van alle features van l-(-)-c2

            # c2l_msg_scatter_index geeft dan voor elk van deze features de index van c1-(+)-l of c1-(-)-l
            # Oftewel, we sommen alle outgoing + edges vanuit l, op naar de index van de inkomende + edge van l

            # NEW PYTORCH
            c2l_msg = torch.zeros((num_edges, self.opts.dim), dtype=c2l_edges_feat.dtype, device=self.opts.device).scatter_reduce(
                0, c2l_msg_scatter_index.unsqueeze(1).expand(-1, self.opts.dim), c2l_edges_feat[c2l_msg_repeat_index], reduce="sum"
            )
            # print("c2l_msg")
            # print(c2l_msg.grad)

            # print(c2l_msg.requires_grad)
            # print(c2l_edges_feat[c2l_msg_repeat_index].requires_grad)
            # input()
            # pdb.set_trace()     
            # c2l_msg = scatter_sum(c2l_edges_feat[c2l_msg_repeat_index], c2l_msg_scatter_index, dim=0, dim_size=num_edges)
            ###


            # Waarom gebruiken we c2l in l2c update, en andersom?

            # De 'edge feature' van literal naar clause, krijgt hier alleen alle andere edge features uit andere
            # clauses als input
            l2c_edges_feat_new = self.l2c_msg_update(c2l_msg)

            # Hier stacken we de + en - edge features in 1 grote per variable (dubbele dimensie)
            v2c_edges_feat_new = l2c_edges_feat_new.reshape(num_edges // 2, -1)
            # En hier halen we ze weer uit elkaar
            pv2c_edges_feat_new, nv2c_edges_feat_new = torch.chunk(v2c_edges_feat_new, 2, 1)
            # cat is concat, dus we plakken - en + achter elkaar
            # Dit was allemaal om van [+,-,+,-,...] naar [-, -, ..., +, +, ...]
            l2c_edges_feat_inv = torch.cat([nv2c_edges_feat_new, pv2c_edges_feat_new], dim=1).reshape(num_edges, -1)

            # Hier plakken we dus [+,-,+,-,...] en [-, -, -, .., +, +, ..] aan elkaar in de feature dimensie
            # maar niet op een logische manier? de + van edge 2 gaat op de - van edge 3?
            # ik heb gechekt dat elems 1 en 0, en elems -2 en -1 dezelfde features zijn 
            # In PAPER: message van assignment naar clause neemt als input de messages van + naar clause en van - naar clause
            # DIT KLOPT DUS NIET!!!!!!!!!           
            l2c_edges_feat = self.l2c_msg_norm(torch.cat([l2c_edges_feat_new, l2c_edges_feat_inv], dim=1))

            # NEW PYTORCH
            l2c_msg_aggr = torch.zeros((l2c_msg_scatter_index.shape[0],self.opts.dim), dtype=l2c_edges_feat.dtype, device=self.opts.device).scatter_reduce(
                0, l2c_msg_aggr_scatter_index.unsqueeze(1).expand(-1, self.opts.dim), l2c_edges_feat[l2c_msg_aggr_repeat_index], reduce="sum"
            )
            # from torch_scatter import scatter_sum, scatter_logsumexp

            # l2c_msg_aggr = scatter_sum(l2c_edges_feat[l2c_msg_aggr_repeat_index], l2c_msg_aggr_scatter_index, dim=0, dim_size=l2c_msg_scatter_index.shape[0])
            ###
            # pdb.set_trace()
            # Step 1: Get the unique indices from A

            unique_indices = torch.unique(l2c_msg_scatter_index)

            # Step 1: Compute the max value for each group across the vector dimensions for numerical stability
            # Initialize max_vals to hold max values per group for each vector dimension
            max_vals = torch.full((len(unique_indices), self.opts.dim), float('-inf'), device=l2c_msg_aggr.device)

            # Use scatter_reduce to find max per group and dimension
            max_vals = max_vals.scatter_reduce(0, l2c_msg_scatter_index.unsqueeze(1).expand(-1, self.opts.dim), l2c_msg_aggr, reduce="amax")

            shifted_vals = l2c_msg_aggr - max_vals[l2c_msg_scatter_index]
            
            
            # # Step 2: Shift `l2c_msg_aggr` by the max values for each group
            # shifted_vals = l2c_msg_aggr - max_vals[l2c_msg_scatter_index]

            # # Step 3: Compute the exponentials of the shifted values and sum within each group
            # exp_vals = shifted_vals.exp()  # (N, M)
            # sum_exp_vals = torch.zeros((num_groups, M), device=l2c_msg_aggr.device)
            # sum_exp_vals = sum_exp_vals.index_add_(0, l2c_msg_scatter_index, exp_vals)

            # # Step 4: Compute logsumexp by taking the log of summed exponentials and adding back max values
            # logsumexp_vals = (sum_exp_vals.log() + max_vals)





            # # Step 2: Create a list to hold the sublist
            # # Collect sublists
            # # pdb.set_trace()
            # sublists = [l2c_msg_aggr[torch.where(l2c_msg_scatter_index == idx)] for idx in unique_indices]

            # print("sublists")
            # print(sublists[0].requires_grad)
            
            # # l2c_msg = torch.zeros((num_edges, self.opts.dim), dtype=l2c_msg_aggr.dtype, device=self.opts.device)
            # l2c_msg = torch.stack([torch.logsumexp(sublist, dim=0) for sublist in sublists], dim=0)
            # # for i, sublist in enumerate(sublists):
            # #     l2c_msg[i] = torch.logsumexp(sublist, dim=0)

            # print("l2c_msg")
            # print(l2c_msg.requires_grad)

            # for row, vec in zip(l2c_msg_aggr_scatter_index, l2c_msg_aggr):
            #     if row != prev_row:
            #         i = 0
            #     else:
            #         i += 1
            #     l2c_msg_aggr_matrix[row, i] = vec

            # l2c_msg = torch.logsumexp(l2c_msg_aggr_matrix, dim=1)

            # Wat gebeurt er als je van deze LSE (max) bijv. een som maakt
            # zodat alle mogelijke satisfying assignments van clause meetellen, en niet alleen de meest waarchijnlijke

            # NEW PYTORCH
            l2c_msg = torch.log(torch.zeros((len(torch.unique(l2c_msg_scatter_index)),self.opts.dim), dtype=l2c_msg_aggr.dtype, device=self.opts.device).scatter_reduce(
                0, l2c_msg_scatter_index.unsqueeze(1).expand(-1, self.opts.dim), torch.exp(shifted_vals), reduce="sum"))
            
            l2c_msg = l2c_msg + max_vals
            # if torch.isnan(l2c_msg).any() or torch.isinf(l2c_msg).any():
            #     raise ValueError("l2c_msg contains NaN or Inf values")
            # l2c_msg = torch.zeros((num_edges,self.opts.dim), dtype=l2c_msg_aggr.dtype, device=self.opts.device).scatter_reduce(
            #     0, l2c_msg_scatter_index.unsqueeze(1).expand(-1, self.opts.dim), l2c_msg_aggr, reduce="sum"  # Replace with appropriate method if logsumexp functionality is needed
            # )    
            # l2c_msg = torch.logsumexp(l2c_msg, dim=1, keepdim=True)
            # l2c_msg = scatter_logsumexp(l2c_msg_aggr, l2c_msg_scatter_index, dim=0, dim_size=num_edges)
            ###

            # In PAPER, message c-l zou LSE (log sum exp) moeten zijn, over elke satisfying assingment voor c
            # per satisfying assignment een som over alle l->c messages voor de matchende literals l
            c2l_edges_feat = self.c2l_msg_update(l2c_msg)

            # Er zijn geen node features??
            # num_edges & c2l_msg shape zijn even groot, c2l_msg_repeat & c2l_msg_scatter ook even groot
            # import pdb; pdb.set_trace()

        if self.opts.task == 'model-counting':
            raise NotImplementedError
            # c_blf_aggr = scatter_sum(l2c_edges_feat[c_blf_repeat_index], c_blf_scatter_index, dim=0, dim_size=c_blf_norm_index.shape[0])
            # c_blf_aggr = self.c_readout(c_blf_aggr)
            # c_blf_norm = scatter_logsumexp(c_blf_aggr, c_blf_norm_index, dim=0, dim_size=c_size)
            # c_norm_blf = c_blf_aggr - c_blf_norm[c_blf_norm_index]
            # c_bethe = -scatter_sum(c_norm_blf * c_norm_blf.exp(), c_blf_norm_index, dim=0, dim_size=c_size).reshape(-1)

            # l_blf_aggr = scatter_sum(c2l_edges_feat, sign_l_edge_index, dim=0, dim_size=l_size)
            # l_blf_aggr = self.l_readout(l_blf_aggr)
            # v_blf_aggr = l_blf_aggr.reshape(-1, 2)
            # v_blf_norm = torch.logsumexp(v_blf_aggr, dim=1, keepdim=True)
            # v_norm_blf = v_blf_aggr - v_blf_norm
            # v_bethe = (v_degrees - 1) * ((v_norm_blf * v_norm_blf.exp()).sum(dim=1))

            # return scatter_sum(c_bethe, c_batch, dim=0, dim_size=data.l_size.shape[0]) + \
            #     scatter_sum(v_bethe, v_batch, dim=0, dim_size=data.l_size.shape[0])
        else:
            # NEW PYTORCH
            l_logit = torch.zeros((l_size, self.opts.dim), dtype=c2l_edges_feat.dtype, device=self.opts.device).scatter_reduce(
                0, sign_l_edge_index.unsqueeze(1).expand(-1, self.opts.dim), c2l_edges_feat, reduce="sum"
            )
            # l_logit = scatter_sum(c2l_edges_feat, sign_l_edge_index, dim=0, dim_size=l_size)
            ####
            l_logit = self.l_readout(l_logit)
            v_logit = l_logit.reshape(-1, 2)
            return self.softmax(v_logit)
            return self.softmax(v_logit)

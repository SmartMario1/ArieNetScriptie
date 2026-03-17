import pdb

import torch


def zero_tensor(dims, device):
    return torch.zeros(dims, dtype=torch.float32, device=device)

def indices_to_2d(index_list, dim_size):
    """
    Change indices [0,1,2] to matrix [[0,0,0], [1,1,1], [2,2,2]],
    where the second dimension has size dim_size.
    """
    if len(index_list.size()) == 1:
        index_list = index_list.unsqueeze(1).expand(-1, dim_size)
    return index_list

# Should add this to a custom geometric package
def scatter_sum(indices, src, n_output, device):
    assert len(indices) == len(src), "Index list and source list must have the same size"

    # make a matrix of 'number of output features' to 'feature dimension'
    res = zero_tensor((n_output, src.size(1)), device=device)
    dim_to_sum = 0

    indices = indices_to_2d(indices, src.size(1))
    return res.scatter_reduce(dim_to_sum, indices, src, reduce="sum")

def scatter_logsumexp(indices, src, n_output, device):
    assert len(indices) == len(src), "Index list and source list must have the same size"

    # Get the 'max' vector for each output group, to subtract
    unique_indices = torch.unique(indices)
    max_vecs = torch.full((len(unique_indices), src.size(1)), float('-inf'), device=device)

    indices_2d = indices_to_2d(indices, src.size(1))
    max_vecs = max_vecs.scatter_reduce(0, indices_2d, src, reduce="amax")

    # Subtract the max vector for each group from all elements in that group
    shifted_src = src - max_vecs[indices]
    res = zero_tensor((n_output, src.size(1)), device=device)

    # LogSumExp, both res and max_vecs have one vector per output group
    res =  torch.log(res.scatter_reduce(0, indices_2d, torch.exp(shifted_src), reduce="sum"))
    # pdb.set_trace()
    return res + max_vecs

def swap_even_odd(tensor):
    indices = torch.arange(tensor.size(0))
    swapped_indices = indices.clone()
    swapped_indices[1::2], swapped_indices[::2] = indices[::2], indices[1::2] # Swap even and odd indices
    return tensor[swapped_indices]


def max_level(lst):
    return isinstance(lst, list) and max(map(max_level, lst)) + 1

from collections.abc import Iterable


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return flatten(x)
        else:
            return x

def split_ind_2_src_matrix(ind2src_matrix):
    """Transform a matrix, where each row has indices for a group,
    to two lists: one with group indices repeated by the number of elements in the row,
    and one with concatenated source indices.
    """
    # list of group inds times the number of elements for each group
    group_inds = torch.tensor([i for i,row in enumerate(ind2src_matrix) for _ in flatten(row)], dtype=torch.long)
    # list of all elements for group in order
    src_inds = torch.tensor(flatten(ind2src_matrix), dtype=torch.long)

    return group_inds, src_inds
    # list of all elements for group in order
    src_inds = torch.tensor(flatten(ind2src_matrix), dtype=torch.long)

    return group_inds, src_inds

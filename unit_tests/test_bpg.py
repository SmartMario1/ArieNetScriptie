import pytest
import numpy as np
import pdb

from nsnet.utils.dataset import BPG, OldBPG, SATDataset, OldTransform2BPG, BPGParamBuilder, MatrixBPG, MatrixBPGParamBuilder
from nsnet.utils.dataloader import get_dataloader
from nsnet.utils.utils import parse_cnf_file
import torch
from nsnet.utils.torch_utils import scatter_sum, scatter_logsumexp, split_ind_2_src_matrix

import os


def random_formulas(n_forms: int):
    all_n_vars = []
    all_clauses = []

    train_dir = '/home/arie/Documents/phd/code/NSNet/SATSolving/3-sat/train_first_4000'
    for _ in range(n_forms):
        file_index = np.random.randint(0, 4000)
        if len(str(file_index)) < 5:
            prefix = '0' * (5 - len(str(file_index)))
            file_index = f'{prefix}{file_index}'

        file_path = os.path.join(train_dir, f'{file_index}.cnf')
        form_n_vars, form_clauses = parse_cnf_file(file_path)
        all_n_vars.append(form_n_vars)
        all_clauses.append(form_clauses)

    return all_n_vars, all_clauses

def neighborhoods_per_edges(edge_indices_per_assignment, assignments_per_neigbhor, neighborhoods):

    assignments_per_edge = [[] for _ in range(len(set(edge_indices_per_assignment)))]
    for assignment_index, edge_index in enumerate(edge_indices_per_assignment):
        assignments_per_edge[int(edge_index)].append(assignment_index)

    neighborhood_per_assignment = [[] for _ in range(len(edge_indices_per_assignment))]
    for assignment_index, neighbor in zip(assignments_per_neigbhor, neighborhoods):
        neighborhood_per_assignment[assignment_index].append(int(neighbor))

    return [[tuple(neighborhood_per_assignment[a_i]) for a_i in assignments]
            for assignments in assignments_per_edge]
            
    # neighborhoods_per_edge = [[] for _ in range(len(edge_indices_per_assignment))]
    # for assignment_index, edge_index in enumerate(edge_indices_per_assignment):
    #     neighborhood_indices_for_assignment = [j for j, x in enumerate(assignments_per_neigbhor) if x == assignment_index]
    #     neighborhood = [int(n) for n in neighborhoods[neighborhood_indices_for_assignment]]
    #     neighborhoods_per_edge[int(edge_index)].append(tuple(neighborhood))

    # return neighborhoods_per_edge



# TODO: singleton clauses don't work
# TODO: make sure that the number of edges matches the receiving indices of l2c
# also note that duplicate clauses are removed in the new setup, but not in the old setup
def test_new_vs_old_bpg():
    hard = True
    if hard:
        n_forms = 5
        all_n_vars, all_clauses = random_formulas(n_forms)

    else:
        #Passes

        # Passes, because for some reason we get 0 assignments, summing to a vector of zeros (because we take the number of edges to sum over as size)
        # and some this vector to one edge (0)
        # clauses = [[2]]

        # Fails, because we again get 0 assignment, and a single edge to sum to.
        # because we use the size of the edges to sum to this gives a single all-0 vector, but then we try to get
        # the group corresponding to edge '1' which doesn't exist.
        # clauses = [[-2]]

        all_clauses = [[[1, 2], [1, 2, 3]]]
        # clauses = [[1, 2, 3], [-1, -2, -3], [1, -2, 3], [-1, 2, -3]]
        all_n_vars = [3]

    for form_index, (clauses, n_vars) in enumerate(zip(all_clauses, all_n_vars)):

        old_bpg = OldBPG(*OldTransform2BPG(n_vars, clauses, 'sat-solving'))
        new_bpg = BPG(*BPGParamBuilder(clauses).params)

        assert torch.equal(new_bpg.literal_indices_per_edge, old_bpg.sign_l_edge_index)
        assert torch.equal(new_bpg.c2l_msg_receiver_indices, old_bpg.c2l_msg_scatter_index)
        assert torch.equal(new_bpg.c2l_msg_sender_indices, old_bpg.c2l_msg_repeat_index)

        num_edges = len(new_bpg.literal_indices_per_edge)
        fake_features = torch.randn(num_edges, 16)

        # Old update

        # The SIZE here SHOULD BE THE NUMBER OF UNIQUE ASSIGNMENTS!! (not the number of edges)
        old_l2c_msg_aggr = scatter_sum(
            old_bpg.l2c_msg_aggr_scatter_index, fake_features[old_bpg.l2c_msg_aggr_repeat_index], len(old_bpg.l2c_msg_scatter_index), 'cpu')

        old_l2c_msg = scatter_logsumexp(old_bpg.l2c_msg_scatter_index, old_l2c_msg_aggr, n_output=num_edges, device='cpu' )

        # New update
        new_l2c_msg_argument_per_assignment = scatter_sum(
            new_bpg.l2c_assignment_indices, fake_features[new_bpg.l2c_assignment_neighborhoods], len(new_bpg.l2c_msg_receiver_indices), 'cpu')

        new_l2c_msg = scatter_logsumexp(new_bpg.l2c_msg_receiver_indices, new_l2c_msg_argument_per_assignment, num_edges, 'cpu')


        # Store the neighborhoods for each edge and check that they are the same:
        old_neighborhoods_per_edge = neighborhoods_per_edges(edge_indices_per_assignment=old_bpg.l2c_msg_scatter_index,
                                                  assignments_per_neigbhor=old_bpg.l2c_msg_aggr_scatter_index,
                                                  neighborhoods=old_bpg.l2c_msg_aggr_repeat_index)
        
        new_neighborhoods_per_edge = neighborhoods_per_edges(edge_indices_per_assignment=new_bpg.l2c_msg_receiver_indices,
                                                  assignments_per_neigbhor=new_bpg.l2c_assignment_indices,
                                                  neighborhoods=new_bpg.l2c_assignment_neighborhoods)
        for i in range(num_edges):
            if not set(old_neighborhoods_per_edge[i]) == set(new_neighborhoods_per_edge[i]):
                print(old_neighborhoods_per_edge[i])
                print(new_neighborhoods_per_edge[i])
                print("\n")
                raise AssertionError("Different neighborhoods mapped to the same edge with old vs new")

        assert old_l2c_msg.shape == new_l2c_msg.shape

        difference = torch.sum(torch.abs(old_l2c_msg - new_l2c_msg))
        assert torch.all(torch.isclose(old_l2c_msg, new_l2c_msg, atol=1e-5)), f"Difference between old and new msgs: {difference}"

        print(f"formula {form_index} ok")
 
    print("BPG test passed")

# TODO add test for MatrixBPG

def test_matrix_bpg():
    hard = True
    if hard:
        n_forms = 1
        all_n_vars, all_clauses = random_formulas(n_forms)
    else:
        all_clauses = [[[1, 2], [1, 2, 3]]]
        all_n_vars = [3]

    for form_index, (clauses, n_vars) in enumerate(zip(all_clauses, all_n_vars)):
        matrix_bpg = MatrixBPG(*MatrixBPGParamBuilder(clauses).params)
        old_bpg = OldBPG(*OldTransform2BPG(n_vars, clauses, 'sat-solving'))

        # THIS FAILS FOR DUPLICATE CLAUSES
        num_edges = sum(len(c) for c in clauses)*2
        fake_features = torch.randn(num_edges, 16)

        #  Check that the c2l maps (from edge to literal neighborhoods) are the same
        matrix_c2l_receiver_indices, matrix_c2l_sender_indices = split_ind_2_src_matrix(matrix_bpg.edge_to_literal_neighborhood_matrix)
        old_c2l_receiver_indices, old_c2l_sender_indices = old_bpg.c2l_msg_scatter_index, old_bpg.c2l_msg_repeat_index

        assert torch.equal(matrix_c2l_receiver_indices, old_c2l_receiver_indices)
        assert torch.equal(matrix_c2l_sender_indices, old_c2l_sender_indices)

        old_literal_to_edges = [[] for _ in range(len(matrix_bpg.literal_to_edges_matrix))]
        for edge_index, literal_index in enumerate(old_bpg.sign_l_edge_index):
            old_literal_to_edges[int(literal_index)].append(edge_index)


        # Check that the literal to edge mapping is the same
        for i in range(len(matrix_bpg.literal_to_edges_matrix)):
            assert set(matrix_bpg.literal_to_edges_matrix[i]) == set(old_literal_to_edges[i])


        # Check that the clause to literal mapping is the same
        old_clause_to_literals = [[] for _ in range(len(matrix_bpg.clause_to_literals_matrix))]
        for c_index, l_index in zip(old_bpg.c_edge_index, old_bpg.l_edge_index):
            old_clause_to_literals[int(c_index)].append(int(l_index))

        for i in range(len(matrix_bpg.clause_to_literals_matrix)):
            assert set(matrix_bpg.clause_to_literals_matrix[i]) == set(old_clause_to_literals[i])

        # Check that the assignment neighborhoods per edge are the same
        matrix_neighborhoods_per_edge = [[tuple(matrix_bpg.clause_assignment_to_neighborhood_matrix[a_i])
                                          for a_i in assignments]
                                        for assignments in matrix_bpg.edge_to_clause_assignments_matrix]

        old_neighborhoods_per_edge = neighborhoods_per_edges(edge_indices_per_assignment=old_bpg.l2c_msg_scatter_index,
                                                  assignments_per_neigbhor=old_bpg.l2c_msg_aggr_scatter_index,
                                                  neighborhoods=old_bpg.l2c_msg_aggr_repeat_index)

        for i in range(num_edges):
            assert set(matrix_neighborhoods_per_edge[i]) == set(old_neighborhoods_per_edge[i])

        print("Matrix BPG test passed")



test_matrix_bpg()
# test_new_vs_old_bpg()
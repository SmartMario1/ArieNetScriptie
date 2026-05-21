import copy
import glob
import itertools
import os
import pdb
import pickle
import random
import time
import typing
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal, NamedTuple, Union

import numpy as np
import torch
from nsnet.utils.utils import literal2l_idx, parse_cnf_file
from torch_geometric.data import Data, Dataset
from tqdm import tqdm


class LCG(Data):
    def __init__(self, 
            l_size=None, 
            c_size=None, 
            c_edge_index=None, 
            l_edge_index=None,
            l_batch=None,
            c_batch=None
        ):
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.c_edge_index = c_edge_index
        self.l_edge_index = l_edge_index
        self.l_batch = l_batch
        self.c_batch = c_batch
       
    @property
    def num_edges(self):
        return self.c_edge_index.size(0)
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'c_edge_index':
            return self.c_size
        elif key == 'l_edge_index':
            return self.l_size
        elif key == 'l_batch' or key == 'c_batch':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)

@dataclass
class VariableNode:
    var_number: int
    edges_around: list = field(default_factory=list)

# Every clause should be connected to both the + and the - literals for each var in it
# since we can use both when summing over the assignments

@dataclass
class ClauseNode:
    """ Node representing a disjunctive clause"""
    content: typing.Tuple[int, ...]
    edges_around: list = field(default_factory=list)

    def satisfied_by(self, assignment: typing.Tuple[bool]):
        assert len(assignment) == len(self.content)
        assert all(type(v)==bool for v in assignment)
        return any([(var>0) == valuation for var, valuation in zip(self.content, assignment)])
    
    def satisfied_by_dict(self, assignment_dct):
        # Removed assertions for performance (hot path)
        for literal in self.content:
            if (literal > 0) == assignment_dct[abs(literal)]:
                return True
        return False


class Edge(NamedTuple):
    variable_node: VariableNode
    clause_node: ClauseNode
    sign: Literal["+", "-"]

    # SHOULD I store those here, as well as in the list?
    index: int

class BPGParams(NamedTuple):
    n_clauses: int
    n_literals: int

    literal_indices_per_edge: torch.Tensor
    literal_indices_per_occurence: torch.Tensor

    clause_indices_per_occurence: torch.Tensor
    local_satisfaction_percentage_per_edge: torch.Tensor

    c2l_msg_receiver_indices: torch.Tensor
    c2l_msg_sender_indices: torch.Tensor
    l2c_msg_receiver_indices: torch.Tensor
    l2c_assignment_indices: torch.Tensor
    l2c_assignment_neighborhoods: torch.Tensor

    # Per-literal UP features, shape (2*n_vars, 4); None when not computed.
    # Column layout: [up_reach_pct, forced_count_pct, conflict_flag, log_up_reach]
    up_features_per_literal: torch.Tensor = None


class MatrixBPG(Data):
    def __init__(self,
        n_clauses = None, # In batching, a list of n_clauses per graph
        n_literals = None,

        literal_to_edges_matrix=None,
        clause_to_literals_matrix=None,
        edge_to_literal_neighborhood_matrix=None,
        edge_to_clause_assignments_matrix=None,
        clause_assignment_to_neighborhood_matrix=None
        ):
        super().__init__()

        # Made into a list of ints when bathing
        self.n_clauses = n_clauses
        self.n_literals = n_literals

        self.literal_to_edges_matrix = literal_to_edges_matrix
        self.clause_to_literals_matrix = clause_to_literals_matrix
        self.edge_to_literal_neighborhood_matrix = edge_to_literal_neighborhood_matrix
        self.edge_to_clause_assignments_matrix = edge_to_clause_assignments_matrix
        self.clause_assignment_to_neighborhood_matrix = clause_assignment_to_neighborhood_matrix

    @property
    def n_edges(self):
        return len(self.edge_to_literal_neighborhood_matrix)
    
    def __inc__(self, key, value, *args, **kwargs):
        """
        Used to facilitate graph stacking, this is used for batching.
        """
        # add the number of literals
        if key == "clause_to_literals_matrix":
            return self.n_literals
        
        # Add the number of edges
        elif key in ["literal_to_edges_matrix", "edge_to_literal_neighborhood_matrix", "clause_assignment_to_neighborhood_matrix"]:
            return self.n_edges

        # Add the number of assignments
        elif key == "edge_to_clause_assignments_matrix":
            return len(self.clause_assignment_to_neighborhood_matrix)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class MatrixBPGParams(NamedTuple):
    n_clauses: int
    n_literals: int

    literal_to_edges_matrix: typing.List
    clause_to_literals_matrix: typing.List
    edge_to_literal_neighborhood_matrix: typing.List
    edge_to_clause_assignments_matrix: typing.List
    clause_assignment_to_neighborhood_matrix: typing.List

class MatrixBPGParamBuilder():
    def __init__(self,
                 cnf: typing.List[typing.List[int]]):

        # Remove duplicates, but keep order (for consistency)
        self.all_clauses = list(dict.fromkeys([tuple(clause) for clause in cnf]))
        self.all_vars = set([abs(l) for clause in self.all_clauses for l in clause])

        self.build_nodes_edges()

        # Fill clause neighborhoods and literal neighborhoods for each edge
        literal_to_edges_matrix = self.literal_to_edge_matrix()

        #output:
        clause_to_literals_matrix = [[self.cnf_literal_to_literal_index(l) for l in c] for c in self.all_clauses]
        edge_to_literal_neighborhood_matrix = self.c2l_message_input()
        edge_to_clause_assignments_matrix, clause_assignment_to_neighborhood_matrix = self.l2c_satisfying_assignment_sum_input()

        self.params = MatrixBPGParams(
            n_clauses = len(self.all_clauses),
            n_literals = len(self.all_vars)*2,
            
            literal_to_edges_matrix = literal_to_edges_matrix,
            clause_to_literals_matrix = clause_to_literals_matrix,
            edge_to_literal_neighborhood_matrix = edge_to_literal_neighborhood_matrix,
            edge_to_clause_assignments_matrix = edge_to_clause_assignments_matrix,
            clause_assignment_to_neighborhood_matrix = clause_assignment_to_neighborhood_matrix
        )

    def cnf_literal_to_literal_index(self, literal: int):
        return 2*(abs(literal)-1) if literal > 0 else 2*(abs(literal)-1)+1

    def edges_around(self, node: Union[ClauseNode,VariableNode]) -> typing.List[Edge]:
        if isinstance(node, VariableNode):
            return tuple([edge for edge in self.edges if edge.variable_node == node])
        
        elif isinstance(node, ClauseNode):
            return tuple([edge for edge in self.edges if edge.clause_node == node])
        raise ValueError("Node type not recognized")

    def build_nodes_edges(self):
        var_node_dict = {var: VariableNode(var_number=var) for var in self.all_vars}
        clause_node_dict = {clause: ClauseNode(content=clause) for clause in self.all_clauses}

        edges = []
        edge_index = 0
        for c in self.all_clauses:
            for l in c:
                edges.append(Edge(variable_node=var_node_dict[abs(l)], clause_node=clause_node_dict[c], index=edge_index, sign="+"))
                edge_index += 1

                edges.append(Edge(variable_node=var_node_dict[abs(l)], clause_node=clause_node_dict[c], index=edge_index, sign="-"))
                edge_index += 1
        self.edges = edges

        # fill edges_around attribute for each node
        for node in list(var_node_dict.values())+list(clause_node_dict.values()):
            node.edges_around = self.edges_around(node)

    def clause_neighborhood(self, edge: Edge) -> typing.List[Edge]:
        """
        Returns all edges around the clause node, + and -
        """
        return edge.clause_node.edges_around

    def literal_neighborhood(self, edge: Edge):
        """
        Returns the edges around the variable node, restricted to edges with the same sign
        Used in l2c message.
        """
        return [neighbor_edge for neighbor_edge in edge.variable_node.edges_around if edge.sign == neighbor_edge.sign]


    def literal_to_edge_matrix(self):
        n_literals = len(self.all_vars)*2
        literal_to_edge_matrix = [[] for _ in range(n_literals)]

        edge_index = 0
        for c in self.all_clauses:
            for l in c:
                literal_to_edge_matrix[self.cnf_literal_to_literal_index(abs(l))].append(edge_index)
                edge_index += 1

                literal_to_edge_matrix[self.cnf_literal_to_literal_index(-abs(l))].append(edge_index)
                edge_index += 1

        return literal_to_edge_matrix

    def c2l_message_input(self) -> typing.Tuple[torch.tensor, torch.tensor]:
        """
        return two lists,
        one with x times the edge index
        one with the x indices of other edges going out from the c
        """
        edge_to_literal_neighborhood_matrix = [[] for _ in self.edges]

        # edge l-c
        for i, edge in enumerate(self.edges):
            # Get all edges to other clauses in which l occurs (with the same sign)
            neighborhood = self.literal_neighborhood(edge)
            punctured_neighborhood = [neighbor_edge for neighbor_edge in neighborhood if neighbor_edge != edge]
            edge_to_literal_neighborhood_matrix[i] = [neighbor_edge.index for neighbor_edge in punctured_neighborhood]

        return edge_to_literal_neighborhood_matrix
    
    def clause_assignment_neighborhood(self, edge: Edge, assignment: typing.List[bool]) -> typing.List[Edge]:
        """
        Return the neighboring edges for some assignment, not including the edge itself

        Parameters
        ----------
        edge : Edge
            The edge to the clause node
        assignment : list[bool]
            List of booleans of the size of the clause, in order of the literals in the clause

        Returns
        -------
            List of one less element than the assignment,
            with all edges to literals matching the assignment except for the given edge.
        """
        different_var_neighbors = self.clause_neighborhood(edge)
        clause_assignment_edges = []
        for valuation, literal in zip(assignment, edge.clause_node.content):

            # We don't return the current edge
            if abs(literal) == edge.variable_node.var_number:
                continue
            # Add the edge matching the assignment (+ or -) and variable
            else:
                sign = "+" if valuation else "-"
                try:
                    matching_edge = next(e for e in different_var_neighbors if e.variable_node.var_number == abs(literal) and e.sign==sign)
                except StopIteration:
                    raise ValueError(f"No matching edge found for clause {edge.clause_node.content} and assignment {assignment}")
                
                clause_assignment_edges.append(matching_edge)

        return clause_assignment_edges
    

    def l2c_satisfying_neighborhoods(self, edge: Edge):
        """
        return a list of lists of edge indices
        each sublist matching a punctured literal neighborhood satisying the clause c

        In oorspronkelijke BPG: 
        voor elke l-c occurence (twee edges)hebben we een lijst van assignment-neiborhoods als [0,0,0], [0,0,1], etc.
        waar 0 '+' betekent
        """

        # The valuation of the var of this edge, according to its sign
        current_var_assignment = edge.sign == "+"
        variable_index_in_clause = [abs(l) for l in edge.clause_node.content].index(edge.variable_node.var_number)

        # All possible assignments of the other variables in the clause
        var_assignments_for_neighbors = list(itertools.product([True, False], repeat=len(edge.clause_node.content)-1))
        var_assignments_for_neighbors = [list(assignment) for assignment in var_assignments_for_neighbors]

        # import pdb; pdb.set_trace()
        satisfying_neighborhoods = []
        for neighborhood_assignment in var_assignments_for_neighbors:

            # Add the current literal, since it may also satisfy the clause
            neighborhood_assignment.insert(variable_index_in_clause, current_var_assignment)
            if edge.clause_node.satisfied_by(neighborhood_assignment):
                # Get the neighborhood for this assignment, not including the edge itself
                satisfying_neighborhoods.append(self.clause_assignment_neighborhood(edge, neighborhood_assignment))

        return satisfying_neighborhoods
    
    def l2c_satisfying_assignment_sum_input(self):
        """
        returns three lists,

        1) the edge numbers in order, repeated for the number of satisfying punctured neighborhoods per edge
        2) a numbering of the satisfying neighborhoods for all edges, each repeated for the size of the punctured neighborhood
        3) all edge numbers of satisfying punctured neighborhoods in order
        """
        edge_to_clause_assignment_matrix = [[] for _ in self.edges]
        clause_assignment_to_neighborhood_matrix = []

        assignment_counter = 0
        for edge_index, edge in enumerate(self.edges):
            satisfying_neighborhoods = self.l2c_satisfying_neighborhoods(edge)

            for neighborhood in satisfying_neighborhoods:
                edge_to_clause_assignment_matrix[edge_index].append(assignment_counter)
                clause_assignment_to_neighborhood_matrix.append([neighbor_edge.index for neighbor_edge in neighborhood])
                assignment_counter += 1

        return edge_to_clause_assignment_matrix, clause_assignment_to_neighborhood_matrix


class BPG(Data):
    """
    Storage class for the data in the GNN, including __inc__ methods that automatically update
    all the indices when stacking multiple data objects into batches.
    """
    def __init__(self,
                 # This should become a map from edge_index to literal_index
        n_clauses = None, # In batching, a list of n_clauses per graph
        n_literals = None,

        literal_indices_per_edge=None,
        literal_indices_per_occurence=None,
        clause_indices_per_occurence=None,
        local_satisfaction_percentage_per_edge=None,

        c2l_msg_receiver_indices=None,
        c2l_msg_sender_indices=None,
        l2c_msg_receiver_indices=None,
        l2c_assignment_indices=None,
        l2c_assignment_neighborhoods=None,

        # Per-literal UP features, shape (2*n_vars, 4); None when not computed.
        # Columns: [up_reach_pct, forced_count_pct, conflict_flag, log_up_reach]
        up_features_per_literal=None,

        # Canonical subgraph features
        subgraphs_p1=None,  # List of EdgeSubgraph objects for p=1
        subgraphs_p0=None,  # List of EdgeSubgraph objects for p=0

        # Literal co-occurrence edges (L2L)
        cooc_src_indices=None,
        cooc_dst_indices=None,
    ):
        super().__init__()

        # Made into a list of ints when bathing
        self.n_clauses = n_clauses
        self.n_literals = n_literals

        self.literal_indices_per_edge = literal_indices_per_edge
        self.literal_indices_per_occurence = literal_indices_per_occurence

        self.clause_indices_per_occurence = clause_indices_per_occurence
        self.local_satisfaction_percentage_per_edge = local_satisfaction_percentage_per_edge

        self.c2l_msg_receiver_indices = c2l_msg_receiver_indices
        self.c2l_msg_sender_indices = c2l_msg_sender_indices

        self.l2c_msg_receiver_indices = l2c_msg_receiver_indices
        self.l2c_assignment_indices = l2c_assignment_indices
        self.l2c_assignment_neighborhoods = l2c_assignment_neighborhoods

        # Per-literal UP features
        self.up_features_per_literal = up_features_per_literal

        # Canonical subgraph features
        self.subgraphs_p1 = subgraphs_p1
        self.subgraphs_p0 = subgraphs_p0

        # Literal co-occurrence edges (L2L)
        self.cooc_src_indices = cooc_src_indices
        self.cooc_dst_indices = cooc_dst_indices


    @property
    def n_edges(self):
        return self.literal_indices_per_edge.size(0)        
    
    def __inc__(self, key, value, *args, **kwargs):
        """
        Used to facilitate graph stacking, this is used for batching.
        """

        # add the number of clauses
        if key == "clause_indices_per_occurence":
            return self.n_clauses
        
        elif key in ("local_satisfaction_percentage_per_edge",
                      "up_features_per_literal"):
            # Feature tensors: values are not indices, do not increment.
            return 0

        # Add the number of literals
        elif key in ["literal_indices_per_edge", "literal_indices_per_occurence"]:
            return self.n_literals

        # Add the number of edges
        elif key == "c2l_msg_receiver_indices" or key == "c2l_msg_sender_indices" \
            or key == "l2c_msg_receiver_indices"  or key == "l2c_assignment_neighborhoods":
            return len(self.literal_indices_per_edge)            

        # Add the number of assignments
        elif key == "l2c_assignment_indices":
            return len(self.l2c_msg_receiver_indices)

        # Literal co-occurrence edges index literals
        elif key in ["cooc_src_indices", "cooc_dst_indices"]:
            return self.n_literals
        else:
            return super().__inc__(key, value, *args, **kwargs)


def _compute_c2l_inputs_fast(
    clauses: typing.List[typing.List[int]],
    n_vars: int,
    show_progress: bool = False,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized c2l message-input computation.

    For each BPG edge (variable v, clause c, sign s), the c2l message input
    lists every other edge that shares the same (variable, sign) — i.e. the
    literal neighbourhood.  The original implementation loops over all edges
    individually and traverses Python objects; here we build the result with
    numpy grouping, cutting the outer loop from n_edges (~100 k) to n_vars
    (~500) iterations.

    Edge indices follow the same convention as BPGParamBuilder:
      + edge for the k-th literal occurrence = index 2k
      − edge for the k-th literal occurrence = index 2k+1
    """
    n_occ = sum(len(c) for c in clauses)
    edge_vars = np.empty(n_occ, dtype=np.int64)
    k = 0
    for clause in clauses:
        for lit in clause:
            edge_vars[k] = abs(lit) - 1
            k += 1

    occ_idx = np.arange(n_occ, dtype=np.int64)
    recv_parts: typing.List[np.ndarray] = []
    send_parts: typing.List[np.ndarray] = []

    for v in tqdm(range(n_vars), desc="c2l", disable=not show_progress, leave=False):
        occ_of_v = occ_idx[edge_vars == v]
        n = len(occ_of_v)
        if n <= 1:
            continue
        for sign_offset in (0, 1):          # 0 → + edges, 1 → − edges
            edge_idxs = 2 * occ_of_v + sign_offset
            # All ordered non-self pairs via repeat / tile
            recv = np.repeat(edge_idxs, n)
            send = np.tile(edge_idxs, n)
            mask = recv != send
            recv_parts.append(recv[mask])
            send_parts.append(send[mask])

    if recv_parts:
        receivers = np.concatenate(recv_parts)
        senders   = np.concatenate(send_parts)
    else:
        receivers = np.array([], dtype=np.int64)
        senders   = np.array([], dtype=np.int64)

    return torch.from_numpy(receivers), torch.from_numpy(senders)


def _compute_l2c_inputs_fast(
    clauses: typing.List[typing.List[int]],
    show_progress: bool = False,
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized l2c satisfying-assignment-sum-input computation.

    For each BPG edge (v, c, s) this enumerates all satisfying assignments of
    the other literals in clause c and records the corresponding edge indices.

    Key differences from the original Python-object implementation:
    * `itertools.product` is called **once per clause** (not once per edge)
      by working in the full 2^k assignment space.
    * Assignment enumeration and clause-satisfaction checks are numpy ops.
    * Output lists are built with np.repeat / np.where rather than Python
      per-assignment loops, then concatenated once at the end.

    For clauses longer than 20 literals (2^20 = 1 M rows) random sampling
    is used instead of exhaustive enumeration.
    """
    MAX_K_EXACT = 20

    edge_idx_parts:  typing.List[np.ndarray] = []
    asgn_idx_parts:  typing.List[np.ndarray] = []
    neigh_parts:     typing.List[np.ndarray] = []
    assignment_counter = 0
    clause_start = 0          # absolute edge-index offset for the current clause

    for clause in tqdm(clauses, desc="l2c", disable=not show_progress, leave=False):
        k = len(clause)
        lit_signs = np.array([l > 0 for l in clause], dtype=bool)  # (k,)
        pos_arr   = np.arange(k, dtype=np.int64)
        edge_plus  = clause_start + 2 * pos_arr        # + edge index per position
        edge_minus = edge_plus + 1                      # − edge index per position

        # ---- generate all 2^k full assignments (or sample for very long clauses) ----
        if k <= MAX_K_EXACT:
            n_asgn   = 1 << k
            all_asgn = (
                (np.arange(n_asgn, dtype=np.int32)[:, None]
                 >> np.arange(k - 1, -1, -1)[None, :]) & 1
            ).astype(bool)                              # (2^k, k)
        else:
            n_asgn   = 1024
            all_asgn = np.random.randint(0, 2, size=(n_asgn, k)).astype(bool)

        # Which full assignments satisfy the clause?
        clause_sat = (all_asgn == lit_signs[np.newaxis, :]).any(axis=1)  # (n_asgn,)

        for pos in range(k):
            # Indices and edge-indices for the other positions
            other_pos  = np.concatenate([pos_arr[:pos], pos_arr[pos + 1:]])
            ep_other   = edge_plus[other_pos]    # (k-1,) + edges for other positions
            em_other   = edge_minus[other_pos]   # (k-1,) − edges

            for s_bool, e_idx in ((True, int(edge_plus[pos])), (False, int(edge_minus[pos]))):
                # Rows where this position is set to s_bool AND the clause is satisfied
                sat_rows = np.where((all_asgn[:, pos] == s_bool) & clause_sat)[0]
                n_sat = len(sat_rows)
                if n_sat == 0:
                    continue

                # Neighbourhood edge indices: (n_sat, k-1)
                sat_other  = all_asgn[np.ix_(sat_rows, other_pos)]
                neigh_edges = np.where(sat_other, ep_other, em_other).astype(np.int64)

                n_neigh   = k - 1
                asgn_nums = np.arange(
                    assignment_counter, assignment_counter + n_sat, dtype=np.int64
                )
                edge_idx_parts.append(np.full(n_sat, e_idx, dtype=np.int64))
                asgn_idx_parts.append(np.repeat(asgn_nums, n_neigh))
                neigh_parts.append(neigh_edges.ravel())
                assignment_counter += n_sat

        clause_start += 2 * k

    def _concat(parts: typing.List[np.ndarray]) -> torch.Tensor:
        arr = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        return torch.from_numpy(arr)

    return _concat(edge_idx_parts), _concat(asgn_idx_parts), _concat(neigh_parts)


def _compute_all_local_sat_percentages(
    clauses: typing.List[typing.List[int]],
    n_vars: int,
    n_samples: int = 500,
    show_progress: bool = False,
    _approx_threshold: int = 10_000,
) -> np.ndarray:
    """Compute local satisfaction percentages for all BPG edges.

    Uses the exact per-occurrence code path with memoization on
    (v, shared_clause_indices).  The approximate per-variable path is
    retained below but intentionally never activated: it has a systematic
    downward bias (empirical mean ≈ 0.062 vs exact ≈ 0.151, RMSE ≈ 0.155,
    Pearson r ≈ 0.89) because it evaluates satisfaction over ALL clauses
    containing v rather than only the shared k2-neighbourhood clauses.
    It may be worth revisiting with a better neighbourhood approximation.

    Returns float32 array of length 2 * sum(len(c) for c in clauses),
    ordered identically to BPGParamBuilder.edges:
      index 2*k   → + edge (v assigned True)
      index 2*k+1 → − edge (v assigned False)
    """
    n_occ = sum(len(c) for c in clauses)
    use_approx = False  # DISABLED: approximate path has unacceptable accuracy

    # Build var → clause-index adjacency (needed by both paths)
    var_to_clause_set: typing.List[set] = [set() for _ in range(n_vars)]
    for ci, clause in enumerate(clauses):
        for lit in clause:
            v = abs(lit) - 1
            if v < n_vars:
                var_to_clause_set[v].add(ci)

    # ------------------------------------------------------------------
    # APPROXIMATE PATH (DISABLED — inaccurate, kept for future reference)
    #
    # Computes one value per variable over ALL clauses containing v, rather
    # than the correct per-occurrence k2-neighbourhood subset.  This causes
    # a systematic downward bias because it enforces a stricter (larger)
    # shared-clause set than the exact definition.
    #
    # Empirical accuracy on crypto/hitag2 instances (n=5):
    #   MAE=0.089, RMSE=0.155, Pearson r=0.89, approx mean=0.062 vs exact mean=0.151
    #
    # To re-enable: change `use_approx = False` above to
    #   use_approx = n_occ > _approx_threshold
    # and investigate whether a tighter neighbourhood approximation
    # (e.g. only clauses sharing ≥2 variables with c) improves accuracy.
    # ------------------------------------------------------------------
    if use_approx:
        local_sat_pos = np.ones(n_vars, dtype=np.float32)
        local_sat_neg = np.ones(n_vars, dtype=np.float32)

        for v in tqdm(range(n_vars), desc="local_sat", disable=not show_progress, leave=False):
            v_clause_idxs = sorted(var_to_clause_set[v])
            if not v_clause_idxs:
                continue

            other_vars_set: set = set()
            for ci in v_clause_idxs:
                for lit in clauses[ci]:
                    other_vars_set.add(abs(lit) - 1)
            other_vars_set.discard(v)
            other_vars_list = sorted(other_vars_set)
            n_other = len(other_vars_list)
            var_to_col = {w: j for j, w in enumerate(other_vars_list)}

            n_vc = len(v_clause_idxs)
            if n_other == 0:
                assignments = np.empty((1, 0), dtype=bool)
            elif (1 << n_other) <= n_samples:
                assignments = (
                    (np.arange(1 << n_other)[:, None] >> np.arange(n_other - 1, -1, -1)[None, :]) & 1
                ).astype(bool)
            else:
                assignments = np.random.randint(0, 2, size=(n_samples, n_other), dtype=np.uint8).view(bool)

            n_asgn = assignments.shape[0]
            pos_mask = np.zeros((n_vc, n_other), dtype=bool)
            neg_mask = np.zeros((n_vc, n_other), dtype=bool)
            v_pos_arr = np.zeros(n_vc, dtype=bool)
            v_neg_arr = np.zeros(n_vc, dtype=bool)
            for i, ci in enumerate(v_clause_idxs):
                for lit in clauses[ci]:
                    w = abs(lit) - 1
                    if w == v:
                        if lit > 0:
                            v_pos_arr[i] = True
                        else:
                            v_neg_arr[i] = True
                    elif n_other > 0:
                        j = var_to_col[w]
                        if lit > 0:
                            pos_mask[i, j] = True
                        else:
                            neg_mask[i, j] = True

            if n_other > 0:
                asgn_b = assignments[np.newaxis, :, :]
                clause_other_sat = (
                    (asgn_b & pos_mask[:, np.newaxis, :]).any(-1) |
                    (~asgn_b & neg_mask[:, np.newaxis, :]).any(-1)
                )
            else:
                clause_other_sat = np.zeros((n_vc, n_asgn), dtype=bool)

            all_sat_pos = (v_pos_arr[:, np.newaxis] | clause_other_sat).all(axis=0)
            all_sat_neg = (v_neg_arr[:, np.newaxis] | clause_other_sat).all(axis=0)
            local_sat_pos[v] = float(all_sat_pos.mean())
            local_sat_neg[v] = float(all_sat_neg.mean())

        result = np.empty(2 * n_occ, dtype=np.float32)
        idx = 0
        for clause in clauses:
            for lit in clause:
                v = abs(lit) - 1
                result[idx]     = local_sat_pos[v]
                result[idx + 1] = local_sat_neg[v]
                idx += 2
        return result

    # ------------------------------------------------------------------
    # EXACT PATH: per-occurrence with memoization on (v, shared_clause_indices)
    #
    # For structured instances (e.g. XOR-gate crypto) many occurrences share
    # identical shared_clause_indices → compute once and cache.  For random
    # 3-SAT every key is unique, but those instances are small and fast anyway.
    # ------------------------------------------------------------------
    results: typing.List[float] = []
    cache: typing.Dict[typing.Tuple, typing.Tuple[float, float]] = {}

    for ci, clause in tqdm(
        enumerate(clauses), desc="local_sat (exact)", total=len(clauses),
        disable=not show_progress, leave=False,
    ):
        for lit in clause:
            v = abs(lit) - 1

            other_vars_in_c = [abs(l2) - 1 for l2 in clause if abs(l2) - 1 != v]
            if not other_vars_in_c:
                results.append(1.0)
                results.append(1.0)
                continue

            # k2 neighbourhood: clauses sharing any non-v variable with ci
            k2_clauses: set = set()
            for w in other_vars_in_c:
                k2_clauses.update(var_to_clause_set[w])

            shared_clause_indices = sorted(var_to_clause_set[v] & k2_clauses)
            if not shared_clause_indices:
                results.append(1.0)
                results.append(1.0)
                continue

            # Cache: many occurrences (especially in XOR-gate crypto) share the
            # same shared_clause_indices for the same variable → reuse result.
            key = (v, tuple(shared_clause_indices))
            if key in cache:
                pos_val, neg_val = cache[key]
                results.append(pos_val)
                results.append(neg_val)
                continue

            # --- first time seeing this (v, neighbourhood) pair ---
            other_vars_set2: set = set()
            for sci in shared_clause_indices:
                for l2 in clauses[sci]:
                    other_vars_set2.add(abs(l2) - 1)
            other_vars_set2.discard(v)
            other_vars_list = sorted(other_vars_set2)
            n_other = len(other_vars_list)
            var_to_col = {w: j for j, w in enumerate(other_vars_list)}

            n_shared = len(shared_clause_indices)
            search_space = 1 << n_other
            if n_other == 0:
                assignments = np.empty((1, 0), dtype=bool)
            elif search_space <= n_samples:
                assignments = (
                    (np.arange(search_space)[:, None] >> np.arange(n_other - 1, -1, -1)[None, :]) & 1
                ).astype(bool)
            else:
                assignments = np.random.randint(0, 2, size=(n_samples, n_other)).astype(bool)

            n_asgn = assignments.shape[0]

            clause_other_sat_list: typing.List[np.ndarray] = []
            clause_v_pos_sat: typing.List[bool] = []
            clause_v_neg_sat: typing.List[bool] = []
            for sci in shared_clause_indices:
                other_sat = np.zeros(n_asgn, dtype=bool)
                v_pos = False
                v_neg = False
                for l2 in clauses[sci]:
                    w = abs(l2) - 1
                    if w == v:
                        if l2 > 0:
                            v_pos = True
                        else:
                            v_neg = True
                    elif n_other > 0:
                        col = var_to_col[w]
                        other_sat |= (assignments[:, col] == (l2 > 0))
                clause_other_sat_list.append(other_sat)
                clause_v_pos_sat.append(v_pos)
                clause_v_neg_sat.append(v_neg)

            for sign_pos, v_sat_flags in ((True, clause_v_pos_sat), (False, clause_v_neg_sat)):
                all_satisfied = np.ones(n_asgn, dtype=bool)
                for i in range(n_shared):
                    if not v_sat_flags[i]:
                        all_satisfied &= clause_other_sat_list[i]
                val = float(all_satisfied.mean())
                if sign_pos:
                    pos_val = val
                else:
                    neg_val = val

            cache[key] = (pos_val, neg_val)
            results.append(pos_val)
            results.append(neg_val)

    return np.array(results, dtype=np.float32)


class BPGParamBuilder():
    """
    This generates the GNN data from a CNF formula.
    I updated it to add the percentages of satisfying assignments for a clause neighborhood
    given that a variable is true or false. (Implementation 1)

    -------
    Possible future plan (Implementation 2):

    Make, in addition, a list of BPGParam objects for each edge neighborhood, matching the edge order.
    Then, to the BPG object, add a list of neighborhood BPG objects (potentially with batched lists, or with separate lists)

    Then, when running a forward pass of the larger GNN, concatenate ALL neighborhood BPG-lists
    make a 'neighborhood data' object with the concatenated lists
    Run the GNN2 on this large object, we get back an output feature for each edge neighborhood

    Use the neighborhood index to use each output feature for the associated edge.
    """
    # Class-level cache that persists across all instances (across all files)
    _persistent_cache = {}
    _persistent_cache_hits = 0
    _persistent_cache_misses = 0
    
    def __init__(self,
                 cnf: typing.List[typing.List[int]],
                 n_vars: int = None,
                 compute_local_satisfaction_percentages: bool = True,
                 compute_up_features: bool = False,
                 compute_c2l: bool = True,
                 compute_l2c: bool = True):
        init_start = time.time()
        
        all_vars = set([abs(l) for clause in cnf for l in clause])
        # Use n_vars from CNF header if provided, otherwise use the max variable number found in clauses
        # This ensures n_literals matches the labels which are based on the CNF header
        if n_vars is None:
            n_vars = max(all_vars) if all_vars else 0
        var_node_dict = {var: VariableNode(var_number=var) for var in all_vars}

        # Remove duplicates, but keep order (for consistency)
        all_clauses = list(dict.fromkeys([tuple(clause) for clause in cnf]))
        clause_node_dict = {clause: ClauseNode(content=clause) for clause in all_clauses}

        edges = []
        edge_index = 0
        literal_indices_per_occurence = []
        for clause in all_clauses:
            for l in clause:
            # Need to make sure that + en - edges have succeeding indices,
            # since this is used in 'forward', this is a bit ugly
            # The SAT-solver output has variables in order of their number
                edges.append(Edge(variable_node=var_node_dict[abs(l)], clause_node=clause_node_dict[clause], index=edge_index, sign="+"))
                edges.append(Edge(variable_node=var_node_dict[abs(l)], clause_node=clause_node_dict[clause], index=edge_index+1, sign="-"))
                edge_index += 2

        self.edges = edges
        
        # Precompute neighborhood lookup tables to avoid O(n_edges) scans
        self._edges_by_variable = {}  # var_number -> list of edges
        self._edges_by_clause = {}    # clause content (tuple) -> list of edges
        for edge in edges:
            var_num = edge.variable_node.var_number
            clause_content = edge.clause_node.content
            
            if var_num not in self._edges_by_variable:
                self._edges_by_variable[var_num] = []
            self._edges_by_variable[var_num].append(edge)
            
            if clause_content not in self._edges_by_clause:
                self._edges_by_clause[clause_content] = []
            self._edges_by_clause[clause_content].append(edge)
        
        setup_time = time.time() - init_start

        # Cache for local satisfaction percentages (label-agnostic)
        self._local_sat_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # A = time.time()
        # Show per-phase progress bars for instances large enough to be slow
        n_edges_approx = 2 * sum(len(c) for c in all_clauses)
        _progress = n_edges_approx > 20_000

        c2l_start = time.time()
        if compute_c2l:
            c2l_msg_receiver_indices, c2l_msg_sender_indices = _compute_c2l_inputs_fast(
                all_clauses, n_vars, show_progress=_progress
            )
        else:
            c2l_msg_receiver_indices = torch.empty(0, dtype=torch.long)
            c2l_msg_sender_indices   = torch.empty(0, dtype=torch.long)
        c2l_time = time.time() - c2l_start

        l2c_start = time.time()
        if compute_l2c:
            l2c_msg_receiver_indices, l2c_assignment_indices, l2c_assignment_neighborhoods = _compute_l2c_inputs_fast(
                all_clauses, show_progress=_progress
            )
        else:
            l2c_msg_receiver_indices      = torch.empty(0, dtype=torch.long)
            l2c_assignment_indices        = torch.empty(0, dtype=torch.long)
            l2c_assignment_neighborhoods  = torch.empty(0, dtype=torch.long)
        l2c_time = time.time() - l2c_start
        # print(f"c2l message input took {C-B} seconds")
        # D = time.time()
        # local_satisfaction_percentages = torch.tensor([0 for _ in edges], dtype=torch.float),

        local_sat_start = time.time()
        if compute_local_satisfaction_percentages:
            lsp_np = _compute_all_local_sat_percentages(
                [list(c) for c in all_clauses], n_vars, show_progress=_progress
            )
            local_satisfaction_percentages = torch.from_numpy(lsp_np)
        else:
            local_satisfaction_percentages = None
        local_sat_time = time.time() - local_sat_start

        # --- UP features (per-literal, shape 2*n_vars × 4) ---
        up_feat_start = time.time()
        if compute_up_features:
            from nsnet.utils.cnf_features import compute_up_features as _compute_up_features
            up_features = _compute_up_features(n_vars, [list(c) for c in all_clauses])
        else:
            up_features = None
        up_feat_time = time.time() - up_feat_start

        total_time = time.time() - init_start

        # Print timing if it takes more than 1 second
        if total_time > 1.0:
            print(f'[BPGParamBuilder] n_vars={n_vars}, n_clauses={len(all_clauses)}, n_edges={len(edges)}')
            print(f'[BPGParamBuilder] Timing: setup={setup_time:.2f}s, c2l={c2l_time:.2f}s, '
                  f'l2c={l2c_time:.2f}s, local_sat={local_sat_time:.2f}s, '
                  f'up_feat={up_feat_time:.2f}s, total={total_time:.2f}s')

        self.params = BPGParams(
            n_clauses = len(all_clauses),
            n_literals = n_vars * 2,
            literal_indices_per_edge = torch.tensor([self.edge_to_literal_index(e) for e in edges], dtype=torch.long),
            
            literal_indices_per_occurence = torch.tensor([2*(abs(l)-1) if l >0 else 2*(abs(l)-1)+1 for c in all_clauses for l in c]),
            clause_indices_per_occurence = torch.tensor([i for i, c in enumerate(all_clauses) for _ in c]),
            local_satisfaction_percentage_per_edge = local_satisfaction_percentages,
            
            c2l_msg_receiver_indices = c2l_msg_receiver_indices,
            c2l_msg_sender_indices = c2l_msg_sender_indices,
            l2c_msg_receiver_indices = l2c_msg_receiver_indices,
            l2c_assignment_indices = l2c_assignment_indices,
            l2c_assignment_neighborhoods = l2c_assignment_neighborhoods,
            up_features_per_literal = up_features,
        )

    def edge_to_literal_index(self, edge: Edge):
        var_index = edge.variable_node.var_number-1
        return 2*var_index + (0 if edge.sign == "+" else 1)
    
    def compute_canonical_subgraphs(self):
        """
        Compute canonical subgraph representations for all edges.
        
        Returns:
            Tuple of (subgraphs_p1, subgraphs_p0) where each is a list of EdgeSubgraph objects
        """
        from nsnet.utils.canonical_subgraph import compute_subgraph_features_for_all_edges
        
        return compute_subgraph_features_for_all_edges(self.edges, self.edges)
    
    def edges_around(self, node: Union[ClauseNode,VariableNode]) -> typing.List[Edge]:
        if isinstance(node, VariableNode):
            return self._edges_by_variable.get(node.var_number, [])
        
        elif isinstance(node, ClauseNode):
            return self._edges_by_clause.get(node.content, [])
        
        raise ValueError("Node type not recognized")
    
    def clause_neighborhood(self, edge: Edge) -> typing.List[Edge]:
        """
        Returns all edges around the clause node, + and -
        """
        return self.edges_around(edge.clause_node)

        # return [neighbor_edge for neighbor_edge in self.edges_around(edge.clause_node)]

    def literal_neighborhood(self, edge: Edge):
        """
        Returns the edges around the variable node, restricted to edges with the same sign
        Used in l2c message.
        """
        return [neighbor_edge for neighbor_edge in self.edges_around(edge.variable_node) if edge.sign == neighbor_edge.sign]

    def variable_neighborhood(self, edge:Edge):
        return self.edges_around(edge.variable_node)

    def _compute_subgraph_hash(self, edge: Edge, shared_clauses: typing.List[ClauseNode]) -> typing.Tuple:
        """
        Compute a label-agnostic hash key for the subgraph around an edge.
        
        The key is based on:
        1. The sign of the edge (+ or -)
        2. The structure of shared clauses, with variables renumbered canonically
        
        This allows us to identify structurally equivalent subgraphs regardless of
        the actual variable labels used.
        """
        hash_start = time.time()
        
        if not shared_clauses:
            return (edge.sign, ())
        
        # Collect all variables in shared clauses
        all_vars_in_shared = set()
        for clause in shared_clauses:
            for lit in clause.content:
                all_vars_in_shared.add(abs(lit))
        
        # Create canonical mapping: current edge's variable is always 1,
        # other variables are numbered 2, 3, ... in order of first appearance
        var_mapping = {edge.variable_node.var_number: 1}
        next_var = 2
        
        # Sort clauses for consistent ordering, then assign variable numbers
        sorted_clause_contents = sorted([clause.content for clause in shared_clauses])
        for clause_content in sorted_clause_contents:
            for lit in clause_content:
                var = abs(lit)
                if var not in var_mapping:
                    var_mapping[var] = next_var
                    next_var += 1
        
        # Convert clauses to canonical form
        canonical_clauses = []
        for clause_content in sorted_clause_contents:
            canonical_clause = tuple(
                var_mapping[abs(lit)] if lit > 0 else -var_mapping[abs(lit)]
                for lit in clause_content
            )
            # Sort literals within each clause for consistent representation
            canonical_clauses.append(tuple(sorted(canonical_clause)))
        
        # Sort the clauses themselves for a fully canonical representation
        canonical_clauses = tuple(sorted(canonical_clauses))
        
        hash_time = time.time() - hash_start
        if hash_time > 0.001:  # Print if it takes more than 1ms
            print(f'[_compute_subgraph_hash] Took {hash_time*1000:.2f}ms for {len(shared_clauses)} clauses')
        
        return (edge.sign, canonical_clauses)

    # This function takes 10**4 as long as l2c and 10 times as long as c2l
    # not unit tested
    def local_satisfaction_percentage(self, edge: Edge):

        # Surrounding clause nodes, including the clause of the edge itself; I remove duplicate contents since they
        # are the same clause.
        clauses_around_variable = list(
            {e.clause_node.content: e.clause_node
            for e in self.variable_neighborhood(edge)}
            .values()
        )

        # # Surrounding clause nodes, including the clause of the edge itself
        # clauses_around_variable = list(set([e.clause_node for e in self.variable_neighborhood(edge)]))

        # Edges around the clause node, excluding the edge itself
        edges_to_other_variables = [e for e in self.clause_neighborhood(edge) if e.variable_node != edge.variable_node]

        k2_clause_neighbor_edges = [k2 for k1 in edges_to_other_variables for k2 in self.variable_neighborhood(k1)]
        # k2_clause_neighbors = list(set([e.clause_node for e in k2_clause_neighbor_edges]))
        k2_clause_neighbors = list(
            {e.clause_node.content: e.clause_node
            for e in k2_clause_neighbor_edges}.values()
        )

        # Shared clauses with current literal
        shared_clauses = [c for c in clauses_around_variable if c in k2_clause_neighbors]

        # Check cache using label-agnostic hash
        cache_key = self._compute_subgraph_hash(edge, shared_clauses)
        
        # Check persistent cache first (shared across all files)
        if cache_key in BPGParamBuilder._persistent_cache:
            BPGParamBuilder._persistent_cache_hits += 1
            self._cache_hits += 1
            return BPGParamBuilder._persistent_cache[cache_key]
        
        # Check instance cache (for current file only)
        if cache_key in self._local_sat_cache:
            self._cache_hits += 1
            return self._local_sat_cache[cache_key]
        
        BPGParamBuilder._persistent_cache_misses += 1
        self._cache_misses += 1

        # Generate all possible assignments for vars in the shared clauses
        if edge.sign == "+":
            assignment_dict = {edge.variable_node.var_number: True}
        else:
            assignment_dict = {edge.variable_node.var_number: False}

        other_vars = list(set([abs(l) for clause in shared_clauses for l in clause.content]))

        # If there are no shared clauses, other_vars can be empty
        if other_vars:
            other_vars.remove(edge.variable_node.var_number)

        MAX_SAMPLES = 1000
        n_other_vars = len(other_vars)
        search_space_size = 2**n_other_vars
        n_satisfying_assignments = 0

        # Pre-extract clause contents for faster iteration
        clause_contents = [c.content for c in shared_clauses]

        # Exact calculation if space is small enough
        if search_space_size <= MAX_SAMPLES:
            total_checked = search_space_size

            for assignment in itertools.product((True, False), repeat=n_other_vars):
                # Update assignment dict in place
                for i, var in enumerate(other_vars):
                    assignment_dict[var] = assignment[i]

                # Check if all shared clauses are satisfied (with early termination)
                all_satisfied = True
                for clause_content in clause_contents:
                    clause_satisfied = False
                    for literal in clause_content:
                        if (literal > 0) == assignment_dict[abs(literal)]:
                            clause_satisfied = True
                            break
                    if not clause_satisfied:
                        all_satisfied = False
                        break
                
                if all_satisfied:
                    n_satisfying_assignments += 1
        
        # Random sampling approximation if space is too large
        else:
            total_checked = MAX_SAMPLES
            for _ in range(MAX_SAMPLES):
                # Generate random assignment for other variables
                for i, var in enumerate(other_vars):
                    assignment_dict[var] = random.random() < 0.5
                
                # Check if all shared clauses are satisfied (with early termination)
                all_satisfied = True
                for clause_content in clause_contents:
                    clause_satisfied = False
                    for literal in clause_content:
                        if (literal > 0) == assignment_dict[abs(literal)]:
                            clause_satisfied = True
                            break
                    if not clause_satisfied:
                        all_satisfied = False
                        break
                
                if all_satisfied:
                    n_satisfying_assignments += 1

        result = n_satisfying_assignments / total_checked
        
        # Store in both caches
        self._local_sat_cache[cache_key] = result
        BPGParamBuilder._persistent_cache[cache_key] = result
        
        return result

    def c2l_message_input(self) -> typing.Tuple[torch.tensor, torch.tensor]:
        """
        return two lists,
        one with x times the edge index
        one with the x indices of other edges going out from the c
        """
        receiving_edge_indices = []
        sending_edge_indices = []

        # edge l-c
        for edge in self.edges:
            # Get all edges to other clauses in which l occurs (with the same sign)
            neighborhood = self.literal_neighborhood(edge)
            punctured_neighborhood = [neighbor_edge for neighbor_edge in neighborhood if neighbor_edge != edge]

            receiving_edge_indices.extend([edge.index for _ in punctured_neighborhood])
            sending_edge_indices.extend([neighbor_edge.index for neighbor_edge in punctured_neighborhood])

        return torch.tensor(receiving_edge_indices, dtype=torch.long), torch.tensor(sending_edge_indices,dtype=torch.long)


    def clause_assignment_neighborhood(self, edge: Edge, assignment: typing.List[bool]) -> typing.List[Edge]:
        """
        Return the neighboring edges for some assignment, not including the edge itself

        Parameters
        ----------
        edge : Edge
            The edge to the clause node
        assignment : list[bool]
            List of booleans of the size of the clause, in order of the literals in the clause

        Returns
        -------
            List of one less element than the assignment,
            with all edges to literals matching the assignment except for the given edge.
        """
        different_var_neighbors = self.clause_neighborhood(edge)

        clause_assignment_edges = []
        for valuation, literal in zip(assignment, edge.clause_node.content):

            # We don't return the current edge
            if abs(literal) == edge.variable_node.var_number:
                continue
            # Add the edge matching the assignment (+ or -) and variable
            else:
                sign = "+" if valuation else "-"
                try:
                    matching_edge = next(e for e in different_var_neighbors if e.variable_node.var_number == abs(literal) and e.sign==sign)
                except StopIteration:
                    raise ValueError(f"No matching edge found for clause {edge.clause_node.content} and assignment {assignment}")
                
                clause_assignment_edges.append(matching_edge)

        return clause_assignment_edges
    

    def l2c_satisfying_neighborhoods(self, edge: Edge):
        """
        return a list of lists of edge indices
        each sublist matching a punctured literal neighborhood satisying the clause c

        In oorspronkelijke BPG: 
        voor elke l-c occurence (twee edges)hebben we een lijst van assignment-neiborhoods als [0,0,0], [0,0,1], etc.
        waar 0 '+' betekent
        """

        # The valuation of the var of this edge, according to its sign
        current_var_assignment = edge.sign == "+"
        variable_index_in_clause = [abs(l) for l in edge.clause_node.content].index(edge.variable_node.var_number)

        # All possible assignments of the other variables in the clause
        var_assignments_for_neighbors = list(itertools.product([True, False], repeat=len(edge.clause_node.content)-1))
        var_assignments_for_neighbors = [list(assignment) for assignment in var_assignments_for_neighbors]


        # import pdb; pdb.set_trace()
        satisfying_neighborhoods = []
        for neighborhood_assignment in var_assignments_for_neighbors:


            # Add the current literal, since it may also satisfy the clause
            neighborhood_assignment.insert(variable_index_in_clause, current_var_assignment)
            if edge.clause_node.satisfied_by(neighborhood_assignment):
                # Get the neighborhood for this assignment, not including the edge itself
                satisfying_neighborhoods.append(self.clause_assignment_neighborhood(edge, neighborhood_assignment))

        return satisfying_neighborhoods

    def l2c_satisfying_assignment_sum_input(self):
        """
        returns three lists,

        1) the edge numbers in order, repeated for the number of satisfying punctured neighborhoods per edge
        2) a numbering of the satisfying neighborhoods for all edges, each repeated for the size of the punctured neighborhood
        3) all edge numbers of satisfying punctured neighborhoods in order
        """
 
        edge_indices = []
        assignment_indices = []
        neighborhood_assignments = []
        assignment_counter = 0

        for edge in self.edges:
            satisfying_neighborhoods = self.l2c_satisfying_neighborhoods(edge)
            for neighborhood in satisfying_neighborhoods:
                # Repeat the edge index, times the number of outgoing assignments
                edge_indices.append(edge.index)
                for neighbor_edge in neighborhood:
                    assignment_indices.append(assignment_counter)
                    neighborhood_assignments.append(neighbor_edge.index)
                assignment_counter += 1

        edge_indices = torch.tensor(edge_indices, dtype=torch.long)
        assignment_indices = torch.tensor(assignment_indices, dtype=torch.long)
        neighborhood_assignments = torch.tensor(neighborhood_assignments, dtype=torch.long)
        return edge_indices, assignment_indices, neighborhood_assignments


def create_bpg_with_subgraphs(
    cnf: typing.List[typing.List[int]], 
    n_vars: int = None,
    compute_subgraphs: bool = True
) -> BPG:
    """
    Create a BPG object with optional canonical subgraph features.
    
    Args:
        cnf: CNF formula as list of clauses
        n_vars: Number of variables (if None, inferred from clauses)
        compute_subgraphs: Whether to compute canonical subgraph features
    
    Returns:
        BPG object with optional subgraph data
    """
    # Build BPG parameters
    builder = BPGParamBuilder(cnf, n_vars)
    
    # Create base BPG
    bpg = BPG(*builder.params)
    
    # Optionally add subgraph features
    if compute_subgraphs:
        subgraphs_p1, subgraphs_p0 = builder.compute_canonical_subgraphs()
        bpg.subgraphs_p1 = subgraphs_p1
        bpg.subgraphs_p0 = subgraphs_p0
    
    return bpg
    

class OriginalBPG(Data):
    def __init__(self,
        # Totaal aantal literals, i.e. n_vars * 2
        l_size=None,
        # Totaal aantal clauses
        c_size=None,


        # Lijst/Tensor van + en - literals, in [1, 2N-1] format, op volgorde van occurence in clauses
        # maar met + en - na elkaar voor elke literal in een clause.
        # Size should be l_size, i.e. 2*len(l_edge_index) KLOPT e.g. 414 edges voor formula 0
        # i.e. voor eerste clause [-3, -9, 13]
        # begint de lijst [4,5,16,17,24,25]
        # hoe is dit 'edge index' want elke edge naar een literal heeft hetzelfde getal

        # Dit is een MAP van edge-index (in +, dan -, op volgorde van literals in clauses)
        # naar literal in [0,2N-1] form
        sign_l_edge_index=None,

        # Size N_edges *
        # Per edge c-l, de 'volgorde index' van l in alle andere clauses waar l of -l in zit,
        # eerst alsof l in elke clause zit, dan alsof -l in elke clause zit
        # i.e. voor edge 0(c)-3(l): [18, 36, 60 ... (volgorde in neighbors van 3), 19, 37, 61 ... (alle vorige getallen +1)]
        # Dit is dus, voor c-l, alle edge indices van c-l (+) c2 en dan c - l (-) c2
        c2l_msg_repeat_index=None,

        # Precies hetzelfde maar met order indices voor d eincoming edge
        # e.e. in vorige voorbeeld [0, 0, 0, ... 1, 1, 1, ...]

        # Als we de bovenstaande lijsten naast elkaar leggen,
        # hebben we voor elke triple c1 - l - c2
        # de 'volgorde index' van l in c2 (repeat_index) en in c1 (scatter_index)
        # Scatter, betekent misschien we sommen alle l-(+)-c2 edges naar de c1-(+)-l en voor - hetzelfde
        c2l_msg_scatter_index=None,

        # Voor eerste formule, zijn zowel repeat als scatter 6000

        l2c_msg_aggr_repeat_index=None,
        l2c_msg_aggr_scatter_index=None,
        l2c_msg_scatter_index=None,
        c_blf_repeat_index=None,
        c_blf_scatter_index=None,
        c_blf_norm_index=None,
        v_degrees=None,
        c_batch=None,
        v_batch=None,

        # List of all literal indices (where literals in [-N, N] are mapped to [0, 2N-1]), f(x) + 1 = f(-x)
        # Hier is onderscheid tussen + en - occurence, i.e. eerste clause [-3, -9, 13]
        # geeft [5, 18, 24]
        # Op volgorde van occurence in clauses, met duplicates
        # Komt overeen met aantal edges, i.e. totaal aantal literals in de CNF
        l_edge_index=None,
        # List van clause indices
        c_edge_index=None
    ):
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.sign_l_edge_index = sign_l_edge_index
        self.c2l_msg_repeat_index = c2l_msg_repeat_index
        self.c2l_msg_scatter_index = c2l_msg_scatter_index
        self.l2c_msg_aggr_repeat_index = l2c_msg_aggr_repeat_index
        self.l2c_msg_aggr_scatter_index = l2c_msg_aggr_scatter_index
        self.l2c_msg_scatter_index = l2c_msg_scatter_index
        self.c_blf_repeat_index = c_blf_repeat_index
        self.c_blf_scatter_index = c_blf_scatter_index
        self.c_blf_norm_index = c_blf_norm_index
        self.v_degrees = v_degrees
        self.c_batch = c_batch
        self.v_batch = v_batch
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index

    # Num edges is totale size van de CNF keer 2. Voor elke literal-to-clause, een + en een -
    @property
    def num_edges(self):
        return self.sign_l_edge_index.size(0)                 

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'c_blf_norm_index' or key == 'c_edge_index':
            return self.c_size
        elif key == 'sign_l_edge_index' or key == 'l_edge_index':
            return self.l_size
        elif key == 'c2l_msg_repeat_index' or key == 'c2l_msg_scatter_index' or key == 'l2c_msg_aggr_repeat_index' \
            or key == 'l2c_msg_scatter_index' or key == 'c_blf_repeat_index':
            return self.sign_l_edge_index.size(0)
        elif key == 'l2c_msg_aggr_scatter_index':
            return self.l2c_msg_scatter_index.size(0)
        elif key == 'c_blf_scatter_index':
            return self.c_blf_norm_index.size(0)
        elif key == 'c_batch' or key == 'v_batch':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)



def OldTransform2BPG(n_vars, clauses, task):
    """
    Deze functie moet ik uitzoeken, dit is representatie voor NSNet
    """
    sign_l_edge_index_list = []
    type_edge_index_list = []

    c2l_msg_aggr_c_index_map = {l: [] for l in range(2 * n_vars)}
    c2l_msg_aggr_edge_index_map = {l: [] for l in range(2 * n_vars)}
    
    c2l_msg_repeat_index_list = []
    c2l_msg_scatter_index_list = []

    l2c_msg_aggr_repeat_index_list = []
    l2c_msg_aggr_scatter_index_list = []
    l2c_msg_scatter_index_list = []

    # auxiliary parameters
    c_blf_repeat_index = None
    c_blf_scatter_index = None
    c_blf_norm_index = None
    v_degrees = None
    c_batch = None
    v_batch = None
    l_edge_index = None
    c_edge_index = None

    if task == 'model-counting':
        c_blf_repeat_index_list = []
        c_blf_scatter_index_list = []
        c_blf_norm_index_list = []
        v_degrees = torch.zeros(n_vars)
    else:
        l_edge_index_list = []
        c_edge_index_list = []
    
    index_base = 0
    msg_aggr_index = 0
    for c_idx, clause in enumerate(clauses):
        used_vars = sorted(list(set([abs(literal)-1 for literal in clause])))
        # literal to clause message
        for msg_idx, v_idx in enumerate(used_vars):
            # Literal ints for variable index, i.e. var 9 krijgt literals 16, 17
            pl_idx = v_idx * 2
            nl_idx = v_idx * 2 + 1

            # Index base telt op met 2*aantal variabelen in clause
            # p_msg_idx, n_msg_idx zijn de indices van literal, op volgorde van variable in clause.... waar is dit voor?
            p_msg_idx = index_base + msg_idx * 2
            n_msg_idx = index_base + msg_idx * 2 + 1
            
            # Lijst met + en - literals voor variable numbers?? Waarom niet gewoon de variabelen opslaan hier?
            sign_l_edge_index_list.append(pl_idx)
            sign_l_edge_index_list.append(nl_idx)
            
            # Map van literal, naar list of clause indices waarin deze voorkomt, maar zonder onderscheid in + en -!
            c2l_msg_aggr_c_index_map[pl_idx].append(c_idx)
            c2l_msg_aggr_c_index_map[nl_idx].append(c_idx)
            
            # Map van literal naar matchende 'volgorde-literal-index' in clause
            # e.g. voor clause (5. 9), is krijgen we voor 9 map van 16 naar 2, en map van 17 naar 3....??
            c2l_msg_aggr_edge_index_map[pl_idx].append(p_msg_idx)
            c2l_msg_aggr_edge_index_map[nl_idx].append(n_msg_idx)
            
            # Hier voegen we '0' of '1' toe, naar gelang de literal + of - is, dit is dus een lijst van size n_edges*2
            # Voor positive instance van l in c appenden we [1,0]
            # Voor negative instance [0,1]
            if (v_idx + 1) in clause:
                type_edge_index_list.append(1)
            else:
                type_edge_index_list.append(0)
            
            if -(v_idx + 1) in clause:
                type_edge_index_list.append(1)
            else:
                type_edge_index_list.append(0)
        
        # clause to literal massage

        # used_vars zijn weer de literal_nummers van de vars in clauses
        # Dus we discarden 1 voor 1 de vars?

        #    var position,    var number
        for scatter_msg_idx, discard_v_idx in enumerate(used_vars):
            # indices, is steeds een keuze van 0 en 1 voor elke variable
            # , dit iterate over alle assignmnets (behalve de gediscarde) (dus tuples van lengte 2 voor 3 vars)
            for indices in np.ndindex(tuple([2] * (len(used_vars)-1))):
                # msg_index is letterlijk de index van de literal in de clause
                # dus msg_table = [(p_msg_index, n_msg_index)], i.e. de indices van de + en - edge voor deze l-c occurences
                # dit voor elke var behalve de 'gediscarde'
                # Dus, als het de eerste clause is [x,y,z], en we discarden 'x'+ (0) of x- (1)
                # [(3,2), (5,4)], als we z discarden [(1,0), (3,2)]
                msg_table = [(index_base + msg_idx * 2 + 1, index_base + msg_idx * 2) 
                    for msg_idx, v_idx in enumerate(used_vars) if v_idx != discard_v_idx]
                
                # Hier pakken we uit msg_table, voor elke '0' of '1' in de assignment, de - (+1) of + (0) edge index
                # Als positive occurence: gaat de (+1) naar '0', de (0) naar 1 in type_edge_index_list
                # Dus als indices = [0,0], en allebei positive occurences, is assign [1,1]
                # DUS 'index 0' betekent, maar 'l' waar -> dat geeft 1 als l positief in c zit
                # Als negative occurence andersom, dus dan is assign [0,0]
                assign = np.array([type_edge_index_list[msg_table[i][idx]] for i, idx in enumerate(indices)])

                # dus, 'index' 0 betekent variable is true, index '1' betekent variable is false
                is_sat = assign.sum() > 0

                # Dit zijn dan de matchende edge indices voor deze assignment
                # Dus voor edge c-l1, zijn dit de edges van alle l2-c, voor deze assignment. 
                msg_aggr_repeat_index = [msg_table[i][idx] for i, idx in enumerate(indices)]
                
                # index_base+scatter_msg_idx*2 is de tweede index voor deze edge in index_list
                # dus deze is 1 als de literal negatief occurred

                # Deze triggered alleen als l positief occurred, or SAT
                if type_edge_index_list[index_base + scatter_msg_idx * 2] or is_sat:       
                    # Dit wordt dus een lijst van alle incoming literal edges,
                    # per c1-l1, per (satisfying?) assignment per l2-c2
                    l2c_msg_aggr_repeat_index_list.append(msg_aggr_repeat_index)

                    # Dit wordt een lijst met, per c1-l1, per assignment per l2-c2 een index die de som stuurt
                    l2c_msg_aggr_scatter_index_list.append([msg_aggr_index] * len(msg_aggr_repeat_index))
                    msg_aggr_index += 1

                    # Hier scatter_som op geeft, per assignment (sommige 0 sommige 2 keer) de som van outgoing literals

                    # Dit is een lijst met de c-l index
                    # scatter_som op output van het vorige met deze indices geeft som over alle assignments
                    # voor c1, naar l1
                    l2c_msg_scatter_index_list.append(index_base + scatter_msg_idx * 2)

                # Deze triggered bij - occurence en SAT
                if type_edge_index_list[index_base + scatter_msg_idx * 2 + 1] or is_sat:
                    l2c_msg_aggr_repeat_index_list.append(msg_aggr_repeat_index)
                    l2c_msg_aggr_scatter_index_list.append([msg_aggr_index] * len(msg_aggr_repeat_index))
                    msg_aggr_index += 1
                    l2c_msg_scatter_index_list.append(index_base + scatter_msg_idx * 2 + 1)

                # import pdb; pdb.set_trace()

                # TODO fix deze terms zodat alleen satisfying assignments worden gebruikt??

        index_base += len(used_vars) * 2

    # Tensor, met lijst van alle + en - literals, op volgorde van clauses, i.e. [9,2,3], [4,5]
    # -> [16, 17, 2, 3, 4, 5, 6, 7, 8, 9]
    sign_l_edge_index = torch.tensor(sign_l_edge_index_list, dtype=torch.long)

    index_base = 0
    for c_idx, clause in enumerate(clauses):
        used_vars = sorted(list(set([abs(literal)-1 for literal in clause])))
        for msg_idx, v_idx in enumerate(used_vars):
            # Zelfde als in vorige loop
            pl_idx = v_idx * 2
            nl_idx = v_idx * 2 + 1

            # Zelfde als in vorige loop, i.e. index representations van literal, o.b.v. volgorde in clauses
            p_msg_idx = index_base + msg_idx * 2
            n_msg_idx = index_base + msg_idx * 2 + 1
            
            #c2l_msg_aggr_c_index_map is map van literal (in [1, 2N-1] format) naar list van clauses
            # Waar l of -l in voorkomt.

            # c2l_msg_aggr_edge_index_map is map van literal naar 'volgorde index' in clause
            # opnieuw maar e.g. map[16] = 2, map[17]=3, onafhankelijk van of occurence + of - is
            for neighbor_c_idx, neighbor_msg_idx in zip(c2l_msg_aggr_c_index_map[pl_idx], c2l_msg_aggr_edge_index_map[pl_idx]):

                # Doe niets, als de neighbor_clause (via literal) hetzelfde is als de huidige clause
                if neighbor_c_idx == c_idx:
                    continue                    

                # Dit wordt een lijst met, per alle literals per clause, de 'volgorde index' van deze variable
                # in elke andere clause waar deze variabele in zit.

                # bijv. voor var literal -3, var komt voor in clauses [0,3,5,10]
                # Matchende volgorde indices zijn [0, 18, 36, 60], maar de 0 skippen we als c_idx==0
                c2l_msg_repeat_index_list.append(neighbor_msg_idx)

                # Dit is per occurence de volgorde index in deze clause, e.g. [0, 0, 0] (size is n_neighbors -1)
                c2l_msg_scatter_index_list.append(p_msg_idx)
            
            for neighbor_c_idx, neighbor_msg_idx in zip(c2l_msg_aggr_c_index_map[nl_idx], c2l_msg_aggr_edge_index_map[nl_idx]):
                if neighbor_c_idx == c_idx:
                    continue

                # Hier voegen we de negatieve 'order index' voor literal in elke neighbor toe
                # i.e. van [18,36,60], naar [18, 36, 60, 19, 27, 61]
                c2l_msg_repeat_index_list.append(neighbor_msg_idx)

                # Hier voegen we voor de negatieve volgorde index van deze occurence toe voor elke neighbor
                # i.e. van [0,0,0] naar [0, 0, 0, 1, 1, 1]
                c2l_msg_scatter_index_list.append(n_msg_idx)
        
        index_base += len(used_vars) * 2

    # Dit is dan, per edge c-l, een entry voor elke occurence van l/-l in een andere clause
    # met de 'volgorde index' van l
    # daarna opnieuw een entry voor elke occurence van l/-l in een andere clause
    # met de 'volgorde index' van -l
    c2l_msg_repeat_index = torch.tensor(c2l_msg_repeat_index_list, dtype=torch.long)

    # Dit is hetzelfde aantal, maar gesorteerd per 'incoming edge' naar l
    # dus voor c-l, en l-c1, l-c2, l-c2
    # geeft dit 3* order van +l in c, dan 3 * van -l in c
    c2l_msg_scatter_index = torch.tensor(c2l_msg_scatter_index_list, dtype=torch.long)

    # concat
    l2c_msg_aggr_repeat_index_list = list(itertools.chain(*l2c_msg_aggr_repeat_index_list))
    l2c_msg_aggr_scatter_index_list = list(itertools.chain(*l2c_msg_aggr_scatter_index_list))

    l2c_msg_aggr_repeat_index = torch.tensor(l2c_msg_aggr_repeat_index_list, dtype=torch.long)
    l2c_msg_aggr_scatter_index = torch.tensor(l2c_msg_aggr_scatter_index_list, dtype=torch.long)
    l2c_msg_scatter_index = torch.tensor(l2c_msg_scatter_index_list, dtype=torch.long)

    if task == 'model-counting':
        index_base = 0
        blf_index = 0
        for c_idx, clause in enumerate(clauses):
            used_vars = set([abs(literal)-1 for literal in clause])
            for indices in np.ndindex(tuple([2] * len(used_vars))):
                msg_table = [(index_base + msg_idx * 2 + 1, index_base + msg_idx * 2) 
                    for msg_idx, v_idx in enumerate(used_vars)]
                assign = np.array([type_edge_index_list[msg_table[i][idx]] for i, idx in enumerate(indices)])
                is_sat = assign.sum() > 0
                blf_repeat_index = [msg_table[i][idx] for i, idx in enumerate(indices)]
                
                if is_sat:
                    c_blf_repeat_index_list.append(blf_repeat_index)
                    c_blf_scatter_index_list.append([blf_index] * len(blf_repeat_index))
                    blf_index += 1
                    c_blf_norm_index_list.append(c_idx)
            index_base += len(used_vars) * 2

            for v_idx in used_vars:
                v_degrees[v_idx] += 1
        
        c_blf_repeat_index_list = list(itertools.chain(*c_blf_repeat_index_list))
        c_blf_scatter_index_list = list(itertools.chain(*c_blf_scatter_index_list))

        c_blf_repeat_index = torch.tensor(c_blf_repeat_index_list, dtype=torch.long)
        c_blf_scatter_index = torch.tensor(c_blf_scatter_index_list, dtype=torch.long)
        c_blf_norm_index = torch.tensor(c_blf_norm_index_list, dtype=torch.long)

        c_batch = torch.zeros(len(clauses), dtype=torch.long)
        v_batch = torch.zeros(n_vars, dtype=torch.long)
    else:
        for c_idx, clause in enumerate(clauses):
            for literal in clause:
                # Zelfde map, i.e. 9 naar 16, -9 naar 17
                l_idx = literal2l_idx(literal)

                # Dit is een lijst van alle literals, op volgorde van occurence in clauses, met duplicates
                l_edge_index_list.append(l_idx)
                # Dit is een lijst van alle clause indices
                c_edge_index_list.append(c_idx)
        
        # 
        l_edge_index = torch.tensor(l_edge_index_list, dtype=torch.long)
        c_edge_index = torch.tensor(c_edge_index_list, dtype=torch.long)
        c_batch = torch.zeros(len(clauses), dtype=torch.long)

    # import pdb; pdb.set_trace()

    return OldBPGParams(
        n_vars*2, 
        len(clauses),
        sign_l_edge_index, 
        c2l_msg_repeat_index,
        c2l_msg_scatter_index,
        l2c_msg_aggr_repeat_index,
        l2c_msg_aggr_scatter_index,
        l2c_msg_scatter_index,
        c_blf_repeat_index,
        c_blf_scatter_index,
        c_blf_norm_index,
        v_degrees,
        c_batch,
        v_batch,
        l_edge_index,
        c_edge_index,
    )

class OldBPGParams(NamedTuple):
    l_size: int
    c_size: int
    sign_l_edge_index: torch.Tensor
    c2l_msg_repeat_index: torch.Tensor
    c2l_msg_scatter_index: torch.Tensor
    l2c_msg_aggr_repeat_index: torch.Tensor
    l2c_msg_aggr_scatter_index: torch.Tensor
    l2c_msg_scatter_index: torch.Tensor
    c_blf_repeat_index: torch.Tensor
    c_blf_scatter_index: torch.Tensor
    c_blf_norm_index: torch.Tensor
    v_degrees: torch.Tensor
    c_batch: torch.Tensor
    v_batch: torch.Tensor
    l_edge_index: torch.Tensor
    c_edge_index: torch.Tensor

class OldBPG(Data):
    def __init__(self, 
        # n_vars = None,
        # clauses = None,
        # task=None,
        # bpg_params: OldBPGParams

        # Totaal aantal literals, i.e. n_vars * 2
        l_size=None,
        # Totaal aantal clauses
        c_size=None,


        # Lijst/Tensor van + en - literals, in [1, 2N-1] format, op volgorde van occurence in clauses
        # maar met + en - na elkaar voor elke literal in een clause.
        # Size should be l_size, i.e. 2*len(l_edge_index) KLOPT e.g. 414 edges voor formula 0
        # i.e. voor eerste clause [-3, -9, 13]
        # begint de lijst [4,5,16,17,24,25]
        # hoe is dit 'edge index' want elke edge naar een literal heeft hetzelfde getal

        # Dit is een MAP van edge-index (in +, dan -, op volgorde van literals in clauses)
        # naar literal in [0,2N-1] form
        sign_l_edge_index=None,

        # Size N_edges *
        # Per edge c-l, de 'volgorde index' van l in alle andere clauses waar l of -l in zit,
        # eerst alsof l in elke clause zit, dan alsof -l in elke clause zit
        # i.e. voor edge 0(c)-3(l): [18, 36, 60 ... (volgorde in neighbors van 3), 19, 37, 61 ... (alle vorige getallen +1)]
        # Dit is dus, voor c-l, alle edge indices van c-l (+) c2 en dan c - l (-) c2
        c2l_msg_repeat_index=None,

        # Precies hetzelfde maar met order indices voor d eincoming edge
        # e.e. in vorige voorbeeld [0, 0, 0, ... 1, 1, 1, ...]

        # Als we de bovenstaande lijsten naast elkaar leggen,
        # hebben we voor elke triple c1 - l - c2
        # de 'volgorde index' van l in c2 (repeat_index) en in c1 (scatter_index)
        # Scatter, betekent misschien we sommen alle l-(+)-c2 edges naar de c1-(+)-l en voor - hetzelfde
        c2l_msg_scatter_index=None,

        # Voor eerste formule, zijn zowel repeat als scatter 6000

        # keyword 'index' means that arguments all get batched along the last dimension?
        l2c_msg_aggr_repeat_index=None,
        l2c_msg_aggr_scatter_index=None,
        l2c_msg_scatter_index=None,
        c_blf_repeat_index=None,
        c_blf_scatter_index=None,
        c_blf_norm_index=None,
        v_degrees=None,
        c_batch=None,
        v_batch=None,

        # List of all literal indices (where literals in [-N, N] are mapped to [0, 2N-1]), f(x) + 1 = f(-x)
        # Hier is onderscheid tussen + en - occurence, i.e. eerste clause [-3, -9, 13]
        # geeft [5, 18, 24]
        # Op volgorde van occurence in clauses, met duplicates
        # Komt overeen met aantal edges, i.e. totaal aantal literals in de CNF
        l_edge_index=None,
        # List van clause indices 'edge_index' is special keyword
        c_edge_index=None
        ):
        super().__init__()
        self.l_size = l_size
        self.c_size = c_size
        self.sign_l_edge_index = sign_l_edge_index
        self.c2l_msg_repeat_index = c2l_msg_repeat_index
        self.c2l_msg_scatter_index = c2l_msg_scatter_index
        self.l2c_msg_aggr_repeat_index = l2c_msg_aggr_repeat_index
        self.l2c_msg_aggr_scatter_index = l2c_msg_aggr_scatter_index
        self.l2c_msg_scatter_index = l2c_msg_scatter_index
        self.c_blf_repeat_index = c_blf_repeat_index
        self.c_blf_scatter_index = c_blf_scatter_index
        self.c_blf_norm_index = c_blf_norm_index
        self.v_degrees = v_degrees
        self.c_batch = c_batch
        self.v_batch = v_batch
        self.l_edge_index = l_edge_index
        self.c_edge_index = c_edge_index

        # pdb.set_trace()

        
    # Num edges is totale size van de CNF keer 2. Voor elke literal-to-clause, een + en een -
    @property
    def num_edges(self):
        return self.sign_l_edge_index.size(0)                 

    # Used to facilitate graph stacking This is used for batching
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'c_blf_norm_index' or key == 'c_edge_index':
            return self.c_size
        elif key == 'sign_l_edge_index' or key == 'l_edge_index':
            return self.l_size
        elif key == 'c2l_msg_repeat_index' or key == 'c2l_msg_scatter_index' or key == 'l2c_msg_aggr_repeat_index' \
            or key == 'l2c_msg_scatter_index' or key == 'c_blf_repeat_index':
            return self.sign_l_edge_index.size(0)
        elif key == 'l2c_msg_aggr_scatter_index':
            return self.l2c_msg_scatter_index.size(0)
        elif key == 'c_blf_scatter_index':
            return self.c_blf_norm_index.size(0)
        elif key == 'c_batch' or key == 'v_batch':
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)


def _process_single_file(args):
    """
    Process a single CNF file into a graph format.
    This is a module-level function for multiprocessing compatibility.
    """
    idx, file_path, graph_type, task, output_path, no_precomputed_local_sat = args
    
    n_vars, clauses = parse_cnf_file(file_path)
    
    if graph_type == 'LCG':
        # LCG transformation inline since it's simple
        c_edge_index_list = []
        l_edge_index_list = []
        l_batch = None
        c_batch = None
        
        if task == 'model-counting':
            l_batch = torch.zeros(n_vars * 2, dtype=torch.long)
        else:
            c_batch = torch.zeros(len(clauses), dtype=torch.long)
        
        for c_idx, clause in enumerate(clauses):
            for literal in clause:
                l_idx = literal2l_idx(literal)
                l_edge_index_list.append(l_idx)
                c_edge_index_list.append(c_idx)
        
        c_edge_index = torch.tensor(c_edge_index_list, dtype=torch.long)
        l_edge_index = torch.tensor(l_edge_index_list, dtype=torch.long)
        
        data = LCG(n_vars * 2, len(clauses), c_edge_index, l_edge_index, l_batch, c_batch)
    elif graph_type == 'OldBPG':
        bpg_params = OldTransform2BPG(n_vars, clauses, task)
        data = OldBPG(*bpg_params)
    elif graph_type == 'BPG':
        bpg_params = BPGParamBuilder(
            clauses,
            n_vars,
            compute_local_satisfaction_percentages=not no_precomputed_local_sat,
        ).params
        data = BPG(*bpg_params)
    elif graph_type == 'MatrixBPG':
        bpg_params = MatrixBPGParamBuilder(clauses).params
        data = MatrixBPG(*bpg_params)
    else:
        raise ValueError(f'Graph type {graph_type} not supported')

    torch.save(_pack_data(data, graph_type), output_path)
    return idx


def _pack_data(data, graph_type: str) -> dict:
    """
    Serialize a graph Data object as a compact dict of downcast tensors.
    Index tensors are stored as int32 (half the size of int64).
    Float tensors are stored as float16.
    Scalar ints are stored as-is.
    """
    def _i32(t):
        """Cast a long tensor to int32 (safe for values < 2^31)."""
        return t.to(torch.int32) if t is not None else None

    def _f16(t):
        """Cast a float tensor to float16."""
        return t.to(torch.float16) if t is not None else None

    d = {'__graph_type__': graph_type}

    if graph_type == 'LCG':
        d['l_size'] = data.l_size
        d['c_size'] = data.c_size
        d['c_edge_index'] = _i32(data.c_edge_index)
        d['l_edge_index'] = _i32(data.l_edge_index)
        if data.l_batch is not None:
            d['l_batch'] = _i32(data.l_batch)
        if data.c_batch is not None:
            d['c_batch'] = _i32(data.c_batch)

    elif graph_type == 'BPG':
        d['n_clauses'] = data.n_clauses
        d['n_literals'] = data.n_literals
        d['literal_indices_per_edge'] = _i32(data.literal_indices_per_edge)
        d['literal_indices_per_occurence'] = _i32(data.literal_indices_per_occurence)
        d['clause_indices_per_occurence'] = _i32(data.clause_indices_per_occurence)
        d['local_satisfaction_percentage_per_edge'] = _f16(data.local_satisfaction_percentage_per_edge)
        d['c2l_msg_receiver_indices'] = _i32(data.c2l_msg_receiver_indices)
        d['c2l_msg_sender_indices'] = _i32(data.c2l_msg_sender_indices)
        d['l2c_msg_receiver_indices'] = _i32(data.l2c_msg_receiver_indices)
        d['l2c_assignment_indices'] = _i32(data.l2c_assignment_indices)
        d['l2c_assignment_neighborhoods'] = _i32(data.l2c_assignment_neighborhoods)
        # subgraphs_p1/p0 are None in the standard processing path

    elif graph_type == 'OldBPG':
        # Fall back to storing the full object for OldBPG (rarely used)
        return data

    elif graph_type == 'MatrixBPG':
        d['n_clauses'] = data.n_clauses
        d['n_literals'] = data.n_literals
        d['literal_to_edges_matrix'] = data.literal_to_edges_matrix
        d['clause_to_literals_matrix'] = data.clause_to_literals_matrix
        d['edge_to_literal_neighborhood_matrix'] = data.edge_to_literal_neighborhood_matrix
        d['edge_to_clause_assignments_matrix'] = data.edge_to_clause_assignments_matrix
        d['clause_assignment_to_neighborhood_matrix'] = data.clause_assignment_to_neighborhood_matrix

    return d


def _unpack_data(d) -> object:
    """
    Reconstruct a graph Data object from the compact dict saved by _pack_data.
    Upcast tensors back to their required dtypes.
    """
    # Backward compat: old files stored the Data object directly
    if not isinstance(d, dict) or '__graph_type__' not in d:
        return d

    graph_type = d['__graph_type__']

    def _long(t):
        return t.to(torch.long) if t is not None else None

    def _float(t):
        return t.to(torch.float) if t is not None else None

    if graph_type == 'LCG':
        return LCG(
            l_size=d['l_size'],
            c_size=d['c_size'],
            c_edge_index=_long(d.get('c_edge_index')),
            l_edge_index=_long(d.get('l_edge_index')),
            l_batch=_long(d.get('l_batch')),
            c_batch=_long(d.get('c_batch')),
        )

    elif graph_type == 'BPG':
        return BPG(
            n_clauses=d['n_clauses'],
            n_literals=d['n_literals'],
            literal_indices_per_edge=_long(d['literal_indices_per_edge']),
            literal_indices_per_occurence=_long(d['literal_indices_per_occurence']),
            clause_indices_per_occurence=_long(d['clause_indices_per_occurence']),
            local_satisfaction_percentage_per_edge=_float(d.get('local_satisfaction_percentage_per_edge')),
            c2l_msg_receiver_indices=_long(d['c2l_msg_receiver_indices']),
            c2l_msg_sender_indices=_long(d['c2l_msg_sender_indices']),
            l2c_msg_receiver_indices=_long(d['l2c_msg_receiver_indices']),
            l2c_assignment_indices=_long(d['l2c_assignment_indices']),
            l2c_assignment_neighborhoods=_long(d['l2c_assignment_neighborhoods']),
        )

    elif graph_type == 'MatrixBPG':
        return MatrixBPG(
            n_clauses=d['n_clauses'],
            n_literals=d['n_literals'],
            literal_to_edges_matrix=d['literal_to_edges_matrix'],
            clause_to_literals_matrix=d['clause_to_literals_matrix'],
            edge_to_literal_neighborhood_matrix=d['edge_to_literal_neighborhood_matrix'],
            edge_to_clause_assignments_matrix=d['edge_to_clause_assignments_matrix'],
            clause_assignment_to_neighborhood_matrix=d['clause_assignment_to_neighborhood_matrix'],
        )

    # Fallback
    return d


class SATDataset(Dataset):
    """
    processed_dir: str
        Directory to save the processed data (LCG or BPG) in .pt files
    """
    def __init__(self, data_dir, data_size, opts, file_indices=None):
        """
        data_dir: str
            Directory with sat input data and labels
        """
        self.opts = opts
        all_files = sorted(glob.glob(data_dir + '/**/*.cnf', recursive=True))
        all_labels = self._get_labels(data_dir)
        if all_labels is None:
            all_labels = [None] * len(all_files)
        
        # Better error message for debugging
        if len(all_labels) != len(all_files):
            print(f'\n[ERROR] Mismatch between files and labels!')
            print(f'[ERROR] Number of .cnf files found: {len(all_files)}')
            print(f'[ERROR] Number of labels found: {len(all_labels)}')
            print(f'[ERROR] Data directory: {data_dir}')
            if hasattr(self.opts, 'loss'):
                label_file = 'marginals.pkl' if self.opts.loss == 'marginal' else 'assignments.pkl'
                print(f'[ERROR] Expected label file: {os.path.join(data_dir, label_file)}')
            raise AssertionError(f'Number of labels ({len(all_labels)}) does not match number of files ({len(all_files)})')
        
        if data_size is not None:
            assert data_size <= len(all_files)

            if file_indices is not None:
                assert len(file_indices) == data_size
                self.file_indices = file_indices
            else:
                self.file_indices = np.random.RandomState(0).permutation(len(all_files))[:data_size]
            self.all_files = all_files[self.file_indices]
            self.all_labels = all_labels[self.file_indices]
        else:
            self.file_indices = list(range(len(all_files)))
            self.all_files = all_files
            self.all_labels = all_labels
        
        if self.opts.model == 'NeuroSAT':
            self.graph = 'LCG'
        elif self.opts.model == "ArieNet":
            self.graph = 'BPG'
        elif self.opts.model == 'GNNSquared':
            self.graph = 'BPG'
        elif self.opts.model == 'MatrixNet':
            self.graph = 'MatrixBPG'
        else:
            self.graph = 'OldBPG'
                    
        # data_dir is 'root' in Dataset
        super().__init__(data_dir)
    
    @property
    def processed_file_names(self):
        return [self._processed_file_name(idx) for idx in self.file_indices]

    def _processed_suffix(self):
        if self.graph == 'BPG' and getattr(self.opts, 'no_precomputed_local_sat', False):
            return '_no_precomputed_local_sat'
        return ''

    def _legacy_processed_file_name(self, idx):
        return f'data_{idx}_{self.graph}_{self.opts.task}.pt'

    def _target_processed_file_name(self, idx):
        return f'data_{idx}_{self.graph}_{self.opts.task}{self._processed_suffix()}.pt'

    def _processed_file_name(self, idx):
        target_name = self._target_processed_file_name(idx)

        # In no-precompute mode, reuse existing legacy BPG cache files (with
        # local satisfaction percentages) and let the model ignore that feature.
        if self.graph == 'BPG' and getattr(self.opts, 'no_precomputed_local_sat', False):
            target_path = os.path.join(self.processed_dir, target_name)
            if os.path.exists(target_path):
                return target_name

            legacy_name = self._legacy_processed_file_name(idx)
            legacy_path = os.path.join(self.processed_dir, legacy_name)
            if os.path.exists(legacy_path):
                return legacy_name

        return target_name
    
    def process(self):
        """
        Stored een GRAPH (LCG of BPG) in een .pt file in self.processed_dir
        Waar wordt dit aangeroepen?? Er zijn geen references...
        -> in super().__init__() Daar wordt wel _process() van Dataset aangeroepen
        En die roept weer deze process()

        
        Stores ALL graphs given to the init function

        
        Mischien in
        for data in train_loader:
            -> self.__next_data()
            -> self._process_data(data)
        """
        print(f'\n[DATASET PROCESS] Starting to process {len(self.all_files)} files into {self.graph} format')
        print(f'[DATASET PROCESS] Processed files will be saved to: {self.processed_dir}')
        print(f'[DATASET PROCESS] Using 4 parallel workers')
        start_process_time = time.time()
        
        # Prepare arguments for parallel processing
        process_args = []
        for idx, (file_path, label) in enumerate(zip(self.all_files, self.all_labels)):
            file_name = self._processed_file_name(idx)
            output_path = os.path.join(self.processed_dir, file_name)
            if os.path.exists(output_path):
                continue

            # If no cache exists yet for this item, write to the mode-specific target.
            output_path = os.path.join(self.processed_dir, self._target_processed_file_name(idx))
            process_args.append((
                idx,
                file_path,
                self.graph,
                self.opts.task,
                output_path,
                getattr(self.opts, 'no_precomputed_local_sat', False),
            ))
        
        # Process files in parallel with 4 workers
        completed = 0
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_process_single_file, args): args[0] for args in process_args}
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    if completed == 1 or completed % 40 == 0:
                        print(f'[DATASET PROCESS] Completed {completed}/{len(self.all_files)} files')
                except Exception as e:
                    print(f'[DATASET PROCESS] Error processing file {idx}: {e}')
                    raise
        
        total_time = time.time() - start_process_time
        print(f'\n[DATASET PROCESS] Finished processing all {completed} files in {total_time:.2f} seconds')
        print(f'[DATASET PROCESS] Average time per file: {total_time/completed:.2f} seconds\n')

    def _transform2LCG(self, n_vars, clauses):
        """
        Representatie voor NeuroSAT
        """
        
        c_edge_index_list = []
        l_edge_index_list = []

        l_batch = None
        c_batch = None

        if self.opts.task == 'model-counting':
            l_batch = torch.zeros(n_vars * 2, dtype=torch.long)
        else:
            c_batch = torch.zeros(len(clauses), dtype=torch.long)
        
        for c_idx, clause in enumerate(clauses):
            for literal in clause:
                l_idx = literal2l_idx(literal)
                c_edge_index_list.append(c_idx)
                l_edge_index_list.append(l_idx)
        
        c_edge_index = torch.tensor(c_edge_index_list, dtype=torch.long)
        l_edge_index = torch.tensor(l_edge_index_list, dtype=torch.long)

        return LCG(
            n_vars * 2,
            len(clauses),
            c_edge_index,
            l_edge_index,
            l_batch,
            c_batch
        )
    
    # def _transform2BPG(self, n_vars, clauses):
    #     """
    #     Deze functie moet ik uitzoeken, dit is representatie voor NSNet


    #     """
    #     sign_l_edge_index_list = []
    #     type_edge_index_list = []

    #     c2l_msg_aggr_c_index_map = {l: [] for l in range(2 * n_vars)}
    #     c2l_msg_aggr_edge_index_map = {l: [] for l in range(2 * n_vars)}
        
    #     c2l_msg_repeat_index_list = []
    #     c2l_msg_scatter_index_list = []

    #     l2c_msg_aggr_repeat_index_list = []
    #     l2c_msg_aggr_scatter_index_list = []
    #     l2c_msg_scatter_index_list = []

    #     # auxiliary parameters
    #     c_blf_repeat_index = None
    #     c_blf_scatter_index = None
    #     c_blf_norm_index = None
    #     v_degrees = None
    #     c_batch = None
    #     v_batch = None
    #     l_edge_index = None
    #     c_edge_index = None

    #     if self.opts.task == 'model-counting':
    #         c_blf_repeat_index_list = []
    #         c_blf_scatter_index_list = []
    #         c_blf_norm_index_list = []
    #         v_degrees = torch.zeros(n_vars)
    #     else:
    #         l_edge_index_list = []
    #         c_edge_index_list = []
        
    #     index_base = 0
    #     msg_aggr_index = 0
    #     for c_idx, clause in enumerate(clauses):
    #         used_vars = sorted(list(set([abs(literal)-1 for literal in clause])))
    #         # literal to clause message
    #         for msg_idx, v_idx in enumerate(used_vars):
    #             # Literal ints for variable index, i.e. var 9 krijgt literals 16, 17
    #             pl_idx = v_idx * 2
    #             nl_idx = v_idx * 2 + 1

    #             # Index base telt op met 2*aantal variabelen in clause
    #             # p_msg_idx, n_msg_idx zijn de indices van literal, op volgorde van variable in clause.... waar is dit voor?
    #             p_msg_idx = index_base + msg_idx * 2
    #             n_msg_idx = index_base + msg_idx * 2 + 1
                
    #             # Lijst met + en - literals voor variable numbers?? Waarom niet gewoon de variabelen opslaan hier?
    #             sign_l_edge_index_list.append(pl_idx)
    #             sign_l_edge_index_list.append(nl_idx)
                
    #             # Map van literal, naar list of clause indices waarin deze voorkomt, maar zonder onderscheid in + en -!
    #             c2l_msg_aggr_c_index_map[pl_idx].append(c_idx)
    #             c2l_msg_aggr_c_index_map[nl_idx].append(c_idx)
                
    #             # Map van literal naar matchende 'volgorde-literal-index' in clause
    #             # e.g. voor clause (5. 9), is krijgen we voor 9 map van 16 naar 2, en map van 17 naar 3....??
    #             c2l_msg_aggr_edge_index_map[pl_idx].append(p_msg_idx)
    #             c2l_msg_aggr_edge_index_map[nl_idx].append(n_msg_idx)
                
    #             # Hier voegen we '0' of '1' toe, naar gelang de literal + of - is, dit is dus een lijst van size n_edges*2
    #             # Voor positive instance van l in c appenden we [1,0]
    #             # Voor negative instance [0,1]
    #             if (v_idx + 1) in clause:
    #                 type_edge_index_list.append(1)
    #             else:
    #                 type_edge_index_list.append(0)
                
    #             if -(v_idx + 1) in clause:
    #                 type_edge_index_list.append(1)
    #             else:
    #                 type_edge_index_list.append(0)
            
    #         # clause to literal massage

    #         # used_vars zijn weer de literal_nummers van de vars in clauses
    #         # Dus we discarden 1 voor 1 de vars?

    #         #    var position,    var number
    #         for scatter_msg_idx, discard_v_idx in enumerate(used_vars):
    #             # indices, is steeds een keuze van 0 en 1 voor elke variable
    #             # , dit iterate over alle assignmnets (behalve de gediscarde) (dus tuples van lengte 2 voor 3 vars)
    #             for indices in np.ndindex(tuple([2] * (len(used_vars)-1))):
    #                 # msg_index is letterlijk de index van de literal in de clause
    #                 # dus msg_table = [(p_msg_index, n_msg_index)], i.e. de indices van de + en - edge voor deze l-c occurences
    #                 # dit voor elke var behalve de 'gediscarde'
    #                 # Dus, als het de eerste clause is [x,y,z], en we discarden 'x'+ (0) of x- (1)
    #                 # [(3,2), (5,4)], als we z discarden [(1,0), (3,2)]
    #                 msg_table = [(index_base + msg_idx * 2 + 1, index_base + msg_idx * 2) 
    #                     for msg_idx, v_idx in enumerate(used_vars) if v_idx != discard_v_idx]
                    
    #                 # Hier pakken we uit msg_table, voor elke '0' of '1' in de assignment, de - (+1) of + (0) edge index
    #                 # Als positive occurence: gaat de (+1) naar '0', de (0) naar 1 in type_edge_index_list
    #                 # Dus als indices = [0,0], en allebei positive occurences, is assign [1,1]
    #                 # DUS 'index 0' betekent, maar 'l' waar -> dat geeft 1 als l positief in c zit
    #                 # Als negative occurence andersom, dus dan is assign [0,0]
    #                 assign = np.array([type_edge_index_list[msg_table[i][idx]] for i, idx in enumerate(indices)])

    #                 # dus, 'index' 0 betekent variable is true, index '1' betekent variable is false
    #                 is_sat = assign.sum() > 0

    #                 # Dit zijn dan de matchende edge indices voor deze assignment
    #                 # Dus voor edge c-l1, zijn dit de edges van alle l2-c, voor deze assignment. 
    #                 msg_aggr_repeat_index = [msg_table[i][idx] for i, idx in enumerate(indices)]
                    
    #                 # index_base+scatter_msg_idx*2 is de tweede index voor deze edge in index_list
    #                 # dus deze is 1 als de literal negatief occurred

    #                 # Deze triggered alleen als l positief occurred, or SAT
    #                 if type_edge_index_list[index_base + scatter_msg_idx * 2] or is_sat:       
    #                     # Dit wordt dus een lijst van alle incoming literal edges,
    #                     # per c1-l1, per (satisfying?) assignment per l2-c2
    #                     l2c_msg_aggr_repeat_index_list.append(msg_aggr_repeat_index)

    #                     # Dit wordt een lijst met, per c1-l1, per assignment per l2-c2 een index die de som stuurt
    #                     l2c_msg_aggr_scatter_index_list.append([msg_aggr_index] * len(msg_aggr_repeat_index))
    #                     msg_aggr_index += 1

    #                     # Hier scatter_som op geeft, per assignment (sommige 0 sommige 2 keer) de som van outgoing literals

    #                     # Dit is een lijst met de c-l index
    #                     # scatter_som op output van het vorige met deze indices geeft som over alle assignments
    #                     # voor c1, naar l1
    #                     l2c_msg_scatter_index_list.append(index_base + scatter_msg_idx * 2)

    #                 # Deze triggered bij - occurence en SAT
    #                 if type_edge_index_list[index_base + scatter_msg_idx * 2 + 1] or is_sat:
    #                     l2c_msg_aggr_repeat_index_list.append(msg_aggr_repeat_index)
    #                     l2c_msg_aggr_scatter_index_list.append([msg_aggr_index] * len(msg_aggr_repeat_index))
    #                     msg_aggr_index += 1
    #                     l2c_msg_scatter_index_list.append(index_base + scatter_msg_idx * 2 + 1)

    #                 # import pdb; pdb.set_trace()

    #                 # TODO fix deze terms zodat alleen satisfying assignments worden gebruikt??

    #         index_base += len(used_vars) * 2

    #     # Tensor, met lijst van alle + en - literals, op volgorde van clauses, i.e. [9,2,3], [4,5]
    #     # -> [16, 17, 2, 3, 4, 5, 6, 7, 8, 9]
    #     sign_l_edge_index = torch.tensor(sign_l_edge_index_list, dtype=torch.long)

    #     index_base = 0
    #     for c_idx, clause in enumerate(clauses):
    #         used_vars = sorted(list(set([abs(literal)-1 for literal in clause])))
    #         for msg_idx, v_idx in enumerate(used_vars):
    #             # Zelfde als in vorige loop
    #             pl_idx = v_idx * 2
    #             nl_idx = v_idx * 2 + 1

    #             # Zelfde als in vorige loop, i.e. index representations van literal, o.b.v. volgorde in clauses
    #             p_msg_idx = index_base + msg_idx * 2
    #             n_msg_idx = index_base + msg_idx * 2 + 1
                
    #             #c2l_msg_aggr_c_index_map is map van literal (in [1, 2N-1] format) naar list van clauses
    #             # Waar l of -l in voorkomt.

    #             # c2l_msg_aggr_edge_index_map is map van literal naar 'volgorde index' in clause
    #             # opnieuw maar e.g. map[16] = 2, map[17]=3, onafhankelijk van of occurence + of - is
    #             for neighbor_c_idx, neighbor_msg_idx in zip(c2l_msg_aggr_c_index_map[pl_idx], c2l_msg_aggr_edge_index_map[pl_idx]):

    #                 # Doe niets, als de neighbor_clause (via literal) hetzelfde is als de huidige clause
    #                 if neighbor_c_idx == c_idx:
    #                     continue                    

    #                 # Dit wordt een lijst met, per alle literals per clause, de 'volgorde index' van deze variable
    #                 # in elke andere clause waar deze variabele in zit.

    #                 # bijv. voor var literal -3, var komt voor in clauses [0,3,5,10]
    #                 # Matchende volgorde indices zijn [0, 18, 36, 60], maar de 0 skippen we als c_idx==0
    #                 c2l_msg_repeat_index_list.append(neighbor_msg_idx)

    #                 # Dit is per occurence de volgorde index in deze clause, e.g. [0, 0, 0] (size is n_neighbors -1)
    #                 c2l_msg_scatter_index_list.append(p_msg_idx)
                
    #             for neighbor_c_idx, neighbor_msg_idx in zip(c2l_msg_aggr_c_index_map[nl_idx], c2l_msg_aggr_edge_index_map[nl_idx]):
    #                 if neighbor_c_idx == c_idx:
    #                     continue

    #                 # Hier voegen we de negatieve 'order index' voor literal in elke neighbor toe
    #                 # i.e. van [18,36,60], naar [18, 36, 60, 19, 27, 61]
    #                 c2l_msg_repeat_index_list.append(neighbor_msg_idx)

    #                 # Hier voegen we voor de negatieve volgorde index van deze occurence toe voor elke neighbor
    #                 # i.e. van [0,0,0] naar [0, 0, 0, 1, 1, 1]
    #                 c2l_msg_scatter_index_list.append(n_msg_idx)
            
    #         index_base += len(used_vars) * 2

    #     # Dit is dan, per edge c-l, een entry voor elke occurence van l/-l in een andere clause
    #     # met de 'volgorde index' van l
    #     # daarna opnieuw een entry voor elke occurence van l/-l in een andere clause
    #     # met de 'volgorde index' van -l
    #     c2l_msg_repeat_index = torch.tensor(c2l_msg_repeat_index_list, dtype=torch.long)

    #     # Dit is hetzelfde aantal, maar gesorteerd per 'incoming edge' naar l
    #     # dus voor c-l, en l-c1, l-c2, l-c2
    #     # geeft dit 3* order van +l in c, dan 3 * van -l in c
    #     c2l_msg_scatter_index = torch.tensor(c2l_msg_scatter_index_list, dtype=torch.long)

    #     # concat
    #     l2c_msg_aggr_repeat_index_list = list(itertools.chain(*l2c_msg_aggr_repeat_index_list))
    #     l2c_msg_aggr_scatter_index_list = list(itertools.chain(*l2c_msg_aggr_scatter_index_list))

    #     l2c_msg_aggr_repeat_index = torch.tensor(l2c_msg_aggr_repeat_index_list, dtype=torch.long)
    #     l2c_msg_aggr_scatter_index = torch.tensor(l2c_msg_aggr_scatter_index_list, dtype=torch.long)
    #     l2c_msg_scatter_index = torch.tensor(l2c_msg_scatter_index_list, dtype=torch.long)

    #     if self.opts.task == 'model-counting':
    #         index_base = 0
    #         blf_index = 0
    #         for c_idx, clause in enumerate(clauses):
    #             used_vars = set([abs(literal)-1 for literal in clause])
    #             for indices in np.ndindex(tuple([2] * len(used_vars))):
    #                 msg_table = [(index_base + msg_idx * 2 + 1, index_base + msg_idx * 2) 
    #                     for msg_idx, v_idx in enumerate(used_vars)]
    #                 assign = np.array([type_edge_index_list[msg_table[i][idx]] for i, idx in enumerate(indices)])
    #                 is_sat = assign.sum() > 0
    #                 blf_repeat_index = [msg_table[i][idx] for i, idx in enumerate(indices)]
                    
    #                 if is_sat:
    #                     c_blf_repeat_index_list.append(blf_repeat_index)
    #                     c_blf_scatter_index_list.append([blf_index] * len(blf_repeat_index))
    #                     blf_index += 1
    #                     c_blf_norm_index_list.append(c_idx)
    #             index_base += len(used_vars) * 2

    #             for v_idx in used_vars:
    #                 v_degrees[v_idx] += 1
            
    #         c_blf_repeat_index_list = list(itertools.chain(*c_blf_repeat_index_list))
    #         c_blf_scatter_index_list = list(itertools.chain(*c_blf_scatter_index_list))

    #         c_blf_repeat_index = torch.tensor(c_blf_repeat_index_list, dtype=torch.long)
    #         c_blf_scatter_index = torch.tensor(c_blf_scatter_index_list, dtype=torch.long)
    #         c_blf_norm_index = torch.tensor(c_blf_norm_index_list, dtype=torch.long)

    #         c_batch = torch.zeros(len(clauses), dtype=torch.long)
    #         v_batch = torch.zeros(n_vars, dtype=torch.long)
    #     else:
    #         for c_idx, clause in enumerate(clauses):
    #             for literal in clause:
    #                 # Zelfde map, i.e. 9 naar 16, -9 naar 17
    #                 l_idx = literal2l_idx(literal)

    #                 # Dit is een lijst van alle literals, op volgorde van occurence in clauses, met duplicates
    #                 l_edge_index_list.append(l_idx)
    #                 # Dit is een lijst van alle clause indices
    #                 c_edge_index_list.append(c_idx)
            
    #         # 
    #         l_edge_index = torch.tensor(l_edge_index_list, dtype=torch.long)
    #         c_edge_index = torch.tensor(c_edge_index_list, dtype=torch.long)
    #         c_batch = torch.zeros(len(clauses), dtype=torch.long)

    #     # import pdb; pdb.set_trace()
    #     params = OldBPGParams(
    #         l_size=n_vars*2, 
    #         c_size=len(clauses),
    #         sign_l_edge_index=sign_l_edge_index, 
    #         c2l_msg_repeat_index=c2l_msg_repeat_index,
    #         c2l_msg_scatter_index=c2l_msg_scatter_index,
    #         l2c_msg_aggr_repeat_index=l2c_msg_aggr_repeat_index,
    #         l2c_msg_aggr_scatter_index=l2c_msg_aggr_scatter_index,
    #         l2c_msg_scatter_index=l2c_msg_scatter_index,
    #         c_blf_repeat_index=c_blf_repeat_index,
    #         c_blf_scatter_index=c_blf_scatter_index,
    #         c_blf_norm_index=c_blf_norm_index,
    #         v_degrees=v_degrees,
    #         c_batch=c_batch,
    #         v_batch=v_batch,
    #         l_edge_index=l_edge_index,
    #         c_edge_index=c_edge_index,
    #     )

    #     # I think it's essential that the Data constructor has its attributes explicitly
    #     # We have to give the params as keywords to the inheriting class
    #     return OldBPG(*params)
    #     return OriginalBPG(*params)


        return OriginalBPG(
            n_vars*2, 
            len(clauses),
            sign_l_edge_index, 
            c2l_msg_repeat_index,
            c2l_msg_scatter_index,
            l2c_msg_aggr_repeat_index,
            l2c_msg_aggr_scatter_index,
            l2c_msg_scatter_index,
            c_blf_repeat_index,
            c_blf_scatter_index,
            c_blf_norm_index,
            v_degrees,
            c_batch,
            v_batch,
            l_edge_index,
            c_edge_index,
        )

    def _get_labels(self, data_dir):
        if self.opts.task == 'model-counting':
            labels = None
            labels_file = os.path.join(data_dir, 'countings.pkl')
            if os.path.exists(labels_file):
                print(f'[DATASET] Loading labels from: {labels_file}')
                with open(labels_file, 'rb') as f:
                    labels = pickle.load(f)
                labels = [torch.tensor(label, dtype=torch.float) for label in labels]
                print(f'[DATASET] Loaded {len(labels)} labels')
            return labels
        elif self.opts.task == 'sat-solving':
            # Pulls assignments.pkl or marginals.pkl
            labels = None
            if hasattr(self.opts, 'loss'):
                if self.opts.loss == 'assignment':
                    labels_file = os.path.join(data_dir, 'assignments.pkl')
                else:
                    assert self.opts.loss == 'marginal'
                    labels_file = os.path.join(data_dir, 'marginals.pkl')
                if os.path.exists(labels_file):
                    print(f'[DATASET] Loading labels from: {labels_file}')
                    with open(labels_file, 'rb') as f:
                        labels = pickle.load(f)
                    labels = [torch.tensor(label, dtype=torch.float) for label in labels]
                    print(f'[DATASET] Loaded {len(labels)} labels')
            return labels
    
    def len(self):
        return len(self.all_files)

    def get(self, idx):
        """
        Pulls a processed GRAPH from a .pt file in self.processed_dir
        """
        file_name = self._processed_file_name(idx)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            raw = torch.load(os.path.join(self.processed_dir, file_name), weights_only=False)
        data = _unpack_data(raw)
        data.y = self.all_labels[idx]
        return data

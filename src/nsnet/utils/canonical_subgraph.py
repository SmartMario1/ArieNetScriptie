"""
Canonical Subgraph Feature Extraction

This module extracts edge neighborhood subgraphs, converts them to canonical 2CNF
representations using Quine-McCluskey, and prepares them for GNN processing.
"""

import typing
from dataclasses import dataclass
import torch
import numpy as np
from torch_geometric.data import Data

from nsnet.utils.quine_mccluskey import generate_2cnf_for_literal_assignment


@dataclass
class EdgeSubgraph:
    """
    Represents a canonical subgraph for an edge with a specific literal assignment.
    
    Attributes:
        edge_index: Index of the original edge in the BPG
        clauses_2cnf: Canonical 2CNF representation (list of 2-literal tuples)
        num_vars: Number of variables in the subformula
        variable_mapping: Mapping from original variable numbers to subgraph variable indices
    """
    edge_index: int
    clauses_2cnf: typing.List[typing.Tuple[int, int]]
    num_vars: int
    variable_mapping: typing.Dict[int, int]


def extract_edge_neighborhood(edge, all_edges) -> typing.Tuple[typing.List[typing.Tuple[int, ...]], typing.Set[int]]:
    """
    Extract the immediate neighborhood of an edge as a list of clauses.
    
    The neighborhood includes:
    1. The clause of the edge itself
    2. All clauses connected to the variable of the edge  
    3. All clauses connected to other variables in the edge's clause (2-hop neighborhood)
    
    Args:
        edge: The Edge object from BPGParamBuilder
        all_edges: List of all edges in the BPG
    
    Returns:
        Tuple of (list of clause contents, set of all variable numbers in neighborhood)
    """
    # Start with the clause of the edge
    neighborhood_clauses = {edge.clause_node.content}
    neighborhood_vars = set(abs(lit) for lit in edge.clause_node.content)
    
    # Get all clauses connected to the edge's variable
    for e in all_edges:
        if e.variable_node.var_number == edge.variable_node.var_number:
            neighborhood_clauses.add(e.clause_node.content)
            neighborhood_vars.update(abs(lit) for lit in e.clause_node.content)
    
    # Get all clauses connected to other variables in the edge's clause (2-hop)
    for other_lit in edge.clause_node.content:
        other_var = abs(other_lit)
        if other_var != edge.variable_node.var_number:
            for e in all_edges:
                if e.variable_node.var_number == other_var:
                    neighborhood_clauses.add(e.clause_node.content)
                    neighborhood_vars.update(abs(lit) for lit in e.clause_node.content)
    
    return list(neighborhood_clauses), neighborhood_vars


def create_canonical_subgraphs_for_edge(
    edge,
    all_edges,
    edge_index: int
) -> typing.Tuple[EdgeSubgraph, EdgeSubgraph]:
    """
    Create two canonical subgraph representations for an edge: one for p=1 and one for p=0.
    
    Args:
        edge: The Edge object from BPGParamBuilder
        all_edges: List of all edges in the BPG
        edge_index: Index of the edge in the BPG
    
    Returns:
        Tuple of (EdgeSubgraph for p=1, EdgeSubgraph for p=0)
    """
    # Extract neighborhood clauses
    neighborhood_clauses, neighborhood_vars = extract_edge_neighborhood(edge, all_edges)
    
    # Create canonical variable mapping (for locality and consistency)
    # Map the edge's variable to 1, other variables to 2, 3, 4, ...
    sorted_vars = sorted(neighborhood_vars)
    variable_mapping = {}
    
    # Put edge variable first
    variable_mapping[edge.variable_node.var_number] = 1
    next_idx = 2
    for var in sorted_vars:
        if var != edge.variable_node.var_number:
            variable_mapping[var] = next_idx
            next_idx += 1
    
    num_vars = len(variable_mapping)
    
    # Remap clauses to use canonical variable indices
    remapped_clauses = []
    for clause in neighborhood_clauses:
        remapped_clause = tuple(
            variable_mapping[abs(lit)] if lit > 0 else -variable_mapping[abs(lit)]
            for lit in clause
        )
        remapped_clauses.append(remapped_clause)
    
    # Generate 2CNF for p=1 (edge's literal is true)
    edge_literal = 1 if edge.sign == "+" else -1
    clauses_2cnf_p1 = generate_2cnf_for_literal_assignment(
        remapped_clauses,
        edge_literal,
        True,
        num_vars
    )
    
    # Generate 2CNF for p=0 (edge's literal is false)
    clauses_2cnf_p0 = generate_2cnf_for_literal_assignment(
        remapped_clauses,
        edge_literal,
        False,
        num_vars
    )
    
    subgraph_p1 = EdgeSubgraph(
        edge_index=edge_index,
        clauses_2cnf=clauses_2cnf_p1,
        num_vars=num_vars,
        variable_mapping=variable_mapping
    )
    
    subgraph_p0 = EdgeSubgraph(
        edge_index=edge_index,
        clauses_2cnf=clauses_2cnf_p0,
        num_vars=num_vars,
        variable_mapping=variable_mapping
    )
    
    return subgraph_p1, subgraph_p0


def edge_subgraph_to_pyg_data(subgraph: EdgeSubgraph) -> Data:
    """
    Convert an EdgeSubgraph to a PyTorch Geometric Data object.
    
    Creates a bipartite graph with:
    - Variable nodes (one per variable in the subgraph)
    - Clause nodes (one per 2CNF clause)
    - Edges connecting variables to clauses they appear in
    
    Args:
        subgraph: EdgeSubgraph to convert
    
    Returns:
        PyTorch Geometric Data object representing the subgraph
    """
    num_vars = subgraph.num_vars
    num_clauses = len(subgraph.clauses_2cnf)
    
    if num_clauses == 0:
        # Empty formula (always satisfiable or tautology).
        # Use typed 2D features: [1, 0] for variable-like nodes.
        x = torch.zeros(max(num_vars, 1), 2)
        x[:, 0] = 1.0
        return Data(
            x=x,
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            num_vars=num_vars,
            num_clauses=0,
            original_edge_index=subgraph.edge_index
        )
    
    # Create edge index for bipartite graph
    # Each clause connects to its 2 variables
    edges_var_to_clause = []
    edges_clause_to_var = []
    
    for clause_idx, clause in enumerate(subgraph.clauses_2cnf):
        for literal in clause:
            var_idx = abs(literal) - 1  # Convert to 0-indexed
            # Variable to clause edge
            edges_var_to_clause.append([var_idx, num_vars + clause_idx])
            # Clause to variable edge
            edges_clause_to_var.append([num_vars + clause_idx, var_idx])
    
    # Combine into edge index
    edge_index = torch.tensor(
        edges_var_to_clause + edges_clause_to_var, 
        dtype=torch.long
    ).t().contiguous()
    
    # Initialize node features as one-hot node types:
    # variables -> [1, 0], clauses -> [0, 1]
    x_var = torch.zeros(num_vars, 2)
    x_var[:, 0] = 1.0

    x_clause = torch.zeros(num_clauses, 2)
    x_clause[:, 1] = 1.0
    
    # Concatenate all node features
    x = torch.cat([x_var, x_clause], dim=0)
    
    return Data(
        x=x,
        edge_index=edge_index,
        num_vars=num_vars,
        num_clauses=num_clauses,
        original_edge_index=subgraph.edge_index
    )


def create_batch_subgraph_data(
    edge_subgraphs: typing.List[EdgeSubgraph]
) -> typing.List[Data]:
    """
    Convert a list of EdgeSubgraph objects to a list of PyG Data objects.
    
    Args:
        edge_subgraphs: List of EdgeSubgraph objects
    
    Returns:
        List of PyTorch Geometric Data objects
    """
    return [edge_subgraph_to_pyg_data(sg) for sg in edge_subgraphs]


def compute_subgraph_features_for_all_edges(
    edges,
    all_edges
) -> typing.Tuple[typing.List[EdgeSubgraph], typing.List[EdgeSubgraph]]:
    """
    Compute canonical subgraph representations for all edges in a BPG.
    
    Args:
        edges: List of Edge objects from BPGParamBuilder
        all_edges: Same as edges (kept for API consistency)
    
    Returns:
        Tuple of (list of subgraphs for p=1, list of subgraphs for p=0)
        Each list has one subgraph per edge, in the same order as the input edges
    """
    subgraphs_p1 = []
    subgraphs_p0 = []
    
    for edge_idx, edge in enumerate(edges):
        sg_p1, sg_p0 = create_canonical_subgraphs_for_edge(edge, all_edges, edge_idx)
        subgraphs_p1.append(sg_p1)
        subgraphs_p0.append(sg_p0)
    
    return subgraphs_p1, subgraphs_p0


def compute_subgraph_hash(subgraph: EdgeSubgraph) -> int:
    """
    Compute a hash for a canonical subgraph for caching/deduplication.
    
    Args:
        subgraph: EdgeSubgraph to hash
    
    Returns:
        Integer hash value
    """
    # Hash based on the canonical 2CNF clauses
    clauses_tuple = tuple(sorted(subgraph.clauses_2cnf))
    return hash((clauses_tuple, subgraph.num_vars))

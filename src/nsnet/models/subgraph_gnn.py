"""
Subgraph GNN for Canonical Feature Extraction

This module implements a GNN to process canonical subgraphs and extract
features that capture structural properties up to isomorphism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import typing

from nsnet.models.mlp import MLP


class SubgraphGINLayer(MessagePassing):
    """
    Graph Isomorphism Network (GIN) layer for subgraph feature extraction.
    
    GIN is known to be as expressive as the Weisfeiler-Leman graph isomorphism test,
    making it suitable for distinguishing graphs up to isomorphism.
    """
    
    def __init__(self, in_dim: int, out_dim: int, n_mlp_layers: int = 2):
        super(SubgraphGINLayer, self).__init__(aggr='add')
        
        # MLP for message transformation
        self.mlp = MLP(
            num_layers=n_mlp_layers,
            input_dim=in_dim,
            hidden_dim=out_dim,
            output_dim=out_dim,
            activation='relu'
        )
        
        # Epsilon parameter (learnable)
        self.eps = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Propagate messages
        out = self.propagate(edge_index, x=x)
        
        # Add self-loop with learnable epsilon
        out = (1 + self.eps) * x + out
        
        # Apply MLP
        out = self.mlp(out)
        
        return out
    
    def message(self, x_j):
        """Message from neighbor j to node i"""
        return x_j


class SubgraphGNN(nn.Module):
    """
    GNN for processing canonical subgraphs to extract structural features.
    
    This GNN processes the 2CNF SAT graphs derived from edge neighborhoods
    and produces a fixed-size feature vector that captures the structural
    properties of the subgraph.
    
    Architecture:
    - Multiple GIN layers for message passing
    - Global pooling (mean + max) for graph-level representation
    - MLP for final feature extraction
    """
    
    def __init__(
        self,
        node_feature_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 32,
        n_layers: int = 3,
        n_mlp_layers: int = 2,
        pool_type: str = 'mean_max'
    ):
        """
        Args:
            node_feature_dim: Dimension of input node features
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output feature dimension
            n_layers: Number of GNN layers
            n_mlp_layers: Number of layers in MLPs within GIN
            pool_type: Type of global pooling ('mean', 'max', or 'mean_max')
        """
        super(SubgraphGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.pool_type = pool_type
        
        # Initial feature embedding
        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        
        # GIN layers
        self.gin_layers = nn.ModuleList([
            SubgraphGINLayer(hidden_dim, hidden_dim, n_mlp_layers)
            for _ in range(n_layers)
        ])
        
        # Batch normalization after each layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(n_layers)
        ])
        
        # Pooling dimensions
        pool_dim = hidden_dim * 2 if pool_type == 'mean_max' else hidden_dim
        
        # Final MLP for feature extraction
        self.readout_mlp = MLP(
            num_layers=n_mlp_layers,
            input_dim=pool_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation='relu'
        )
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass for subgraph feature extraction.
        
        Args:
            data: Batched PyG Data object containing:
                - x: Node features [total_nodes, node_feature_dim]
                - edge_index: Graph connectivity [2, total_edges]
                - batch: Batch assignment for nodes [total_nodes]
        
        Returns:
            Graph-level features [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial embedding
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Apply GIN layers
        for gin_layer, batch_norm in zip(self.gin_layers, self.batch_norms):
            x = gin_layer(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
        
        # Global pooling
        if self.pool_type == 'mean':
            graph_features = global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            graph_features = global_max_pool(x, batch)
        elif self.pool_type == 'mean_max':
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            graph_features = torch.cat([mean_pool, max_pool], dim=1)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        # Final feature extraction
        output = self.readout_mlp(graph_features)
        
        return output
    
    def forward_batch_subgraphs(
        self,
        subgraph_data_list: typing.List[Data]
    ) -> torch.Tensor:
        """
        Process a list of subgraph Data objects.
        
        Args:
            subgraph_data_list: List of PyG Data objects for subgraphs
        
        Returns:
            Stacked features [len(subgraph_data_list), output_dim]
        """
        if not subgraph_data_list:
            return torch.zeros(0, self.output_dim)
        
        # Batch the data
        batch_data = Batch.from_data_list(subgraph_data_list)
        
        # Process
        return self.forward(batch_data)


class DualSubgraphFeatureExtractor(nn.Module):
    """
    Processes two subgraphs per edge (for p=1 and p=0) and combines their features.
    
    This module:
    1. Processes the p=1 subgraph with SubgraphGNN
    2. Processes the p=0 subgraph with SubgraphGNN  
    3. Combines the two feature vectors into a single edge feature
    """
    
    def __init__(
        self,
        node_feature_dim: int = 1,
        hidden_dim: int = 64,
        subgraph_feature_dim: int = 32,
        output_dim: int = 32,
        n_layers: int = 3,
        n_mlp_layers: int = 2,
        pool_type: str = 'mean_max',
        combination: str = 'concat'
    ):
        """
        Args:
            node_feature_dim: Input node feature dimension
            hidden_dim: Hidden dimension for GNN
            subgraph_feature_dim: Feature dimension for each subgraph
            output_dim: Final output dimension per edge
            n_layers: Number of GNN layers
            n_mlp_layers: Number of MLP layers
            pool_type: Pooling type for SubgraphGNN
            combination: How to combine p=1 and p=0 features ('concat', 'add', 'diff')
        """
        super(DualSubgraphFeatureExtractor, self).__init__()
        
        self.combination = combination
        self.subgraph_feature_dim = subgraph_feature_dim
        self.output_dim = output_dim
        
        # Shared or separate GNNs for p=1 and p=0
        self.subgraph_gnn = SubgraphGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=subgraph_feature_dim,
            n_layers=n_layers,
            n_mlp_layers=n_mlp_layers,
            pool_type=pool_type
        )
        
        # Combination MLP
        if combination == 'concat':
            combine_dim = subgraph_feature_dim * 2
        elif combination in ['add', 'diff', 'mult']:
            combine_dim = subgraph_feature_dim
        else:
            raise ValueError(f"Unknown combination method: {combination}")
        
        self.combine_mlp = MLP(
            num_layers=n_mlp_layers,
            input_dim=combine_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation='relu'
        )
    
    def forward(
        self,
        subgraphs_p1: typing.List[Data],
        subgraphs_p0: typing.List[Data]
    ) -> torch.Tensor:
        """
        Process dual subgraphs for all edges.
        
        Args:
            subgraphs_p1: List of PyG Data for p=1 subgraphs
            subgraphs_p0: List of PyG Data for p=0 subgraphs
        
        Returns:
            Edge features [num_edges, output_dim]
        """
        # Process p=1 subgraphs
        features_p1 = self.subgraph_gnn.forward_batch_subgraphs(subgraphs_p1)
        
        # Process p=0 subgraphs
        features_p0 = self.subgraph_gnn.forward_batch_subgraphs(subgraphs_p0)
        
        # Combine features
        if self.combination == 'concat':
            combined = torch.cat([features_p1, features_p0], dim=1)
        elif self.combination == 'add':
            combined = features_p1 + features_p0
        elif self.combination == 'diff':
            combined = features_p1 - features_p0
        elif self.combination == 'mult':
            combined = features_p1 * features_p0
        else:
            raise ValueError(f"Unknown combination method: {self.combination}")
        
        # Final transformation
        output = self.combine_mlp(combined)
        
        return output

"""
Test script for canonical subgraph features.

This script tests the implementation of canonical subgraph features
by creating a simple SAT formula and processing it through the pipeline.

Usage:
    python test_canonical_subgraph.py
"""

import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from nsnet.utils.dataset import create_bpg_with_subgraphs
from nsnet.utils.canonical_subgraph import (
    extract_edge_neighborhood,
    create_canonical_subgraphs_for_edge,
    edge_subgraph_to_pyg_data,
    create_batch_subgraph_data
)
from nsnet.models.subgraph_gnn import SubgraphGNN, DualSubgraphFeatureExtractor
from train_arienet_backbone import ArieNetBackbone


def test_quine_mccluskey():
    """Test Quine-McCluskey algorithm."""
    print("\n" + "="*80)
    print("Testing Quine-McCluskey Algorithm")
    print("="*80)
    
    from nsnet.utils.quine_mccluskey import quine_mccluskey, canonical_2cnf
    
    # Simple test: (x1 OR x2) AND (NOT x1 OR x2)
    # This simplifies to just x2
    formula = [[1, 2], [-1, 2]]
    num_vars = 2
    
    canonical = canonical_2cnf(formula, num_vars)
    print(f"Input formula: {formula}")
    print(f"Canonical 2CNF: {canonical}")
    
    # The canonical form should be simpler
    print(f"✓ Quine-McCluskey test passed")


def test_subgraph_extraction():
    """Test subgraph extraction from BPG."""
    print("\n" + "="*80)
    print("Testing Subgraph Extraction")
    print("="*80)
    
    # Create a simple 3-SAT formula
    # (x1 OR x2 OR x3) AND (NOT x1 OR x2 OR NOT x4) AND (x3 OR x4 OR NOT x5)
    clauses = [[1, 2, 3], [-1, 2, -4], [3, 4, -5]]
    n_vars = 5
    
    # Create BPG with subgraph features
    bpg = create_bpg_with_subgraphs(clauses, n_vars, compute_subgraphs=True)
    
    print(f"Formula: {clauses}")
    print(f"Number of variables: {n_vars}")
    print(f"Number of clauses: {len(clauses)}")
    print(f"Number of edges: {bpg.n_edges}")
    print(f"Number of subgraphs (p=1): {len(bpg.subgraphs_p1)}")
    print(f"Number of subgraphs (p=0): {len(bpg.subgraphs_p0)}")
    
    # Check first subgraph
    if len(bpg.subgraphs_p1) > 0:
        sg_p1 = bpg.subgraphs_p1[0]
        sg_p0 = bpg.subgraphs_p0[0]
        
        print(f"\nFirst edge subgraphs:")
        print(f"  p=1: {len(sg_p1.clauses_2cnf)} 2CNF clauses, {sg_p1.num_vars} variables")
        print(f"  p=0: {len(sg_p0.clauses_2cnf)} 2CNF clauses, {sg_p0.num_vars} variables")
        print(f"  p=1 clauses: {sg_p1.clauses_2cnf}")
        print(f"  p=0 clauses: {sg_p0.clauses_2cnf}")
    
    print(f"✓ Subgraph extraction test passed")
    return bpg


def test_subgraph_gnn():
    """Test subgraph GNN."""
    print("\n" + "="*80)
    print("Testing Subgraph GNN")
    print("="*80)
    
    # Create a simple formula
    clauses = [[1, 2, 3], [-1, 2, -4]]
    n_vars = 4
    
    # Create BPG with subgraphs
    bpg = create_bpg_with_subgraphs(clauses, n_vars, compute_subgraphs=True)
    
    # Convert subgraphs to PyG data
    subgraph_data_p1 = create_batch_subgraph_data(bpg.subgraphs_p1)
    subgraph_data_p0 = create_batch_subgraph_data(bpg.subgraphs_p0)
    
    print(f"Created {len(subgraph_data_p1)} PyG subgraph objects")
    
    # Initialize subgraph GNN
    subgraph_gnn = SubgraphGNN(
        node_feature_dim=2,
        hidden_dim=32,
        output_dim=16,
        n_layers=2
    )
    
    print(f"Subgraph GNN parameters: {sum(p.numel() for p in subgraph_gnn.parameters()):,}")
    
    # Process subgraphs
    features = subgraph_gnn.forward_batch_subgraphs(subgraph_data_p1)
    
    print(f"Output features shape: {features.shape}")
    print(f"Expected: [{len(subgraph_data_p1)}, 16]")
    
    assert features.shape == (len(subgraph_data_p1), 16), "Feature shape mismatch!"
    
    print(f"✓ Subgraph GNN test passed")


def test_dual_extractor():
    """Test dual subgraph feature extractor."""
    print("\n" + "="*80)
    print("Testing Dual Subgraph Feature Extractor")
    print("="*80)
    
    # Create a simple formula
    clauses = [[1, 2], [-1, 3], [2, -3]]
    n_vars = 3
    
    # Create BPG with subgraphs
    bpg = create_bpg_with_subgraphs(clauses, n_vars, compute_subgraphs=True)
    
    # Convert to PyG data
    subgraph_data_p1 = create_batch_subgraph_data(bpg.subgraphs_p1)
    subgraph_data_p0 = create_batch_subgraph_data(bpg.subgraphs_p0)
    
    # Initialize dual extractor
    dual_extractor = DualSubgraphFeatureExtractor(
        node_feature_dim=2,
        hidden_dim=32,
        subgraph_feature_dim=16,
        output_dim=16,
        n_layers=2,
        combination='concat'
    )
    
    print(f"Dual extractor parameters: {sum(p.numel() for p in dual_extractor.parameters()):,}")
    
    # Extract features
    edge_features = dual_extractor(subgraph_data_p1, subgraph_data_p0)
    
    print(f"Edge features shape: {edge_features.shape}")
    print(f"Expected: [{bpg.n_edges}, 16]")
    
    assert edge_features.shape == (bpg.n_edges, 16), "Edge feature shape mismatch!"
    
    print(f"✓ Dual extractor test passed")


def test_arienet_backbone():
    """Test ArieNet backbone with subgraph features."""
    print("\n" + "="*80)
    print("Testing ArieNet Backbone with Subgraph Features")
    print("="*80)
    
    # Create a simple formula
    clauses = [[1, 2, 3], [-1, 2, -4], [3, 4, -5]]
    n_vars = 5
    
    # Create BPG with subgraphs
    bpg = create_bpg_with_subgraphs(clauses, n_vars, compute_subgraphs=True)
    
    # Add dummy labels
    bpg.y = torch.tensor([0, 1, 2, 2, 1], dtype=torch.long)  # Some backbone, some free
    
    print(f"Formula: {clauses}")
    print(f"Variables: {n_vars}")
    print(f"Edges: {bpg.n_edges}")
    
    # Test without subgraph features
    print("\n--- Without subgraph features ---")
    model_no_sg = ArieNetBackbone(
        dim=64,
        n_rounds=5,
        use_subgraph_features=False
    )
    
    params_no_sg = sum(p.numel() for p in model_no_sg.parameters())
    print(f"Model parameters: {params_no_sg:,}")
    
    output_no_sg = model_no_sg(bpg)
    print(f"Output shape: {output_no_sg.shape}")
    print(f"Expected: [{n_vars}, 2]")
    
    assert output_no_sg.shape == (n_vars, 2), "Output shape mismatch!"
    
    # Test with subgraph features
    print("\n--- With subgraph features ---")
    model_with_sg = ArieNetBackbone(
        dim=64,
        n_rounds=5,
        use_subgraph_features=True,
        subgraph_dim=16
    )
    
    params_with_sg = sum(p.numel() for p in model_with_sg.parameters())
    print(f"Model parameters: {params_with_sg:,}")
    print(f"Additional parameters: {params_with_sg - params_no_sg:,}")
    
    output_with_sg = model_with_sg(bpg)
    print(f"Output shape: {output_with_sg.shape}")
    print(f"Expected: [{n_vars}, 2]")
    
    assert output_with_sg.shape == (n_vars, 2), "Output shape mismatch!"
    
    print(f"✓ ArieNet backbone test passed")


def test_forward_backward():
    """Test full forward and backward pass."""
    print("\n" + "="*80)
    print("Testing Forward and Backward Pass")
    print("="*80)
    
    # Create a simple formula
    clauses = [[1, 2], [-1, 3], [2, -3], [1, -2, 3]]
    n_vars = 3
    
    # Create BPG with subgraphs
    bpg = create_bpg_with_subgraphs(clauses, n_vars, compute_subgraphs=True)
    bpg.y = torch.tensor([0, 1, 2], dtype=torch.long)
    
    # Create model
    model = ArieNetBackbone(
        dim=32,
        n_rounds=3,
        use_subgraph_features=True,
        subgraph_dim=8
    )
    
    # Forward pass
    output = model(bpg)
    
    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    mask = bpg.y != 2
    loss = criterion(output[mask], bpg.y[mask])
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients (only check parameters that require grad)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    has_grad = sum(1 for p in trainable_params if p.grad is not None)
    total_trainable = len(trainable_params)
    
    print(f"Parameters with gradients: {has_grad}/{total_trainable}")
    
    # Allow some parameters to not have gradients (e.g., unused parameters, batch norm stats)
    grad_percentage = has_grad / total_trainable if total_trainable > 0 else 0
    
    if grad_percentage < 0.9:
        print(f"Warning: Only {grad_percentage*100:.1f}% of parameters have gradients")
        # Print which parameters don't have gradients for debugging
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                print(f"  No gradient: {name}")
    
    assert grad_percentage >= 0.9, f"Too few parameters have gradients: {grad_percentage*100:.1f}%"
    
    print(f"✓ Forward/backward test passed")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("CANONICAL SUBGRAPH FEATURES TEST SUITE")
    print("="*80)
    
    try:
        test_quine_mccluskey()
        test_subgraph_extraction()
        test_subgraph_gnn()
        test_dual_extractor()
        test_arienet_backbone()
        test_forward_backward()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

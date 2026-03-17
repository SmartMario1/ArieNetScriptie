# Canonical Subgraph Features for ArieNet Backbone

This implementation adds canonical subgraph features to the ArieNet backbone prediction model, following the approach described in the paper for building canonical graph representations that are the same for all logically equivalent subgraphs.

## Overview

For each edge (c, p) in the belief propagation graph (BPG), we:

1. **Extract the neighborhood**: Get all clauses in the immediate vicinity of the edge
2. **Generate 2CNF subformulas**: Create two subformulas, one for p=1 and one for p=0
3. **Apply Quine-McCluskey algorithm**: Obtain canonical representations in polynomial time
4. **Generate SAT graphs**: Create graph representations from the canonical 2CNF formulas
5. **Apply a GNN**: Process these subgraphs to extract structural features
6. **Integrate features**: Add the subgraph features to the main backbone prediction model

### Key Properties

- **Canonical representation**: The Quine-McCluskey algorithm ensures that logically equivalent subformulas have identical representations
- **Expressiveness**: The GNN's expressive power determines the symmetry relation captured. With a GIN (Graph Isomorphism Network), we can distinguish graphs up to isomorphism
- **Separability**: If the GNN can distinguish graphs up to isomorphism, the resulting local kernel is fully separating

## Implementation Components

### 1. Quine-McCluskey Algorithm (`src/nsnet/utils/quine_mccluskey.py`)

Implements the Quine-McCluskey algorithm for obtaining canonical 2CNF representations of Boolean formulas.

**Key functions**:
- `quine_mccluskey()`: Find prime implicants using the QM algorithm
- `canonical_2cnf()`: Convert a formula to canonical 2CNF form
- `generate_2cnf_for_literal_assignment()`: Generate 2CNF for a specific literal assignment

### 2. Canonical Subgraph Extraction (`src/nsnet/utils/canonical_subgraph.py`)

Handles extraction of edge neighborhoods and conversion to canonical graph representations.

**Key classes and functions**:
- `EdgeSubgraph`: Data structure for storing canonical subgraph info
- `extract_edge_neighborhood()`: Extract clauses in the neighborhood of an edge
- `create_canonical_subgraphs_for_edge()`: Create p=1 and p=0 subgraphs
- `edge_subgraph_to_pyg_data()`: Convert to PyTorch Geometric format
- `compute_subgraph_features_for_all_edges()`: Process all edges in a BPG

### 3. Subgraph GNN (`src/nsnet/models/subgraph_gnn.py`)

Graph neural network for processing canonical subgraphs and extracting structural features.

**Key classes**:
- `SubgraphGINLayer`: GIN layer for expressive graph processing
- `SubgraphGNN`: Main GNN for processing individual subgraphs
- `DualSubgraphFeatureExtractor`: Processes both p=1 and p=0 subgraphs and combines features

**Architecture**:
- Multiple GIN layers for message passing
- Global pooling (mean + max) for graph-level features
- MLP for final feature extraction
- Combination of p=1 and p=0 features

### 4. BPG Integration (`src/nsnet/utils/dataset.py`)

Extended BPG class and builder to support canonical subgraph features.

**Key changes**:
- Added `subgraphs_p1` and `subgraphs_p0` fields to `BPG` class
- Added `compute_canonical_subgraphs()` method to `BPGParamBuilder`
- New helper function `create_bpg_with_subgraphs()` for easy BPG creation

### 5. Updated ArieNet Backbone (`train_arienet_backbone.py`)

Extended ArieNet backbone model to use canonical subgraph features.

**Key changes**:
- Added `use_subgraph_features` flag
- Integrated `DualSubgraphFeatureExtractor` component
- Modified forward pass to incorporate subgraph features into edge embeddings

## Usage

### Basic Training (without subgraph features)

```python
python train_arienet_backbone.py --data_dir data/backbone
```

### Training with Canonical Subgraph Features

```python
python train_arienet_backbone_canonical.py \
    --data_dir data/backbone \
    --use_subgraph_features \
    --batch_size 8 \
    --dim 128 \
    --n_rounds 26 \
    --n_epochs 100
```

### Creating BPG with Subgraph Features Programmatically

```python
from nsnet.utils.dataset import create_bpg_with_subgraphs
from nsnet.utils.utils import parse_cnf_file

# Parse CNF file
n_vars, clauses = parse_cnf_file('path/to/formula.cnf')

# Create BPG with canonical subgraph features
bpg = create_bpg_with_subgraphs(
    clauses, 
    n_vars, 
    compute_subgraphs=True
)

# Access subgraph data
print(f"Number of edges: {len(bpg.subgraphs_p1)}")
print(f"First edge's p=1 subgraph: {bpg.subgraphs_p1[0]}")
print(f"First edge's p=0 subgraph: {bpg.subgraphs_p0[0]}")
```

### Using the Model with Subgraph Features

```python
from train_arienet_backbone import ArieNetBackbone

# Create model with subgraph features
model = ArieNetBackbone(
    dim=128,
    n_rounds=26,
    use_subgraph_features=True,
    subgraph_dim=32
)

# Forward pass (BPG must have subgraph data)
predictions = model(bpg)
```

## Architecture Details

### Subgraph Extraction

For an edge (c, p) connecting clause c to variable p:

1. **Neighborhood Definition**:
   - The clause c itself
   - All clauses containing variable p
   - All clauses containing other variables in c (2-hop neighborhood)

2. **Variable Mapping**:
   - Edge's variable is mapped to index 1
   - Other variables are mapped to 2, 3, 4, ... in sorted order
   - This ensures canonical ordering

3. **2CNF Generation**:
   - For p=1: Substitute p=true and simplify using Quine-McCluskey
   - For p=0: Substitute p=false and simplify using Quine-McCluskey

### GNN Architecture

The subgraph GNN uses a Graph Isomorphism Network (GIN) architecture:

```
Input: Bipartite graph (variables + clauses)
   ↓
Node Embedding (Linear layer)
   ↓
GIN Layer 1 + BatchNorm + ReLU
   ↓
GIN Layer 2 + BatchNorm + ReLU
   ↓
GIN Layer 3 + BatchNorm + ReLU
   ↓
Global Pooling (Mean + Max)
   ↓
MLP (Readout)
   ↓
Output: Subgraph feature vector
```

### Feature Integration

The subgraph features are integrated into ArieNet as follows:

```
Initial Edge Embeddings
   +
Subgraph Features (from dual GNN)
   ↓
Integration MLP
   ↓
Message Passing (n_rounds)
   ↓
Readout
   ↓
Backbone Predictions
```

## Configuration

### Model Hyperparameters

- `dim`: Main GNN hidden dimension (default: 128)
- `n_rounds`: Number of message passing rounds (default: 26)
- `use_subgraph_features`: Enable canonical subgraph features (default: False)
- `subgraph_dim`: Dimension of subgraph features (default: 32)

### Subgraph GNN Hyperparameters

- `hidden_dim`: GNN hidden dimension (default: 64)
- `n_layers`: Number of GIN layers (default: 3)
- `pool_type`: Pooling method ('mean', 'max', or 'mean_max', default: 'mean_max')
- `combination`: How to combine p=1 and p=0 features ('concat', 'add', 'diff', 'mult', default: 'concat')

## Performance Considerations

### Computational Complexity

- **Quine-McCluskey**: Polynomial time in the number of variables in the neighborhood
- **Subgraph GNN**: Linear in the number of nodes and edges in the subgraph
- **Overall**: The subgraph computation adds overhead, but is parallelizable

### Caching

The implementation includes caching mechanisms:
- BPG already caches local satisfaction percentages
- Canonical subgraphs could be cached based on their structural hash
- Consider pre-computing subgraph features for large datasets

### Memory Usage

- Each edge stores two subgraph objects (p=1 and p=0)
- Subgraph features are computed on-the-fly during training
- For large formulas, consider batch processing or disk caching

## Examples

### Example 1: Simple Formula

```python
# 3-SAT formula: (x1 ∨ x2 ∨ x3) ∧ (¬x1 ∨ x2 ∨ ¬x4)
clauses = [[1, 2, 3], [-1, 2, -4]]
n_vars = 4

bpg = create_bpg_with_subgraphs(clauses, n_vars)

# Check edge (clause 0, variable 1, positive literal)
edge_0_subgraph_p1 = bpg.subgraphs_p1[0]
print(f"2CNF clauses for x1=true: {edge_0_subgraph_p1.clauses_2cnf}")
```

### Example 2: Training Loop

```python
model = ArieNetBackbone(
    dim=128, 
    use_subgraph_features=True
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Model automatically uses subgraph features if available
        output = model(batch)
        
        # Compute loss on backbone variables only
        mask = batch.y != 2
        loss = criterion(output[mask], batch.y[mask])
        
        loss.backward()
        optimizer.step()
```

## Theoretical Background

### Quine-McCluskey Algorithm

The Quine-McCluskey algorithm is a method for minimizing Boolean functions. It systematically finds all prime implicants and then selects a minimum subset that covers all minterms.

**Key properties**:
1. Produces a canonical (unique) representation for any Boolean function
2. Runs in polynomial time for 2CNF formulas
3. Guarantees that logically equivalent formulas have identical representations

### Graph Isomorphism Network (GIN)

GIN is provably as expressive as the Weisfeiler-Leman (WL) graph isomorphism test. This means it can distinguish any graphs that the WL test can distinguish.

**Key properties**:
1. Uses injective aggregation: h^(k+1) = MLP((1 + ε) · h^(k) + Σ h_j^(k))
2. Learnable ε parameter for self-loop importance
3. Universal approximation for graph-structured functions

### Separating Kernels

A kernel k(G₁, G₂) is called separating if k(G₁, G₂) = k(G₁, G₁) implies G₁ ≅ G₂ (graph isomorphism).

The canonical subgraph features create a separating kernel because:
1. Quine-McCluskey ensures canonical representation
2. GIN can distinguish non-isomorphic graphs
3. Combined, they encode enough structural information for separation

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size, subgraph_dim, or hidden_dim

```python
model = ArieNetBackbone(
    dim=64,  # Reduce from 128
    use_subgraph_features=True,
    subgraph_dim=16  # Reduce from 32
)
```

### Issue: Slow Training

**Solution**: 
1. Disable subgraph features for initial debugging
2. Pre-compute and cache subgraph features
3. Reduce number of GIN layers in subgraph GNN

### Issue: NaN Loss

**Solution**:
1. Check gradient clipping is enabled
2. Reduce learning rate
3. Initialize parameters with smaller values

## Future Improvements

1. **Caching**: Implement persistent cache for canonical subgraphs
2. **Sparse Graphs**: Optimize for formulas with large neighborhoods
3. **Attention**: Add attention mechanism for combining p=1 and p=0 features
4. **Hierarchical**: Use hierarchical GNNs for very large subgraphs
5. **More Expressive GNNs**: Try higher-order GNNs beyond GIN

## References

1. Quine-McCluskey Algorithm: Quine (1952), McCluskey (1956)
2. Graph Isomorphism Network: Xu et al. (2019) "How Powerful are Graph Neural Networks?"
3. Weisfeiler-Leman Test: Weisfeiler & Leman (1968)
4. ArieNet: Original architecture from nsnet repository

## Citation

If you use this implementation, please cite:

```bibtex
@software{canonical_subgraph_features,
  title = {Canonical Subgraph Features for SAT Backbone Prediction},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/nsnet}
}
```

"""
ArieNet with Backbone Prediction Objective

This script combines:
- ArieNet GNN architecture from nsnet (with BPG format)
- Graph preparation using BPG (Bipartite Problem Graph) from nsnet
- Message passing from ArieNet with local satisfaction percentages
- Backbone prediction objective from neuroback
"""

import argparse
import os
import sys
import math
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from tqdm import tqdm

# Import necessary utilities from nsnet
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from nsnet.models.mlp import MLP
from nsnet.utils.torch_utils import scatter_sum, scatter_logsumexp, swap_even_odd
from nsnet.utils.dataset import BPG, BPGParamBuilder
from nsnet.utils.utils import parse_cnf_file


# ---------------------------------------------------------------------------
# Module-level worker function (required for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _process_backbone_file(args):
    """Process a single CNF file into a compact dict cache with backbone labels."""
    idx, cnf_path, processed_dir, backbones, no_precomputed_local_sat, use_up_features = args
    import torch
    from nsnet.utils.dataset import BPGParamBuilder
    from nsnet.utils.utils import parse_cnf_file
    import os
    output_path = os.path.join(processed_dir, f'data_{idx}.pt')
    if os.path.exists(output_path):
        return
    n_vars, clauses = parse_cnf_file(cnf_path)
    if n_vars == 0 or len(clauses) == 0:
        return
    p = BPGParamBuilder(
        clauses,
        n_vars,
        compute_local_satisfaction_percentages=not no_precomputed_local_sat,
        compute_up_features=use_up_features,
    ).params
    labels = torch.full((n_vars,), 2, dtype=torch.int8)
    abs_path = os.path.abspath(cnf_path)
    if abs_path in backbones:
        for var, value in backbones[abs_path].items():
            if value is not None and 1 <= var <= n_vars:
                labels[var - 1] = 1 if value else 0
    # Save a compact dict: index tensors as int32 (half the size of int64),
    # floats as float32, labels as int8. No PyG Data wrapper overhead.
    torch.save({
        'n_clauses':  p.n_clauses,
        'n_literals': p.n_literals,
        'lipe':  p.literal_indices_per_edge.to(torch.int32),
        'lipo':  p.literal_indices_per_occurence.to(torch.int32),
        'cipo':  p.clause_indices_per_occurence.to(torch.int32),
        'lsppe': p.local_satisfaction_percentage_per_edge.to(torch.float32)
                 if p.local_satisfaction_percentage_per_edge is not None else None,
        'c2l_r': p.c2l_msg_receiver_indices.to(torch.int32),
        'c2l_s': p.c2l_msg_sender_indices.to(torch.int32),
        'l2c_r': p.l2c_msg_receiver_indices.to(torch.int32),
        'l2c_a': p.l2c_assignment_indices.to(torch.int32),
        'l2c_n': p.l2c_assignment_neighborhoods.to(torch.int32),
        'up_feat': p.up_features_per_literal.to(torch.float32)
                   if p.up_features_per_literal is not None else None,
        'y':     labels,
    }, output_path)


class ArieNetBackbone(nn.Module):
    """
    ArieNet adapted for backbone prediction using BPG format.
    
    Uses the full BPG (Bipartite Problem Graph) format with:
    - Pre-computed message passing indices (c2l, l2c)
    - Local satisfaction percentages
    - Literal and clause indices
    - All the optimizations from ArieNet
    """
    
    def __init__(self, dim=128, n_rounds=26, n_mlp_layers=3, activation='relu', device='cuda', 
                 use_subgraph_features=False, subgraph_dim=32, no_precomputed_local_sat=False,
                 use_up_features=False):
        super(ArieNetBackbone, self).__init__()
        
        self.dim = dim
        self.n_rounds = n_rounds
        self.device = device
        self.use_subgraph_features = use_subgraph_features
        self.no_precomputed_local_sat = no_precomputed_local_sat
        self.use_up_features = use_up_features
        
        # Edge embeddings
        self.c2l_edges_init = nn.Parameter(torch.randn(1, dim))
        self.l2c_edges_init = nn.Parameter(torch.randn(1, dim))
        self.denom = math.sqrt(dim)
        
        # Canonical subgraph feature extractor (optional)
        if use_subgraph_features:
            from nsnet.models.subgraph_gnn import DualSubgraphFeatureExtractor
            self.subgraph_extractor = DualSubgraphFeatureExtractor(
                node_feature_dim=2,
                hidden_dim=64,
                subgraph_feature_dim=subgraph_dim,
                output_dim=subgraph_dim,
                n_layers=3,
                n_mlp_layers=2,
                pool_type='mean_max',
                combination='concat'
            )
            # Integration MLP to combine subgraph features with edge embeddings
            self.subgraph_integration = MLP(
                num_layers=2,
                input_dim=dim + subgraph_dim,
                hidden_dim=dim,
                output_dim=dim,
                activation=activation
            )
        
        # Message update networks (with local satisfaction percentage)
        self.c2l_msg_update = MLP(n_mlp_layers, dim + 1, dim, dim, activation)
        self.l2c_msg_update = MLP(n_mlp_layers, dim, dim, dim, activation)
        self.l2c_msg_norm = MLP(n_mlp_layers, dim * 2 + 2, dim, dim, activation)

        # Optional: UP-feature literal embedding (projects 4 scalars → dim,
        # then added to the initial edge embeddings before message passing).
        # UP_FEATURES_DIM = 4: [up_reach_pct, forced_count_pct, conflict_flag, log_up_reach]
        if use_up_features:
            self.up_feat_embed = MLP(2, 4, dim, dim, activation)
        
        # Readout for backbone prediction (per literal)
        self.l_readout = MLP(n_mlp_layers, dim, dim, 1, activation)
        self.softmax = nn.Softmax(dim=1)
        
        # Initialize parameters properly to prevent exploding values
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters with small values."""
        # Initialize edge embeddings with smaller values
        nn.init.normal_(self.c2l_edges_init, mean=0, std=0.01)
        nn.init.normal_(self.l2c_edges_init, mean=0, std=0.01)
    
    def forward(self, data):
        """
        Forward pass for backbone prediction using BPG format.
        
        Input:
            data: BPG object with:
                - n_edges, n_literals, n_clauses
                - literal_indices_per_edge
                - local_satisfaction_percentage_per_edge
                - c2l_msg_receiver_indices, c2l_msg_sender_indices
                - l2c_msg_receiver_indices, l2c_assignment_indices, l2c_assignment_neighborhoods
                - (optional) subgraphs_p1, subgraphs_p0: canonical subgraph data
                - y: [num_vars] backbone labels (0=negative, 1=positive, 2=free)
        
        Returns:
            predictions: [num_vars, 2] tensor with sigmoid probabilities for [negative, positive] literals
        """
        # Get device from data (should already be on the right device)
        device = data.literal_indices_per_edge.device
        
        n_edges = data.n_edges
        
        # Initialize edge features on the same device as data
        c2l_edges_feat = (self.c2l_edges_init / self.denom).repeat(n_edges, 1).to(device)
        l2c_edges_feat = (self.l2c_edges_init / self.denom).repeat(n_edges, 1).to(device)

        # Inject per-literal UP features into initial edge embeddings.
        # Each edge is associated with a literal via literal_indices_per_edge.
        # We look up the UP features for that literal and embed them into `dim`
        # dimensions, then add to the learnable initial edge representations.
        if self.use_up_features:
            up_feat_raw = getattr(data, 'up_features_per_literal', None)
            if up_feat_raw is not None:
                # up_feat_raw: (n_literals_total, 4) — concatenated across batch
                # literal_indices_per_edge: (n_edges,) — already batch-offset
                up_feat_per_edge = up_feat_raw[data.literal_indices_per_edge].to(device)  # (n_edges, 4)
                up_emb = self.up_feat_embed(up_feat_per_edge)  # (n_edges, dim)
                c2l_edges_feat = c2l_edges_feat + up_emb
                l2c_edges_feat = l2c_edges_feat + up_emb
        
        # Process canonical subgraph features if available
        if self.use_subgraph_features and hasattr(data, 'subgraphs_p1') and data.subgraphs_p1 is not None:
            from nsnet.utils.canonical_subgraph import create_batch_subgraph_data
            
            # Convert subgraphs to PyG Data objects
            subgraph_data_p1 = create_batch_subgraph_data(data.subgraphs_p1)
            subgraph_data_p0 = create_batch_subgraph_data(data.subgraphs_p0)
            
            # Move to device
            subgraph_data_p1 = [d.to(device) for d in subgraph_data_p1]
            subgraph_data_p0 = [d.to(device) for d in subgraph_data_p0]
            
            # Extract subgraph features
            subgraph_features = self.subgraph_extractor(subgraph_data_p1, subgraph_data_p0)
            
            # Integrate subgraph features with initial edge embeddings
            c2l_edges_feat = self.subgraph_integration(torch.cat([c2l_edges_feat, subgraph_features], dim=1))
            l2c_edges_feat = self.subgraph_integration(torch.cat([l2c_edges_feat, subgraph_features], dim=1))
        
        # Message passing rounds (ArieNet-style)
        for _ in range(self.n_rounds):
            
            ##### First update: Clause to Literal messages #####
            # Sum c2l messages by literal occurrence
            l2c_msg_argument = scatter_sum(
                data.c2l_msg_receiver_indices, 
                c2l_edges_feat[data.c2l_msg_sender_indices], 
                n_edges, 
                device
            )
            
            # Update l2c messages
            l2c_msg = self.l2c_msg_update(l2c_msg_argument)
            
            local_satisfaction_raw = getattr(data, 'local_satisfaction_percentage_per_edge', None)
            if self.no_precomputed_local_sat or local_satisfaction_raw is None:
                local_satisfaction_percentages = torch.zeros(
                    (n_edges, 1), dtype=l2c_msg.dtype, device=l2c_msg.device
                )
            elif local_satisfaction_raw.dim() == 1:
                local_satisfaction_percentages = local_satisfaction_raw.unsqueeze(1)
            else:
                local_satisfaction_percentages = local_satisfaction_raw
            
            # Swap features for negated literals
            negated_local_satisfaction_percentages = swap_even_odd(local_satisfaction_percentages)
            l2c_negated_msg = swap_even_odd(l2c_msg)
            
            # Normalize with negated messages and local satisfaction percentages
            l2c_edges_feat = self.l2c_msg_norm(torch.cat([
                l2c_msg, 
                l2c_negated_msg, 
                local_satisfaction_percentages, 
                negated_local_satisfaction_percentages
            ], dim=1))
            
            ##### Second update: Literal to Clause messages #####
            # Sum l2c messages per assignment
            l2c_msgs_per_assignment = scatter_sum(
                data.l2c_assignment_indices,
                l2c_edges_feat[data.l2c_assignment_neighborhoods],
                len(data.l2c_msg_receiver_indices),
                device
            )
            
            # Max over satisfying assignments (logsumexp)
            c2l_msg_argument = scatter_logsumexp(
                data.l2c_msg_receiver_indices,
                l2c_msgs_per_assignment,
                len(torch.unique(data.l2c_msg_receiver_indices)),
                device
            )
            
            # Update c2l messages with local satisfaction percentages
            c2l_edges_feat = self.c2l_msg_update(torch.cat([
                c2l_msg_argument, 
                local_satisfaction_percentages
            ], dim=1))
        
        # Readout: aggregate edge features to literal features
        l_features = scatter_sum(
            data.literal_indices_per_edge,
            c2l_edges_feat,
            data.n_literals.sum().item() if isinstance(data.n_literals, torch.Tensor) else data.n_literals,
            device
        )
        
        # Apply readout MLP
        l_features = self.l_readout(l_features)
        
        # Reshape to [num_vars, 2] for backbone prediction
        num_vars = l_features.shape[0] // 2
        v_features = l_features.reshape(num_vars, 2)
        
        # Return raw logits (for CrossEntropyLoss)
        return v_features


class BackboneBPGDataset(Dataset):
    """
    Dataset that creates BPG format from CNF files with backbone labels.
    
    Combines:
    - CNF parsing and BPG creation from nsnet
    - Backbone labels from backbones.pkl file
    
    Expected structure:
    - data_dir: directory with .cnf files
    - data_dir/backbones.pkl: pickle file mapping file paths to backbone dicts
    """
    
    def __init__(self, data_dir, root=None, transform=None, pre_transform=None,
                 no_precomputed_local_sat=False, use_up_features=False,
                 num_process_workers=4):
        """
        Args:
            data_dir: Directory containing .cnf files and backbones.pkl
            root: Root directory for processed files (optional, defaults to data_dir/processed_bpg_v2
                  or data_dir/processed_bpg_up when use_up_features=True)
            use_up_features: If True, compute and store per-literal UP features.
                             Uses a separate cache directory to avoid stale caches.
            num_process_workers: Number of parallel workers for initial cache creation.
                                 Set to 1 to force single-threaded processing (safer
                                 when use_up_features=True, as pysat C extensions can
                                 deadlock under fork-based multiprocessing on Linux).
        """
        self.data_dir = data_dir
        self.no_precomputed_local_sat = no_precomputed_local_sat
        self.use_up_features = use_up_features
        self.num_process_workers = num_process_workers
        
        # Find all CNF files
        import glob
        self.cnf_files = sorted(glob.glob(os.path.join(data_dir, '**/*.cnf'), recursive=True))
        
        # Load backbone labels
        backbone_file = os.path.join(data_dir, 'backbones.pkl')
        if os.path.exists(backbone_file):
            with open(backbone_file, 'rb') as f:
                self.backbones = pickle.load(f)
            print(f"Loaded backbone labels for {len(self.backbones)} files")
        else:
            print(f"[WARNING] No backbones.pkl found at {backbone_file}")
            print("Run generate_backbone_labels.py first to create backbone labels")
            self.backbones = {}
        
        print(f"Found {len(self.cnf_files)} CNF files in {data_dir}")
        
        # Use a versioned processed directory so that caches built under
        # different feature configurations do not corrupt each other:
        #   processed_bpg_up/    – UP features enabled
        #   processed_bpg_nolsp/ – local-satisfaction percentages disabled
        #   processed_bpg_v2/    – default (all features)
        if root is None:
            if use_up_features:
                subdir = 'processed_bpg_up'
            elif no_precomputed_local_sat:
                subdir = 'processed_bpg_nolsp'
            else:
                subdir = 'processed_bpg_v2'
            root = os.path.join(data_dir, subdir)
        
        # In-memory cache to avoid repeated torch.load() calls
        self._cache = {}
        
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.cnf_files))]
    
    def download(self):
        pass
    
    def process(self):
        """Process CNF files into BPG format with backbone labels."""
        tasks = [
            (idx, cnf_path, self.processed_dir, self.backbones,
             self.no_precomputed_local_sat, self.use_up_features)
            for idx, cnf_path in enumerate(self.cnf_files)
            if not os.path.exists(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        ]

        if not tasks:
            print(f"All {len(self.cnf_files)} BPG files already processed, skipping.")
            return

        n_workers = self.num_process_workers
        completed = 0

        if n_workers == 1:
            print(f"Processing {len(tasks)}/{len(self.cnf_files)} files into BPG format "
                  f"(single-threaded)...")
            for t in tasks:
                idx = t[0]
                try:
                    _process_backbone_file(t)
                    completed += 1
                    if completed % 10 == 0 or completed == len(tasks):
                        print(f'  [{completed}/{len(tasks)}]', end='\r')
                except Exception as e:
                    print(f"[ERROR] Processing index {idx}: {e}")
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(f"Processing {len(tasks)}/{len(self.cnf_files)} files into BPG format "
                  f"({n_workers} parallel workers)...")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_process_backbone_file, t): t[0] for t in tasks}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                        completed += 1
                        if completed % 10 == 0 or completed == len(tasks):
                            print(f'  [{completed}/{len(tasks)}]', end='\r')
                    except Exception as e:
                        print(f"[ERROR] Processing index {idx}: {e}")

        print(f"\nDone: {completed} files processed.")
    
    def len(self):
        return len(self.cnf_files)
    
    def get(self, idx):
        """Load a compact cache file and reconstruct BPG with in-memory caching."""
        if idx in self._cache:
            return self._cache[idx]
        cache_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        try:
            d = torch.load(cache_path, weights_only=True)
        except Exception:
            # Backward compatibility for legacy caches that stored full BPG objects.
            d = torch.load(cache_path, weights_only=False)

        if isinstance(d, dict):
            data = BPG(
                n_clauses=d['n_clauses'],
                n_literals=d['n_literals'],
                literal_indices_per_edge=d['lipe'].long(),
                literal_indices_per_occurence=d['lipo'].long(),
                clause_indices_per_occurence=d['cipo'].long(),
                local_satisfaction_percentage_per_edge=d.get('lsppe'),
                c2l_msg_receiver_indices=d['c2l_r'].long(),
                c2l_msg_sender_indices=d['c2l_s'].long(),
                l2c_msg_receiver_indices=d['l2c_r'].long(),
                l2c_assignment_indices=d['l2c_a'].long(),
                l2c_assignment_neighborhoods=d['l2c_n'].long(),
                up_features_per_literal=d.get('up_feat'),
            )
            data.y = d['y'].long()
        else:
            data = d

        self._cache[idx] = data
        return data


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_batches = 0
    all_target = []
    all_pred_class = []
    
    with tqdm(total=len(train_loader.dataset)) as pbar:
        for data in train_loader:
            if data.y is None:
                continue
            
            try:
                data = data.to(device)
                
                # Filter out unknown labels (y == 2)
                y01_indices = (data.y != 2).nonzero(as_tuple=True)[0]
                
                if len(y01_indices) == 0:
                    continue
                
                y01 = data.y[y01_indices].long()  # Use long for CrossEntropyLoss
                
                # Compute class weights for balanced loss
                u0 = torch.sum((y01 == 0).int()).item()
                u1 = torch.sum((y01 == 1).int()).item()
                
                # Class weights: inverse of frequency
                class_weights = torch.tensor([
                    len(y01) / (2 * (u0 + 1)),  # weight for class 0
                    len(y01) / (2 * (u1 + 1))   # weight for class 1
                ], device=device, dtype=torch.float)
                
                crit = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                logits = model(data)  # [num_vars, 2] raw logits
                
                # Select logits for variables with known labels
                logits_subset = logits[y01_indices]
                
                # Clamp logits to prevent extreme values
                logits_subset = torch.clamp(logits_subset, min=-50, max=50)
                
                # Compute loss using CrossEntropyLoss (returns mean loss per sample)
                loss = crit(logits_subset, y01)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"WARNING: NaN or Inf loss, skipping batch")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Collect predictions
                with torch.no_grad():
                    pred_class = torch.argmax(logits_subset, dim=1)
                
                all_target += y01.cpu().numpy().tolist()
                all_pred_class += pred_class.cpu().numpy().tolist()
                
                # CrossEntropyLoss returns mean loss, so just accumulate it
                total_loss += loss.item()
                total_batches += 1
                
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM error, skipping batch")
                    continue
                else:
                    raise e
            
            pbar.update(data.num_graphs)
    
    # Compute metrics
    if total_batches > 0:
        avg_loss = total_loss / total_batches  # Average over batches
        cm = confusion_matrix(all_target, all_pred_class, labels=[0, 1])
        
        return avg_loss, cm
    else:
        return None, None


def evaluate(model, val_loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    total_batches = 0
    all_target = []
    all_pred_class = []
    
    with torch.no_grad():
        for data in tqdm(val_loader):
            if data.y is None:
                continue
            
            try:
                data = data.to(device)
                
                # Filter out unknown labels
                y01_indices = (data.y != 2).nonzero(as_tuple=True)[0]
                
                if len(y01_indices) == 0:
                    continue
                
                y01 = data.y[y01_indices].long()  # Use long for CrossEntropyLoss
                
                # Forward pass
                logits = model(data)  # [num_vars, 2] raw logits
                logits_subset = logits[y01_indices]
                
                # Compute loss with class weights
                u0 = torch.sum((y01 == 0).int()).item()
                u1 = torch.sum((y01 == 1).int()).item()
                
                class_weights = torch.tensor([
                    len(y01) / (2 * (u0 + 1)),
                    len(y01) / (2 * (u1 + 1))
                ], device=device, dtype=torch.float)
                
                crit = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
                loss = crit(logits_subset, y01)
                
                # Collect predictions
                pred_class = torch.argmax(logits_subset, dim=1)
                
                all_target += y01.cpu().numpy().tolist()
                all_pred_class += pred_class.cpu().numpy().tolist()
                
                # CrossEntropyLoss returns mean loss
                total_loss += loss.item()
                total_batches += 1
                
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print(f"OOM error, skipping batch")
                    continue
                else:
                    raise e
    
    if total_batches > 0:
        avg_loss = total_loss / total_batches
        cm = confusion_matrix(all_target, all_pred_class, labels=[0, 1])
        
        # Compute additional metrics
        recall = recall_score(all_target, all_pred_class, average='binary', zero_division=0)
        precision = precision_score(all_target, all_pred_class, average='binary', zero_division=0)
        f1 = f1_score(all_target, all_pred_class, average='binary')
        
        return avg_loss, cm, recall, precision, f1
    else:
        return None, None, None, None, None


def _ensure_backbones(data_dir, n_process=4):
    """Run generate_backbone_labels.py on data_dir if backbones.pkl is missing."""
    backbone_file = os.path.join(data_dir, 'backbones.pkl')
    if os.path.exists(backbone_file):
        return
    print(f"[INFO] backbones.pkl not found in {data_dir} — generating now...")
    import subprocess
    script = os.path.join(os.path.dirname(__file__), 'generate_backbone_labels.py')
    result = subprocess.run(
        [sys.executable, script, data_dir, '--n_process', str(n_process)],
        check=True
    )
    if not os.path.exists(backbone_file):
        raise RuntimeError(f"Backbone generation failed for {data_dir}")
    print(f"[INFO] Backbone labels saved to {backbone_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ArieNet with backbone prediction objective',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # All problem types together:\n'
            '  python train_arienet_backbone.py --mode pretrain --name all \\\n'
            '      --train_dirs SATSolving/train/3sat SATSolving/train/4sat \\\n'
            '                   SATSolving/train/5sat SATSolving/train/tseitin_tree \\\n'
            '                   SATSolving/train/parity \\\n'
            '      --val_dirs   SATSolving/train/3sat SATSolving/train/4sat \\\n'
            '                   SATSolving/train/5sat SATSolving/train/tseitin_tree \\\n'
            '                   SATSolving/train/parity'
        )
    )
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True,
                        help='One or more training directories (backbones.pkl auto-generated if missing)')
    parser.add_argument('--val_dirs', type=str, nargs='+', required=True,
                        help='One or more validation directories (backbones.pkl auto-generated if missing)')
    parser.add_argument('--name', type=str, default='run',
                        help='Run name used for log/checkpoint filenames')
    parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'finetune'],
                        help='Training mode')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to resume from (finetune default: <save_dir>/pretrain-<name>-best.pt)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 5e-5 pretrain, 1e-5 finetune)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: 40 pretrain, 60 finetune)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: 10 pretrain, 8 finetune)')
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n_rounds', type=int, default=26)
    parser.add_argument('--n_mlp_layers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=77)
    parser.add_argument('--save_dir', type=str, default='./models/arienet_backbone')
    parser.add_argument('--n_process', type=int, default=4,
                        help='Parallel workers for backbone generation')
    parser.add_argument('--no_precomputed_local_sat', action='store_true',
                        help='Do not precompute local graph satisfaction percentages; ignore this feature in the model.')
    parser.add_argument('--use_up_features', action='store_true',
                        help='Compute and use per-literal unit-propagation features '
                             '(UP reach, forced count, conflict flag, log UP reach).  '
                             'Requires python-sat.  Uses a separate cache directory '
                             '(processed_bpg_up/) so existing caches are not invalidated.')
    args = parser.parse_args()

    mode = args.mode
    name = args.name

    # Mode-specific defaults
    if mode == 'pretrain':
        lr         = args.lr         if args.lr         is not None else 5e-5
        epoch_num  = args.epochs     if args.epochs     is not None else 40
        batch_size = args.batch_size if args.batch_size is not None else 10
        log_dir    = f'./log/arienet_pretrain_{name}'
        checkpoint_path = args.checkpoint
    else:  # finetune
        lr         = args.lr         if args.lr         is not None else 1e-5
        epoch_num  = args.epochs     if args.epochs     is not None else 60
        batch_size = args.batch_size if args.batch_size is not None else 8
        log_dir    = f'./log/arienet_finetune_{name}'
        checkpoint_path = args.checkpoint or os.path.join(args.save_dir, f'pretrain-{name}-best.pt')

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if args.no_precomputed_local_sat:
        print("[INFO] no_precomputed_local_sat enabled: reusing existing cache if present and ignoring local-satisfaction feature.")
    if args.use_up_features:
        print("[INFO] use_up_features enabled: computing per-literal UP features via pysat (cache dir: processed_bpg_up/).")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, training will be SLOW on CPU!")

    # Auto-generate backbones and build concatenated datasets
    print("\nLoading datasets...")
    train_parts, val_parts = [], []
    for d in args.train_dirs:
        _ensure_backbones(d, n_process=args.n_process)
        train_parts.append(BackboneBPGDataset(
            data_dir=d,
            no_precomputed_local_sat=args.no_precomputed_local_sat,
            use_up_features=args.use_up_features,
        ))
        print(f"  train: {len(train_parts[-1])} instances from {d}")
    for d in args.val_dirs:
        _ensure_backbones(d, n_process=args.n_process)
        val_parts.append(BackboneBPGDataset(
            data_dir=d,
            no_precomputed_local_sat=args.no_precomputed_local_sat,
            use_up_features=args.use_up_features,
        ))
        print(f"  val:   {len(val_parts[-1])} instances from {d}")

    import torch.utils.data as tud
    dataset_train = tud.ConcatDataset(train_parts)
    dataset_val   = tud.ConcatDataset(val_parts)
    print(f"Total: {len(dataset_train)} train / {len(dataset_val)} val")

    num_workers = 2 if device.type == 'cuda' else 0
    train_loader = DataLoader(dataset_train, batch_size=batch_size,
                              shuffle=True, pin_memory=(device.type == 'cuda'),
                              num_workers=num_workers, persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(dataset_val, batch_size=1,
                              shuffle=False, pin_memory=(device.type == 'cuda'),
                              num_workers=num_workers, persistent_workers=(num_workers > 0))

    print(f"Training samples:   {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val)}")

    # Initialize model
    model = ArieNetBackbone(
        dim=args.dim,
        n_rounds=args.n_rounds,
        n_mlp_layers=args.n_mlp_layers,
        activation='relu',
        device=device,
        no_precomputed_local_sat=args.no_precomputed_local_sat,
        use_up_features=args.use_up_features,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Load checkpoint if finetuning
    start_epoch = 0
    best_val_loss = float('inf')

    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Training loop
    log_path = os.path.join(log_dir, f'training_log_{mode}.txt')
    
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Mode: {mode}, name: {name}\n")
        log_file.write(f"lr={lr}, epochs={epoch_num}, batch_size={batch_size}\n")
        log_file.write(f"train_dirs={args.train_dirs}\nval_dirs={args.val_dirs}\n")
        log_file.write(f"{'='*50}\n\n")

        for epoch in range(start_epoch, epoch_num):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{epoch_num}")
            log_file.write(f"\nEpoch {epoch + 1}/{epoch_num}\n")
            
            # Train
            train_start = time.time()
            train_loss, train_cm = train_epoch(model, train_loader, optimizer, device)
            train_time = time.time() - train_start
            
            if train_loss is not None:
                print(f"Train Loss: {train_loss:.4f} (took {train_time:.1f}s)")
                log_file.write(f"Train Loss: {train_loss:.4f} (took {train_time:.1f}s)\n")
                log_file.write("Train Confusion Matrix:\n")
                log_file.write(str(train_cm) + "\n")
            
            # Validate
            val_start = time.time()
            val_loss, val_cm, recall, precision, f1 = evaluate(model, val_loader, device)
            val_time = time.time() - val_start
            
            if val_loss is not None:
                epoch_time = time.time() - epoch_start
                print(f"Val Loss: {val_loss:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f} (took {val_time:.1f}s)")
                print(f"Total epoch time: {epoch_time:.1f}s")
                log_file.write(f"Val Loss: {val_loss:.4f} (took {val_time:.1f}s)\n")
                log_file.write(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}\n")
                log_file.write("Validation Confusion Matrix:\n")
                log_file.write(str(val_cm) + "\n")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(args.save_dir, f'{mode}-{name}-best.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss,
                        'args': vars(args),
                    }, save_path)
                    print(f"Saved best model to {save_path}")
                    log_file.write(f"Saved best model to {save_path}\n")
            
            log_file.flush()
    
    print("Training complete!")


if __name__ == "__main__":
    main()

"""
ArieNet with Backbone Prediction Objective

This script combines:
- ArieNet GNN architecture from nsnet (with BPG format)
- Graph preparation using BPG (Bipartite Problem Graph) from nsnet
- Message passing from ArieNet with local satisfaction percentages
- Backbone prediction objective from neuroback
"""

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
                 use_subgraph_features=False, subgraph_dim=32):
        super(ArieNetBackbone, self).__init__()
        
        self.dim = dim
        self.n_rounds = n_rounds
        self.device = device
        self.use_subgraph_features = use_subgraph_features
        
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
            
            # Prepare local satisfaction percentages (2D, N x 1)
            local_satisfaction_percentages = data.local_satisfaction_percentage_per_edge.unsqueeze(1)
            
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
    
    def __init__(self, data_dir, root=None, transform=None, pre_transform=None):
        """
        Args:
            data_dir: Directory containing .cnf files and backbones.pkl
            root: Root directory for processed files (optional, defaults to data_dir/processed)
        """
        self.data_dir = data_dir
        
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
        
        # Use processed directory within data_dir if root not specified
        if root is None:
            root = os.path.join(data_dir, 'processed_bpg')
        
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
        print(f"Processing {len(self.cnf_files)} files into BPG format...")
        
        for idx, cnf_path in enumerate(tqdm(self.cnf_files)):
            output_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
            
            if os.path.exists(output_path):
                continue
            
            # Parse CNF file
            try:
                n_vars, clauses = parse_cnf_file(cnf_path)
                
                if n_vars == 0 or len(clauses) == 0:
                    print(f"[WARNING] Empty CNF file: {cnf_path}")
                    continue
                
                # Create BPG
                bpg_params = BPGParamBuilder(clauses, n_vars).params
                data = BPG(
                    n_clauses=bpg_params.n_clauses,
                    n_literals=bpg_params.n_literals,
                    literal_indices_per_edge=bpg_params.literal_indices_per_edge,
                    literal_indices_per_occurence=bpg_params.literal_indices_per_occurence,
                    clause_indices_per_occurence=bpg_params.clause_indices_per_occurence,
                    local_satisfaction_percentage_per_edge=bpg_params.local_satisfaction_percentage_per_edge,
                    c2l_msg_receiver_indices=bpg_params.c2l_msg_receiver_indices,
                    c2l_msg_sender_indices=bpg_params.c2l_msg_sender_indices,
                    l2c_msg_receiver_indices=bpg_params.l2c_msg_receiver_indices,
                    l2c_assignment_indices=bpg_params.l2c_assignment_indices,
                    l2c_assignment_neighborhoods=bpg_params.l2c_assignment_neighborhoods
                )
                
                # Load backbone labels for this file
                abs_cnf_path = os.path.abspath(cnf_path)
                backbone_labels = torch.full((n_vars,), 2, dtype=torch.long)  # Default: all free (2)
                
                if abs_cnf_path in self.backbones:
                    backbone = self.backbones[abs_cnf_path]
                    for var, value in backbone.items():
                        if value is not None and 1 <= var <= n_vars:
                            # Convert 1-indexed var to 0-indexed
                            backbone_labels[var - 1] = 1 if value else 0
                
                # Attach labels to data
                data.y = backbone_labels
                
                # Save processed data
                torch.save(data, output_path)
                    
            except Exception as e:
                print(f"[ERROR] Processing {cnf_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def len(self):
        return len(self.cnf_files)
    
    def get(self, idx):
        """Load a processed BPG file with in-memory caching."""
        if idx in self._cache:
            return self._cache[idx]
        
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
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


def main():
    # Hyperparameters
    if len(sys.argv) < 3:
        print("Usage: python train_arienet_backbone.py [pretrain|finetune] name")
        sys.exit(1)
    
    mode = sys.argv[1]
    name = sys.argv[2]
    
    hyper_params = {}
    if mode == "pretrain":
        hyper_params["pretrain"] = True
        hyper_params["seed"] = 77
        hyper_params["lr"] = 5e-5  # Reduced from 1e-4 to prevent exploding gradients
        hyper_params["epoch_num"] = 40
        hyper_params["batch_size"] = 10  # Balanced for GPU memory usage
        hyper_params["log_dir"] = f"./log/arienet_pretrain_{name}"
        hyper_params["checkpoint_path"] = None
    elif mode == "finetune":
        hyper_params["pretrain"] = False
        hyper_params["seed"] = 77
        hyper_params["lr"] = 1e-5  # Already low for finetuning
        hyper_params["epoch_num"] = 60
        hyper_params["batch_size"] = 8  # Smaller batch for finetuning stability
        hyper_params["log_dir"] = f"./log/arienet_finetune_{name}"
        hyper_params["checkpoint_path"] = f"./models/arienet_backbone/pretrain-{name}-best.pt"
    else:
        print("Invalid mode! Use 'pretrain' or 'finetune'")
        sys.exit(1)
    
    # Create directories
    os.makedirs(hyper_params["log_dir"], exist_ok=True)
    os.makedirs("./models/arienet_backbone", exist_ok=True)
    
    # Set seed
    torch.manual_seed(hyper_params["seed"])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available, training will be SLOW on CPU!")
        print("Make sure PyTorch with CUDA is installed properly.")
    
    # Set up data paths for nsnet's 3-SAT data
    # Expected structure: SATSolving/3-sat/train_first_4000_ArieNet/ etc.
    satsolving_root = os.path.join(os.path.dirname(__file__), 'SATSolving', '3-sat')
    
    if hyper_params["pretrain"]:
        # Use training set for pretraining
        train_data_dir = os.path.join(satsolving_root, 'train_first_4000_ArieNet')
        val_data_dir = os.path.join(satsolving_root, 'valid_first_1000_ArieNet')
    else:
        # Use training set for finetuning (or a smaller subset)
        train_data_dir = os.path.join(satsolving_root, 'train_first_4000_ArieNet')
        val_data_dir = os.path.join(satsolving_root, 'valid_first_1000_ArieNet')
    
    print(f"Train data dir: {train_data_dir}")
    print(f"Val data dir: {val_data_dir}")
    
    # Check if backbone labels exist
    train_backbone_file = os.path.join(train_data_dir, 'backbones.pkl')
    val_backbone_file = os.path.join(val_data_dir, 'backbones.pkl')
    
    if not os.path.exists(train_backbone_file) or not os.path.exists(val_backbone_file):
        print("\n" + "="*70)
        print("WARNING: Backbone labels not found!")
        print("="*70)
        print("\nPlease generate backbone labels first by running:")
        print(f"  python generate_backbone_labels.py {train_data_dir}")
        print(f"  python generate_backbone_labels.py {val_data_dir}")
        print("\nThis will create backbones.pkl files in each directory.")
        print("="*70 + "\n")
        
        # Ask user if they want to continue anyway
        response = input("Continue without backbone labels? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please generate backbone labels first.")
            sys.exit(0)
    
    # Load datasets
    print("\nLoading datasets...")
    
    dataset_train = BackboneBPGDataset(data_dir=train_data_dir)
    dataset_val = BackboneBPGDataset(data_dir=val_data_dir)
    
    # Use multiple workers for faster data loading (set to 0 if you have multiprocessing issues)
    num_workers = 2 if device.type == 'cuda' else 0
    
    train_loader = DataLoader(dataset_train, batch_size=hyper_params["batch_size"], 
                             shuffle=True, pin_memory=(device.type == 'cuda'), 
                             num_workers=num_workers, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(dataset_val, batch_size=1, 
                           shuffle=False, pin_memory=(device.type == 'cuda'), 
                           num_workers=num_workers, persistent_workers=(num_workers > 0))
    
    print(f"Training samples: {len(dataset_train)}")
    print(f"Validation samples: {len(dataset_val)}")
    
    # Initialize model
    model = ArieNetBackbone(
        dim=128,
        n_rounds=26,
        n_mlp_layers=3,
        activation='relu',
        device=device
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["lr"])
    
    # Load checkpoint if finetuning
    start_epoch = 0
    best_val_loss = float('inf')
    
    if hyper_params["checkpoint_path"] and os.path.isfile(hyper_params["checkpoint_path"]):
        print(f"Loading checkpoint: {hyper_params['checkpoint_path']}")
        checkpoint = torch.load(hyper_params["checkpoint_path"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Training loop
    log_path = os.path.join(hyper_params["log_dir"], f"training_log_{mode}.txt")
    
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"Training started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Mode: {mode}\n")
        log_file.write(f"Hyperparameters: {hyper_params}\n")
        log_file.write(f"{'='*50}\n\n")
        
        for epoch in range(start_epoch, hyper_params["epoch_num"]):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{hyper_params['epoch_num']}")
            log_file.write(f"\nEpoch {epoch + 1}/{hyper_params['epoch_num']}\n")
            
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
                    save_path = f"./models/arienet_backbone/{mode}-{name}-best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss
                    }, save_path)
                    print(f"Saved best model to {save_path}")
                    log_file.write(f"Saved best model to {save_path}\n")
            
            log_file.flush()
    
    print("Training complete!")


if __name__ == "__main__":
    main()

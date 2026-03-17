"""
Training script for ArieNet Backbone with Canonical Subgraph Features

This script demonstrates how to train the ArieNet backbone prediction model
with the new canonical subgraph features that capture local graph structure.

The canonical subgraph features:
1. Extract the neighborhood around each edge (c, p)
2. Generate two 2CNF subformulas: one for p=1 and one for p=0
3. Apply Quine-McCluskey algorithm for canonical representation
4. Use a GNN to extract features from these canonical subgraphs
5. Integrate these features into the backbone prediction model

Usage:
    # Single directory (will split 80/20 for train/val)
    python train_arienet_backbone_canonical.py --data_dir data/backbone --use_subgraph_features
    
    # Separate train and validation directories
    python train_arienet_backbone_canonical.py --train_dir data/train --val_dir data/val --use_subgraph_features
"""

import os
import sys
import argparse
import time
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from tqdm import tqdm

# Import from nsnet
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from nsnet.utils.dataset import create_bpg_with_subgraphs, BPGParamBuilder, BPG
from nsnet.utils.utils import parse_cnf_file
from train_arienet_backbone import ArieNetBackbone, BackboneBPGDataset


class CanonicalBackboneBPGDataset(BackboneBPGDataset):
    """
    Extended dataset that includes canonical subgraph features.
    """
    
    def __init__(self, data_dir, root=None, transform=None, pre_transform=None,
                 compute_subgraphs=True):
        self.compute_subgraphs = compute_subgraphs
        super().__init__(data_dir, root, transform, pre_transform)
    
    def process(self):
        """Process CNF files into BPG format with canonical subgraph features."""
        print(f"Processing {len(self.cnf_files)} files into BPG format with subgraph features...")
        
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
                
                # Create BPG with canonical subgraph features
                data = create_bpg_with_subgraphs(
                    clauses, 
                    n_vars, 
                    compute_subgraphs=self.compute_subgraphs
                )
                
                # Add backbone labels
                backbone_key = os.path.relpath(cnf_path, self.data_dir)
                if backbone_key in self.backbones:
                    backbone = self.backbones[backbone_key]
                    
                    # Create label tensor: 0=negative backbone, 1=positive backbone, 2=free variable
                    labels = torch.full((n_vars,), 2, dtype=torch.long)
                    
                    for var in range(1, n_vars + 1):
                        if var in backbone.get('positive', []):
                            labels[var - 1] = 1
                        elif var in backbone.get('negative', []):
                            labels[var - 1] = 0
                    
                    data.y = labels
                else:
                    print(f"[WARNING] No backbone labels for {cnf_path}")
                    data.y = torch.full((n_vars,), 2, dtype=torch.long)
                
                # Save
                torch.save(data, output_path)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {cnf_path}: {e}")
                import traceback
                traceback.print_exc()


def train_epoch(model, loader, optimizer, criterion, device, use_subgraphs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch)
        
        # Compute loss (only on backbone literals, not free variables)
        mask = batch.y != 2  # Mask out free variables
        if mask.sum() == 0:
            continue
        
        loss = criterion(output[mask], batch.y[mask])
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * mask.sum().item()
        total_samples += mask.sum().item()
    
    return total_loss / total_samples if total_samples > 0 else 0


def evaluate(model, loader, device):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            
            # Forward pass
            output = model(batch)
            
            # Get predictions
            preds = torch.argmax(output, dim=1)
            
            # Filter out free variables
            mask = batch.y != 2
            
            if mask.sum() > 0:
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(batch.y[mask].cpu().numpy())
    
    if len(all_preds) == 0:
        return 0, 0, 0, 0
    
    # Compute metrics
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    return accuracy, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description='Train ArieNet Backbone with Canonical Subgraph Features')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing CNF files and backbones.pkl (will be split 80/20 for train/val)')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Directory containing training CNF files and backbones.pkl')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Directory containing validation CNF files and backbones.pkl')
    parser.add_argument('--use_subgraph_features', action='store_true',
                        help='Use canonical subgraph features')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--n_rounds', type=int, default=26,
                        help='Number of message passing rounds')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='models/arienet_backbone_canonical',
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.data_dir is None and (args.train_dir is None or args.val_dir is None):
        parser.error("Either --data_dir OR both --train_dir and --val_dir must be provided")
    if args.data_dir is not None and (args.train_dir is not None or args.val_dir is not None):
        parser.error("Cannot specify both --data_dir and --train_dir/--val_dir")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 80)
    print("ArieNet Backbone Training with Canonical Subgraph Features")
    print("=" * 80)
    if args.data_dir:
        print(f"Data directory: {args.data_dir} (will split 80/20)")
    else:
        print(f"Train directory: {args.train_dir}")
        print(f"Val directory: {args.val_dir}")
    print(f"Use subgraph features: {args.use_subgraph_features}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hidden dimension: {args.dim}")
    print(f"Message passing rounds: {args.n_rounds}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    
    if args.data_dir:
        # Single directory - load and split
        if args.use_subgraph_features:
            dataset = CanonicalBackboneBPGDataset(
                args.data_dir,
                compute_subgraphs=True
            )
        else:
            dataset = BackboneBPGDataset(args.data_dir)
        
        print(f"Loaded {len(dataset)} samples")
        
        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    else:
        # Separate directories for train and val
        print("Loading training dataset...")
        if args.use_subgraph_features:
            train_dataset = CanonicalBackboneBPGDataset(
                args.train_dir,
                compute_subgraphs=True
            )
        else:
            train_dataset = BackboneBPGDataset(args.train_dir)
        
        print(f"Loaded {len(train_dataset)} training samples")
        
        print("Loading validation dataset...")
        if args.use_subgraph_features:
            val_dataset = CanonicalBackboneBPGDataset(
                args.val_dir,
                compute_subgraphs=True
            )
        else:
            val_dataset = BackboneBPGDataset(args.val_dir)
        
        print(f"Loaded {len(val_dataset)} validation samples")
        print(f"Total: Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\nInitializing model...")
    model = ArieNetBackbone(
        dim=args.dim,
        n_rounds=args.n_rounds,
        n_mlp_layers=3,
        activation='relu',
        device=args.device,
        use_subgraph_features=args.use_subgraph_features,
        subgraph_dim=32
    ).to(args.device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0
    
    for epoch in range(args.n_epochs):
        print(f"\nEpoch {epoch + 1}/{args.n_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            args.device, args.use_subgraph_features
        )
        
        # Evaluate
        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, args.device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}, Precision: {val_prec:.4f}, "
              f"Recall: {val_rec:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'args': args
            }, save_path)
            print(f"Saved best model with F1: {val_f1:.4f}")
    
    print("\nTraining complete!")
    print(f"Best validation F1: {best_val_f1:.4f}")


if __name__ == '__main__':
    main()

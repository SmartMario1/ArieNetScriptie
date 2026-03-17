"""
Train an ArieNet-style backbone predictor on WLIG graphs.

This script reuses the existing backbone objective and training flow,
but replaces BPG inputs with the Weighted Literal-Incidence Graph (WLIG)
encoding (Chen & Wang, 2025).

Expected data layout per split directory:
- *.cnf files (recursive)
- backbones.pkl mapping CNF path -> {var_index(1-based): True/False/None}

Usage:
  python train_arienet_backbone_wlig.py --data_dir SATSolving/3-sat/train_first_4000_ArieNet
  python train_arienet_backbone_wlig.py --train_dir <train_dir> --val_dir <val_dir>
"""

import argparse
import glob
import os
import random
import sys
import time
import pickle

from collections import defaultdict

import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

from graph_encodings import cnf_to_wlig

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from nsnet.models.mlp import MLP
from nsnet.utils.torch_utils import scatter_sum, swap_even_odd
from nsnet.utils.utils import parse_cnf_file


class ArieNetWLIGBackbone(nn.Module):
    """ArieNet-style iterative message passing over WLIG literal nodes."""

    def __init__(self, dim=128, n_rounds=16, n_mlp_layers=3, activation="relu"):
        super().__init__()
        self.dim = dim
        self.n_rounds = n_rounds

        # Node initialization from WLIG literal features: [weighted_degree, polarity]
        self.node_init = nn.Linear(2, dim)

        # Keep ArieNet-like edge-message updates, but over WLIG directed edges.
        self.c2l_edges_init = nn.Parameter(torch.randn(1, dim))
        self.l2c_edges_init = nn.Parameter(torch.randn(1, dim))
        self.denom = dim ** 0.5

        self.l2c_msg_update = MLP(n_mlp_layers, dim, dim, dim, activation)
        self.l2c_msg_norm = MLP(n_mlp_layers, dim * 2, dim, dim, activation)
        self.c2l_msg_update = MLP(n_mlp_layers, dim, dim, dim, activation)

        # Per-literal readout, then reshape to [num_vars, 2] logits.
        self.l_readout = MLP(n_mlp_layers, dim, dim, 1, activation)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.xavier_uniform_(self.node_init.weight)
        nn.init.zeros_(self.node_init.bias)
        nn.init.normal_(self.c2l_edges_init, mean=0.0, std=0.01)
        nn.init.normal_(self.l2c_edges_init, mean=0.0, std=0.01)

    def forward(self, data: Data):
        device = data.x.device
        n_literals = data.x.size(0)
        h0 = self.node_init(data.x.float())

        n_edges = int(data.n_edges.sum().item()) if isinstance(data.n_edges, torch.Tensor) else int(data.n_edges)
        c2l_edges_feat = (self.c2l_edges_init / self.denom).repeat(n_edges, 1).to(device)
        l2c_edges_feat = (self.l2c_edges_init / self.denom).repeat(n_edges, 1).to(device)

        c2l_receiver = data.c2l_msg_receiver_indices
        c2l_sender = data.c2l_msg_sender_indices
        literal_indices_per_edge = data.literal_indices_per_edge
        edge_weight = data.edge_weight_per_directed.unsqueeze(1)

        for _ in range(self.n_rounds):
            # List-index message passing: explicit receiver/sender edge lists.
            l2c_msg_argument = scatter_sum(
                c2l_receiver,
                c2l_edges_feat[c2l_sender] * edge_weight[c2l_sender],
                n_edges,
                device,
            )

            l2c_msg = self.l2c_msg_update(l2c_msg_argument)

            # Inject opposite-literal context similarly to ArieNet's +/− coupling.
            lit_features = scatter_sum(literal_indices_per_edge, l2c_msg, n_literals, device)
            lit_negated = swap_even_odd(lit_features)
            l2c_edges_feat = self.l2c_msg_norm(torch.cat([
                l2c_msg,
                lit_negated[literal_indices_per_edge],
            ], dim=1))

            c2l_edges_feat = self.c2l_msg_update(l2c_edges_feat)

        literal_logits = self.l_readout(scatter_sum(literal_indices_per_edge, c2l_edges_feat, n_literals, device) + h0)
        var_logits = literal_logits.reshape(-1, 2)
        return var_logits


class WLIGData(Data):
    """PyG Data with custom batching increments for edge-index list fields."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("c2l_msg_receiver_indices", "c2l_msg_sender_indices"):
            return int(self.n_edges.item()) if isinstance(self.n_edges, torch.Tensor) else int(self.n_edges)
        if key == "literal_indices_per_edge":
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class BackboneWLIGDataset(Dataset):
    """Create WLIG PyG samples from CNF files with backbone labels."""

    def __init__(self, data_dir, root=None, transform=None, pre_transform=None):
        self.data_dir = os.path.abspath(data_dir)
        self.cnf_files = sorted(glob.glob(os.path.join(self.data_dir, "**", "*.cnf"), recursive=True))

        backbone_file = os.path.join(self.data_dir, "backbones.pkl")
        if os.path.exists(backbone_file):
            with open(backbone_file, "rb") as f:
                self.backbones = pickle.load(f)
            print(f"Loaded backbone labels for {len(self.backbones)} files from {backbone_file}")
        else:
            print(f"[WARNING] No backbones.pkl found at {backbone_file}")
            self.backbones = {}

        if root is None:
            root = os.path.join(self.data_dir, "processed_wlig")

        self._cache = {}
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(len(self.cnf_files))]

    def download(self):
        pass

    def _resolve_backbone_key(self, cnf_path):
        abs_path = os.path.abspath(cnf_path)
        rel_path = os.path.relpath(abs_path, self.data_dir)
        if abs_path in self.backbones:
            return abs_path
        if rel_path in self.backbones:
            return rel_path
        return None

    def process(self):
        print(f"Processing {len(self.cnf_files)} CNF files into WLIG format...")
        os.makedirs(self.processed_dir, exist_ok=True)

        for idx, cnf_path in enumerate(tqdm(self.cnf_files)):
            output_path = os.path.join(self.processed_dir, f"data_{idx}.pt")
            if os.path.exists(output_path):
                continue

            try:
                n_vars, clauses = parse_cnf_file(cnf_path)
                if n_vars == 0 or len(clauses) == 0:
                    continue

                G = cnf_to_wlig(n_vars, clauses)
                n_literals = 2 * n_vars

                # Node features: weighted degree + polarity bit.
                weighted_degree = torch.zeros(n_literals, dtype=torch.float)
                polarity = torch.zeros(n_literals, dtype=torch.float)

                for lit_idx in range(n_literals):
                    node_name = f"lit_{lit_idx}"
                    weighted_degree[lit_idx] = float(G.degree(node_name, weight="weight"))
                    polarity[lit_idx] = 1.0 if lit_idx % 2 == 0 else 0.0

                max_deg = torch.max(weighted_degree)
                if max_deg > 0:
                    weighted_degree = weighted_degree / max_deg
                x = torch.stack([weighted_degree, polarity], dim=1)

                directed_src = []
                directed_dst = []
                directed_weights = []
                for u, v, edge_data in G.edges(data=True):
                    i = int(u.split("_")[1])
                    j = int(v.split("_")[1])
                    w = float(edge_data.get("weight", 1.0))
                    directed_src.extend([i, j])
                    directed_dst.extend([j, i])
                    directed_weights.extend([w, w])

                n_directed = len(directed_src)

                incoming_per_literal = defaultdict(list)
                for e_idx, dst_lit in enumerate(directed_dst):
                    incoming_per_literal[dst_lit].append(e_idx)

                # Explicit receiver/sender list-index mapping.
                c2l_msg_receiver_indices = []
                c2l_msg_sender_indices = []
                for recv_idx, src_lit in enumerate(directed_src):
                    for send_idx in incoming_per_literal.get(src_lit, []):
                        c2l_msg_receiver_indices.append(recv_idx)
                        c2l_msg_sender_indices.append(send_idx)

                literal_indices_per_edge = torch.tensor(directed_src, dtype=torch.long)
                edge_weight_per_directed = torch.tensor(directed_weights, dtype=torch.float)

                labels = torch.full((n_vars,), 2, dtype=torch.long)
                backbone_key = self._resolve_backbone_key(cnf_path)
                if backbone_key is not None:
                    backbone = self.backbones[backbone_key]
                    for var, value in backbone.items():
                        if value is not None and 1 <= var <= n_vars:
                            labels[var - 1] = 1 if value else 0

                data = WLIGData(
                    x=x,
                    literal_indices_per_edge=literal_indices_per_edge,
                    c2l_msg_receiver_indices=torch.tensor(c2l_msg_receiver_indices, dtype=torch.long),
                    c2l_msg_sender_indices=torch.tensor(c2l_msg_sender_indices, dtype=torch.long),
                    edge_weight_per_directed=edge_weight_per_directed,
                    y=labels,
                    n_edges=torch.tensor([n_directed], dtype=torch.long),
                    n_vars=torch.tensor([n_vars], dtype=torch.long),
                )

                torch.save(data, output_path)

            except Exception as e:
                print(f"[ERROR] Processing {cnf_path}: {e}")

    def len(self):
        return len(self.cnf_files)

    def get(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        self._cache[idx] = data
        return data


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_class_weights(y01, device):
    u0 = torch.sum((y01 == 0).int()).item()
    u1 = torch.sum((y01 == 1).int()).item()
    return torch.tensor(
        [len(y01) / (2 * (u0 + 1)), len(y01) / (2 * (u1 + 1))],
        dtype=torch.float,
        device=device,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0
    all_target = []
    all_pred = []

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        y01_indices = (batch.y != 2).nonzero(as_tuple=True)[0]
        if len(y01_indices) == 0:
            continue

        y01 = batch.y[y01_indices].long()
        class_weights = _batch_class_weights(y01, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        logits_subset = torch.clamp(logits[y01_indices], min=-50, max=50)
        loss = criterion(logits_subset, y01)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits_subset, dim=1)
            all_target.extend(y01.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        return None, None

    return total_loss / total_batches, confusion_matrix(all_target, all_pred, labels=[0, 1])


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_target = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            y01_indices = (batch.y != 2).nonzero(as_tuple=True)[0]
            if len(y01_indices) == 0:
                continue

            y01 = batch.y[y01_indices].long()
            class_weights = _batch_class_weights(y01, device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

            logits = model(batch)
            logits_subset = logits[y01_indices]
            loss = criterion(logits_subset, y01)

            preds = torch.argmax(logits_subset, dim=1)
            all_target.extend(y01.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        return None, None, None, None, None

    avg_loss = total_loss / total_batches
    cm = confusion_matrix(all_target, all_pred, labels=[0, 1])
    recall = recall_score(all_target, all_pred, average="binary", zero_division=0)
    precision = precision_score(all_target, all_pred, average="binary", zero_division=0)
    f1 = f1_score(all_target, all_pred, average="binary", zero_division=0)
    return avg_loss, cm, recall, precision, f1


def main():
    parser = argparse.ArgumentParser(description="Train ArieNet backbone prediction on WLIG")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Single directory with CNF files and backbones.pkl (uses 80/20 split)")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Training directory with CNF files and backbones.pkl")
    parser.add_argument("--val_dir", type=str, default=None,
                        help="Validation directory with CNF files and backbones.pkl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n_rounds", type=int, default=16)
    parser.add_argument("--n_mlp_layers", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--n_epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="models/arienet_backbone_wlig")
    parser.add_argument("--run_name", type=str, default="default")
    args = parser.parse_args()

    if args.data_dir is None and (args.train_dir is None or args.val_dir is None):
        parser.error("Either --data_dir OR both --train_dir and --val_dir must be provided")
    if args.data_dir is not None and (args.train_dir is not None or args.val_dir is not None):
        parser.error("Use either --data_dir OR --train_dir/--val_dir, not both")

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.data_dir:
        dataset = BackboneWLIGDataset(args.data_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=generator
        )
        print(f"Loaded dataset from {args.data_dir}: train={len(train_dataset)} val={len(val_dataset)}")
    else:
        train_dataset = BackboneWLIGDataset(args.train_dir)
        val_dataset = BackboneWLIGDataset(args.val_dir)
        print(f"Loaded train={len(train_dataset)} from {args.train_dir}")
        print(f"Loaded val={len(val_dataset)} from {args.val_dir}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0 and device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0 and device.type == "cuda"),
    )

    model = ArieNetWLIGBackbone(
        dim=args.dim,
        n_rounds=args.n_rounds,
        n_mlp_layers=args.n_mlp_layers,
        activation=args.activation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    log_path = os.path.join(args.save_dir, f"training_log_{args.run_name}.txt")
    ckpt_path = os.path.join(args.save_dir, f"arienet_wlig_{args.run_name}_best.pt")

    with open(log_path, "a") as log_file:
        log_file.write(f"\n{'=' * 60}\n")
        log_file.write(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Args: {vars(args)}\n")
        log_file.write(f"{'=' * 60}\n")

        for epoch in range(args.n_epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch + 1}/{args.n_epochs}")

            train_loss, train_cm = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_cm, recall, precision, f1 = evaluate(model, val_loader, device)

            epoch_time = time.time() - epoch_start

            if train_loss is not None:
                print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(
                    f"Val Loss: {val_loss:.4f}, Recall: {recall:.4f}, "
                    f"Precision: {precision:.4f}, F1: {f1:.4f}, Time: {epoch_time:.1f}s"
                )

            log_file.write(f"\nEpoch {epoch + 1}/{args.n_epochs}\n")
            if train_loss is not None:
                log_file.write(f"Train Loss: {train_loss:.6f}\n")
                log_file.write(f"Train CM:\n{train_cm}\n")
            if val_loss is not None:
                log_file.write(f"Val Loss: {val_loss:.6f}\n")
                log_file.write(f"Val CM:\n{val_cm}\n")
                log_file.write(
                    f"Recall: {recall:.6f} Precision: {precision:.6f} F1: {f1:.6f} Time: {epoch_time:.1f}s\n"
                )

            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Saved best model to {ckpt_path}")
                log_file.write(f"Saved best model to {ckpt_path}\n")

            log_file.flush()

    print("Training complete.")


if __name__ == "__main__":
    main()

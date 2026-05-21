"""
Training script for ArieNetCooc — ArieNet with literal co-occurrence (L2L) message passing.

ArieNetCooc extends ArieNet with an additional message-passing step over
literal co-occurrence edges (two literals that appear together in at least one
clause).  These edges carry a separate learnable starting vector
(cooc_edges_init) so the model can distinguish co-occurrence relationships from
the standard literal-clause relationships.

The co-occurrence indices are computed automatically by BPGParamBuilder and
stored in the BPG data objects.

Usage:
    # Single directory (split 80/20 for train/val)
    python train_arienet_cooc.py --data_dir data/backbone

    # Separate train and validation directories
    python train_arienet_cooc.py --train_dir data/train --val_dir data/val
"""

import argparse
import glob
import os
import pickle
import sys
import traceback

import torch
import torch.nn as nn
from torch_geometric.data import Dataset, Batch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from nsnet.models.arienet import ArieNetCooc
from nsnet.utils.dataset import BPG, BPGParamBuilder
from nsnet.utils.utils import parse_cnf_file


def _collate_skip_none(data_list):
    """Collate that drops None items (skipped/trivial instances) before PyG batching."""
    data_list = [d for d in data_list if d is not None]
    if not data_list:
        return None
    return Batch.from_data_list(data_list)


# ---------------------------------------------------------------------------
# Module-level worker function (required for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _process_cooc_file(args):
    """Process a single CNF file into a compact dict cache with backbone labels."""
    idx, cnf_path, processed_dir, backbones = args
    import torch
    from nsnet.utils.dataset import BPGParamBuilder
    from nsnet.utils.utils import parse_cnf_file, unit_propagation
    import os
    output_path = os.path.join(processed_dir, f'data_{idx}.pt')
    if os.path.exists(output_path):
        return
    n_vars, clauses = parse_cnf_file(cnf_path)
    # Save a sentinel for unparseable or trivially-empty instances so that
    # get() can skip them rather than crashing on a missing file.
    if n_vars == 0 or len(clauses) == 0:
        torch.save({'skip': True}, output_path)
        return
    # Unit-propagate to eliminate unit clauses. We keep the original variable
    # numbering (no renaming) so backbone labels remain valid. Variables forced
    # by unit propagation are simply no longer present in any clause; they keep
    # their backbone label (already set to 2/unknown by default, and overwritten
    # from backbones.pkl below if known).
    clauses, _ = unit_propagation(clauses)
    if len(clauses) == 0:
        # Formula is trivially SAT after propagation — save sentinel.
        torch.save({'skip': True}, output_path)
        return
    p = BPGParamBuilder(clauses, n_vars).params
    labels = torch.full((n_vars,), 2, dtype=torch.int8)
    abs_path = os.path.abspath(cnf_path)
    if abs_path in backbones:
        for var, value in backbones[abs_path].items():
            if value is not None and 1 <= var <= n_vars:
                labels[var - 1] = 1 if value else 0
    # Compute literal co-occurrence edges: for each clause, connect every pair
    # of literals (both directions) that appear together in that clause.
    # Literal index: variable v → positive = 2*(v-1), negative = 2*(v-1)+1.
    cooc_src_list, cooc_dst_list = [], []
    seen = set()
    for clause in clauses:
        lit_indices = [2*(abs(l)-1) if l > 0 else 2*(abs(l)-1)+1 for l in clause]
        for i in range(len(lit_indices)):
            for j in range(len(lit_indices)):
                if i != j:
                    s, d = lit_indices[i], lit_indices[j]
                    if (s, d) not in seen:
                        seen.add((s, d))
                        cooc_src_list.append(s)
                        cooc_dst_list.append(d)
    cooc_src = torch.tensor(cooc_src_list, dtype=torch.int32)
    cooc_dst = torch.tensor(cooc_dst_list, dtype=torch.int32)
    # Save a compact dict: index tensors as int32 (half the size of int64),
    # floats as float32, labels as int8. No PyG Data wrapper overhead.
    torch.save({
        'n_clauses':  p.n_clauses,
        'n_literals': p.n_literals,
        'lipe':  p.literal_indices_per_edge.to(torch.int32),
        'lipo':  p.literal_indices_per_occurence.to(torch.int32),
        'cipo':  p.clause_indices_per_occurence.to(torch.int32),
        'lsppe': p.local_satisfaction_percentage_per_edge.to(torch.float32),
        'c2l_r': p.c2l_msg_receiver_indices.to(torch.int32),
        'c2l_s': p.c2l_msg_sender_indices.to(torch.int32),
        'l2c_r': p.l2c_msg_receiver_indices.to(torch.int32),
        'l2c_a': p.l2c_assignment_indices.to(torch.int32),
        'l2c_n': p.l2c_assignment_neighborhoods.to(torch.int32),
        'cooc_src': cooc_src,
        'cooc_dst': cooc_dst,
        'y':     labels,
    }, output_path)


# ---------------------------------------------------------------------------
# Minimal opts container expected by ArieNet / ArieNetCooc
# ---------------------------------------------------------------------------

class Opts:
    def __init__(self, dim, n_rounds, n_mlp_layers, activation, device, task='satisfiability'):
        self.dim = dim
        self.n_rounds = n_rounds
        self.n_mlp_layers = n_mlp_layers
        self.activation = activation
        self.device = device
        self.task = task


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CoocBPGDataset(Dataset):
    """
    Dataset that creates BPG objects (including co-occurrence indices) from CNF
    files with backbone labels.

    Expected structure:
        data_dir/          — CNF files (recursively found)
        data_dir/backbones.pkl  — dict mapping abs path → {var: bool|None}
    """

    def __init__(self, data_dir, root=None, transform=None, pre_transform=None,
                 process_workers=1):
        self.data_dir = data_dir
        self.process_workers = max(1, int(process_workers))
        self.cnf_files = sorted(glob.glob(os.path.join(data_dir, '**/*.cnf'), recursive=True))

        backbone_file = os.path.join(data_dir, 'backbones.pkl')
        if os.path.exists(backbone_file):
            with open(backbone_file, 'rb') as f:
                self.backbones = pickle.load(f)
            print(f"Loaded backbone labels for {len(self.backbones)} files")
        else:
            print(f"[WARNING] No backbones.pkl found at {backbone_file}")
            self.backbones = {}

        print(f"Found {len(self.cnf_files)} CNF files in {data_dir}")

        if root is None:
            root = os.path.join(data_dir, 'processed_cooc_v2')

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
        from concurrent.futures import ProcessPoolExecutor, as_completed

        tasks = [
            (idx, cnf_path, self.processed_dir, self.backbones)
            for idx, cnf_path in enumerate(self.cnf_files)
            if not os.path.exists(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        ]

        if not tasks:
            print(f"All {len(self.cnf_files)} BPG (cooc) files already processed, skipping.")
            return

        workers = self.process_workers
        print(f"Processing {len(tasks)}/{len(self.cnf_files)} files into BPG format "
              f"(with co-occurrence, {workers} worker{'s' if workers != 1 else ''})...")

        completed = 0
        if workers == 1:
            for task in tasks:
                idx = task[0]
                try:
                    _process_cooc_file(task)
                    completed += 1
                    if completed % 10 == 0 or completed == len(tasks):
                        print(f'  [{completed}/{len(tasks)}]', end='\r')
                except Exception as e:
                    print(f"[ERROR] Failed to process index {idx}: {e}")
                    traceback.print_exc()
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_process_cooc_file, t): t[0] for t in tasks}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()
                        completed += 1
                        if completed % 10 == 0 or completed == len(tasks):
                            print(f'  [{completed}/{len(tasks)}]', end='\r')
                    except Exception as e:
                        print(f"[ERROR] Failed to process index {idx}: {e}")
                        traceback.print_exc()
        print(f"\nDone: {completed} files processed.")

    def len(self):
        return len(self.cnf_files)

    def get(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        d = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'), weights_only=True)
        if d.get('skip', False):
            return None
        data = BPG(
            n_clauses=d['n_clauses'],
            n_literals=d['n_literals'],
            literal_indices_per_edge=d['lipe'].long(),
            literal_indices_per_occurence=d['lipo'].long(),
            clause_indices_per_occurence=d['cipo'].long(),
            local_satisfaction_percentage_per_edge=d['lsppe'],
            c2l_msg_receiver_indices=d['c2l_r'].long(),
            c2l_msg_sender_indices=d['c2l_s'].long(),
            l2c_msg_receiver_indices=d['l2c_r'].long(),
            l2c_assignment_indices=d['l2c_a'].long(),
            l2c_assignment_neighborhoods=d['l2c_n'].long(),
            cooc_src_indices=d['cooc_src'].long() if 'cooc_src' in d else None,
            cooc_dst_indices=d['cooc_dst'].long() if 'cooc_dst' in d else None,
        )
        data.y = d['y'].long()
        self._cache[idx] = data
        return data


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for data in tqdm(loader, desc="Training"):
        if data is None:
            continue
        data = data.to(device)
        y01_idx = (data.y != 2).nonzero(as_tuple=True)[0]
        if len(y01_idx) == 0:
            continue

        y01 = data.y[y01_idx].long()
        u0 = (y01 == 0).sum().item()
        u1 = (y01 == 1).sum().item()
        weights = torch.tensor(
            [len(y01) / (2 * (u0 + 1)), len(y01) / (2 * (u1 + 1))],
            device=device, dtype=torch.float
        )
        criterion = nn.CrossEntropyLoss(weight=weights)

        optimizer.zero_grad(set_to_none=True)
        logits = model(data)
        logits_subset = torch.clamp(logits[y01_idx], min=-50, max=50)
        loss = criterion(logits_subset, y01)

        if torch.isnan(loss) or torch.isinf(loss):
            print("[WARNING] NaN/Inf loss, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y01_idx)
        total_samples += len(y01_idx)

    return total_loss / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for data in tqdm(loader, desc="Evaluating"):
        if data is None:
            continue
        data = data.to(device)
        logits = model(data)
        preds = torch.argmax(logits, dim=1)
        mask = data.y != 2
        if mask.sum() > 0:
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(data.y[mask].cpu().tolist())

    if not all_preds:
        return 0.0, 0.0, 0.0, 0.0

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec  = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1   = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    return acc, prec, rec, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ensure_backbones(data_dir, n_process=4):
    """Run generate_backbone_labels.py on data_dir if backbones.pkl is missing."""
    backbone_file = os.path.join(data_dir, 'backbones.pkl')
    if os.path.exists(backbone_file):
        return
    print(f"[INFO] backbones.pkl not found in {data_dir} — generating now...")
    import subprocess, sys
    script = os.path.join(os.path.dirname(__file__), 'generate_backbone_labels.py')
    subprocess.run(
        [sys.executable, script, data_dir, '--n_process', str(n_process)],
        check=True
    )
    if not os.path.exists(backbone_file):
        raise RuntimeError(f"Backbone generation failed for {data_dir}")
    print(f"[INFO] Backbone labels saved to {backbone_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Train ArieNetCooc (with literal co-occurrence edges)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  # Single directory, auto-split 80/20:\n'
            '  python train_arienet_cooc.py --data_dir SATSolving/train/3sat\n'
            '  # Multiple train dirs + multiple val dirs:\n'
            '  python train_arienet_cooc.py \\\n'
            '      --train_dirs SATSolving/train/3sat SATSolving/train/4sat SATSolving/train/tseitin_tree \\\n'
            '      --val_dirs   SATSolving/train/parity'
        )
    )
    # Three mutually exclusive ways to specify data:
    #   --data_dir DIR          : single dir, auto-split 80/20
    #   --train_dirs D1 D2 ...  : one or more train dirs
    #   --val_dirs   D1 D2 ...  : one or more val dirs  (required with --train_dirs)
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Single directory with CNF files + backbones.pkl (split 80/20)')
    parser.add_argument('--train_dirs', type=str, nargs='+', default=None,
                        help='One or more training directories')
    parser.add_argument('--val_dirs', type=str, nargs='+', default=None,
                        help='One or more validation directories')
    parser.add_argument('--n_process', type=int, default=4,
                        help='Parallel workers for backbone generation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n_rounds', type=int, default=26)
    parser.add_argument('--n_mlp_layers', type=int, default=3)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='models/arienet_cooc')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a .pt checkpoint file to resume training from')
    args = parser.parse_args()

    n_modes = sum([
        args.data_dir is not None,
        args.train_dirs is not None,
    ])
    if n_modes == 0:
        parser.error("Provide --data_dir OR --train_dirs/--val_dirs")
    if n_modes > 1:
        parser.error("Cannot mix --data_dir with --train_dirs/--val_dirs")
    if args.train_dirs is not None and args.val_dirs is None:
        parser.error("--train_dirs requires --val_dirs")

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("ArieNetCooc Training")
    print("=" * 70)
    print(f"dim={args.dim}  rounds={args.n_rounds}  mlp_layers={args.n_mlp_layers}")
    print(f"epochs={args.n_epochs}  lr={args.lr}  batch_size={args.batch_size}")
    print(f"device={args.device}  save_dir={args.save_dir}")
    print("=" * 70)

    # Auto-generate backbones and build datasets
    if args.data_dir:
        _ensure_backbones(args.data_dir, n_process=args.n_process)
        full_dataset = CoocBPGDataset(args.data_dir)
        train_size = int(0.8 * len(full_dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, len(full_dataset) - train_size]
        )
        print(f"Split: {len(train_dataset)} train / {len(val_dataset)} val")
    else:
        train_parts, val_parts = [], []
        for d in args.train_dirs:
            _ensure_backbones(d, n_process=args.n_process)
            train_parts.append(CoocBPGDataset(d))
            print(f"  train: {len(train_parts[-1])} instances from {d}")
        for d in args.val_dirs:
            _ensure_backbones(d, n_process=args.n_process)
            val_parts.append(CoocBPGDataset(d))
            print(f"  val:   {len(val_parts[-1])} instances from {d}")
        train_dataset = torch.utils.data.ConcatDataset(train_parts)
        val_dataset   = torch.utils.data.ConcatDataset(val_parts)
        print(f"Total: {len(train_dataset)} train / {len(val_dataset)} val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate_skip_none)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=_collate_skip_none)

    # Model
    opts = Opts(
        dim=args.dim,
        n_rounds=args.n_rounds,
        n_mlp_layers=args.n_mlp_layers,
        activation='relu',
        device=args.device,
    )
    model = ArieNetCooc(opts).to(args.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nArieNetCooc: {n_params:,} trainable parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_f1 = 0.0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_f1 = ckpt.get('val_f1', 0.0)
        print(f"Resuming from epoch {start_epoch}, best F1 so far: {best_f1:.4f}")

    for epoch in range(start_epoch, args.n_epochs):
        print(f"\nEpoch {epoch + 1}/{args.n_epochs}")
        loss = train_epoch(model, train_loader, optimizer, args.device)
        acc, prec, rec, f1 = evaluate(model, val_loader, args.device)
        print(f"Loss: {loss:.4f}  |  Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': f1,
                'args': args,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  → Saved best model (F1: {f1:.4f})")

    print(f"\nDone. Best validation F1: {best_f1:.4f}")


if __name__ == '__main__':
    main()

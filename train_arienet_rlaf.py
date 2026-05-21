"""
train_arienet_rlaf.py
=====================
Train ArieNet (with optional literal co-occurrence graph) using the RLAF
(Reinforcement Learning with Advantages and Feedback) objective.

Replaces the backbone-prediction objective of train_arienet_backbone.py /
train_arienet_cooc.py with GRPO or DPO training against solver performance
(decisions, conflicts, …).

Model architecture kept intact
-------------------------------
  * BPG message passing (clause ↔ literal bipartite graph)
  * Local satisfaction percentages on every c2l edge
  * Optional literal co-occurrence (L2L) edges  →  --use_cooc / cfg.use_cooc

Objective replaced
------------------
  * GRPO  (default) — Group Relative Policy Optimisation
  * DPO            — Direct Preference Optimisation

Usage
-----
  python train_arienet_rlaf.py                          # use defaults in config
  python train_arienet_rlaf.py method=dpo               # switch to DPO
  python train_arienet_rlaf.py use_cooc=true            # enable COOC graph
  python train_arienet_rlaf.py from_checkpoint=runs/…/best.pt

Hydra config: configs/config_train_arienet_rlaf.yaml
"""

import os
import sys

from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd
import torch
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── nsnet package on sys.path ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nsnet.models.arienet_rlaf import ArieNetRLAF, ArieNetRLAFCooc
from nsnet.datasets.rlaf_dataset import RLAFBPGDataset, RLAFTrainingDataset, _collate_skip_none
from nsnet.policy.evaluate_bpg import sample_var_params, compute_solver_stats
from nsnet.training.grpo_bpg import train_grpo, get_grpo_advantage
from nsnet.training.dpo_bpg import train_dpo


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(cfg: DictConfig) -> torch.nn.Module:
    model_cls = ArieNetRLAFCooc if cfg.use_cooc else ArieNetRLAF
    return model_cls(
        dim=cfg.model.dim,
        n_rounds=cfg.model.n_rounds,
        n_mlp_layers=cfg.model.n_mlp_layers,
        activation=cfg.model.activation,
        no_precomputed_local_sat=cfg.model.no_precomputed_local_sat,
        use_up_features=cfg.model.use_up_features,
    )


def save_model(model: torch.nn.Module, cfg: DictConfig, name: str = "last") -> None:
    os.makedirs(cfg.model_dir, exist_ok=True)
    cfg_path = os.path.join(cfg.model_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        OmegaConf.save(cfg, f)
    torch.save(model.state_dict(), os.path.join(cfg.model_dir, f"{name}.pt"))


def load_checkpoint(ckpt_path: str, cfg: DictConfig) -> torch.nn.Module:
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_solver_metrics(
    solver_stats: pd.DataFrame,
    iteration: int,
    global_step: int,
    prefix: str = "train",
    target_stat: str = "decisions",
) -> None:
    keys = ["decisions", "conflicts", "propagations", "restarts", "CPU time"]
    metrics = {
        f"{prefix}/{k}": solver_stats[k].mean()
        for k in keys
        if k in solver_stats.columns
    }
    metrics["iteration"] = iteration
    metrics["global_step"] = global_step
    for k in keys:
        if k in solver_stats.columns:
            metrics[f"{prefix}/{k}_hist"] = wandb.Histogram(solver_stats[k])
    wandb.log(metrics, step=global_step)
    print(
        f"[{prefix}] iter={iteration} "
        + "  ".join(f"{k}={v:.1f}" for k, v in metrics.items() if isinstance(v, float))
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    if cfg.from_checkpoint is not None:
        model = load_checkpoint(cfg.from_checkpoint, cfg)
        print(f"Loaded checkpoint from {cfg.from_checkpoint}")
    else:
        model = build_model(cfg)

    # ── Datasets ───────────────────────────────────────────────────────────
    dataset_kwargs = dict(
        use_cooc=cfg.use_cooc,
        no_precomputed_local_sat=cfg.model.no_precomputed_local_sat,
        use_up_features=cfg.model.use_up_features,
        num_workers=cfg.dataset.num_process_workers,
    )

    dataset_train = RLAFBPGDataset(
        path=cfg.dataset.train_dirs if cfg.dataset.train_dirs is not None else cfg.dataset.train_path,
        **dataset_kwargs,
    )
    dataset_val = RLAFBPGDataset(
        path=cfg.dataset.val_dirs if cfg.dataset.val_dirs is not None else cfg.dataset.val_path,
        **dataset_kwargs,
    )

    loader_train = DataLoader(
        dataset=dataset_train,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=True,
        collate_fn=_collate_skip_none,
    )
    loader_val = DataLoader(
        dataset=dataset_val,
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        shuffle=False,
        collate_fn=_collate_skip_none,
    )

    assert cfg.training.cnf_per_iter % cfg.loader.batch_size == 0
    train_batches = cfg.training.cnf_per_iter // cfg.loader.batch_size

    # ── Optimiser ──────────────────────────────────────────────────────────
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        maximize=True,          # we maximise the policy objective
    )

    warmup_iters = 5

    def lr_lambda(step):
        return float(step + 1) / warmup_iters if step < warmup_iters else 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ── Solver kwargs ───────────────────────────────────────────────────────
    solver_kwargs = dict(
        solver_dir=cfg.solver.solver_dir,
        num_workers=cfg.solver.num_workers,
        solver=cfg.solver.solver,
        **OmegaConf.to_container(cfg.solver.params),
    )

    # ── Training loop ───────────────────────────────────────────────────────
    if cfg.ckpt_interval is not None:
        save_model(model, cfg, "iter=0")

    best_score = np.inf
    global_step = 0
    method = cfg.method.lower()

    for iteration in range(cfg.training.iterations):
        print(f"\n{'─'*20} {method.upper()} Iteration {iteration} {'─'*20}")

        # ---- Validation -------------------------------------------------------
        if iteration % cfg.val_interval == 0:
            val_data = sample_var_params(
                model=model,
                loader=loader_val,
                num_samples=1,
                device=device,
                use_mode=True,
                scale_sigma=cfg.scale_sigma,
            )
            val_stats = compute_solver_stats(
                data_list=val_data,
                cnf_clauses=dataset_val.cnf_clauses,
                **solver_kwargs,
            )
            log_solver_metrics(
                val_stats, iteration, global_step, prefix="val",
                target_stat=cfg.training.target_stat,
            )
            score = val_stats[cfg.training.target_stat].mean()
            if score < best_score:
                print("  → New best checkpoint")
                save_model(model, cfg, "best")
                best_score = score

        # ---- Sample training data --------------------------------------------
        train_data = sample_var_params(
            model=model,
            loader=loader_train,
            num_samples=cfg.training.num_samples,
            max_num_batches=train_batches,
            device=device,
            scale_sigma=cfg.scale_sigma,
        )

        train_stats = compute_solver_stats(
            data_list=train_data,
            cnf_clauses=dataset_train.cnf_clauses,
            **solver_kwargs,
        )

        log_solver_metrics(
            train_stats, iteration, global_step, prefix="train",
            target_stat=cfg.training.target_stat,
        )

        # ---- Build iteration dataset -----------------------------------------
        if method == "grpo":
            train_stats["advantage"] = get_grpo_advantage(
                train_stats, cfg.training.target_stat
            )
            iter_dataset = RLAFTrainingDataset(
                data_list=train_data,
                solver_stats=train_stats,
                target_stat="advantage",
                objective="maximize",
            )
        else:
            iter_dataset = RLAFTrainingDataset(
                data_list=train_data,
                solver_stats=train_stats,
                target_stat=cfg.training.target_stat,
                objective="minimize",
            )

        iter_loader = DataLoader(
            dataset=iter_dataset,
            batch_size=cfg.loader.batch_size,
            num_workers=cfg.loader.num_workers,
            shuffle=True,
            collate_fn=_collate_skip_none,
        )

        # ---- Optimise --------------------------------------------------------
        train_fn = train_grpo if method == "grpo" else train_dpo
        common_kwargs = dict(
            model=model,
            loader=iter_loader,
            optim=optim,
            sched=sched,
            steps=cfg.training.steps_per_iter,
            kl_penalty=cfg.training.kl_penalty,
            global_step=global_step,
            scale_sigma=cfg.scale_sigma,
            device=device,
            use_amp=cfg.training.use_amp,
            accum_steps=cfg.training.accum_steps,
        )
        if method == "grpo":
            global_step = train_grpo(clip_ratio=cfg.training.clip_ratio, **common_kwargs)
        else:
            global_step = train_dpo(beta=cfg.training.beta, **common_kwargs)

        if cfg.ckpt_interval is not None and iteration % cfg.ckpt_interval == 0:
            save_model(model, cfg, f"iter={iteration}")

    save_model(model, cfg, "last")
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-n", "--name", default=None, help="Model run name (sets model_name and model_dir)")
    _args, _remaining = parser.parse_known_args()

    _cfg_path = os.path.join(os.path.dirname(__file__), "configs", "config_train_arienet_rlaf.yaml")
    _cfg = OmegaConf.load(_cfg_path)
    # Accept Hydra-style dot-notation overrides from CLI, e.g. training.iterations=3
    _overrides = [a for a in _remaining if "=" in a]
    if _overrides:
        _cfg = OmegaConf.merge(_cfg, OmegaConf.from_dotlist(_overrides))
    if _args.name is not None:
        _cfg.model_name = _args.name
        _cfg.model_dir = f"runs/{_args.name}"
    OmegaConf.resolve(_cfg)
    main(_cfg)

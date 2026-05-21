"""GRPO (Group Relative Policy Optimisation) training loop for BPG-format data.

Adapted from RLAF/src/training/grpo.py.

Key differences from the original:
  - var_batch is derived from data.n_literals (BPG) instead of data["lit"].batch
  - Model parameters are stored as data.var_params / data.y_var_ref / data.log_prob
    instead of the HeteroData-dict style used in the original RLAF repo.
"""

import time

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
from typing import Union, Tuple

import wandb

from nsnet.policy import policy_bpg as policy
from nsnet.policy.evaluate_bpg import get_var_batch


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def get_grpo_advantage(
    solver_stats: pd.DataFrame,
    target_stat: str = "decisions",
) -> np.ndarray:
    """Normalise per-group (per-CNF) solver costs into GRPO advantages."""
    max_cost = solver_stats[target_stat].max()
    solver_stats = solver_stats.copy()
    solver_stats[target_stat] = solver_stats[target_stat].fillna(max_cost)

    grouped = solver_stats[["cnf_id", target_stat]].groupby("cnf_id")
    mean = grouped.mean().loc[solver_stats["cnf_id"]][target_stat].to_numpy()
    std = grouped.std().loc[solver_stats["cnf_id"]][target_stat].to_numpy()

    advantage = -(solver_stats[target_stat].to_numpy() - mean) / (std + 1e-8)
    return advantage


# ---------------------------------------------------------------------------
# Clipped PPO-style objective
# ---------------------------------------------------------------------------

def objective(
    log_prob: Tensor,
    log_prob_ref: Tensor,
    advantage: Tensor,
    clip_ratio: float = 0.2,
) -> Tuple[Tensor, Tensor]:
    g = advantage.clone()
    g[advantage >= 0.0] *= 1 + clip_ratio
    g[advantage < 0.0] *= 1 - clip_ratio

    prob_ratio = torch.exp(log_prob - log_prob_ref)
    L = torch.minimum(prob_ratio * advantage, g)
    return L.mean(), prob_ratio


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_grpo(
    model: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    steps: int,
    clip_ratio: float = 0.2,
    kl_penalty: float = 0.1,
    global_step: int = 0,
    scale_sigma: float = 0.1,
    device: Union["torch.device", str] = "cpu",
    use_amp: bool = True,
    accum_steps: int = 1,
) -> int:
    scaler = torch.amp.GradScaler() if use_amp else None
    model.to(device)
    model.train()

    epochs = max(1, (steps * accum_steps) // len(loader))
    L_all, prob_ratio_all, kl_all, ent_all = [], [], [], []
    num_steps = 0
    accum_count = 0

    t0 = time.time()
    for _ in range(epochs):
        for data in loader:
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                if accum_count == 0:
                    optim.zero_grad()
                data = data.to(device)
                y_var = model(data)

            with torch.amp.autocast(device_type="cuda", enabled=False):
                y_var = y_var.float()
                # [n_vars_batch, num_samples, 2] → transpose to [num_samples, n_vars, 2]
                var_params = data.var_params.transpose(0, 1).float()
                log_prob_ref = data.log_prob.transpose(0, 1).float()   # [num_samples, B]
                y_var_ref = data.y_var_ref.float()                      # [n_vars_batch, 2]
                advantage = data.stats.transpose(0, 1).float()          # [num_samples, B]
                var_batch = get_var_batch(data, device=device)

                log_prob = policy.log_prob(
                    y_var, var_params, var_batch, scale_sigma=scale_sigma
                )  # [num_samples, B]
                L, prob_ratio = objective(log_prob, log_prob_ref, advantage, clip_ratio)

                kl = policy.kl_div(y_var, y_var_ref, var_batch, scale_sigma=scale_sigma)
                kl_mean = kl.mean()
                ent = policy.entropy(y_var, var_batch, scale_sigma=scale_sigma)

                L_total = (L - kl_penalty * kl_mean) / accum_steps

                if use_amp:
                    scaler.scale(L_total).backward()
                else:
                    L_total.backward()

                accum_count += 1
                L_all.append(L.item())
                prob_ratio_all.append(prob_ratio.detach().cpu().numpy())
                kl_all.append(kl_mean.item())
                ent_all.append(ent.item())

                if accum_count >= accum_steps:
                    if use_amp:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optim)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optim.step()
                    accum_count = 0
                    global_step += 1
                    num_steps += 1

    wandb.log(
        {
            "train/L": np.mean(L_all),
            "train/prob_ratio": wandb.Histogram(np.concatenate(prob_ratio_all)),
            "train/kl_div": np.mean(kl_all),
            "train/entropy": np.mean(ent_all),
            "train/lr": sched.get_last_lr()[0],
        },
        step=global_step,
    )
    print(f"GRPO: {num_steps} steps in {time.time() - t0:.1f}s")
    sched.step()
    return global_step

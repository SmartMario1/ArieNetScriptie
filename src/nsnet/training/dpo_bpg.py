"""DPO (Direct Preference Optimisation) training loop for BPG-format data.

Adapted from RLAF/src/training/dpo.py.

Key differences from the original:
  - var_batch is derived from data.n_literals (BPG) instead of data["lit"].batch
  - Model parameters are stored as data.var_params / data.y_var_ref / data.log_prob
    instead of the HeteroData-dict style used in the original RLAF repo.
"""

import time

import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from typing import Union

import wandb

from nsnet.policy import policy_bpg as policy
from nsnet.policy.evaluate_bpg import get_var_batch


def train_dpo(
    model: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    steps: int,
    beta: float = 1.0,
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
    L_all, kl_all, ent_all = [], [], []
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
                var_params = data.var_params.transpose(0, 1).float()   # [S, n_vars, 2]
                log_prob_ref = data.log_prob.transpose(0, 1).float()   # [S, B]
                y_var_ref = data.y_var_ref.float()
                var_batch = get_var_batch(data, device=device)

                log_prob = policy.log_prob(
                    y_var, var_params, var_batch, scale_sigma=scale_sigma
                )  # [S, B]

                # DPO ranking loss: prefer lower-cost samples over higher-cost ones
                # (data.stats is sorted ascending by cost in RLAFTrainingDataset)
                log_prob_ratio = (log_prob - log_prob_ref).transpose(0, 1)  # [B, S]
                B, N = log_prob_ratio.shape
                # All pairwise preference scores: higher-ranked (lower-cost) vs lower-ranked
                score = (
                    log_prob_ratio[:, 1:].view(B, N - 1, 1)
                    - log_prob_ratio[:, :-1].view(B, 1, N - 1)
                )
                tril = torch.tril_indices(N - 1, N - 1, device=log_prob.device)
                score = score[:, tril[0], tril[1]]
                L = F.logsigmoid(beta * score).mean()

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
            "train/kl_div": np.mean(kl_all),
            "train/entropy": np.mean(ent_all),
            "train/lr": sched.get_last_lr()[0],
        },
        step=global_step,
    )
    print(f"DPO: {num_steps} steps in {time.time() - t0:.1f}s")
    sched.step()
    return global_step

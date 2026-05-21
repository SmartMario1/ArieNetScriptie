"""Policy functions for the RLAF training objective.

Each variable is parameterised by a pair (rho, mu):
    phase ~ Binomial(logits=rho)           -- which truth value to prefer
    scale ~ LogNormal(mu, scale_sigma)     -- branching-weight magnitude

These are identical to the policy functions in the RLAF repo
(RLAF/src/policy/policy.py), reproduced here so the nsnet package is
self-contained.
"""

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch_scatter import scatter_sum
from typing import Optional, Tuple


def distributions(
    y_var: Tensor, scale_sigma: float = 0.1
) -> Tuple[Distribution, Distribution]:
    rho, mu = y_var[:, 0], y_var[:, 1]
    rho = rho.clamp(-8, 8)
    phase_dist = torch.distributions.Binomial(logits=rho, total_count=1)
    scale_dist = torch.distributions.LogNormal(mu, scale_sigma)
    return phase_dist, scale_dist


def sample(y_var: Tensor, num_samples: int = 1, scale_sigma: float = 0.1) -> Tensor:
    """Sample `num_samples` variable parameterisations from the policy.

    Returns shape [num_samples, n_vars, 2] where the last dim is (phase, scale).
    """
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    sample_shape = torch.Size((num_samples,))
    phase = phase_dist.sample(sample_shape=sample_shape)
    scale = scale_dist.sample(sample_shape=sample_shape)
    return torch.stack([phase, scale], dim=-1)  # [num_samples, n_vars, 2]


def mode(y_var: Tensor, scale_sigma: float = 0.1) -> Tensor:
    """Return the modal parameterisation (shape [1, n_vars, 2])."""
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    return torch.stack(
        [phase_dist.mode, scale_dist.mode], dim=-1
    ).unsqueeze(0)  # [1, n_vars, 2]


def log_prob(
    y_var: Tensor,
    var_params: Tensor,
    var_batch: Optional[Tensor] = None,
    scale_sigma: float = 0.1,
) -> Tensor:
    """Log-probability of `var_params` under the policy defined by `y_var`.

    var_params: [num_samples, n_vars, 2]

    Returns:
        If var_batch is provided: [num_samples, n_graphs] (summed per graph)
        Otherwise:                [num_samples, n_vars, 2]
    """
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    phase_lp = phase_dist.log_prob(var_params[:, :, 0])  # [num_samples, n_vars]
    scale_lp = scale_dist.log_prob(var_params[:, :, 1])  # [num_samples, n_vars]

    if var_batch is not None:
        lp = phase_lp + scale_lp  # [num_samples, n_vars]
        return scatter_sum(lp, var_batch, dim=1)  # [num_samples, n_graphs]
    return torch.stack([phase_lp, scale_lp], dim=-1)


def kl_div(
    y_var: Tensor,
    y_var_ref: Tensor,
    var_batch: Optional[Tensor] = None,
    scale_sigma: float = 0.1,
) -> Tensor:
    """KL divergence KL(current || reference) per graph (or summed if no batch)."""
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    phase_ref, scale_ref = distributions(y_var_ref, scale_sigma=scale_sigma)

    kl = (
        torch.distributions.kl_divergence(phase_dist, phase_ref)
        + torch.distributions.kl_divergence(scale_dist, scale_ref)
    )

    if var_batch is None:
        return kl.sum()
    return scatter_sum(kl, var_batch, dim=0)  # [n_graphs]


def entropy(
    y_var: Tensor,
    var_batch: Optional[Tensor] = None,
    scale_sigma: float = 0.1,
) -> Tensor:
    """Mean entropy of the policy across graphs."""
    phase_dist, scale_dist = distributions(y_var, scale_sigma=scale_sigma)
    ent = phase_dist.entropy() + scale_dist.entropy()
    if var_batch is None:
        return ent.sum()
    return scatter_sum(ent, var_batch, dim=0).mean()

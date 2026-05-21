"""
ArieNet variants adapted for RLAF (Reinforcement Learning with Advantages and
Feedback) training.

These models use the BPG (Bipartite Problem Graph) format from nsnet — including
local satisfaction percentages — and output per-variable parameters (rho, mu)
for the RLAF policy:
    rho  : logit for a Binomial phase distribution (which truth value to prefer)
    mu   : mean of a LogNormal weight distribution (how much to weight the var)

No softmax is applied.  The output layer is zero-initialised so the initial
policy is near-uniform (rho ≈ 0  →  50/50 phase; mu ≈ 0  →  unit weight).

Two model variants are provided:
    ArieNetRLAF        — base model, standard BPG message passing
    ArieNetRLAFCooc    — extends base with literal co-occurrence (L2L) edges
"""

import math
import torch
import torch.nn as nn

from nsnet.models.mlp import MLP
from nsnet.utils.torch_utils import scatter_sum, scatter_logsumexp, swap_even_odd


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _zero_init_last_linear(module: nn.Module) -> None:
    """Zero-initialise the weights and bias of the last nn.Linear in `module`."""
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------

class ArieNetRLAF(nn.Module):
    """
    BPG message-passing network for RLAF policy training.

    Architecture mirrors ArieNetBackbone (local satisfaction percentages, c2l /
    l2c alternating messages) but outputs raw (rho, mu) logits for each variable
    rather than backbone-class scores.

    Forward output shape: [total_vars_in_batch, 2]
        column 0 – rho : phase logit  (positive-literal readout)
        column 1 – mu  : log-normal mean (negative-literal readout)
    """

    def __init__(
        self,
        dim: int = 128,
        n_rounds: int = 26,
        n_mlp_layers: int = 3,
        activation: str = "relu",
        no_precomputed_local_sat: bool = False,
        use_up_features: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_rounds = n_rounds
        self.no_precomputed_local_sat = no_precomputed_local_sat
        self.use_up_features = use_up_features
        self.denom = math.sqrt(dim)

        # Learnable initial edge embeddings (small init for stability)
        self.c2l_edges_init = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.l2c_edges_init = nn.Parameter(torch.randn(1, dim) * 0.01)

        # Message-update MLPs (same structure as ArieNetBackbone)
        self.c2l_msg_update = MLP(n_mlp_layers, dim + 1, dim, dim, activation)
        self.l2c_msg_update = MLP(n_mlp_layers, dim, dim, dim, activation)
        # input: [l2c_msg | l2c_negated | local_sat | neg_local_sat]
        self.l2c_msg_norm = MLP(n_mlp_layers, dim * 2 + 2, dim, dim, activation)

        # Optional UP-feature encoder (4 scalars → dim)
        if use_up_features:
            self.up_feat_embed = MLP(2, 4, dim, dim, activation)

        # Readout: aggregate edge features → literal logits → reshape [n_vars, 2]
        self.l_readout = MLP(n_mlp_layers, dim, dim, 1, activation)
        # Zero-init output layer for a stable initial (near-uniform) policy
        _zero_init_last_linear(self.l_readout)

    # ------------------------------------------------------------------ #

    def _get_local_sat(self, data, n_edges: int, dtype, device) -> torch.Tensor:
        """Return local satisfaction percentages as shape [n_edges, 1]."""
        raw = getattr(data, "local_satisfaction_percentage_per_edge", None)
        if self.no_precomputed_local_sat or raw is None:
            return torch.zeros((n_edges, 1), dtype=dtype, device=device)
        return raw.unsqueeze(1) if raw.dim() == 1 else raw

    def _n_lit(self, data) -> int:
        """Total number of literals in the (possibly batched) data object."""
        nl = data.n_literals
        if isinstance(nl, torch.Tensor):
            return int(nl.sum().item())
        return int(nl)

    # ------------------------------------------------------------------ #

    def forward(self, data):
        device = data.literal_indices_per_edge.device
        # data.n_edges is a property: len(literal_indices_per_edge) — works after batching
        n_edges = data.n_edges
        n_lit = self._n_lit(data)

        # Initialise edge embeddings
        c2l = (self.c2l_edges_init / self.denom).repeat(n_edges, 1)
        l2c = (self.l2c_edges_init / self.denom).repeat(n_edges, 1)

        # Optional UP-feature injection into initial edge representations
        if self.use_up_features:
            up_raw = getattr(data, "up_features_per_literal", None)
            if up_raw is not None:
                up_emb = self.up_feat_embed(up_raw[data.literal_indices_per_edge])
                c2l = c2l + up_emb
                l2c = l2c + up_emb

        # Message-passing rounds
        for _ in range(self.n_rounds):
            # ---- Clause → Literal ------------------------------------------------
            l2c_arg = scatter_sum(
                data.c2l_msg_receiver_indices,
                c2l[data.c2l_msg_sender_indices],
                n_edges,
                device,
            )
            l2c_msg = self.l2c_msg_update(l2c_arg)

            local_sat = self._get_local_sat(data, n_edges, l2c_msg.dtype, device)
            neg_sat = swap_even_odd(local_sat)
            l2c_neg = swap_even_odd(l2c_msg)
            l2c = self.l2c_msg_norm(
                torch.cat([l2c_msg, l2c_neg, local_sat, neg_sat], dim=1)
            )

            # ---- Literal → Clause ------------------------------------------------
            per_assign = scatter_sum(
                data.l2c_assignment_indices,
                l2c[data.l2c_assignment_neighborhoods],
                len(data.l2c_msg_receiver_indices),
                device,
            )
            c2l_arg = scatter_logsumexp(
                data.l2c_msg_receiver_indices,
                per_assign,
                n_edges,
                device,
            )
            c2l = self.c2l_msg_update(torch.cat([c2l_arg, local_sat], dim=1))

        # Readout: edge features → literal features → [n_vars, 2]
        lit_feats = scatter_sum(data.literal_indices_per_edge, c2l, n_lit, device)
        lit_logits = self.l_readout(lit_feats)  # [n_lit, 1]
        return lit_logits.reshape(-1, 2)         # [n_vars, 2]


# ---------------------------------------------------------------------------
# Co-occurrence extension
# ---------------------------------------------------------------------------

class ArieNetRLAFCooc(ArieNetRLAF):
    """
    ArieNetRLAF extended with a literal co-occurrence (L2L) message-passing step.

    Co-occurrence edges connect pairs of literals that appear together in at
    least one clause.  Their aggregated signal is injected into the c2l edge
    features each round, giving the model direct access to literal co-occurrence
    structure.

    The BPG data must contain `cooc_src_indices` and `cooc_dst_indices`
    (populated by RLAFBPGDataset when use_cooc=True).
    """

    def __init__(
        self,
        dim: int = 128,
        n_rounds: int = 26,
        n_mlp_layers: int = 3,
        activation: str = "relu",
        no_precomputed_local_sat: bool = False,
        use_up_features: bool = False,
    ):
        super().__init__(
            dim=dim,
            n_rounds=n_rounds,
            n_mlp_layers=n_mlp_layers,
            activation=activation,
            no_precomputed_local_sat=no_precomputed_local_sat,
            use_up_features=use_up_features,
        )
        # Separate learnable start vector and update MLP for co-occurrence edges
        self.cooc_edges_init = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.l2l_update = MLP(n_mlp_layers, dim, dim, dim, activation)

    def forward(self, data):
        device = data.literal_indices_per_edge.device
        n_edges = data.n_edges
        n_lit = self._n_lit(data)

        c2l = (self.c2l_edges_init / self.denom).repeat(n_edges, 1)
        l2c = (self.l2c_edges_init / self.denom).repeat(n_edges, 1)

        # Co-occurrence edge features (one per directed literal pair)
        cooc_src = getattr(data, "cooc_src_indices", None)
        cooc_dst = getattr(data, "cooc_dst_indices", None)
        if cooc_src is not None and len(cooc_src) > 0:
            n_cooc = len(cooc_src)
            cooc_feats = (self.cooc_edges_init / self.denom).repeat(n_cooc, 1)
        else:
            cooc_feats = None

        # Optional UP-feature injection
        if self.use_up_features:
            up_raw = getattr(data, "up_features_per_literal", None)
            if up_raw is not None:
                up_emb = self.up_feat_embed(up_raw[data.literal_indices_per_edge])
                c2l = c2l + up_emb
                l2c = l2c + up_emb

        for _ in range(self.n_rounds):
            # ---- Clause → Literal ------------------------------------------------
            l2c_arg = scatter_sum(
                data.c2l_msg_receiver_indices,
                c2l[data.c2l_msg_sender_indices],
                n_edges,
                device,
            )
            l2c_msg = self.l2c_msg_update(l2c_arg)

            local_sat = self._get_local_sat(data, n_edges, l2c_msg.dtype, device)
            neg_sat = swap_even_odd(local_sat)
            l2c_neg = swap_even_odd(l2c_msg)
            l2c = self.l2c_msg_norm(
                torch.cat([l2c_msg, l2c_neg, local_sat, neg_sat], dim=1)
            )

            # ---- Literal → Clause ------------------------------------------------
            per_assign = scatter_sum(
                data.l2c_assignment_indices,
                l2c[data.l2c_assignment_neighborhoods],
                len(data.l2c_msg_receiver_indices),
                device,
            )
            c2l_arg = scatter_logsumexp(
                data.l2c_msg_receiver_indices,
                per_assign,
                n_edges,
                device,
            )
            c2l = self.c2l_msg_update(torch.cat([c2l_arg, local_sat], dim=1))

            # ---- L2L: co-occurrence step -----------------------------------------
            # For each literal l, aggregate cooc edge features from all edges
            # where cooc_dst == l, then inject into c2l edges incident to l.
            if cooc_feats is not None and cooc_dst is not None:
                cooc_agg = scatter_sum(cooc_dst, cooc_feats, n_lit, device)
                c2l = c2l + self.l2l_update(cooc_agg[data.literal_indices_per_edge])

        lit_feats = scatter_sum(data.literal_indices_per_edge, c2l, n_lit, device)
        lit_logits = self.l_readout(lit_feats)
        return lit_logits.reshape(-1, 2)

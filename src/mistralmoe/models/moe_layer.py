"""The MoE FFN replacement layer and its auxiliary load-balancing loss.

Ported from moe_complete.ipynb. Two versions of `MoELayer` existed in the
notebook (lines 2139 and 6182): the second is a deliberate fix, not a
copy-paste duplicate — it generalizes router-bias initialization to respect
arbitrary `num_experts_per_tok` and `router_jitter_noise` (needed for the
top1_8x1, efficient_4x1, and routing_noisy_8x2 variants), where the first
version hardcoded a top-2/8-expert bias pattern. This module keeps only the
second (canonical) version. `compute_moe_auxiliary_loss`/`compute_moe_loss`
are from lines 2515-2574.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """Mixture of Experts layer with configurable routing.

    Supports different routing strategies, expert counts, and placements.
    Each expert is a 3-linear-layer FFN (gate_proj -> SiLU, up_proj, down_proj)
    matching Mistral's FFN structure, intended to be initialized from a dense
    FFN's weights via `replace_ffn_with_moe` (sparse upcycling).
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts=8,
        num_experts_per_tok=2,
        router_jitter_noise=0.0,
        router_aux_loss_coef=0.001,
        bnb_config=None,
        device="cuda",
        init_on_cpu=True,
        dtype=torch.bfloat16,
        enable_cpu_offload=True,
        layer_index=None,
        total_layers=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_jitter_noise = router_jitter_noise
        self.router_aux_loss_coef = router_aux_loss_coef
        self.bnb_config = bnb_config
        self.device = device
        self.dtype = dtype
        self.enable_cpu_offload = False
        self.layer_index = layer_index  # Track which layer this is
        self.total_layers = total_layers  # Total number of layers

        LinearClass = nn.Linear
        self.compute_dtype = dtype
        init_device = "cpu" if init_on_cpu else device
        linear_kwargs = {"device": init_device, "dtype": dtype}

        self.gate_proj = nn.ModuleList(
            [LinearClass(hidden_size, intermediate_size, bias=False, **linear_kwargs) for _ in range(num_experts)]
        )
        self.up_proj = nn.ModuleList(
            [LinearClass(hidden_size, intermediate_size, bias=False, **linear_kwargs) for _ in range(num_experts)]
        )
        self.down_proj = nn.ModuleList(
            [LinearClass(intermediate_size, hidden_size, bias=False, **linear_kwargs) for _ in range(num_experts)]
        )

        # Router: maps hidden state to expert logits.
        # Adaptive initialization that respects config parameters.
        self.router = nn.Linear(hidden_size, num_experts, bias=True, device=device, dtype=dtype)
        with torch.no_grad():
            # Zero out weights so logits don't depend on input content initially.
            self.router.weight.zero_()

            # Respect num_experts_per_tok from config: top-1 configs only use
            # expert 0, top-2 use experts 0 & 1, etc.
            k = min(num_experts_per_tok, num_experts)

            # Top-k experts get positive bias (decreasing values for proper
            # ordering); expert 0 gets highest bias, expert 1 second, etc.
            for i in range(k):
                self.router.bias[i] = 10.0 - (i * 0.5)

            # Remaining experts get negative bias to ensure they're not selected.
            if num_experts > k:
                self.router.bias[k:] = -10.0

            # Add initialization noise if router_jitter_noise is specified.
            # Creates slight variation for routing strategies that use noise
            # (e.g. routing_noisy_8x2).
            if router_jitter_noise > 0:
                noise_scale = router_jitter_noise * 0.1
                self.router.bias += torch.randn(num_experts, device=device, dtype=dtype) * noise_scale

        self._last_router_probs = None
        self._collect_router_logits = False
        self._experts_on_gpu = set()

    def forward(self, hidden_states):
        """Vectorized forward pass (no per-token Python loop)."""
        original_dtype = hidden_states.dtype
        hidden_states_reshaped = hidden_states.view(-1, hidden_states.size(-1))  # [B*S, H]

        router_input = hidden_states_reshaped.to(self.router.weight.dtype)
        router_logits = self.router(router_input)  # [B*S, num_experts]

        if self.training and self.router_jitter_noise > 0:
            router_logits = router_logits + torch.normal(
                0, self.router_jitter_noise, size=router_logits.shape, device=router_logits.device
            )

        router_probs = torch.softmax(router_logits, dim=-1)  # [B*S, num_experts]
        top_k_probs, top_k_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)  # [B*S, k]
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        if self.training:
            self._last_router_probs = router_probs.view(hidden_states.size(0), hidden_states.size(1), -1)

        output = torch.zeros_like(hidden_states_reshaped)

        for expert_idx in range(self.num_experts):
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [B*S]
            if expert_mask.sum() == 0:
                continue

            expert_selected_mask = top_k_indices == expert_idx  # [B*S, k]
            expert_weights = (top_k_probs * expert_selected_mask.float()).sum(dim=-1)  # [B*S]

            expert_hidden = hidden_states_reshaped[expert_mask]
            if self.compute_dtype is not None and expert_hidden.dtype != self.compute_dtype:
                expert_hidden = expert_hidden.to(dtype=self.compute_dtype)

            gate_out = F.silu(self.gate_proj[expert_idx](expert_hidden))
            up_out = self.up_proj[expert_idx](expert_hidden)
            expert_out = gate_out * up_out
            expert_out = self.down_proj[expert_idx](expert_out)
            expert_out = expert_out.to(dtype=output.dtype)

            output[expert_mask] += expert_weights[expert_mask].unsqueeze(-1) * expert_out

        output = output.view_as(hidden_states)
        return output.to(original_dtype)

    def compute_auxiliary_loss(self):
        """Load-balancing auxiliary loss for this layer, from the last forward pass."""
        if self._last_router_probs is None:
            return torch.tensor(0.0, device=self.router.weight.device)
        expert_freq = self._last_router_probs.mean(dim=[0, 1])
        router_confidence = self._last_router_probs.mean(dim=[0, 1])
        aux_loss = torch.sum(expert_freq * router_confidence) * self.num_experts
        return aux_loss


def compute_moe_auxiliary_loss(model, router_aux_loss_coef: float = 0.001):
    """Sum the load-balancing auxiliary loss across all MoE layers in a model.

    L_aux = sum(D_i * P_i) where D_i is expert frequency and P_i is router
    confidence, summed over every replaced FFN layer.
    """
    total_aux_loss = torch.tensor(0.0, device=next(model.parameters()).device)

    for layer in model.model.layers:
        if hasattr(layer.mlp, "_last_router_probs") and layer.mlp._last_router_probs is not None:
            router_probs = layer.mlp._last_router_probs
            expert_freq = router_probs.mean(dim=0)
            router_confidence = router_probs.mean(dim=0)
            layer_aux_loss = torch.sum(expert_freq * router_confidence) * layer.mlp.num_experts
            total_aux_loss = total_aux_loss + layer_aux_loss

    return total_aux_loss


def compute_moe_loss(model, outputs, router_aux_loss_coef: float = 0.001):
    """Total MoE loss: L_total = L_NTP + lambda * L_aux.

    Returns (total_loss, ntp_loss, aux_loss).
    """
    ntp_loss = outputs.loss if hasattr(outputs, "loss") else None
    aux_loss = compute_moe_auxiliary_loss(model, router_aux_loss_coef)

    if ntp_loss is not None:
        total_loss = ntp_loss + router_aux_loss_coef * aux_loss
    else:
        total_loss = router_aux_loss_coef * aux_loss

    return total_loss, ntp_loss, aux_loss

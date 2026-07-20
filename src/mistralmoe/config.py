"""Environment setup and all static configuration for the MistralMoE pipeline.

Ported from moe_complete.ipynb cells: env setup (lines 1-24), MAX_LENGTH (258),
dense KD configs (1618-1647), MoE baseline constants (2588-2590), MoE training/
LoRA/KD configs (3122-3157), and the variant configuration system (6347-6503).

The notebook is left untouched; this module is a faithful, de-globalized
extraction so these values can be imported instead of relying on notebook
global state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Literal, Optional


def configure_environment() -> None:
    """Set the process env vars the notebook sets before importing torch/transformers."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Model / data constants
# ---------------------------------------------------------------------------

MODEL_ID = "mistralai/Mistral-7B-v0.1"
MAX_LENGTH = 512

ANSWER_LETTERS = ("A", "B", "C", "D")
ANSWER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def build_answer_tokens(tokenizer) -> dict:
    """Recreate the notebook's ANSWER_TOKENS dict (cell 12) for a given tokenizer."""
    return {
        letter: tokenizer.encode(letter, add_special_tokens=False)[0]
        for letter in ANSWER_LETTERS
    }


# ---------------------------------------------------------------------------
# MoE baseline architecture constants (notebook lines 2588-2590)
# ---------------------------------------------------------------------------

NUM_EXPERTS = 8  # Full 8 experts like Mixtral
NUM_EXPERTS_PER_TOK = 2  # Top-2 routing like Mixtral
ROUTER_JITTER_NOISE = 0.0

# ---------------------------------------------------------------------------
# LoRA configs
# ---------------------------------------------------------------------------

# Applied to the dense student model in the Dense+KD control experiment
# (notebook lines 1635-1641). Target modules cover attention + FFN.
LORA_CONFIG_DENSE = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

# Applied to MoE models (attention + router only; expert FFN weights stay
# frozen so sparse-upcycled knowledge is preserved). Notebook lines 3140-3144.
LORA_CONFIG_MOE = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}

# ---------------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------------

TRAINING_CONFIG_DENSE = {
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "num_train_epochs": 1,
    "logging_steps": 25,
    "save_steps": 1000,
    "max_steps": 250,
    "fp16": True,
    "bf16": False,
    "save_total_limit": 2,
}

TRAINING_CONFIG_MOE = {
    "num_experts": NUM_EXPERTS,
    "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
    "router_jitter_noise": ROUTER_JITTER_NOISE,
    "router_aux_loss_coef": 0.01,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "num_train_epochs": 1,
    "logging_steps": 25,
    "eval_steps": 50,
    "save_steps": 100,
    "max_steps": 250,
    "max_length": MAX_LENGTH,
}

# ---------------------------------------------------------------------------
# Knowledge Distillation configs
# ---------------------------------------------------------------------------

KD_CONFIG_DENSE = {
    "kd_alpha": 0.5,
    "temperature": 4.0,
    "name": "Standard KD",
}

# Output-only distillation (stable). Notebook lines 1454-1466.
KD_CONFIG_STANDARD = {
    "kd_alpha": 0.5,
    "temperature": 4.0,
    "routing_kd_weight": 0.0,
    "expert_spec_weight": 0.0,
    "enable_routing_kd": False,
    "enable_ka": False,
    "enable_sar": False,
    "enable_non_activated": False,
    "router_aux_loss_coef": 0.001,
    "name": "Standard KD",
}

# Output + light router constraints (MoE-stable). Notebook lines 1468-1480.
KD_CONFIG_ROUTER_STABLE = {
    "kd_alpha": 0.6,
    "temperature": 5.0,
    "routing_kd_weight": 0.1,
    "expert_spec_weight": 0.0,
    "enable_routing_kd": True,
    "enable_ka": False,
    "enable_sar": False,
    "enable_non_activated": False,
    "router_aux_loss_coef": 0.01,
    "name": "Router-Stable KD",
}

# Used by the MoE variant training driver (notebook lines 3147-3154).
KD_CONFIG_MOE_DEFAULT = {
    "kd_alpha": 0.5,
    "temperature": 4.0,
    "routing_kd_weight": 0.0,
    "enable_routing_kd": False,
    "router_aux_loss_coef": TRAINING_CONFIG_MOE["router_aux_loss_coef"],
    "name": "Standard KD",
}


# ---------------------------------------------------------------------------
# MoE variant configuration system (notebook lines 6347-6503)
# ---------------------------------------------------------------------------


@dataclass
class MoEExperimentConfig:
    """Configuration for a single MoE architectural variant."""

    num_experts: int = 8
    num_experts_per_tok: int = 2

    router_jitter_noise: float = 0.0
    router_aux_loss_coef: float = 0.001

    expert_layers: Literal["all", "every_2", "every_4", "selected"] = "all"
    layer_indices: Optional[List[int]] = None

    load_balancing_loss_coef: float = 0.01

    experiment_name: str = "default"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "router_jitter_noise": self.router_jitter_noise,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "expert_layers": self.expert_layers,
            "layer_indices": self.layer_indices,
            "load_balancing_loss_coef": self.load_balancing_loss_coef,
            "experiment_name": self.experiment_name,
            "description": self.description,
        }

    def save(self, filepath) -> None:
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath) -> "MoEExperimentConfig":
        with open(filepath, "r") as f:
            return cls(**json.load(f))


EXPERIMENT_CONFIGS: dict[str, MoEExperimentConfig] = {
    "moe_baseline": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=2,
        router_jitter_noise=0.0,
        router_aux_loss_coef=0.001,
        expert_layers="all",
        experiment_name="moe_baseline",
        description="Baseline MoE: 8 experts, top-2 routing, all layers, standard config (matches baseline setup)",
    ),
    # --- 1. Routing Variants ---
    "top1_8x1": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=1,
        expert_layers="all",
        experiment_name="top1_8x1",
        description="Top-1 routing: 8 experts, single expert per token",
    ),
    "top1_16x1": MoEExperimentConfig(
        num_experts=16,
        num_experts_per_tok=1,
        expert_layers="all",
        experiment_name="top1_16x1",
        description="Top-1 routing: 16 experts, single expert per token",
    ),
    "routing_noisy_8x2": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=2,
        router_jitter_noise=0.2,
        expert_layers="all",
        experiment_name="routing_noisy_8x2",
        description="Noisy routing: 8 experts, top-2, high jitter (0.2)",
    ),
    "balanced_8x2": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.05,  # Higher coefficient for strict balancing
        expert_layers="all",
        experiment_name="balanced_8x2",
        description="Load Balanced: 8 experts, top-2, high aux loss coef (0.05)",
    ),
    # --- 2. Expert Count Variants ---
    "efficient_4x1": MoEExperimentConfig(
        num_experts=4,
        num_experts_per_tok=1,
        expert_layers="all",
        experiment_name="efficient_4x1",
        description="Efficient: 4 experts, top-1 routing",
    ),
    "large_16x2": MoEExperimentConfig(
        num_experts=16,
        num_experts_per_tok=2,
        expert_layers="all",
        experiment_name="large_16x2",
        description="Large: 16 experts, top-2 routing",
    ),
    # --- 3. Placement Variants ---
    "sparse_8x2": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=2,
        expert_layers="every_2",
        experiment_name="sparse_8x2",
        description="Sparse placement: experts every 2nd layer",
    ),
    "placement_early_8x2": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=2,
        expert_layers="selected",
        layer_indices=list(range(0, 16)),  # First 16 layers
        experiment_name="placement_early_8x2",
        description="Early placement: Experts in first 16 layers only",
    ),
    "placement_middle_8x2": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=2,
        expert_layers="selected",
        layer_indices=list(range(8, 24)),  # Middle 16 layers
        experiment_name="placement_middle_8x2",
        description="Middle placement: Experts in middle 16 layers (8-23)",
    ),
    "placement_late_8x2": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=2,
        expert_layers="selected",
        layer_indices=list(range(16, 32)),  # Last 16 layers
        experiment_name="placement_late_8x2",
        description="Late placement: Experts in last 16 layers only",
    ),
    "placement_mixed_8x2": MoEExperimentConfig(
        num_experts=8,
        num_experts_per_tok=2,
        expert_layers="selected",
        layer_indices=[0, 1, 2, 3, 14, 15, 16, 17, 28, 29, 30, 31],  # First 4, Middle 4, Last 4
        experiment_name="placement_mixed_8x2",
        description="Mixed placement: Experts in first 4, middle 4, and last 4 layers",
    ),
}

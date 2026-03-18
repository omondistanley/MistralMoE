# MistralMoE: Sparse Upcycling & Knowledge Distillation for Mixture of Experts

**Link to paper:** [Knowledge Distillation for Scalable Sparse Upcycled Mixture of Experts](https://github.com/omondistanley/MistralMoE/blob/master/Mixtral%20Final%20Papaer.pdf)

A research framework for converting **Mistral-7B-v0.1** into Mixture of Experts (MoE) architectures and training them with **Knowledge Distillation (KD)**. Experiments are benchmarked on the MMLU dataset across 10+ architectural variants, measuring accuracy, FLOPs, throughput, and memory efficiency.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Experiment Variants](#experiment-variants)
- [Results](#results)
- [Training Pipeline](#training-pipeline)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [References](#references)

---

## Overview

Dense LLMs activate all parameters for every token — expensive and inefficient at inference time. This project explores **sparse upcycling**: replacing the Feed-Forward Network (FFN) layers in Mistral-7B with MoE modules, where each token only activates a subset of "expert" sub-networks.

**Key research questions:**
- How much FLOPs can be saved with sparse activation, and at what accuracy cost?
- Does Knowledge Distillation from the dense teacher improve MoE accuracy?
- Where in the network should experts be placed (early, middle, late, distributed)?
- How do routing strategy and expert count affect performance?

**Key finding:** MoE reduces FLOPs by ~66% (8.8T → 3.0T) while maintaining or improving MMLU accuracy with fine-tuning.

---

## Architecture

### Sparse Upcycling

Mistral-7B's 32 transformer layers each contain a dense FFN. This project replaces selected layers with `MoELayer` modules using weights copied directly from the original FFN (sparse upcycling — no training from scratch).

```
Input Hidden State
       │
       ▼
  ┌─────────────┐
  │   Router    │  Linear → softmax → top-k selection
  └──────┬──────┘
         │  selects k experts
    ┌────┴────┐
    ▼         ▼
 Expert 1  Expert k    (each: gate_proj → SiLU → up_proj → down_proj)
    │         │
    └────┬────┘
         │  weighted sum
         ▼
  Output Hidden State
```

### MoELayer Design

| Component | Detail |
|---|---|
| Experts | 4–16 per layer (configurable) |
| Expert structure | 3-layer FFN: `gate_proj`, `up_proj`, `down_proj` with SiLU activation |
| Router | Linear → softmax → top-k |
| Routing strategies | Top-1, Top-2, Noisy Top-2 |
| Load balancing | Auxiliary loss (coefficient: 0.001–0.05) |
| Initialization | All experts start as copies of the original dense FFN weights |
| Quantization | 4-bit (bitsandbytes) — reduces ~116 GB (bf16) to ~25 GB |

### Training Loss

```
L_total = (1 - α) · L_NTP  +  α · L_KD  +  λ · L_aux  +  β · L_routing_KD
```

| Term | Description | Default |
|---|---|---|
| `L_NTP` | Next-token prediction (standard LM loss) | — |
| `L_KD` | KL divergence from frozen dense teacher logits | α = 0.5, T = 4.0 |
| `L_aux` | Load balancing loss (prevents expert collapse) | λ = 0.001 |
| `L_routing_KD` | Entropy regularization on router outputs | optional |

LoRA adapters (r=16, alpha=32) are applied to attention and router layers for parameter-efficient fine-tuning (~13.6M trainable params).

---

## Experiment Variants

Ten architectural configurations are studied, each run through 3 phases: pre-training eval → standard fine-tuning → KD fine-tuning.

| Variant | Experts | Top-k | Layer Placement | Focus |
|---|---|---|---|---|
| `moe_baseline` | 8 | 2 | All 32 layers | Reference MoE |
| `efficient_4x1` | 4 | 1 | All 32 layers | Minimal overhead |
| `top1_8x1` | 8 | 1 | All 32 layers | Sparsest routing |
| `large_16x2` | 16 | 2 | All 32 layers | Maximum capacity |
| `balanced_8x2` | 8 | 2 | All 32 layers | Strict load balancing (λ=0.05) |
| `routing_noisy_8x2` | 8 | 2 | All 32 layers | Jitter noise (0.2) for exploration |
| `sparse_8x2` | 8 | 2 | Every 2nd layer (16 layers) | Sparse placement |
| `placement_early_8x2` | 8 | 2 | Layers 0–15 | Early-layer experts |
| `placement_middle_8x2` | 8 | 2 | Layers 8–23 | Mid-network experts |
| `placement_late_8x2` | 8 | 2 | Layers 16–31 | Late-layer experts |
| `placement_mixed_8x2` | 8 | 2 | Layers 0–3, 14–17, 28–31 | Distributed placement |

---

## Results

### Baseline Comparison

| Model | MMLU Accuracy | Top-2 Accuracy | FLOPs | Throughput | GPU Memory |
|---|---|---|---|---|---|
| Dense baseline | 66.4% | 83.1% | 8,796 G | ~33,175 tok/s | 13.5 GB |
| MoE baseline (untrained) | ~66.7% | — | ~3,020 G | — | ~25 GB (4-bit) |
| MoE baseline + standard FT | ~69–70% | ~86–87% | ~3,020 G | ~5,000–7,000 tok/s | — |
| MoE baseline + KD | ~69–70% | ~86–87% | ~3,020 G | ~5,000–7,000 tok/s | — |

**FLOPs reduction: ~66%** compared to the dense baseline.

### Variant Highlights

| Variant | Training | MMLU Accuracy | Notes |
|---|---|---|---|
| `placement_mixed_8x2` | KD | 68.9% | Best KD variant |
| `balanced_8x2` | KD | 68.0% | Strong load balancing |
| `efficient_4x1` | Standard | ~68% | Best efficiency/accuracy tradeoff |
| `placement_early_8x2` | Standard | ~68% | Effective, lower FLOPs |

---

## Training Pipeline

### Three-Phase Workflow

```
Phase 0: Create MoE model → evaluate on MMLU (no training)
           ↓
Phase 1: Apply LoRA → standard fine-tuning (NTP + L_aux) → evaluate
           ↓
Phase 2: Fresh model → Apply LoRA → KD fine-tuning (NTP + KD + L_aux) → evaluate
```

All three phases are run per variant by default (`train_both_modes=True`).

### Training Configuration

| Parameter | Value |
|---|---|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate | 2e-4 |
| Training steps | 250 |
| Batch size | 4 |
| Gradient accumulation | 4 steps |
| Max sequence length | 512 |
| Warmup ratio | 0.1 |

### Evaluation Metrics

- **Accuracy:** Top-1 and Top-2 on MMLU (1000 samples per experiment)
- **Calibration:** Expected Calibration Error (ECE)
- **Efficiency:** FLOPs, tokens/sec, ms/token, samples/sec
- **Parameters:** Total, active (per token), trainable, sparsity ratio
- **Memory:** Model size (MB), GPU allocated/reserved (GB)

---

## Project Structure

```
MistralMoE/
├── moe_complete.ipynb                        # Main end-to-end pipeline (66 cells)
├── Mixtral Final Paper.pdf                   # Reference: Mixtral architecture
├── MoE Final Paper.pdf                       # Reference: MoE research
│
├── results/
│   ├── baseline_comprehensive.json           # Dense model evaluation
│   ├── dense_kd_comprehensive.json           # Dense + KD evaluation
│   ├── moe_standard_comprehensive.json       # MoE baseline standard training
│   ├── moe_kd_comprehensive.json             # MoE baseline KD training
│   └── unified_model_comparison.json         # Consolidated cross-model comparison
│
├── experiments/
│   ├── {variant}_standard_results.json       # Per-variant standard training results
│   ├── {variant}_kd_results.json             # Per-variant KD results
│   ├── {variant}_unified_results.json        # All phases combined
│   ├── {variant}_standard_checkpoints/       # LoRA checkpoints (standard)
│   ├── {variant}_kd_checkpoints/             # LoRA checkpoints (KD)
│   └── all_variants_results.json             # Consolidated results
│
├── wandb/                                    # Weights & Biases experiment logs
└── stuff/                                    # Earlier iteration notebooks
    ├── MoEKD.ipynb
    ├── HPML.ipynb
    ├── MistralMOE.ipynb
    └── uhmm/
```

### Result File Schema

```json
{
  "experiment_name": "top1_8x1",
  "config": { "num_experts": 8, "top_k": 1, "..." : "..." },
  "phases": {
    "pre_training":      { "accuracy": 0.667, "flops": 3020, "..." : "..." },
    "standard_training": { "accuracy": 0.695, "accuracy_gain": 0.028, "..." : "..." },
    "kd_training":       { "accuracy": 0.681, "kd_config": { "..." : "..." }, "..." : "..." }
  },
  "timestamp": 1766002532.112663
}
```

---

## Setup & Usage

### Requirements

- Python 3.x
- CUDA 12.x (tested on 80 GB GPU; 4-bit quantization required for consumer GPUs)
- Kaggle account (for MMLU dataset download)

### Installation

```bash
pip install torch transformers peft datasets accelerate bitsandbytes
pip install wandb kagglehub scikit-learn pandas matplotlib seaborn
```

### Running Experiments

Open `moe_complete.ipynb` in Jupyter and run cells sequentially:

```python
# 1. Configure which variants to run
experiments_to_run = ["moe_baseline", "efficient_4x1", "placement_mixed_8x2"]

# 2. Enable variant experiments
RUN_VARIANT_EXPERIMENTS = True

# 3. Run the experiment loop
runner = MoEExperimentRunner(
    base_model=None,
    tokenizer=tokenizer,
    eval_dataset=eval_dataset,
    answer_tokens=ANSWER_TOKENS,
    train_dataset=train_dataset
)

for exp_name in experiments_to_run:
    config = EXPERIMENT_CONFIGS[exp_name]
    results = runner.run_experiment(
        config,
        max_samples=1000,
        train=True,
        train_steps=250,
        train_both_modes=True,
        kd_config=KD_CONFIG_STANDARD
    )

# 4. Compare results
runner.compare_experiments()
```

Results are automatically saved to `experiments/` as JSON files.

### WandB Integration

```python
import wandb
wandb.init(project="mistral-moe", name="your-experiment")
```

---

## References

1. **Jiang et al. (2024). "Mixtral of Experts."** [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
   Introduces the Mixtral architecture and top-k sparse routing strategy.

2. **He et al. (2024). "Upcycling Large Language Models into Mixture of Experts."** [arXiv:2410.07524](https://arxiv.org/abs/2410.07524)
   Guides the dense-to-MoE conversion methodology used in `_create_moe_model`.

3. **Komatsuzaki et al. (2023). "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints."**
   Informs the expert initialization strategy (identical weights copied from original FFN).

4. **Hinton et al. (2015). "Distilling the Knowledge in a Neural Network."** [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
   Foundational KD work; basis for the temperature-scaled KL divergence loss.

5. **MoE-KD (2024). "Knowledge Distillation for Mixture-of-Experts."** NLPCC 2024
   Guides integration of KD with MoE-specific losses in `IntegratedMoEKDTrainer`.

6. **Li et al. (2025). "Leave It to the Experts: Detecting Knowledge Distillation via MoE Expert Signatures."** [arXiv:2510.16968](https://arxiv.org/abs/2510.16968)
   Informs analysis of how KD affects expert routing patterns and specialization.

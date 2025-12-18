# MoE Experimentation README

## Overview

The `moe_complete.ipynb` notebook implements a comprehensive end-to-end workflow for Mixture of Experts (MoE) model experimentation on the MMLU dataset. It includes baseline evaluation, MoE model creation, standard training with LoRA, Knowledge Distillation (KD) training, and multi-phase experiment orchestration for comparing different MoE variants.

This implementation draws inspiration from recent advances in MoE architectures and knowledge distillation techniques, building upon foundational research in sparse model training and efficient neural network design.

## Implementation Structure

### Baseline Models

The notebook begins with establishing baseline models that serve as reference points and teachers for knowledge distillation.

#### 1. Dense Model Baseline

**Architecture:**

- Base model: Mistral-7B-v0.1 (7.24B parameters)
- Dense architecture: All parameters active for every token
- 4-bit quantization for memory efficiency
- No MoE routing - standard transformer FFN layers

**Evaluation:**

- Comprehensive evaluation on MMLU dataset (1000 samples)
- Metrics computed: accuracy, top-2 accuracy, ECE, FLOPs, throughput, parameters, memory usage
- Results stored in: `results/baseline_comprehensive.json`

**Results Summary:**

- MMLU Accuracy: ~0.664
- Top-2 Accuracy: ~0.831
- FLOPs: ~8796G per forward pass
- Throughput: ~33,175 tokens/second
- GPU Memory: ~13.52 GB allocated
- Model Size: ~13.8 GB

**Usage:**

The dense baseline serves as:

1. Reference point for comparing MoE variants
2. Teacher model for knowledge distillation experiments
3. Baseline for measuring efficiency gains from MoE architectures

#### 2. Dense Model with Knowledge Distillation

**Implementation:**

- Student model: Same dense architecture as baseline
- Teacher model: Dense baseline (frozen)
- Training: LoRA fine-tuning with KD loss
- KD Configuration: Standard KD (α=0.5, temperature=4.0)

**Training Process:**

- Applies LoRA adapters (r=16, alpha=32)
- Trains with combined loss: `L = (1-α)*L_NTP + α*L_KD`
- Next-token prediction loss + knowledge distillation loss
- Teacher model provides soft targets for student

**Evaluation:**

- Post-training evaluation on MMLU
- Comparison with dense baseline
- Results stored in: `results/dense_kd_comprehensive.json` (if available)

**Results Summary:**

- Typically shows similar or slightly lower accuracy than baseline
- Demonstrates KD effectiveness in dense-to-dense transfer
- Provides foundation for understanding KD in MoE context

#### 3. MoE Baseline Implementation

**Architecture:**

The MoE baseline uses a standard Mixtral-style architecture:

- **Base Model**: Mistral-7B-v0.1 converted to MoE
- **Expert Configuration**:
  - Number of experts: 8 per layer
  - Experts per token: 2 (top-2 routing)
  - Expert width: Full intermediate_size (14336 dimensions)
  - Total layers with MoE: All 32 layers

- **Router Design**:
  - Top-k routing with k=2
  - Router initialized with bias-based preference (experts 0 and 1 preferred initially)
  - Load balancing via auxiliary loss (coefficient: 0.001)
  - No jitter noise (router_jitter_noise=0.0)

- **Expert Structure**:

Each expert is a standard FFN with:

  - `gate_proj`: Linear(hidden_size → intermediate_size)
  - `up_proj`: Linear(hidden_size → intermediate_size)
  - `down_proj`: Linear(intermediate_size → hidden_size)
  - Activation: SiLU (Swish) on gate output, multiplied with up output

- **Weight Initialization**:
  - Experts initialized from dense FFN weights (sparse upcycling)
  - All experts start with identical weights copied from original dense layer
  - Router weights zeroed, biases set for initial expert preference

- **Memory Optimization**:
  - 4-bit quantization applied to expert weights
  - Reduces memory from ~116GB (bf16) to ~25GB (4-bit)
  - Enables training on 80GB GPUs

**Pre-Training Evaluation:**

- MoE model evaluated before any fine-tuning
- Results stored in: `experiments/moe_baseline_results.json` (if available)
- Typically shows accuracy similar to dense baseline (~0.664-0.67)
- Demonstrates that MoE architecture maintains performance with sparse activation

#### 4. MoE Baseline Standard Fine-Tuning

**Training Configuration:**

- LoRA adapters applied to attention and MoE layers
- LoRA rank: 16, alpha: 32, dropout: 0.05
- Training steps: 250 (configurable)
- Loss: Next-token prediction + auxiliary load balancing loss
- No knowledge distillation

**Training Process:**

1. Apply LoRA to model (trainable params: ~13.6M)
2. Train with standard trainer (`IntegratedMoETrainer`)
3. Monitor router statistics and expert utilization
4. Save checkpoints to: `experiments/moe_baseline_standard_checkpoints/`

**Post-Training Evaluation:**

- Comprehensive evaluation after training
- Results stored in: `experiments/moe_baseline_standard_results.json`
- Metrics include accuracy gain vs. pre-training baseline

**Results Summary:**

- Accuracy typically improves from ~0.67 to ~0.69-0.70
- Top-2 accuracy: ~0.86-0.87
- Throughput: ~5,000-7,000 tokens/second
- Demonstrates effectiveness of standard fine-tuning for MoE

#### 5. MoE Baseline Knowledge Distillation

**Training Configuration:**

- Teacher model: Dense baseline (frozen)
- Student model: MoE baseline with LoRA
- KD Configuration: Standard KD (α=0.5, temperature=4.0)
- Training steps: 250 (same as standard)

**Training Process:**

1. Apply LoRA to MoE model
2. Train with KD trainer (`IntegratedMoEKDTrainer`)
3. Loss: `L = (1-α)*L_NTP + α*L_KD + λ*L_aux`

   - Next-token prediction loss
   - Knowledge distillation loss (KL divergence on logits)
   - Auxiliary load balancing loss

4. Save checkpoints to: `experiments/moe_baseline_kd_checkpoints/`

**Post-Training Evaluation:**

- Comprehensive evaluation after KD training
- Results stored in: `experiments/moe_baseline_kd_results.json`
- Comparison with standard training and pre-training baselines

**Results Summary:**

- Accuracy typically: ~0.69-0.70
- Top-2 accuracy: ~0.86-0.87
- Accuracy gain vs. pre-training: ~0.02-0.03
- Throughput: ~5,000-7,000 tokens/second
- Demonstrates knowledge transfer from dense teacher to MoE student

**Key Insights:**

- KD helps MoE models learn from dense teacher's knowledge
- Router patterns may reflect knowledge transfer (as discussed in Li et al., 2025)
- Expert utilization patterns differ between standard and KD training

### Key Components (Variant Experiments)

1. **MoEExperimentRunner Class** (`MoEExperimentRunner`)

   - Main orchestrator for running variant experiments
   - Handles model creation, training, evaluation, and result storage
   - Manages memory cleanup between experiments
   - Located around line 8983 in the notebook

2. **Experiment Configuration** (`EXPERIMENT_CONFIGS`)

   - Dictionary of pre-defined experiment configurations
   - Each config specifies: number of experts, experts per token, layer placement, routing strategy
   - Examples: `efficient_4x1`, `top1_8x1`, `balanced_8x2`, `placement_early_8x2`, etc.
   - Located around line 8863

3. **Training Modes**

   - **Standard Training**: Uses `IntegratedMoETrainer` with next-token prediction loss
   - **Knowledge Distillation**: Uses `IntegratedMoEKDTrainer` with teacher-student distillation
   - Both use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning

4. **Evaluation Functions**

   - `evaluate_mmlu_comprehensive`: MMLU accuracy, top-2 accuracy, ECE (calibration)
   - `compute_model_flops`: Computational cost estimation
   - `compute_throughput_metrics`: Inference speed (tokens/sec, samples/sec)
   - `compute_parameter_efficiency`: Parameter counts and sparsity
   - `compute_memory_metrics`: Model size and GPU memory usage

## Experiment Flow

### Three-Phase Workflow

When `train_both_modes=True`, each experiment runs three phases:

#### Phase 0: Pre-Training Evaluation

- Creates MoE model according to configuration
- Evaluates on MMLU dataset (no training)
- Records baseline metrics: accuracy, throughput, FLOPs, memory usage
- Results stored in `phases.pre_training`

#### Phase 1: Standard Training

- Applies LoRA to the model
- Trains with standard next-token prediction loss
- Evaluates after training
- Computes accuracy gain vs. pre-training baseline
- Results stored in `phases.standard_training` and `{experiment_name}_standard`

#### Phase 2: Knowledge Distillation Training

- Creates fresh model (same config as Phase 0)
- Applies LoRA
- Trains with KD loss: `L_total = (1-α)*L_NTP + α*L_KD + λ*L_aux`
- Uses teacher model (dense baseline) for distillation
- Evaluates after training
- Results stored in `phases.kd_training` and `{experiment_name}_kd`

### Single-Mode Workflow

When `train_both_modes=False`, runs either standard or KD training (not both).

## Results Storage

### Directory Structure

```
results/
├── baseline_comprehensive.json              # Dense baseline evaluation
├── dense_kd_comprehensive.json              # Dense model with KD (if available)

experiments/
├── moe_baseline_results.json                # MoE baseline pre-training (if available)
├── moe_baseline_standard_results.json        # MoE baseline standard training
├── moe_baseline_kd_results.json             # MoE baseline KD training
├── moe_baseline_standard_checkpoints/       # Standard training checkpoints
├── moe_baseline_kd_checkpoints/             # KD training checkpoints
├── {experiment_name}_standard_results.json  # Variant standard training results
├── {experiment_name}_kd_results.json         # Variant KD training results
├── {experiment_name}_unified_results.json    # All phases combined
└── all_variants_results.json                 # Consolidated results from all experiments
```

### Result File Format

**Unified Results** (`{experiment_name}_unified_results.json`):

```json
{
  "experiment_name": "top1_8x1",
  "config": { /* experiment configuration */ },
  "phases": {
    "pre_training": { /* metrics */ },
    "standard_training": { /* metrics */ },
    "kd_training": { /* metrics */ }
  },
  "{experiment_name}_standard": { /* standard results */ },
  "{experiment_name}_kd": { /* KD results */ },
  "timestamp": 1766002532.112663
}
```

**Individual Results** (`{experiment_name}_standard_results.json` or `{experiment_name}_kd_results.json`):

- Contains metrics for a single training phase
- Includes: accuracy, top-2 accuracy, ECE, FLOPs, throughput, parameters, memory usage
- Includes training metadata: `training_mode`, `pre_train_accuracy`, `accuracy_gain`, `kd_config` (if KD)

### Metrics Recorded

Each result file contains:

**Accuracy Metrics:**

- `accuracy`: Top-1 MMLU accuracy
- `top2_accuracy`: Top-2 MMLU accuracy  
- `ece`: Expected Calibration Error

**Performance Metrics:**

- `flops`: Floating point operations per forward pass
- `tokens_per_second`: Inference throughput
- `ms_per_token`: Latency per token
- `samples_per_second`: Samples processed per second

**Parameter Metrics:**

- `total_params`: Total model parameters
- `active_params`: Parameters used per token (for MoE)
- `trainable_params`: LoRA trainable parameters
- `sparsity_ratio`: Model sparsity

**Memory Metrics:**

- `model_size_mb`: Model file size
- `gpu_memory_allocated_gb`: GPU memory allocated
- `gpu_memory_reserved_gb`: GPU memory reserved

**Training Metrics (if trained):**

- `pre_train_accuracy`: Baseline accuracy before training
- `accuracy_gain`: Improvement over baseline
- `kd_config`: Knowledge distillation configuration (if KD mode)

## How the Flow Works

### 1. Initialization

- Downloads MMLU dataset from Kaggle
- Loads base model (Mistral-7B-v0.1) with 4-bit quantization
- Creates dense baseline teacher model for KD
- Sets up tokenizer and answer token mappings

### 2. Baseline Establishment

- Evaluates dense baseline model
- Creates and evaluates MoE baseline (pre-training)
- Trains MoE baseline with standard fine-tuning
- Trains MoE baseline with knowledge distillation
- All baseline results saved for comparison

### 3. Variant Experiment Execution

```python
runner = MoEExperimentRunner(
    base_model=None,  # Loads fresh model per experiment
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
        train_both_modes=True,  # Runs all 3 phases
        kd_config=KD_CONFIG_STANDARD
    )
```

### 4. Model Creation (`_create_moe_model`)

- Loads base model with quantization
- Replaces specified FFN layers with MoE layers
- Configures router (top-k routing, load balancing)
- Handles layer placement strategies (early, middle, late, mixed, sparse)

### 5. Training (`_train_model`)

- Applies LoRA configuration
- Creates trainer (standard or KD)
- Runs training loop with evaluation callbacks
- Saves checkpoints to `experiments/{experiment_name}_{mode}_checkpoints/`

### 6. Evaluation (`_evaluate_comprehensive`)

- Runs MMLU evaluation on subset of test set
- Computes all metrics (accuracy, FLOPs, throughput, memory)
- Returns comprehensive results dictionary

### 7. Result Storage (`_save_results`)

- Saves individual results to JSON files
- Updates consolidated `all_variants_results.json`
- Creates unified results file combining all phases

### 8. Memory Management

- Aggressive cleanup between experiments
- Preserves teacher model across experiments
- Clears GPU cache and forces garbage collection

## Example Experiment Configurations

- **efficient_4x1**: 4 experts, top-1 routing (efficient)
- **top1_8x1**: 8 experts, top-1 routing
- **balanced_8x2**: 8 experts, top-2 routing (balanced)
- **large_16x2**: 16 experts, top-2 routing (large)
- **placement_early_8x2**: MoE in early layers only
- **placement_middle_8x2**: MoE in middle layers only
- **placement_late_8x2**: MoE in late layers only
- **placement_mixed_8x2**: MoE distributed across layers
- **sparse_8x2**: Sparse layer placement
- **routing_noisy_8x2**: Router with jitter noise

## Usage

1. Open `moe_complete.ipynb` in Jupyter
2. Run cells sequentially to set up environment
3. Baseline models are evaluated first (dense, MoE baseline)
4. Configure experiments in `experiments_to_run` list
5. Set `RUN_VARIANT_EXPERIMENTS = True`
6. Execute experiment loop
7. Results automatically saved to `experiments/` directory
8. Use `runner.compare_experiments()` to view comparison table

## Key Features

- **Comprehensive Metrics**: Tracks accuracy, efficiency, and resource usage
- **Multi-Phase Evaluation**: Pre-training, standard training, and KD training
- **Memory Efficient**: Aggressive cleanup enables running multiple experiments
- **Flexible Configuration**: Easy to add new experiment variants
- **Result Consolidation**: Automatic aggregation of all experiment results
- **WandB Integration**: Optional logging to Weights & Biases
- **TensorBoard Support**: Training metrics logged for visualization

## References

This implementation builds upon and extends the following foundational research:

### Mixture of Experts Architectures

1. **Jiang et al. (2024). "Mixtral of Experts."** arXiv:2401.04088

   - Introduces the Mixtral architecture, demonstrating effective sparse expert routing
   - Influences the top-k routing strategy and expert selection mechanisms in this implementation

2. **He et al. (2024). "Upcycling Large Language Models into Mixture of Experts."** arXiv:2410.07524 (NVIDIA)

   - Presents methods for converting dense LLMs into MoE architectures
   - Guides the model conversion approach used in `_create_moe_model` for replacing FFN layers with MoE layers

3. **Komatsuzaki et al. (2023). "Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints."**

   - Describes techniques for training MoE models from pre-trained dense model checkpoints
   - Informs the initialization strategy where experts start from identical weights copied from the original dense FFN

### Knowledge Distillation

4. **Hinton et al. (2015). "Distilling the Knowledge in a Neural Network."** arXiv:1503.02531

   - Foundational work on knowledge distillation, introducing the temperature scaling and soft target concepts
   - Provides the theoretical basis for the KD loss formulation: `L_KD = KL(softmax(S/T) || softmax(T/T)) * T²`

5. **MoE-KD (2024). "Knowledge Distillation for Mixture-of-Experts."** NLPCC 2024

   - Applies knowledge distillation specifically to MoE architectures
   - Guides the implementation of `IntegratedMoEKDTrainer` and the integration of KD with MoE-specific losses (auxiliary load balancing loss)

6. **Li et al. (2025). "Leave It to the Experts: Detecting Knowledge Distillation via MoE Expert Signatures."** arXiv:2510.16968

   - Authors: Pingzhi Li†, Morris Yu-Chao Huang†, Zhen Tan, Qingquan Song, Jie Peng, Kai Zou, Yu Cheng, Kaidi Xu, and Tianlong Chen
   - Affiliations: UNC-Chapel Hill, Arizona State University, NetMind.AI, The Chinese University of Hong Kong, City University of Hong Kong
   - Published: October 21, 2025
   - Analyzes how knowledge distillation affects MoE expert routing patterns and specialization behaviors
   - Informs the analysis of router statistics and expert utilization patterns in `collect_router_statistics` and `visualize_router_statistics` functions, helping interpret differences between standard and KD training results

### Implementation Connections

- **Router Design**: The top-k routing mechanism follows principles from Mixtral (Jiang et al., 2024), with load balancing inspired by auxiliary loss formulations in MoE literature
- **Layer Placement**: The various placement strategies (early, middle, late, mixed) explore the impact of expert location on model performance, building on insights from upcycling research (He et al., 2024; Komatsuzaki et al., 2023)
- **KD Integration**: The knowledge distillation implementation combines Hinton's foundational temperature-based distillation with MoE-specific considerations from MoE-KD (2024), including handling of router outputs and expert activations
- **Expert Analysis**: The router statistics collection and visualization functions (`collect_router_statistics`, `visualize_router_statistics`) enable analysis of expert utilization patterns that can reveal KD-induced changes in routing behavior, as discussed in Li et al. (2025)
- **Training Efficiency**: The use of LoRA for parameter-efficient fine-tuning enables rapid experimentation across multiple MoE configurations while maintaining computational feasibility

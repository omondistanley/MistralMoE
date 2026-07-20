"""Accuracy, calibration, FLOPs, throughput, parameter, memory, and KD metrics.

Ported from moe_complete.ipynb lines 353-888. `evaluate_mmlu_comprehensive`
and `compute_throughput_metrics` originally read a bare `MAX_LENGTH` notebook
global and `answer_to_idx`/`idx_to_letter` globals; both are now explicit
parameters (defaulting to the same values from `config.py`) instead.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from ..config import ANSWER_TO_IDX, IDX_TO_LETTER, MAX_LENGTH


def compute_ece(confidences, predictions, labels, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE): how well confidence tracks accuracy."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def evaluate_mmlu_comprehensive(
    model,
    tokenizer,
    eval_dataset,
    answer_tokens,
    device="cuda",
    max_samples=None,
    show_progress=True,
    max_length: int = MAX_LENGTH,
    answer_to_idx: dict = ANSWER_TO_IDX,
    idx_to_letter: dict = IDX_TO_LETTER,
) -> dict:
    """Comprehensive MMLU evaluation: accuracy, top-2 accuracy, ECE, throughput.

    Returns a dict with accuracy, top2_accuracy, correct, total, ece,
    confidences, predictions, true_labels, confusion_matrix, throughput,
    avg_latency.
    """
    model.eval()

    correct = 0
    top2_correct = 0
    total = 0

    all_confidences = []
    all_predictions = []
    all_true_labels = []

    samples = eval_dataset
    if max_samples is not None:
        samples = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    answer_token_ids = torch.tensor(
        [answer_tokens["A"], answer_tokens["B"], answer_tokens["C"], answer_tokens["D"]]
    )

    start_time = time.time()
    iterator = tqdm(samples, desc="Evaluating", disable=not show_progress)

    with torch.no_grad():
        for example in iterator:
            prompt = example["formatted_prompt"]
            true_answer = example["answer"]
            true_idx = answer_to_idx[true_answer]

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            outputs = model(**inputs)
            last_logits = outputs.logits[0, -1, :]

            answer_logits = last_logits[answer_token_ids]
            answer_probs = torch.softmax(answer_logits, dim=0)

            pred_idx = answer_probs.argmax().item()
            pred_answer = idx_to_letter[pred_idx]
            confidence = answer_probs[pred_idx].item()

            top2_indices = answer_probs.topk(2).indices.tolist()

            all_confidences.append(confidence)
            all_predictions.append(pred_idx)
            all_true_labels.append(true_idx)

            if pred_answer == true_answer:
                correct += 1
            if true_idx in top2_indices:
                top2_correct += 1
            total += 1

    duration = time.time() - start_time

    confidences = np.array(all_confidences)
    predictions = np.array(all_predictions)
    true_labels = np.array(all_true_labels)

    accuracy = correct / total if total > 0 else 0.0
    top2_accuracy = top2_correct / total if total > 0 else 0.0
    ece = compute_ece(confidences, predictions, true_labels)
    conf_matrix = confusion_matrix(true_labels, predictions, labels=[0, 1, 2, 3])

    throughput = total / duration if duration > 0 else 0.0
    avg_latency = duration / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "top2_accuracy": top2_accuracy,
        "correct": correct,
        "total": total,
        "ece": ece,
        "confidences": confidences,
        "predictions": predictions,
        "true_labels": true_labels,
        "confusion_matrix": conf_matrix,
        "throughput": throughput,
        "avg_latency": avg_latency,
    }


def compute_model_flops(model, seq_length: int = MAX_LENGTH) -> float:
    """Estimate FLOPs per forward pass.

    Dense: FLOPs ~= 2 * active_params * seq_length (attention + FFN terms below).
    MoE: FFN term scaled by sparsity factor (active_experts / total_experts).
    """
    config = model.config
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    intermediate_size = config.intermediate_size

    attention_flops = 4 * seq_length * d_model * d_model * n_layers
    ffn_flops = 8 * seq_length * d_model * intermediate_size * n_layers

    total_flops = attention_flops + ffn_flops

    is_moe = False
    sparsity_factor = 1.0

    try:
        base_model = model
        if hasattr(model, "base_model"):
            base_model = model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model

        layers = base_model.layers if hasattr(base_model, "layers") else base_model.model.layers

        for layer in layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "num_experts"):
                is_moe = True
                num_experts = layer.mlp.num_experts
                num_experts_per_tok = layer.mlp.num_experts_per_tok
                sparsity_factor = num_experts_per_tok / num_experts
                break
    except Exception:
        pass

    if is_moe:
        total_flops = attention_flops + (ffn_flops * sparsity_factor)

    return total_flops


def compute_throughput_metrics(model, tokenizer, eval_dataset, max_samples: int = 100, max_length: int = MAX_LENGTH) -> dict:
    """Measure throughput/latency: tokens/sec, ms/token, samples/sec, total_time."""
    model.eval()

    samples = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

    total_tokens = 0
    sample_times = []

    with torch.no_grad():
        for example in samples:
            prompt = example["formatted_prompt"]

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
            num_tokens = inputs["input_ids"].shape[1]
            total_tokens += num_tokens

            start = time.time()
            _ = model(**inputs)
            sample_times.append(time.time() - start)

    total_time = sum(sample_times)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    ms_per_token = (total_time / total_tokens * 1000) if total_tokens > 0 else 0
    samples_per_second = len(samples) / total_time if total_time > 0 else 0

    return {
        "tokens_per_second": tokens_per_second,
        "ms_per_token": ms_per_token,
        "samples_per_second": samples_per_second,
        "total_time": total_time,
    }


def compute_parameter_efficiency(model, num_experts_per_tok: int = 1) -> dict:
    """Total/active/trainable parameter counts and sparsity ratio (active/total)."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    is_moe = False
    try:
        base_model = model
        if hasattr(model, "base_model"):
            base_model = model.base_model
        if hasattr(base_model, "model"):
            base_model = base_model.model

        layers = base_model.layers if hasattr(base_model, "layers") else base_model.model.layers

        for layer in layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "num_experts"):
                is_moe = True
                num_experts = layer.mlp.num_experts
                break
    except Exception:
        pass

    if is_moe:
        # Active = attention params + (k/n * expert params).
        total_expert_params = 0
        for layer in layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "num_experts"):
                if hasattr(layer.mlp, "gate_proj") and isinstance(layer.mlp.gate_proj, list):
                    for expert_idx in range(num_experts):
                        total_expert_params += sum(p.numel() for p in layer.mlp.gate_proj[expert_idx].parameters())
                        total_expert_params += sum(p.numel() for p in layer.mlp.up_proj[expert_idx].parameters())
                        total_expert_params += sum(p.numel() for p in layer.mlp.down_proj[expert_idx].parameters())

        sparsity = num_experts_per_tok / num_experts
        active_expert_params = int(total_expert_params * sparsity)

        non_expert_params = total_params - total_expert_params
        active_params = non_expert_params + active_expert_params
    else:
        active_params = total_params

    sparsity_ratio = active_params / total_params if total_params > 0 else 1.0

    return {
        "total_params": total_params,
        "active_params": active_params,
        "trainable_params": trainable_params,
        "sparsity_ratio": sparsity_ratio,
    }


def compute_memory_metrics(model) -> dict:
    """Model size in memory plus peak GPU allocated/reserved."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024

    if torch.cuda.is_available():
        gpu_memory_allocated_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        gpu_memory_reserved_gb = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
    else:
        gpu_memory_allocated_gb = 0
        gpu_memory_reserved_gb = 0

    return {
        "model_size_mb": model_size_mb,
        "gpu_memory_allocated_gb": gpu_memory_allocated_gb,
        "gpu_memory_reserved_gb": gpu_memory_reserved_gb,
    }


def compute_kd_specific_metrics(
    teacher_metrics: dict,
    student_kd_metrics: dict,
    student_no_kd_metrics: dict | None = None,
    kd_config: dict | None = None,
    alpha: float = 0.5,
) -> dict:
    """Knowledge Distillation-specific metrics: retention, gap, distillation score,
    knowledge transfer efficiency, calibration, and efficiency-per-parameter.

    Args:
        teacher_metrics: dict with 'accuracy', 'top2_accuracy', 'ece', 'total_params',
            'active_params', 'flops', 'tokens_per_second'.
        student_kd_metrics: same keys, for the KD-trained student.
        student_no_kd_metrics: same keys, for a standard-fine-tuned student (optional).
        kd_config: dict with 'kd_alpha', 'temperature', 'name' (optional).
        alpha: weight for the distillation score, balancing size vs accuracy.
    """
    kd_metrics = {}

    kd_metrics["accuracy_retention_rate"] = (
        (student_kd_metrics["accuracy"] / teacher_metrics["accuracy"]) * 100 if teacher_metrics["accuracy"] > 0 else 0.0
    )

    kd_metrics["top2_accuracy_retention_rate"] = (
        (student_kd_metrics["top2_accuracy"] / teacher_metrics["top2_accuracy"]) * 100
        if teacher_metrics["top2_accuracy"] > 0
        else 0.0
    )

    kd_metrics["teacher_student_accuracy_gap"] = teacher_metrics["accuracy"] - student_kd_metrics["accuracy"]

    kd_metrics["teacher_student_accuracy_gap_pct"] = (
        (kd_metrics["teacher_student_accuracy_gap"] / teacher_metrics["accuracy"]) * 100
        if teacher_metrics["accuracy"] > 0
        else 0.0
    )

    kd_metrics["teacher_student_top2_gap"] = teacher_metrics["top2_accuracy"] - student_kd_metrics["top2_accuracy"]

    if student_no_kd_metrics is not None:
        kd_metrics["kd_accuracy_improvement"] = student_kd_metrics["accuracy"] - student_no_kd_metrics["accuracy"]
        kd_metrics["kd_accuracy_improvement_pct"] = (
            (kd_metrics["kd_accuracy_improvement"] / student_no_kd_metrics["accuracy"]) * 100
            if student_no_kd_metrics["accuracy"] > 0
            else 0.0
        )

        kd_metrics["kd_top2_improvement"] = student_kd_metrics["top2_accuracy"] - student_no_kd_metrics["top2_accuracy"]

        kd_metrics["kd_ece_improvement"] = student_no_kd_metrics["ece"] - student_kd_metrics["ece"]

        kd_metrics["kd_improvement_ratio"] = (
            (student_kd_metrics["accuracy"] / student_no_kd_metrics["accuracy"])
            if student_no_kd_metrics["accuracy"] > 0
            else 0.0
        )

    # Distillation Score (composite metric): higher is better (small model, high accuracy).
    size_ratio = (
        (student_kd_metrics["total_params"] / teacher_metrics["total_params"]) if teacher_metrics["total_params"] > 0 else 1.0
    )
    accuracy_ratio = (
        (student_kd_metrics["accuracy"] / teacher_metrics["accuracy"]) if teacher_metrics["accuracy"] > 0 else 0.0
    )

    kd_metrics["distillation_score"] = (1 - alpha) * accuracy_ratio - alpha * size_ratio
    kd_metrics["distillation_score_normalized"] = accuracy_ratio / (1 + size_ratio)

    # Knowledge Transfer Efficiency: accuracy ratio per unit of size ratio.
    kd_metrics["knowledge_transfer_efficiency"] = (accuracy_ratio / size_ratio) if size_ratio > 0 else 0.0

    kd_metrics["calibration_improvement"] = teacher_metrics["ece"] - student_kd_metrics["ece"]
    kd_metrics["calibration_retention"] = (
        (student_kd_metrics["ece"] / teacher_metrics["ece"]) if teacher_metrics["ece"] > 0 else 1.0
    )

    kd_metrics["compression_ratio"] = size_ratio
    kd_metrics["efficiency_score"] = (student_kd_metrics["accuracy"] / size_ratio) if size_ratio > 0 else 0.0
    kd_metrics["accuracy_per_million_params"] = (
        (student_kd_metrics["accuracy"] / (student_kd_metrics["total_params"] / 1e6))
        if student_kd_metrics["total_params"] > 0
        else 0.0
    )

    if "flops" in teacher_metrics and "flops" in student_kd_metrics:
        if teacher_metrics["flops"] > 0 and student_kd_metrics["flops"] > 0:
            flops_ratio = student_kd_metrics["flops"] / teacher_metrics["flops"]
            kd_metrics["flops_ratio"] = flops_ratio
            kd_metrics["accuracy_per_flop"] = student_kd_metrics["accuracy"] / student_kd_metrics["flops"]

    if "tokens_per_second" in teacher_metrics and "tokens_per_second" in student_kd_metrics:
        if teacher_metrics["tokens_per_second"] > 0 and student_kd_metrics["tokens_per_second"] > 0:
            kd_metrics["throughput_ratio"] = (
                student_kd_metrics["tokens_per_second"] / teacher_metrics["tokens_per_second"]
            )

    if kd_config is not None:
        kd_metrics["kd_alpha"] = kd_config.get("kd_alpha", None)
        kd_metrics["kd_temperature"] = kd_config.get("temperature", None)
        kd_metrics["kd_config_name"] = kd_config.get("name", "Unknown")

    # Overall KD effectiveness score: weighted combination of normalized components.
    acc_retention_norm = kd_metrics["accuracy_retention_rate"] / 100.0
    calib_improvement_norm = max(0, min(1, kd_metrics["calibration_improvement"] + 0.5))
    efficiency_norm = min(1.0, kd_metrics["knowledge_transfer_efficiency"])

    kd_metrics["kd_effectiveness_score"] = (
        0.5 * acc_retention_norm + 0.3 * calib_improvement_norm + 0.2 * efficiency_norm
    )

    if student_no_kd_metrics is not None:
        kd_improvement_norm = max(-1, min(1, kd_metrics["kd_accuracy_improvement"] * 10))
        kd_metrics["kd_effectiveness_score"] += 0.2 * (kd_improvement_norm + 1) / 2
        kd_metrics["kd_effectiveness_score"] = min(1.0, kd_metrics["kd_effectiveness_score"])

    return kd_metrics

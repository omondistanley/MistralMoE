"""Router behavior diagnostics: per-expert utilization, load balance, confidence.

Ported from moe_complete.ipynb lines 889-1112. The notebook's
`collect_router_statistics` fell back to `globals().get('NUM_EXPERTS', 8)` /
`globals().get('NUM_EXPERTS_PER_TOK', 2)` when a model had no MoE layers;
here those are explicit parameters (defaulting to the same config.py values)
instead of a globals() lookup.
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from ..config import MAX_LENGTH, NUM_EXPERTS, NUM_EXPERTS_PER_TOK


def collect_router_statistics(
    model,
    eval_dataset,
    tokenizer,
    answer_tokens,
    max_samples: int = 500,
    device: str = "cuda",
    max_length: int = MAX_LENGTH,
    default_num_experts: int = NUM_EXPERTS,
    default_num_experts_per_tok: int = NUM_EXPERTS_PER_TOK,
) -> dict:
    """Collect per-layer, per-subject router statistics from an MoE model during inference."""
    model.eval()

    first_moe_layer = None
    for layer in model.model.layers:
        if hasattr(layer.mlp, "num_experts"):
            first_moe_layer = layer.mlp
            break

    if first_moe_layer:
        num_experts = first_moe_layer.num_experts
        num_experts_per_tok = first_moe_layer.num_experts_per_tok
    else:
        num_experts = default_num_experts
        num_experts_per_tok = default_num_experts_per_tok

    for layer in model.model.layers:
        if hasattr(layer.mlp, "forward"):
            layer.mlp._collect_router_logits = True

    router_stats = {
        "expert_selections": defaultdict(lambda: np.zeros(num_experts)),
        "expert_confidence": defaultdict(list),
        "per_layer_selection": [np.zeros(num_experts) for _ in range(len(model.model.layers))],
        "per_subject_routing": defaultdict(lambda: defaultdict(lambda: np.zeros(num_experts))),
    }

    samples = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    print(f"Collecting router statistics from {len(samples)} samples (Experts: {num_experts}, k: {num_experts_per_tok})...")

    with torch.no_grad():
        for example in tqdm(samples, desc="Collecting router stats"):
            prompt = example["formatted_prompt"]
            subject = example.get("subject", "default")

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            outputs = model(**inputs)

            for layer_idx, layer in enumerate(model.model.layers):
                if hasattr(layer.mlp, "_last_router_probs") and layer.mlp._last_router_probs is not None:
                    router_probs = layer.mlp._last_router_probs.cpu().numpy()
                    avg_probs = router_probs.mean(axis=0)
                    top_experts = np.argsort(avg_probs)[-num_experts_per_tok:]

                    router_stats["per_layer_selection"][layer_idx] += avg_probs
                    for expert_idx in top_experts:
                        router_stats["expert_selections"]["overall"][expert_idx] += 1
                        router_stats["per_subject_routing"][subject][layer_idx][expert_idx] += 1

                    router_stats["expert_confidence"][layer_idx].append(avg_probs.max())

    for layer_idx in range(len(router_stats["per_layer_selection"])):
        total = router_stats["per_layer_selection"][layer_idx].sum()
        if total > 0:
            router_stats["per_layer_selection"][layer_idx] /= total

    for layer in model.model.layers:
        if hasattr(layer.mlp, "_collect_router_logits"):
            layer.mlp._collect_router_logits = False

    print("Router statistics collected!")
    return router_stats


def visualize_router_statistics(router_stats: dict, title: str = "MoE Router Analysis") -> None:
    """Plot expert utilization, load-balance-by-layer, confidence distribution, and
    specialization, saving to router_visualization.png; also warns on router collapse."""
    expert_usage = router_stats["expert_selections"]["overall"]
    num_experts = len(expert_usage)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Expert utilization across all layers.
    ax1 = fig.add_subplot(gs[0, :2])
    expert_usage_norm = expert_usage / expert_usage.sum() if expert_usage.sum() > 0 else expert_usage

    bars = ax1.bar(range(num_experts), expert_usage_norm, color="steelblue", alpha=0.7)
    ax1.axhline(1 / num_experts, color="red", linestyle="--", label="Uniform distribution")
    ax1.set_xlabel("Expert ID", fontsize=12)
    ax1.set_ylabel("Selection Frequency", fontsize=12)
    ax1.set_title("Overall Expert Utilization (All Layers)", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(num_experts))
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height * 100:.1f}%", ha="center", va="bottom", fontsize=9)

    # 2. Expert utilization heatmap across layers.
    ax2 = fig.add_subplot(gs[0, 2])
    layer_expert_matrix = np.array(router_stats["per_layer_selection"])
    sns.heatmap(layer_expert_matrix.T, cmap="YlOrRd", ax=ax2, cbar_kws={"label": "Selection Prob"})
    ax2.set_xlabel("Layer", fontsize=10)
    ax2.set_ylabel("Expert ID", fontsize=10)
    ax2.set_title("Expert Selection Heatmap\n(Layer vs Expert)", fontsize=12, fontweight="bold")

    # 3. Load balance score per layer (normalized entropy).
    ax3 = fig.add_subplot(gs[1, 0])
    load_balance_scores = []
    for layer_probs in router_stats["per_layer_selection"]:
        if layer_probs.sum() > 0:
            layer_probs_norm = layer_probs / layer_probs.sum()
            entropy = -np.sum(layer_probs_norm * np.log(layer_probs_norm + 1e-10))
            normalized_entropy = entropy / np.log(num_experts)
        else:
            normalized_entropy = 0
        load_balance_scores.append(normalized_entropy)

    ax3.plot(load_balance_scores, marker="o", linewidth=2, markersize=4)
    ax3.axhline(1.0, color="green", linestyle="--", alpha=0.5, label="Perfect balance")
    ax3.axhline(0.5, color="orange", linestyle="--", alpha=0.5, label="50% balance")
    ax3.set_xlabel("Layer", fontsize=10)
    ax3.set_ylabel("Load Balance Score", fontsize=10)
    ax3.set_title("Load Balancing Across Layers\n(1.0 = perfect balance)", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1.1])

    # 4. Router confidence distribution.
    ax4 = fig.add_subplot(gs[1, 1])
    all_confidences = []
    for layer_confs in router_stats["expert_confidence"].values():
        all_confidences.extend(layer_confs)

    ax4.hist(all_confidences, bins=50, color="purple", alpha=0.7, edgecolor="black")
    ax4.axvline(np.mean(all_confidences), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(all_confidences):.3f}")
    ax4.set_xlabel("Router Confidence (Max Prob)", fontsize=10)
    ax4.set_ylabel("Frequency", fontsize=10)
    ax4.set_title("Distribution of Router Confidence", fontsize=12, fontweight="bold")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    # 5. Expert specialization: variance across layers.
    ax5 = fig.add_subplot(gs[1, 2])
    expert_variances = [np.var(layer_expert_matrix[:, expert_id]) for expert_id in range(num_experts)]

    ax5.bar(range(num_experts), expert_variances, color="teal", alpha=0.7)
    ax5.set_xlabel("Expert ID", fontsize=10)
    ax5.set_ylabel("Variance Across Layers", fontsize=10)
    ax5.set_title("Expert Specialization\n(Higher = more layer-specific)", fontsize=12, fontweight="bold")
    ax5.set_xticks(range(num_experts))
    ax5.grid(axis="y", alpha=0.3)

    # 6. Per-layer confidence box plots.
    ax6 = fig.add_subplot(gs[2, :])
    layer_confidence_data = [router_stats["expert_confidence"][i] for i in range(len(router_stats["expert_confidence"]))]
    bp = ax6.boxplot(layer_confidence_data, patch_artist=True, showmeans=True)

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    ax6.set_xlabel("Layer", fontsize=12)
    ax6.set_ylabel("Router Confidence", fontsize=12)
    ax6.set_title("Router Confidence Distribution Across Layers", fontsize=14, fontweight="bold")
    ax6.grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig("router_visualization.png", dpi=300, bbox_inches="tight")
    print("\nVisualization saved to 'router_visualization.png'")
    plt.show()

    print("ROUTER STATISTICS SUMMARY")
    print(f"Average load balance score: {np.mean(load_balance_scores):.4f}")
    print(f"Average router confidence: {np.mean(all_confidences):.4f}")
    print(f"Std dev of expert usage: {np.std(expert_usage_norm):.4f}")

    max_expert_usage = expert_usage_norm.max()
    if max_expert_usage > 0.3:
        print("\nWARNING: Potential router collapse detected!")
        print(f"   Expert {expert_usage_norm.argmax()} is selected {max_expert_usage * 100:.1f}% of the time")

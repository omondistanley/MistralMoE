"""Result loading, KD-metric aggregation, and plotting for the results/ and
experiments/ directories.

Ported from moe_complete.ipynb's "Visualizations" section (lines 4102-8377).
Several functions were defined twice in the notebook with byte-identical
bodies from two separate reporting passes (verified via diff):
`plot_training_losses_all_models`, `plot_retention_rates_all_models`,
`plot_kte_all_models`, `plot_moe_baseline_knowledge_transfer`, and
`extract_variant_name`. Each keeps a single copy here; both original driver
flows (`load_all_metrics_with_correct_teachers` and
`load_all_metrics_with_dense_teacher`, which differ in which baseline they
use as the KD teacher) are preserved separately since those genuinely differ.

`plot_variant_comparison_grid` condenses the notebook's final ~300-line
9-panel chart cell (lines 7762-8377), which computed each of 9 metrics
(accuracy, FLOPs, throughput, latency, size, ECE, accuracy gain, ...) via a
copy-pasted if/else block per metric, into a small data-driven loop over the
same 9 metric definitions. The metrics, panels, and output PNG are unchanged
-- only the repeated boilerplate is condensed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import KD_CONFIG_DENSE, KD_CONFIG_MOE_DEFAULT
from ..evaluation.metrics import compute_kd_specific_metrics

sns.set_style("whitegrid")


def load_results(filepath) -> dict | None:
    """Load a results JSON file, returning None (with a warning) if missing."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    print(f"Warning: {filepath} not found. Returning None.")
    return None


# ---------------------------------------------------------------------------
# Main 4-model (Dense / Dense+KD / MoE Standard / MoE KD) comparison
# ---------------------------------------------------------------------------

DEFAULT_RESULTS_FILES = {
    "Dense (Baseline)": "results/baseline_comprehensive.json",
    "Dense (KD)": "results/trained_dense_kd_comprehensive.json",
    "MoE (Standard Training)": "results/trained_moe_comprehensive.json",
    "MoE (KD)": "results/trained_moe_kd_comprehensive.json",
}


def build_main_comparison_table(
    results_files: dict[str, str] = DEFAULT_RESULTS_FILES,
    save_path: str = "results/model_comparison_visualization.png",
) -> pd.DataFrame:
    """Build the core accuracy/ECE/FLOPs/throughput/params comparison across the
    4 top-level models and render a 7-panel bar chart. Returns the underlying
    DataFrame (unformatted, numeric) for further use.

    Note: the notebook version used `display(HTML(...))` for two formatted
    tables; that's IPython-only and is replaced here with a returned
    DataFrame plus printed summaries.
    """
    all_results = {name: load_results(path) for name, path in results_files.items()}
    all_results = {name: data for name, data in all_results.items() if data is not None}

    rows = []
    for model_name, data in all_results.items():
        rows.append(
            {
                "Model": model_name,
                "MMLU Accuracy": data.get("accuracy", np.nan),
                "Top-2 Accuracy": data.get("top2_accuracy", np.nan),
                "ECE": data.get("ece", np.nan),
                "FLOPs (G)": data.get("flops", 0) / 1e9 if data.get("flops") else np.nan,
                "Throughput (samples/sec)": data.get("samples_per_second", np.nan),
                "Avg Latency (sec)": data.get("avg_latency", np.nan),
                "Tokens/sec": data.get("tokens_per_second", np.nan),
                "ms/token": data.get("ms_per_token", np.nan),
                "Total Params (B)": data.get("total_params", 0) / 1e9 if data.get("total_params") else np.nan,
                "Active Params (B)": data.get("active_params", 0) / 1e9 if data.get("active_params") else np.nan,
                "Trainable Params (M)": data.get("trainable_params", 0) / 1e6 if data.get("trainable_params") else np.nan,
                "Model Size (MB)": data.get("model_size_mb", np.nan),
                "GPU Memory (GB)": data.get("gpu_memory_allocated_gb", np.nan),
            }
        )

    df = pd.DataFrame(rows)

    if "Dense (Baseline)" in df["Model"].values:
        baseline_acc = df.loc[df["Model"] == "Dense (Baseline)", "MMLU Accuracy"].iloc[0]
        df["Accuracy vs Baseline"] = df["MMLU Accuracy"] - baseline_acc
        df["Accuracy % Change"] = ((df["MMLU Accuracy"] - baseline_acc) / baseline_acc * 100).round(2)

    print("COMPREHENSIVE MODEL EVALUATION RESULTS")
    print(df.to_string(index=False))

    valid_df = df.dropna(subset=["MMLU Accuracy"])
    if len(valid_df) == 0:
        print("No valid data found to visualize.")
        return df

    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle("Model Evaluation Results Comparison", fontsize=18, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(valid_df)]

    panels = [
        (axes[0, 0], "MMLU Accuracy", "MMLU Accuracy", "{:.4f}"),
        (axes[0, 1], "ECE", "Expected Calibration Error (Lower is Better)", "{:.4f}"),
        (axes[0, 2], "FLOPs (G)", "Computational Cost (FLOPs)", "{:.0f}G"),
        (axes[1, 0], "Throughput (samples/sec)", "Throughput (Samples/sec)", "{:.2f}"),
        (axes[1, 1], "Avg Latency (sec)", "Average Latency (Lower is Better)", "{:.4f}s"),
        (axes[1, 2], "Total Params (B)", "Total Parameters", "{:.2f}B"),
        (axes[2, 0], "Trainable Params (M)", "Trainable Parameters", "{:.2f}M"),
    ]

    for ax, col, title, fmt in panels:
        bars = ax.bar(valid_df["Model"], valid_df[col], color=colors)
        ax.set_ylabel(col, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, valid_df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

    for ax in (axes[2, 1], axes[2, 2]):
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nKEY INSIGHTS")
    if "MMLU Accuracy" in valid_df.columns:
        idx = valid_df["MMLU Accuracy"].idxmax()
        print(f"Best MMLU Accuracy: {valid_df.loc[idx, 'Model']} ({valid_df.loc[idx, 'MMLU Accuracy']:.4f})")
    if "ECE" in valid_df.columns:
        idx = valid_df["ECE"].idxmin()
        print(f"Best Calibration (Lowest ECE): {valid_df.loc[idx, 'Model']} ({valid_df.loc[idx, 'ECE']:.4f})")
    if "FLOPs (G)" in valid_df.columns:
        idx = valid_df["FLOPs (G)"].idxmin()
        print(f"Most Efficient (Lowest FLOPs): {valid_df.loc[idx, 'Model']} ({valid_df.loc[idx, 'FLOPs (G)']:.2f}G FLOPs)")

    return df


# ---------------------------------------------------------------------------
# KD metric aggregation across Dense KD / MoE Standard / MoE KD
# ---------------------------------------------------------------------------


def load_all_metrics_with_correct_teachers(results_dir: str | Path = "results") -> dict:
    """Load all metrics with per-model-appropriate teachers:
    Dense KD -> Dense Baseline teacher; MoE Standard / MoE KD -> MoE Baseline teacher.
    """
    metrics = {"training": {}, "post_training": {}, "comprehensive": {}, "baselines": {}}
    results_dir = Path(results_dir)

    baseline_files = {
        "dense_baseline": results_dir / "baseline_comprehensive.json",
        "moe_baseline": results_dir / "moe_baseline_comprehensive.json",
    }
    for name, path in baseline_files.items():
        if path.exists():
            with open(path, "r") as f:
                metrics["baselines"][name] = json.load(f)

    if "dense_baseline" not in metrics["baselines"]:
        print("Warning: Dense baseline not found.")
    if "moe_baseline" not in metrics["baselines"]:
        print("Warning: MoE baseline not found.")

    training_metrics_file = results_dir / "training_metrics_comparison.json"
    if training_metrics_file.exists():
        with open(training_metrics_file, "r") as f:
            metrics["training"] = json.load(f)

    eval_files = {
        "dense_kd": results_dir / "trained_dense_kd_comprehensive.json",
        "moe_standard": results_dir / "trained_moe_standard_comprehensive.json",
        "moe_kd": results_dir / "trained_moe_kd_comprehensive.json",
    }
    for model_name, path in eval_files.items():
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                metrics["comprehensive"][model_name] = data
                if "training_metrics" in data and model_name not in metrics["training"]:
                    metrics["training"][model_name] = data["training_metrics"]

    dense_baseline = metrics["baselines"].get("dense_baseline")
    moe_baseline = metrics["baselines"].get("moe_baseline")

    if "dense_kd" in metrics["comprehensive"] and dense_baseline:
        try:
            metrics["post_training"]["dense_kd"] = compute_kd_specific_metrics(
                teacher_metrics=dense_baseline,
                student_kd_metrics=metrics["comprehensive"]["dense_kd"],
                student_no_kd_metrics=None,
                kd_config=KD_CONFIG_DENSE,
                alpha=0.5,
            )
        except Exception as e:
            print(f"Warning: Could not compute KD metrics for dense_kd: {e}")

    if "moe_standard" in metrics["comprehensive"] and moe_baseline:
        try:
            metrics["post_training"]["moe_standard"] = compute_kd_specific_metrics(
                teacher_metrics=moe_baseline,
                student_kd_metrics=metrics["comprehensive"]["moe_standard"],
                student_no_kd_metrics=None,
                kd_config=None,
                alpha=0.5,
            )
        except Exception as e:
            print(f"Warning: Could not compute KD metrics for moe_standard: {e}")

    if "moe_kd" in metrics["comprehensive"] and moe_baseline:
        try:
            student_no_kd = metrics["comprehensive"].get("moe_standard")
            metrics["post_training"]["moe_kd"] = compute_kd_specific_metrics(
                teacher_metrics=moe_baseline,
                student_kd_metrics=metrics["comprehensive"]["moe_kd"],
                student_no_kd_metrics=student_no_kd,
                kd_config=KD_CONFIG_MOE_DEFAULT,
                alpha=0.5,
            )
        except Exception as e:
            print(f"Warning: Could not compute KD metrics for moe_kd: {e}")

    if dense_baseline and moe_baseline:
        try:
            metrics["post_training"]["moe_baseline_knowledge_transfer"] = compute_kd_specific_metrics(
                teacher_metrics=dense_baseline,
                student_kd_metrics=moe_baseline,
                student_no_kd_metrics=None,
                kd_config=None,
                alpha=0.5,
            )
        except Exception as e:
            print(f"Warning: Could not compute knowledge transfer metrics for MoE baseline: {e}")

    return metrics


def load_all_metrics_with_dense_teacher(results_dir: str | Path = "results") -> dict:
    """Load all metrics using Dense Baseline as the teacher for ALL models
    (Dense KD, MoE Standard, MoE KD), for a single-teacher comparison view.

    KD-improvement comparisons: Dense KD vs Dense Baseline; MoE KD vs MoE Standard.
    """
    metrics = {"training": {}, "post_training": {}, "comprehensive": {}, "baselines": {}}
    results_dir = Path(results_dir)

    baseline_file = results_dir / "baseline_comprehensive.json"
    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            metrics["baselines"]["dense_baseline"] = json.load(f)

    if "dense_baseline" not in metrics["baselines"]:
        print("Warning: Dense baseline not found.")
        return metrics

    dense_baseline = metrics["baselines"]["dense_baseline"]

    moe_baseline_file = results_dir / "moe_baseline_comprehensive.json"
    if moe_baseline_file.exists():
        with open(moe_baseline_file, "r") as f:
            metrics["baselines"]["moe_baseline"] = json.load(f)

    training_metrics_file = results_dir / "training_metrics_comparison.json"
    if training_metrics_file.exists():
        with open(training_metrics_file, "r") as f:
            metrics["training"] = json.load(f)

    eval_files = {
        "dense_kd": results_dir / "trained_dense_kd_comprehensive.json",
        "moe_standard": results_dir / "trained_moe_standard_comprehensive.json",
        "moe_kd": results_dir / "trained_moe_kd_comprehensive.json",
    }
    for model_name, path in eval_files.items():
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                metrics["comprehensive"][model_name] = data
                if "training_metrics" in data and model_name not in metrics["training"]:
                    metrics["training"][model_name] = data["training_metrics"]

    for model_name in ["dense_kd", "moe_standard", "moe_kd"]:
        if model_name not in metrics["comprehensive"]:
            continue

        student_metrics = metrics["comprehensive"][model_name]

        student_no_kd_metrics = None
        if model_name == "dense_kd":
            student_no_kd_metrics = dense_baseline
        elif model_name == "moe_kd" and "moe_standard" in metrics["comprehensive"]:
            student_no_kd_metrics = metrics["comprehensive"]["moe_standard"]

        kd_config = None
        if model_name == "dense_kd":
            kd_config = KD_CONFIG_DENSE
        elif model_name == "moe_kd":
            kd_config = KD_CONFIG_MOE_DEFAULT

        try:
            metrics["post_training"][model_name] = compute_kd_specific_metrics(
                teacher_metrics=dense_baseline,
                student_kd_metrics=student_metrics,
                student_no_kd_metrics=student_no_kd_metrics,
                kd_config=kd_config,
                alpha=0.5,
            )
        except Exception as e:
            print(f"Warning: Could not compute KD metrics for {model_name}: {e}")

    moe_baseline = metrics["baselines"].get("moe_baseline")
    if dense_baseline and moe_baseline:
        try:
            metrics["post_training"]["moe_baseline_knowledge_transfer"] = compute_kd_specific_metrics(
                teacher_metrics=dense_baseline,
                student_kd_metrics=moe_baseline,
                student_no_kd_metrics=None,
                kd_config=None,
                alpha=0.5,
            )
        except Exception as e:
            print(f"Warning: Could not compute knowledge transfer metrics for MoE baseline: {e}")

    return metrics


_MODELS = ["dense_kd", "moe_standard", "moe_kd"]
_MODEL_LABELS = {"dense_kd": "Dense KD", "moe_standard": "MoE Standard", "moe_kd": "MoE KD"}
_MODEL_COLORS = {"dense_kd": "#2E86AB", "moe_standard": "#A23B72", "moe_kd": "#F18F01"}


def plot_training_losses_all_models(metrics_data: dict, save_path: str = "results/training_losses_all_models.png") -> None:
    """Bar chart of final/average NTP, KD, and total training losses across models."""
    if not metrics_data:
        print("No training metrics data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training Losses: All Models Comparison", fontsize=16, fontweight="bold")

    available_models = [m for m in _MODELS if m in metrics_data]
    available_labels = [_MODEL_LABELS[m] for m in available_models]
    if not available_models:
        print("No training metrics available for plotting")
        return

    x = np.arange(len(available_models))
    width = 0.25

    for ax, prefix, title in [(axes[0], "final", "Final Training Losses"), (axes[1], "avg", "Average Training Losses")]:
        ntp = [metrics_data[m].get(f"{prefix}_ntp_loss", 0) for m in available_models]
        kd = [metrics_data[m].get(f"{prefix}_kd_loss", 0) for m in available_models]
        total = [metrics_data[m].get(f"{prefix}_total_loss", 0) for m in available_models]

        ax.bar(x - width, ntp, width, label="NTP Loss", color="#3498db", alpha=0.8)
        ax.bar(x, kd, width, label="KD Loss", color="#e74c3c", alpha=0.8)
        ax.bar(x + width, total, width, label="Total Loss", color="#2ecc71", alpha=0.8)
        ax.set_xlabel("Model")
        ax.set_ylabel("Loss Value")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(available_labels, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        for i, (n, k, t) in enumerate(zip(ntp, kd, total)):
            ax.text(i - width, n, f"{n:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
            if k > 0:
                ax.text(i, k, f"{k:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
            ax.text(i + width, t, f"{t:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved training losses visualization to {save_path}")
    plt.show()


def plot_retention_rates_all_models(kd_metrics_data: dict, save_path: str = "results/retention_rates_all_models.png") -> None:
    """Bar chart of accuracy/top-2 retention rate and teacher-student gap across models."""
    if not kd_metrics_data:
        print("No post-training KD metrics data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Performance Retention Rates: All Models", fontsize=16, fontweight="bold")

    labels = ["Dense KD\n(vs Dense Baseline)", "MoE Standard\n(vs MoE Baseline)", "MoE KD\n(vs MoE Baseline)"]
    available_models = [m for m in _MODELS if m in kd_metrics_data]
    available_labels = [labels[_MODELS.index(m)] for m in available_models]
    if not available_models:
        print("No retention rate data available")
        return

    x = np.arange(len(available_models))
    width = 0.35

    ax = axes[0]
    acc_retention = [kd_metrics_data[m].get("accuracy_retention_rate", 0) for m in available_models]
    top2_retention = [kd_metrics_data[m].get("top2_accuracy_retention_rate", 0) for m in available_models]
    bars1 = ax.bar(x - width / 2, acc_retention, width, label="Accuracy Retention", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, top2_retention, width, label="Top-2 Accuracy Retention", color="#2ecc71", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Retention Rate (%)")
    ax.set_title("Performance Retention Rates", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(available_labels, rotation=15, ha="right")
    ax.axhline(y=100, color="r", linestyle="--", alpha=0.5, label="100% (Perfect Retention)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax = axes[1]
    acc_gap = [kd_metrics_data[m].get("teacher_student_accuracy_gap", 0) for m in available_models]
    acc_gap_pct = [kd_metrics_data[m].get("teacher_student_accuracy_gap_pct", 0) for m in available_models]
    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.2, acc_gap, width=0.4, label="Absolute Gap", color="#e74c3c", alpha=0.7)
    ax2.bar(x + 0.2, acc_gap_pct, width=0.4, label="Relative Gap (%)", color="#c0392b", alpha=0.7)
    ax.set_xlabel("Model")
    ax.set_ylabel("Absolute Accuracy Gap", color="#e74c3c", fontweight="bold")
    ax2.set_ylabel("Relative Gap (%)", color="#c0392b", fontweight="bold")
    ax.set_title("Teacher-Student Performance Gap", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(available_labels, rotation=15, ha="right")
    ax.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    for bar, gap in zip(bars1, acc_gap):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{gap:.4f}", ha="center", va="bottom" if height >= 0 else "top", fontsize=9, fontweight="bold", color="#e74c3c")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved retention rates visualization to {save_path}")
    plt.show()


def plot_kte_all_models(kd_metrics_data: dict, save_path: str = "results/kte_all_models.png") -> None:
    """Bar charts of Knowledge Transfer Efficiency, Efficiency Score, and normalized
    Distillation Score across models."""
    if not kd_metrics_data:
        print("No post-training KD metrics data available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Knowledge Transfer Efficiency: All Models", fontsize=16, fontweight="bold")

    labels = ["Dense KD\n(vs Dense Baseline)", "MoE Standard\n(vs MoE Baseline)", "MoE KD\n(vs MoE Baseline)"]
    available_models = [m for m in _MODELS if m in kd_metrics_data]
    available_labels = [labels[_MODELS.index(m)] for m in available_models]
    if not available_models:
        print("No KTE data available")
        return

    panels = [
        (axes[0], "knowledge_transfer_efficiency", "Knowledge Transfer Efficiency\n(Higher = Better)"),
        (axes[1], "efficiency_score", "Efficiency Score\n(Accuracy per Unit Size)"),
        (axes[2], "distillation_score_normalized", "Normalized Distillation Score\n(Higher = Better)"),
    ]
    for ax, key, title in panels:
        values = [kd_metrics_data[m].get(key, 0) for m in available_models]
        bars = ax.bar(available_models, values, color=[_MODEL_COLORS.get(m, "#95a5a6") for m in available_models], alpha=0.8)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        ax.set_xticklabels(available_labels, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, score in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{score:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved KTE visualization to {save_path}")
    plt.show()


def plot_moe_baseline_knowledge_transfer(kd_metrics_data: dict, save_path: str = "results/moe_baseline_knowledge_transfer.png") -> None:
    """4-panel visualization of the untrained MoE baseline's knowledge transfer from
    the dense baseline (retention, KTE, gap, calibration)."""
    if "moe_baseline_knowledge_transfer" not in kd_metrics_data:
        print("MoE baseline knowledge transfer metrics not available")
        return

    kt = kd_metrics_data["moe_baseline_knowledge_transfer"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("MoE Baseline: Knowledge Transfer from Dense Baseline", fontsize=16, fontweight="bold", y=0.995)

    panels = [
        (axes[0, 0], ["Accuracy\nRetention", "Top-2 Accuracy\nRetention"], [kt.get("accuracy_retention_rate", 0), kt.get("top2_accuracy_retention_rate", 0)], ["#3498db", "#2ecc71"], "Performance Retention: MoE Baseline vs Dense Baseline", "{:.2f}%"),
        (axes[0, 1], ["KTE Score", "Efficiency\nScore", "Distillation\nScore"], [kt.get("knowledge_transfer_efficiency", 0), kt.get("efficiency_score", 0), kt.get("distillation_score_normalized", 0)], ["#16a085", "#27ae60", "#9b59b6"], "Knowledge Transfer Efficiency Metrics", "{:.4f}"),
        (axes[1, 0], ["Absolute\nGap", "Relative\nGap (%)"], [kt.get("teacher_student_accuracy_gap", 0), kt.get("teacher_student_accuracy_gap_pct", 0)], ["#e74c3c", "#c0392b"], "Performance Gap: Dense Baseline - MoE Baseline", "{:.4f}"),
        (axes[1, 1], ["Calibration\nImprovement", "Calibration\nRetention"], [kt.get("calibration_improvement", 0), kt.get("calibration_retention", 0)], ["#e67e22", "#d35400"], "Calibration Metrics", "{:.4f}"),
    ]

    for ax, categories, values, colors, title, fmt in panels:
        bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.6)
        ax.set_ylabel("Value")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, fmt.format(val), ha="center", va="bottom" if height >= 0 else "top", fontsize=10, fontweight="bold")

    footnote_text = (
        "Why MoE Baseline is Considered to Have Knowledge Transfer from Dense Baseline:\n\n"
        "MoE models are initialized from a pre-trained dense model via sparse upcycling: the dense FFN "
        "weights are duplicated across every expert, so the MoE inherits the dense model's learned "
        "representations and language understanding rather than learning from scratch."
    )
    fig.text(0.5, 0.02, footnote_text, ha="center", va="bottom", fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3), wrap=True)

    plt.tight_layout(rect=[0, 0.15, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved MoE baseline knowledge transfer visualization to {save_path}")
    plt.show()


def create_prioritized_kd_metrics_table(all_metrics: dict, save_path: str = "results/kd_metrics_prioritized_table.csv") -> pd.DataFrame:
    """Build the prioritized (Tier 1-4) KD metrics table and save to CSV."""
    table_data = []

    for model_name in _MODELS:
        if model_name not in all_metrics["post_training"]:
            continue

        kd_metrics = all_metrics["post_training"][model_name]
        training_metrics = all_metrics["training"].get(model_name, {})

        row = {
            "Model": _MODEL_LABELS[model_name],
            "Teacher (Knowledge Transfer)": "Dense Baseline",
            "Comparison Baseline (KD Improvement)": {"dense_kd": "Dense Baseline", "moe_kd": "MoE Standard"}.get(model_name, "N/A"),
            # Tier 1: core impact.
            "KD Accuracy Improvement": kd_metrics.get("kd_accuracy_improvement"),
            "KD Accuracy Improvement (%)": kd_metrics.get("kd_accuracy_improvement_pct"),
            "Accuracy Retention Rate (%)": kd_metrics.get("accuracy_retention_rate"),
            "Top-2 Accuracy Retention (%)": kd_metrics.get("top2_accuracy_retention_rate"),
            "Knowledge Transfer Efficiency": kd_metrics.get("knowledge_transfer_efficiency"),
            # Tier 2: comparative analysis.
            "Distillation Score (Normalized)": kd_metrics.get("distillation_score_normalized"),
            "Teacher-Student Accuracy Gap": kd_metrics.get("teacher_student_accuracy_gap"),
            "Teacher-Student Gap (%)": kd_metrics.get("teacher_student_accuracy_gap_pct"),
            "KD Effectiveness Score": kd_metrics.get("kd_effectiveness_score"),
            # Tier 3: training dynamics.
            "Avg KD Loss": training_metrics.get("avg_kd_loss"),
            "Final KD Loss": training_metrics.get("final_kd_loss"),
            "KD/Total Loss Ratio": training_metrics.get("avg_kd_total_ratio"),
            # Tier 4: additional context.
            "Calibration Improvement": kd_metrics.get("calibration_improvement"),
            "Compression Ratio": kd_metrics.get("compression_ratio"),
            "Efficiency Score": kd_metrics.get("efficiency_score"),
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)
    df.to_csv(save_path, index=False)
    print(f"Saved prioritized KD metrics table to {save_path}")
    print(df.to_string(index=False))
    return df


def plot_tier1_core_impact_metrics(kd_metrics_data: dict, save_path: str = "results/tier1_core_kd_impact.png") -> None:
    """4-panel Tier-1 KD impact chart: accuracy improvement, retention, KTE, gap."""
    if not kd_metrics_data:
        print("No post-training KD metrics data available")
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("TIER 1: Core KD Impact Metrics (All vs Dense Baseline)", fontsize=18, fontweight="bold", y=0.995)

    available_models = [m for m in _MODELS if m in kd_metrics_data]
    available_labels = [_MODEL_LABELS[m] for m in available_models]
    if not available_models:
        print("No KD metrics available for plotting")
        return

    x = np.arange(len(available_models))
    width = 0.35

    # 1. KD accuracy improvement.
    ax = axes[0, 0]
    improvements = [kd_metrics_data[m].get("kd_accuracy_improvement", 0) for m in available_models]
    improvement_pcts = [kd_metrics_data[m].get("kd_accuracy_improvement_pct", 0) for m in available_models]
    has_data = any("kd_accuracy_improvement" in kd_metrics_data[m] for m in available_models)
    if has_data:
        ax2 = ax.twinx()
        bars1 = ax.bar(x - 0.2, improvements, width=0.4, label="Absolute Improvement", color="#2ecc71", alpha=0.8)
        ax2.bar(x + 0.2, improvement_pcts, width=0.4, label="Relative Improvement (%)", color="#27ae60", alpha=0.8)
        ax.set_ylabel("Absolute Accuracy Improvement", color="#2ecc71", fontweight="bold")
        ax2.set_ylabel("Relative Improvement (%)", color="#27ae60", fontweight="bold")
        ax.tick_params(axis="y", labelcolor="#2ecc71")
        ax2.tick_params(axis="y", labelcolor="#27ae60")
        ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
        for bar, val in zip(bars1, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:+.4f}", ha="center", va="bottom" if height >= 0 else "top", fontsize=9, fontweight="bold", color="#2ecc71")
    else:
        ax.text(0.5, 0.5, "KD Improvement data not available", ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_xlabel("Model")
    ax.set_title("1. KD Accuracy Improvement\n(Dense KD vs Dense Baseline; MoE KD vs MoE Standard)", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(available_labels, rotation=15, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Retention rates.
    ax = axes[0, 1]
    acc_retention = [kd_metrics_data[m].get("accuracy_retention_rate", 0) for m in available_models]
    top2_retention = [kd_metrics_data[m].get("top2_accuracy_retention_rate", 0) for m in available_models]
    bars1 = ax.bar(x - width / 2, acc_retention, width, label="Accuracy Retention", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, top2_retention, width, label="Top-2 Accuracy Retention", color="#2ecc71", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Retention Rate (%)")
    ax.set_title("2. Performance Retention Rates\n(All vs Dense Baseline)", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(available_labels, rotation=15, ha="right")
    ax.axhline(y=100, color="r", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 3. Knowledge transfer efficiency.
    ax = axes[1, 0]
    kte = [kd_metrics_data[m].get("knowledge_transfer_efficiency", 0) for m in available_models]
    dist_score = [kd_metrics_data[m].get("distillation_score_normalized", 0) for m in available_models]
    bars1 = ax.bar(x - width / 2, kte, width, label="Knowledge Transfer Efficiency", color="#16a085", alpha=0.8)
    bars2 = ax.bar(x + width / 2, dist_score, width, label="Distillation Score (Normalized)", color="#9b59b6", alpha=0.8)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("3. Knowledge Transfer Efficiency\n(All vs Dense Baseline)", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(available_labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bars, vals in ((bars1, kte), (bars2, dist_score)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Teacher-student gap.
    ax = axes[1, 1]
    acc_gap = [kd_metrics_data[m].get("teacher_student_accuracy_gap", 0) for m in available_models]
    acc_gap_pct = [kd_metrics_data[m].get("teacher_student_accuracy_gap_pct", 0) for m in available_models]
    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.2, acc_gap, width=0.4, label="Absolute Gap", color="#e74c3c", alpha=0.7)
    ax2.bar(x + 0.2, acc_gap_pct, width=0.4, label="Relative Gap (%)", color="#c0392b", alpha=0.7)
    ax.set_xlabel("Model")
    ax.set_ylabel("Absolute Accuracy Gap", color="#e74c3c", fontweight="bold")
    ax2.set_ylabel("Relative Gap (%)", color="#c0392b", fontweight="bold")
    ax.set_title("4. Teacher-Student Performance Gap\n(All vs Dense Baseline)", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(available_labels, rotation=15, ha="right")
    ax.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    for bar, gap in zip(bars1, acc_gap):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{gap:.4f}", ha="center", va="bottom" if height >= 0 else "top", fontsize=9, fontweight="bold", color="#e74c3c")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved Tier 1 core impact metrics to {save_path}")
    plt.show()


def plot_tier2_comparative_metrics(kd_metrics_data: dict, save_path: str = "results/tier2_comparative_kd_metrics.png") -> None:
    """3-panel Tier-2 chart: distillation score, KD effectiveness score, efficiency score."""
    if not kd_metrics_data:
        print("No post-training KD metrics data available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("TIER 2: Comparative Analysis Metrics (All vs Dense Baseline)", fontsize=16, fontweight="bold")

    available_models = [m for m in _MODELS if m in kd_metrics_data]
    available_labels = [_MODEL_LABELS[m] for m in available_models]
    if not available_models:
        return

    panels = [
        (axes[0], "distillation_score_normalized", "Distillation Score (Normalized)\n(Higher = Better)", "{:.4f}", None),
        (axes[1], "kd_effectiveness_score", "KD Effectiveness Score\n(Higher = Better)", "{:.3f}", (0, 1)),
        (axes[2], "efficiency_score", "Efficiency Score\n(Accuracy per Unit Size)", "{:.4f}", None),
    ]
    for ax, key, title, fmt, ylim in panels:
        values = [kd_metrics_data[m].get(key, 0) for m in available_models]
        bars = ax.bar(available_models, values, color=[_MODEL_COLORS.get(m, "#95a5a6") for m in available_models], alpha=0.8)
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        ax.set_xticklabels(available_labels, rotation=15, ha="right")
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, score in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), fmt.format(score), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved Tier 2 comparative metrics to {save_path}")
    plt.show()


def plot_tier3_training_dynamics(training_metrics_data: dict, save_path: str = "results/tier3_training_dynamics.png") -> None:
    """2-panel Tier-3 chart: final and average training losses (NTP/KD/total)."""
    if not training_metrics_data:
        print("No training metrics data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("TIER 3: Training Dynamics Metrics", fontsize=16, fontweight="bold")

    available_models = [m for m in _MODELS if m in training_metrics_data]
    available_labels = [_MODEL_LABELS[m] for m in available_models]
    if not available_models:
        return

    x = np.arange(len(available_models))
    width = 0.25

    for ax, prefix, title in [(axes[0], "final", "Final Training Losses"), (axes[1], "avg", "Average Training Losses")]:
        ntp = [training_metrics_data[m].get(f"{prefix}_ntp_loss", 0) for m in available_models]
        kd = [training_metrics_data[m].get(f"{prefix}_kd_loss", 0) for m in available_models]
        total = [training_metrics_data[m].get(f"{prefix}_total_loss", 0) for m in available_models]

        ax.bar(x - width, ntp, width, label=f"{prefix.upper()} NTP Loss", color="#3498db", alpha=0.8)
        ax.bar(x, kd, width, label=f"{prefix.upper()} KD Loss", color="#e74c3c", alpha=0.8)
        ax.bar(x + width, total, width, label=f"{prefix.upper()} Total Loss", color="#2ecc71", alpha=0.8)
        ax.set_xlabel("Model")
        ax.set_ylabel("Loss Value")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(available_labels, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved Tier 3 training dynamics to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Cross-variant (10 MoE architectures) comparison
# ---------------------------------------------------------------------------


def extract_variant_name(filename) -> str:
    """Extract a variant name from a results filename, e.g. 'top1_8x1' from
    'top1_8x1_standard_results.json'."""
    return filename.stem.replace("_standard_results", "").replace("_kd_results", "")


def _load_variant_results_dict(experiments_dir: str | Path = "experiments") -> dict:
    """Load every {variant}_standard_results.json / {variant}_kd_results.json into
    {variant_name: {'standard': {...}, 'kd': {...}}}."""
    experiments_dir = Path(experiments_dir)
    standard_files = sorted(experiments_dir.glob("*_standard_results.json"))
    kd_files = sorted(experiments_dir.glob("*_kd_results.json"))

    results_dict: dict = {}
    for file in standard_files:
        with open(file, "r") as f:
            results_dict.setdefault(extract_variant_name(file), {})["standard"] = json.load(f)
    for file in kd_files:
        with open(file, "r") as f:
            results_dict.setdefault(extract_variant_name(file), {})["kd"] = json.load(f)

    return results_dict


_VARIANT_TABLE_METRICS = [
    ("accuracy", "Accuracy"),
    ("top2_accuracy", "Top-2 Accuracy"),
    ("ece", "ECE"),
    ("accuracy_gain", "Accuracy Gain"),
    ("flops", "FLOPs"),
    ("tokens_per_second", "Tokens/sec"),
    ("total_params", "Total Params"),
    ("active_params", "Active Params"),
    ("model_size_mb", "Model Size (MB)"),
    ("gpu_memory_allocated_gb", "GPU Memory (GB)"),
]


def build_variant_comparison_table(experiments_dir: str | Path = "experiments") -> pd.DataFrame:
    """Build the side-by-side Standard vs KD table across all MoE variants."""
    results_dict = _load_variant_results_dict(experiments_dir)

    def _fmt(key, value):
        if not isinstance(value, float):
            return value
        if key in ("accuracy", "top2_accuracy", "ece", "accuracy_gain"):
            return f"{value:.4f}"
        if key == "flops":
            return f"{value:.2e}"
        if key == "tokens_per_second":
            return f"{value:.2f}"
        if key in ("total_params", "active_params"):
            return f"{int(value):,}"
        if key in ("model_size_mb", "gpu_memory_allocated_gb"):
            return f"{value:.2f}"
        return f"{value:.4f}"

    rows = []
    for variant_name in sorted(results_dict.keys()):
        row = {"Variant": variant_name}
        variant_data = results_dict[variant_name]
        for key, label in _VARIANT_TABLE_METRICS:
            for mode in ("standard", "kd"):
                mode_label = "Standard" if mode == "standard" else "KD"
                if mode in variant_data and key in variant_data[mode]:
                    row[f"{mode_label} {label}"] = _fmt(key, variant_data[mode][key])
                else:
                    row[f"{mode_label} {label}"] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("MoE Variants Experiment Results Comparison (Standard vs Knowledge Distillation)")
    print(df.to_string(index=False))
    return df


def add_value_labels(ax, bars, format_str: str = "{:.3f}", offset_factor: float = 0.02) -> None:
    """Add a value label above (or below, if negative) each bar in `bars`."""
    max_height = max([abs(bar.get_height()) for bar in bars if bar.get_height() != 0], default=0)
    y_offset = max_height * offset_factor if max_height > 0 else 0.01

    for bar in bars:
        height = bar.get_height()
        if height != 0:
            label_y = height + (y_offset if height >= 0 else -y_offset)
            ax.text(bar.get_x() + bar.get_width() / 2.0, label_y, format_str.format(height), ha="center", va="bottom" if height >= 0 else "top", fontsize=8)


# (metric_key, panel title, y-label, value format, unit divisor, is "diff vs baseline" panel)
_VARIANT_GRID_METRICS = [
    ("accuracy", "Accuracy Comparison (Baseline Reference Lines)", "Accuracy", "{:.3f}", 1, False),
    ("accuracy", "Accuracy Difference vs MoE Baseline", "Accuracy Difference vs Baseline", "{:.4f}", 1, True),
    ("flops", "FLOPs Comparison (Baseline Reference Lines)", "FLOPs (TFLOPs)", "{:.2f}", 1e12, False),
    ("tokens_per_second", "Throughput Comparison", "Tokens/sec", "{:.1f}", 1, False),
    ("ms_per_token", "Latency Comparison", "ms/token", "{:.2f}", 1, False),
    ("model_size_mb", "Model Size Comparison", "Size (GB)", "{:.2f}", 1024, False),
    ("ece", "ECE Comparison (Lower is Better)", "ECE", "{:.4f}", 1, False),
    ("accuracy_gain", "Accuracy Gain vs Pre-Training", "Accuracy Gain", "{:.4f}", 1, False),
]


def plot_variant_comparison_grid(
    experiments_dir: str | Path = "experiments",
    results_dir: str | Path = "results",
    save_path: str = "results/moe_variants.png",
) -> None:
    """Comprehensive Standard-vs-KD grid across all MoE variants, referenced against
    the moe_baseline standard/KD results as horizontal reference lines.

    Condenses the notebook's per-metric copy-pasted blocks (lines 7869-8377) into
    one loop over `_VARIANT_GRID_METRICS`; produces the same set of metrics/panels.
    """
    experiments_dir = Path(experiments_dir)
    results_dir = Path(results_dir)
    results_dict = _load_variant_results_dict(experiments_dir)
    variants = sorted(results_dict.keys())
    if not variants:
        print("No variant results found.")
        return

    moe_baseline_standard = load_results(experiments_dir / "moe_baseline_standard_results.json") or {}
    moe_baseline_kd = load_results(experiments_dir / "moe_baseline_kd_results.json") or {}

    n_metrics = len(_VARIANT_GRID_METRICS)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6.5 * n_rows))
    fig.suptitle(
        "MoE Variants Comprehensive Comparison: Standard vs Knowledge Distillation (vs MoE Baseline)",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    axes = np.array(axes).reshape(-1)

    x = np.arange(len(variants))
    width = 0.35

    for panel_idx, (key, title, ylabel, fmt, divisor, is_diff) in enumerate(_VARIANT_GRID_METRICS):
        ax = axes[panel_idx]
        baseline_std = moe_baseline_standard.get(key, 0) / divisor if moe_baseline_standard else 0
        baseline_kd = moe_baseline_kd.get(key, 0) / divisor if moe_baseline_kd else 0

        standard_vals, kd_vals = [], []
        for variant in variants:
            vdata = results_dict[variant]
            std_val = vdata.get("standard", {}).get(key, 0) / divisor if "standard" in vdata else 0
            kd_val = vdata.get("kd", {}).get(key, 0) / divisor if "kd" in vdata else 0
            standard_vals.append(std_val - baseline_std if is_diff else std_val)
            kd_vals.append(kd_val - baseline_kd if is_diff else kd_val)

        bars_std = ax.bar(x - width / 2, standard_vals, width, label="Standard", color="#3498db", alpha=0.8, edgecolor="black", linewidth=0.5)
        bars_kd = ax.bar(x + width / 2, kd_vals, width, label="KD", color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.5)

        if not is_diff:
            if baseline_std:
                ax.axhline(y=baseline_std, color="#3498db", linestyle="--", linewidth=2, alpha=0.6, label=f"Baseline Std ({baseline_std:.4g})")
            if baseline_kd:
                ax.axhline(y=baseline_kd, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.6, label=f"Baseline KD ({baseline_kd:.4g})")
        else:
            ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

        ax.set_xlabel("Variant", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        add_value_labels(ax, list(bars_std) + list(bars_kd), fmt)

    for extra_idx in range(n_metrics, len(axes)):
        axes[extra_idx].axis("off")

    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved variant comparison grid to {save_path}")
    plt.show()

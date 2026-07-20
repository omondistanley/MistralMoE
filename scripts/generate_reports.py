#!/usr/bin/env python
"""CLI: regenerate the results/*.png and *.csv reports from existing results/ and
experiments/ JSON files (no GPU/model required -- pure post-processing).

Usage:
    python scripts/generate_reports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mistralmoe.viz.plots import (
    build_main_comparison_table,
    build_variant_comparison_table,
    create_prioritized_kd_metrics_table,
    load_all_metrics_with_correct_teachers,
    load_all_metrics_with_dense_teacher,
    plot_kte_all_models,
    plot_moe_baseline_knowledge_transfer,
    plot_retention_rates_all_models,
    plot_tier1_core_impact_metrics,
    plot_tier2_comparative_metrics,
    plot_tier3_training_dynamics,
    plot_training_losses_all_models,
    plot_variant_comparison_grid,
)


def main() -> None:
    print("=" * 80)
    print("Building main 4-model comparison table + chart")
    print("=" * 80)
    build_main_comparison_table()

    print("=" * 80)
    print("Loading KD metrics (per-model-appropriate teachers)")
    print("=" * 80)
    metrics_correct_teachers = load_all_metrics_with_correct_teachers()
    if metrics_correct_teachers["training"]:
        plot_training_losses_all_models(metrics_correct_teachers["training"])
    if metrics_correct_teachers["post_training"]:
        plot_retention_rates_all_models(metrics_correct_teachers["post_training"])
        plot_kte_all_models(metrics_correct_teachers["post_training"])
        plot_moe_baseline_knowledge_transfer(metrics_correct_teachers["post_training"])

    print("=" * 80)
    print("Loading KD metrics (dense baseline as teacher for all models)")
    print("=" * 80)
    metrics_dense_teacher = load_all_metrics_with_dense_teacher()
    create_prioritized_kd_metrics_table(metrics_dense_teacher)
    if metrics_dense_teacher["post_training"]:
        plot_tier1_core_impact_metrics(metrics_dense_teacher["post_training"])
        plot_tier2_comparative_metrics(metrics_dense_teacher["post_training"])
    if metrics_dense_teacher["training"]:
        plot_tier3_training_dynamics(metrics_dense_teacher["training"])

    print("=" * 80)
    print("Building cross-variant comparison table + grid")
    print("=" * 80)
    build_variant_comparison_table()
    plot_variant_comparison_grid()

    print("Done. See results/*.png and results/*.csv")


if __name__ == "__main__":
    main()

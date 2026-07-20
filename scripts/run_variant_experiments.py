#!/usr/bin/env python
"""CLI: run one or more MoE architectural variants end-to-end (pre-train eval ->
standard fine-tuning -> KD fine-tuning), matching the README's usage example.

Usage:
    python scripts/run_variant_experiments.py --list
    python scripts/run_variant_experiments.py moe_baseline efficient_4x1 placement_mixed_8x2
    python scripts/run_variant_experiments.py moe_baseline --max-samples 1000 --train-steps 250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mistralmoe.config import EXPERIMENT_CONFIGS, KD_CONFIG_STANDARD, configure_environment
from mistralmoe.data import load_base_model_and_tokenizer, load_mmlu_dataset
from mistralmoe.experiments.runner import MoEExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="*", help="Variant names from EXPERIMENT_CONFIGS to run")
    parser.add_argument("--list", action="store_true", help="List available variant names and exit")
    parser.add_argument("--max-samples", type=int, default=1000, help="MMLU samples per evaluation phase")
    parser.add_argument("--train-steps", type=int, default=250, help="Training steps per phase")
    parser.add_argument("--standard-only", action="store_true", help="Skip the KD phase (standard fine-tuning only)")
    args = parser.parse_args()

    if args.list or not args.variants:
        print("Available variants:")
        for name, config in EXPERIMENT_CONFIGS.items():
            print(f"  {name}: {config.description}")
        if not args.variants:
            return

    unknown = [v for v in args.variants if v not in EXPERIMENT_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {unknown}. Use --list to see available names.")

    configure_environment()

    print("Loading MMLU dataset...")
    mmlu = load_mmlu_dataset()

    print("Loading dense teacher model + tokenizer...")
    teacher_model, tokenizer, answer_tokens = load_base_model_and_tokenizer()
    teacher_model.eval()

    runner = MoEExperimentRunner(
        base_model=None,
        tokenizer=tokenizer,
        eval_dataset=mmlu.eval_dataset,
        answer_tokens=answer_tokens,
        train_dataset=mmlu.train_dataset,
        teacher_model=teacher_model,
    )

    for name in args.variants:
        config = EXPERIMENT_CONFIGS[name]
        runner.run_experiment(
            config,
            max_samples=args.max_samples,
            train=True,
            train_steps=args.train_steps,
            train_both_modes=not args.standard_only,
            training_mode="standard" if args.standard_only else "kd",
            kd_config=dict(KD_CONFIG_STANDARD),
        )

    runner.compare_experiments()


if __name__ == "__main__":
    main()

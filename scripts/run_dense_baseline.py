#!/usr/bin/env python
"""CLI: evaluate the dense Mistral-7B baseline, optionally fine-tune it with KD.

Usage:
    python scripts/run_dense_baseline.py                 # baseline eval only
    python scripts/run_dense_baseline.py --kd             # + Dense+KD training
    python scripts/run_dense_baseline.py --kd --steps 250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mistralmoe.config import KD_CONFIG_DENSE, TRAINING_CONFIG_DENSE, configure_environment
from mistralmoe.data import load_base_model_and_tokenizer, load_mmlu_dataset
from mistralmoe.experiments.pipelines import evaluate_dense_baseline, train_dense_with_kd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kd", action="store_true", help="Also train a Dense+KD student after the baseline eval")
    parser.add_argument("--steps", type=int, default=250, help="KD training steps (only used with --kd)")
    parser.add_argument("--max-eval-samples", type=int, default=1000, help="MMLU samples for evaluation")
    args = parser.parse_args()

    configure_environment()

    print("Loading MMLU dataset...")
    mmlu = load_mmlu_dataset()

    print("Loading dense base model + tokenizer...")
    model, tokenizer, answer_tokens = load_base_model_and_tokenizer()

    baseline_comprehensive = evaluate_dense_baseline(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=mmlu.eval_dataset,
        answer_tokens=answer_tokens,
        max_samples=args.max_eval_samples,
    )

    if args.kd:
        training_config = {**TRAINING_CONFIG_DENSE, "max_steps": args.steps}
        train_dense_with_kd(
            teacher_model=model,
            tokenizer=tokenizer,
            dataset=mmlu.dataset,
            eval_dataset=mmlu.eval_dataset,
            answer_tokens=answer_tokens,
            baseline_comprehensive=baseline_comprehensive,
            kd_config=dict(KD_CONFIG_DENSE),
            training_config=training_config,
        )


if __name__ == "__main__":
    main()

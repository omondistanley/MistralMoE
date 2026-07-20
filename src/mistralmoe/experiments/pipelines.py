"""Top-level dense-model pipelines: baseline evaluation and Dense+KD training.

Ported from moe_complete.ipynb's procedural (non-function) cells:
- evaluate_dense_baseline: lines 1349-1442
- train_dense_with_kd: lines 1604-2130

The notebook versions were sequential script cells that read/wrote bare
globals (`dataset`, `train_dataset`, `OUTPUT_DIR`, `baseline_comprehensive`,
...) and branched on `if 'x' in globals()` existence checks. Here each is a
single function taking its inputs as explicit parameters — the existence
checks are gone because a function's parameters are always defined.
"""

from __future__ import annotations

import json
import os

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments

from ..config import KD_CONFIG_DENSE, LORA_CONFIG_DENSE, MAX_LENGTH, MODEL_ID, TRAINING_CONFIG_DENSE
from ..data import make_tokenize_fn
from ..evaluation.metrics import (
    compute_kd_specific_metrics,
    compute_memory_metrics,
    compute_model_flops,
    compute_parameter_efficiency,
    compute_throughput_metrics,
    evaluate_mmlu_comprehensive,
)
from ..training.trainers import DenseKDTrainer


def evaluate_dense_baseline(model, tokenizer, eval_dataset, answer_tokens, max_samples: int = 1000) -> dict:
    """Run the full comprehensive evaluation suite on the (untrained) dense baseline
    and save it to results/baseline_comprehensive.json.
    """
    print("BASELINE EVALUATION")
    baseline_results = evaluate_mmlu_comprehensive(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        answer_tokens=answer_tokens,
        device="cuda",
        max_samples=max_samples,
        show_progress=True,
    )

    print("\nComputing FLOPs...")
    baseline_flops = compute_model_flops(model, seq_length=MAX_LENGTH)

    print("Measuring throughput...")
    baseline_throughput = compute_throughput_metrics(model=model, tokenizer=tokenizer, eval_dataset=eval_dataset, max_samples=100)

    print("Analyzing parameter efficiency...")
    baseline_params = compute_parameter_efficiency(model=model, num_experts_per_tok=1)  # Dense model

    print("Collecting memory metrics...")
    baseline_memory = compute_memory_metrics(model)

    baseline_comprehensive = {
        **baseline_results,
        "flops": baseline_flops,
        **baseline_throughput,
        **baseline_params,
        **baseline_memory,
    }

    print("COMPREHENSIVE BASELINE METRICS")
    print("Accuracy Metrics:")
    print(f"  MMLU Accuracy: {baseline_comprehensive['accuracy']:.4f}")
    print(f"  Top-2 Accuracy: {baseline_comprehensive['top2_accuracy']:.4f}")
    print(f"  ECE: {baseline_comprehensive['ece']:.4f}")
    print("\n Computational Efficiency:")
    print(f"  FLOPs per forward pass: {baseline_comprehensive['flops'] / 1e9:.2f}G")
    print(f"  Tokens/second: {baseline_comprehensive['tokens_per_second']:.2f}")
    print(f"  ms/token: {baseline_comprehensive['ms_per_token']:.2f}")
    print(f"  Samples/second: {baseline_comprehensive['samples_per_second']:.2f}")
    print("\n Parameter Efficiency:")
    print(f"  Total parameters: {baseline_comprehensive['total_params'] / 1e9:.2f}B")
    print(f"  Active parameters: {baseline_comprehensive['active_params'] / 1e9:.2f}B")
    print(f"  Trainable parameters: {baseline_comprehensive['trainable_params'] / 1e6:.2f}M")
    print(f"  Sparsity ratio: {baseline_comprehensive['sparsity_ratio']:.2%}")
    print("\n Memory Usage:")
    print(f"  Model size: {baseline_comprehensive['model_size_mb']:.2f} MB")
    print(f"  GPU allocated: {baseline_comprehensive['gpu_memory_allocated_gb']:.2f} GB")
    print(f"  GPU reserved: {baseline_comprehensive['gpu_memory_reserved_gb']:.2f} GB")

    os.makedirs("results", exist_ok=True)
    with open("results/baseline_comprehensive.json", "w") as f:
        json.dump(
            {k: v for k, v in baseline_comprehensive.items() if not hasattr(v, "tolist")},
            f,
            indent=2,
            default=str,
        )

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "baseline/accuracy": baseline_comprehensive["accuracy"],
                    "baseline/flops_billions": baseline_comprehensive["flops"] / 1e9,
                    "baseline/tokens_per_second": baseline_comprehensive["tokens_per_second"],
                    "baseline/ms_per_token": baseline_comprehensive["ms_per_token"],
                    "baseline/active_params_billions": baseline_comprehensive["active_params"] / 1e9,
                    "baseline/gpu_memory_gb": baseline_comprehensive["gpu_memory_allocated_gb"],
                }
            )
    except Exception:
        pass

    print("\n Comprehensive baseline evaluation complete!")
    return baseline_comprehensive


def train_dense_with_kd(
    teacher_model,
    tokenizer,
    dataset,
    eval_dataset,
    answer_tokens,
    baseline_comprehensive: dict | None = None,
    model_id: str = MODEL_ID,
    kd_config: dict = KD_CONFIG_DENSE,
    lora_config: dict = LORA_CONFIG_DENSE,
    training_config: dict = TRAINING_CONFIG_DENSE,
    output_dir: str = "./dense_model_kd",
    use_subset: bool = True,
    subset_percentage: float = 0.2,
) -> dict:
    """Train a fresh dense student with LoRA + Knowledge Distillation from `teacher_model`,
    evaluate it comprehensively, and save results to results/trained_dense_kd_comprehensive.json.

    `dataset` is the `DatasetDict` from `data.load_mmlu_dataset().dataset`
    (with 'train'/'test' splits); `eval_dataset` is its 'test' split (or an
    already-tokenized eval set for accuracy).
    """
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    print("=" * 80)
    print("DENSE MODEL TRAINING CONFIGURATION\n")
    print(f"KD Alpha: {kd_config['kd_alpha']}")
    print(f"Temperature: {kd_config['temperature']}")
    print(f"Output Directory: {output_dir}")
    print(f"Max Steps: {training_config['max_steps']}")
    print("=" * 80)

    print("\nSetting up student model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    student_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    student_model.config.use_cache = False
    student_model = prepare_model_for_kbit_training(student_model)
    print("Student model prepared for k-bit training")

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **lora_config)
    student_model = get_peft_model(student_model, peft_config)
    print("LoRA adapters applied")

    student_model.gradient_checkpointing_enable()
    student_model.train()

    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e9:.2f}B ({100 * trainable_params / total_params:.2f}%)")

    print("Tokenizing training data (with answers)...")
    tokenize_train_fn = make_tokenize_fn(tokenizer, max_length=MAX_LENGTH, with_answer=True)
    tokenized_train = dataset["train"].map(
        tokenize_train_fn, batched=True, remove_columns=dataset["train"].column_names, desc="Tokenizing training data"
    )

    print("Tokenizing eval data (without answers)...")
    tokenize_eval_fn = make_tokenize_fn(tokenizer, max_length=MAX_LENGTH, with_answer=False)
    tokenized_eval = dataset["test"].map(
        tokenize_eval_fn, batched=True, remove_columns=dataset["test"].column_names, desc="Tokenizing eval data"
    )
    print(f"Created tokenized datasets: {len(tokenized_train):,} train, {len(tokenized_eval):,} eval")

    if use_subset:
        subset_size = int(len(tokenized_train) * subset_percentage)
        train_dataset_subset = tokenized_train.select(range(subset_size))
        print(f"Using {subset_percentage * 100}% of training data: {len(train_dataset_subset)} samples")
    else:
        train_dataset_subset = tokenized_train

    training_args = TrainingArguments(
        output_dir=output_dir,
        **training_config,
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=f"dense_kd_{kd_config['kd_alpha']}",
        logging_first_step=True,
        logging_strategy="steps",
        eval_strategy="no",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("\nInitializing trainer...")
    trainer = DenseKDTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset_subset,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        teacher_model=teacher_model,
        kd_config=kd_config,
    )
    print("Dense student model Trainer with Knowledge Distillation initialized successfully!")

    print("STARTING DENSE MODEL TRAINING")
    torch.cuda.empty_cache()
    train_result = trainer.train()

    print("DENSE MODEL TRAINING COMPLETE")
    print("\nFinal training metrics:")
    for key, value in train_result.metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print(f"\nSaving model to {output_dir}/final_model...")
    trainer.save_model(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    # Comprehensive evaluation of the trained student.
    print("TRAINED DENSE (KD) MODEL COMPREHENSIVE EVALUATION")
    student_model.eval()

    print("1. MMLU ACCURACY EVALUATION")
    trained_results = evaluate_mmlu_comprehensive(
        model=student_model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        answer_tokens=answer_tokens,
        device="cuda",
        max_samples=1000,
        show_progress=True,
    )

    print("2. COMPUTATIONAL EFFICIENCY (FLOPs)")
    trained_flops = compute_model_flops(student_model, seq_length=MAX_LENGTH)

    print("3. THROUGHPUT METRICS")
    trained_throughput = compute_throughput_metrics(model=student_model, tokenizer=tokenizer, eval_dataset=eval_dataset, max_samples=100)

    print("4. PARAMETER EFFICIENCY")
    trained_params = compute_parameter_efficiency(model=student_model, num_experts_per_tok=1)

    print("5. MEMORY USAGE")
    trained_memory = compute_memory_metrics(student_model)

    trained_comprehensive = {
        **trained_results,
        "flops": trained_flops,
        **trained_throughput,
        **trained_params,
        **trained_memory,
    }

    print("6. Training metrics and logs for dense KD model")
    training_metrics = trainer.get_training_metrics_summary()
    if training_metrics:
        trained_comprehensive["training_metrics"] = training_metrics
        print(f"  Final NTP Loss: {training_metrics['final_ntp_loss']:.4f}")
        print(f"  Final KD Loss: {training_metrics['final_kd_loss']:.4f}")
        print(f"  Average KD/NTP Ratio: {training_metrics['avg_kd_ntp_ratio']:.4f}")

    print("7. KNOWLEDGE DISTILLATION SPECIFIC METRICS")
    kd_specific_metrics = None
    if baseline_comprehensive is not None:
        try:
            kd_specific_metrics = compute_kd_specific_metrics(
                teacher_metrics=baseline_comprehensive,
                student_kd_metrics=trained_comprehensive,
                student_no_kd_metrics=None,
                kd_config=kd_config,
                alpha=0.5,
            )
            trained_comprehensive["kd_metrics"] = kd_specific_metrics
            print(f"  Accuracy Retention: {kd_specific_metrics['accuracy_retention_rate']:.2f}%")
            print(f"  Knowledge Transfer Efficiency: {kd_specific_metrics['knowledge_transfer_efficiency']:.4f}")
            print(f"  KD Effectiveness Score: {kd_specific_metrics['kd_effectiveness_score']:.4f}")
        except Exception as e:
            print(f"  Error computing KD-specific metrics: {e}")
    else:
        print("  No baseline_comprehensive provided. Skipping KD-specific metrics computation.")

    print("COMPREHENSIVE TRAINED DENSE (KD) EVALUATION COMPLETE!")

    os.makedirs("results", exist_ok=True)
    out_path = "results/trained_dense_kd_comprehensive.json"
    with open(out_path, "w") as f:
        json.dump({k: v for k, v in trained_comprehensive.items() if not hasattr(v, "tolist")}, f, indent=2, default=str)
    print(f"\nSaved KD dense evaluation to {out_path}")

    try:
        import wandb

        if wandb.run is not None:
            wandb_log_dict = {
                "dense_kd/accuracy": trained_comprehensive["accuracy"],
                "dense_kd/top2_accuracy": trained_comprehensive["top2_accuracy"],
                "dense_kd/ece": trained_comprehensive["ece"],
                "dense_kd/flops_billions": trained_comprehensive["flops"] / 1e9,
                "dense_kd/tokens_per_second": trained_comprehensive["tokens_per_second"],
                "dense_kd/total_params_billions": trained_comprehensive["total_params"] / 1e9,
                "dense_kd/active_params_billions": trained_comprehensive["active_params"] / 1e9,
                "dense_kd/gpu_memory_allocated_gb": trained_comprehensive["gpu_memory_allocated_gb"],
            }
            if training_metrics:
                wandb_log_dict.update(
                    {
                        "dense_kd_training/final_ntp_loss": training_metrics["final_ntp_loss"],
                        "dense_kd_training/final_kd_loss": training_metrics["final_kd_loss"],
                        "dense_kd_training/avg_kd_ntp_ratio": training_metrics["avg_kd_ntp_ratio"],
                    }
                )
            if kd_specific_metrics:
                wandb_log_dict.update(
                    {
                        "dense_kd_metrics/accuracy_retention_rate": kd_specific_metrics["accuracy_retention_rate"],
                        "dense_kd_metrics/knowledge_transfer_efficiency": kd_specific_metrics["knowledge_transfer_efficiency"],
                        "dense_kd_metrics/kd_effectiveness_score": kd_specific_metrics["kd_effectiveness_score"],
                    }
                )
            wandb.log(wandb_log_dict)
            print(" Logged KD dense evaluation to WandB")
    except Exception as e:
        print(f" Could not log to WandB: {e}")

    return trained_comprehensive

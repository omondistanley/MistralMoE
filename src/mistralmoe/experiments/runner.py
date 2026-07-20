"""MoEExperimentRunner: the three-phase (pre-train eval -> standard FT -> KD FT)
experiment orchestrator used to sweep all EXPERIMENT_CONFIGS variants.

Ported from moe_complete.ipynb lines 6509-7517.

Notebook-specific adaptations (per the modularization plan):
- `globals()['teacher_model']` reads/writes (used across `_clear_all_models_except_teacher`,
  `_cleanup_model`, `_create_moe_model`, `_train_model`, `run_experiment`) are replaced with
  a real `self.teacher_model` instance attribute — the notebook's globals()-poking was a
  Jupyter-only workaround for keeping the teacher alive across cells and has no equivalent
  outside a notebook kernel.
- The bare `model_id` global read in `_create_moe_model` becomes `self.model_id`.
- `is_model_object` / the "Memory check and clean-up" cell (notebook lines 6010-6173) is
  intentionally NOT ported: it's a standalone utility that scans the notebook's `globals()`
  dict for stray model references to garbage-collect between cells. It is never called by
  `MoEExperimentRunner` itself, and its entire purpose (scanning interpreter globals) doesn't
  translate outside a Jupyter kernel.
"""

from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments

from ..config import MAX_LENGTH, MODEL_ID, MoEExperimentConfig
from ..evaluation.metrics import (
    compute_memory_metrics,
    compute_model_flops,
    compute_parameter_efficiency,
    compute_throughput_metrics,
    evaluate_mmlu_comprehensive,
)
from ..models.upcycle import replace_ffn_with_moe
from ..training.trainers import IntegratedMoEKDTrainer, MoETrainer


def _make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.float32):
        return float(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj) if not isinstance(obj, (dict, list, str, int, float, bool, type(None))) else obj


def _clean_for_json(d):
    """Recursively clean a results dict/list for JSON serialization."""
    if isinstance(d, dict):
        return {k: _clean_for_json(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean_for_json(item) for item in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, (np.integer, np.floating)):
        return float(d)
    if isinstance(d, set):
        return list(d)
    return _make_json_serializable(d)


class MoEExperimentRunner:
    """Runs and tracks MoE variant experiments (pre-train eval, standard FT, KD FT)."""

    def __init__(
        self,
        base_model,
        tokenizer,
        eval_dataset,
        answer_tokens,
        train_dataset=None,
        teacher_model=None,
        model_id: str = MODEL_ID,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.answer_tokens = answer_tokens
        self.train_dataset = train_dataset
        self.teacher_model = teacher_model
        self.model_id = model_id
        self.results = {}
        self.current_model = None

    # -- Memory management -------------------------------------------------

    def _clear_all_models_except_teacher(self):
        """Aggressively clear GPU memory (except the frozen teacher) before a new experiment."""
        print("  Clearing GPU memory (preserving teacher_model)...")

        if self.current_model is not None and self.current_model is not self.teacher_model:
            self._cleanup_model(self.current_model, preserve_teacher=True)
            self.current_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"    GPU Memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    def _cleanup_model(self, model, preserve_teacher=True):
        """Aggressively free a model's GPU memory (unless it is the teacher model)."""
        if model is None:
            return

        teacher_model = self.teacher_model if preserve_teacher else None

        if model is teacher_model:
            print("  Skipping cleanup of teacher_model (preserved for KD)")
            return

        try:
            if hasattr(model, "cpu"):
                model.cpu()
            elif hasattr(model, "to"):
                model.to("cpu")
        except Exception as e:
            print(f"  Warning: Could not move model to CPU: {e}")

        try:
            if hasattr(model, "parameters"):
                for param in list(model.parameters()):
                    if param is not None:
                        del param
            if hasattr(model, "buffers"):
                for buffer in list(model.buffers()):
                    if buffer is not None:
                        del buffer
        except Exception as e:
            print(f"  Warning: Error deleting parameters: {e}")

        try:
            del model
        except Exception as e:
            print(f"  Warning: Error deleting model: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        for _ in range(2):
            gc.collect()

        print("  Model cleanup completed")

    # -- Model construction --------------------------------------------------

    def _create_moe_model(self, config: MoEExperimentConfig):
        """Create a fresh MoE model (bf16, sparse-upcycled) according to `config`."""
        print("  Clearing memory before loading model...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        for _ in range(2):
            gc.collect()

        # No quantization: loads in bf16 for correct weight extraction (matches baseline).
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        total_layers = len(model.model.layers)
        if config.expert_layers == "all":
            target_layers = None
        elif config.expert_layers == "every_2":
            target_layers = list(range(0, total_layers, 2))
        elif config.expert_layers == "every_4":
            target_layers = list(range(0, total_layers, 4))
        elif config.expert_layers == "selected" and config.layer_indices:
            target_layers = config.layer_indices
        else:
            target_layers = None

        # bnb_config=None: experts are trainable bf16 Linear layers (matches baseline).
        model = replace_ffn_with_moe(
            model=model,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            router_jitter_noise=config.router_jitter_noise,
            router_aux_loss_coef=config.router_aux_loss_coef,
            bnb_config=None,
            ram_threshold=80.0,
            use_disk_offload=True,
            layer_indices=target_layers,
            half_width=False,
            enable_cpu_offload=False,
        )

        torch.cuda.empty_cache()
        return model

    def _apply_lora(self, model):
        """Apply LoRA adapters (attention only) for training."""
        is_quantized = False
        try:
            for _name, module in model.named_modules():
                if hasattr(module, "weight") and hasattr(module.weight, "quant_state"):
                    is_quantized = True
                    break
        except Exception:
            pass

        if is_quantized:
            model = prepare_model_for_kbit_training(model)
        else:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        model = get_peft_model(model, lora_config)
        model.train()
        return model

    # -- Training / evaluation -----------------------------------------------

    def _train_model(self, model, config: MoEExperimentConfig, steps, training_mode="standard", kd_config=None, skip_eval=True):
        """Run the training loop with MoETrainer (standard) or IntegratedMoEKDTrainer (KD)."""
        teacher_model = None
        if training_mode == "kd":
            if self.teacher_model is None:
                raise ValueError("teacher_model must be set on the runner for KD training")
            teacher_model = self.teacher_model

        eval_dataset_for_trainer = None
        if not skip_eval and self.eval_dataset is not None:
            if hasattr(self.eval_dataset, "column_names") and "input_ids" in self.eval_dataset.column_names:
                eval_dataset_for_trainer = self.eval_dataset
            else:

                def tokenize_eval_function(examples):
                    return self.tokenizer(
                        examples["formatted_prompt"],
                        truncation=True,
                        max_length=MAX_LENGTH,
                        padding=False,
                    )

                eval_dataset_for_trainer = self.eval_dataset.map(
                    tokenize_eval_function,
                    batched=True,
                    remove_columns=self.eval_dataset.column_names,
                    desc="Tokenizing eval dataset for Trainer",
                )
        elif skip_eval:
            print("  Skipping training evaluation for faster training")

        output_dir = f"experiments/{config.experiment_name}_{training_mode}_checkpoints"
        os.makedirs(output_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=output_dir,
            max_steps=steps,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_ratio=0.1,
            logging_dir=f"{output_dir}/logs",
            logging_steps=25,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=steps,
            save_total_limit=1,
            load_best_model_at_end=False,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to=["tensorboard", "wandb"],
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
        )

        if training_mode == "kd":
            print("  Using IntegratedMoEKDTrainer for knowledge distillation")
            print(
                f"  KD Config: alpha={kd_config.get('kd_alpha', 0.5)}, "
                f"temp={kd_config.get('temperature', 4.0)}, "
                f"routing_kd={kd_config.get('enable_routing_kd', False)}"
            )

            trainer_kwargs = {
                "model": model,
                "args": args,
                "train_dataset": self.train_dataset,
                "tokenizer": self.tokenizer,
                "data_collator": DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                "teacher_model": teacher_model,
                "kd_config": kd_config,
                "router_aux_loss_coef": config.router_aux_loss_coef,
            }
            if not skip_eval and eval_dataset_for_trainer is not None:
                trainer_kwargs["eval_dataset"] = eval_dataset_for_trainer

            trainer = IntegratedMoEKDTrainer(**trainer_kwargs)
        else:
            print("  Using MoETrainer for standard training")
            trainer_kwargs = {
                "model": model,
                "args": args,
                "train_dataset": self.train_dataset,
                "tokenizer": self.tokenizer,
                "data_collator": DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
                "router_aux_loss_coef": config.router_aux_loss_coef,
            }
            if not skip_eval and eval_dataset_for_trainer is not None:
                trainer_kwargs["eval_dataset"] = eval_dataset_for_trainer

            trainer = MoETrainer(**trainer_kwargs)

        trainer.train()

    def _evaluate_comprehensive(self, model, config: MoEExperimentConfig, max_samples):
        """Run the full accuracy/FLOPs/throughput/parameter/memory evaluation suite."""
        print(f"  Running MMLU evaluation (n={max_samples})...")
        mmlu_results = evaluate_mmlu_comprehensive(
            model=model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            answer_tokens=self.answer_tokens,
            device="cuda",
            max_samples=max_samples,
            show_progress=False,
        )

        flops = compute_model_flops(model, seq_length=MAX_LENGTH)

        throughput_metrics = compute_throughput_metrics(
            model=model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset,
            max_samples=50,
        )

        param_metrics = compute_parameter_efficiency(model=model, num_experts_per_tok=config.num_experts_per_tok)
        memory_metrics = compute_memory_metrics(model)

        results = {
            "accuracy": mmlu_results["accuracy"],
            "top2_accuracy": mmlu_results["top2_accuracy"],
            "ece": mmlu_results["ece"],
            "flops": flops,
            **throughput_metrics,
            **param_metrics,
            **memory_metrics,
        }

        print(f"  -> Accuracy: {results['accuracy']:.4f} | Throughput: {results['tokens_per_second']:.1f} tok/s")
        return results

    # -- Persistence -----------------------------------------------------------

    def _save_results(self, exp_name, results):
        """Save results to disk, both as an individual file and merged into the consolidated file."""
        os.makedirs("experiments", exist_ok=True)

        clean_results = _clean_for_json(results)

        individual_filepath = f"experiments/{exp_name}_results.json"
        with open(individual_filepath, "w") as f:
            json.dump(clean_results, f, indent=2, default=_make_json_serializable)

        consolidated_filepath = "experiments/all_variants_results.json"
        all_results = {}
        if os.path.exists(consolidated_filepath):
            try:
                with open(consolidated_filepath, "r") as f:
                    all_results = json.load(f)
            except Exception:
                all_results = {}

        all_results[exp_name] = clean_results
        with open(consolidated_filepath, "w") as f:
            json.dump(all_results, f, indent=2, default=_make_json_serializable)

        print(f"  Saved results: {individual_filepath}")
        print(f"  Updated consolidated file: {consolidated_filepath}")

    def save_all_results(self):
        """Explicitly save all current in-memory results to the consolidated file."""
        os.makedirs("experiments", exist_ok=True)

        all_results_clean = {name: _clean_for_json(results) for name, results in self.results.items()}

        consolidated_filepath = "experiments/all_variants_results.json"
        with open(consolidated_filepath, "w") as f:
            json.dump(all_results_clean, f, indent=2, default=_make_json_serializable)

        print(f"Saved all {len(all_results_clean)} experiment results to {consolidated_filepath}")
        return consolidated_filepath

    # -- Main entrypoint ---------------------------------------------------

    def run_experiment(
        self,
        config: MoEExperimentConfig,
        max_samples=100,
        train=False,
        train_steps=150,
        training_mode="standard",
        kd_config=None,
        skip_training_eval=True,
        train_both_modes=False,
    ):
        """Run a complete experiment for one config.

        `train_both_modes=True` runs all three phases (pre-train eval -> standard
        FT -> KD FT) and returns a dict with all three under `phases`, matching
        the README's `{variant}_unified_results.json` schema. Otherwise runs a
        single mode (`training_mode`) and returns just that phase's results.
        """
        if train_both_modes and train:
            return self._run_experiment_both_modes(config, max_samples, train_steps, kd_config, skip_training_eval)
        return self._run_experiment_single_mode(
            config, max_samples, train, train_steps, training_mode, kd_config, skip_training_eval
        )

    def _run_experiment_both_modes(self, config: MoEExperimentConfig, max_samples, train_steps, kd_config, skip_training_eval):
        print(f"\n{'=' * 70}")
        print(f"RUNNING EXPERIMENT WITH BOTH TRAINING MODES: {config.experiment_name}")
        print(f"{'=' * 70}\n")

        all_results = {"experiment_name": config.experiment_name, "config": config.to_dict(), "phases": {}}

        if kd_config is None:
            kd_config = {
                "kd_alpha": 0.5,
                "temperature": 4.0,
                "routing_kd_weight": 0.0,
                "enable_routing_kd": False,
                "name": "Standard KD",
            }
            print(f"  Using default KD config: {kd_config['name']}")

        if self.teacher_model is None:
            raise ValueError("teacher_model must be set on the runner for KD training")

        # PHASE 0: Pre-training Evaluation (shared for both modes).
        print(f"\n{'#' * 70}\n# PHASE 0: PRE-TRAINING EVALUATION (Shared)\n{'#' * 70}\n")

        self._clear_all_models_except_teacher()
        moe_model = self._create_moe_model(config)
        self.current_model = moe_model

        pre_results = self._evaluate_comprehensive(model=moe_model, config=config, max_samples=max_samples)
        all_results["phases"]["pre_training"] = {**pre_results, "phase": "pre_training", "timestamp": time.time()}

        print("\nPre-training Results:")
        print(f"  Accuracy: {pre_results['accuracy']:.4f}")
        print(f"  Top-2 Accuracy: {pre_results['top2_accuracy']:.4f}")
        print(f"  ECE: {pre_results['ece']:.4f}")

        # PHASE 1: Standard training.
        print(f"\n{'#' * 70}\n# PHASE 1: STANDARD TRAINING\n{'#' * 70}\n")

        self._cleanup_model(moe_model, preserve_teacher=True)
        self.current_model = None
        self._clear_all_models_except_teacher()

        moe_model_std = self._create_moe_model(config)
        self.current_model = moe_model_std
        moe_model_std = self._apply_lora(moe_model_std)
        self.current_model = moe_model_std

        self._train_model(moe_model_std, config, train_steps, training_mode="standard", kd_config=None, skip_eval=skip_training_eval)

        standard_results = self._evaluate_comprehensive(model=moe_model_std, config=config, max_samples=max_samples)
        standard_results.update(
            {
                "phase": "standard_training",
                "training_mode": "standard",
                "pre_train_accuracy": pre_results["accuracy"],
                "accuracy_gain": standard_results["accuracy"] - pre_results["accuracy"],
                "timestamp": time.time(),
            }
        )

        all_results["phases"]["standard_training"] = dict(standard_results)
        standard_key = f"{config.experiment_name}_standard"
        all_results[standard_key] = standard_results
        self.results[standard_key] = standard_results
        self._save_results(standard_key, standard_results)

        print("\nStandard Training Results:")
        print(f"  Accuracy: {standard_results['accuracy']:.4f}")
        print(f"  Accuracy Gain: {standard_results['accuracy_gain']:+.4f}")
        print(f"  Top-2 Accuracy: {standard_results['top2_accuracy']:.4f}")

        # Cleanup between training modes.
        print(f"\n{'#' * 70}\n# CLEANUP BETWEEN TRAINING MODES\n{'#' * 70}\n")
        self._cleanup_model(moe_model_std, preserve_teacher=True)
        self.current_model = None
        self._clear_all_models_except_teacher()

        # PHASE 2: KD training.
        print(f"\n{'#' * 70}\n# PHASE 2: KNOWLEDGE DISTILLATION TRAINING\n{'#' * 70}\n")

        moe_model_kd = self._create_moe_model(config)
        self.current_model = moe_model_kd
        moe_model_kd = self._apply_lora(moe_model_kd)
        self.current_model = moe_model_kd

        self._train_model(moe_model_kd, config, train_steps, training_mode="kd", kd_config=kd_config, skip_eval=skip_training_eval)

        kd_results = self._evaluate_comprehensive(model=moe_model_kd, config=config, max_samples=max_samples)
        kd_results.update(
            {
                "phase": "kd_training",
                "training_mode": "kd",
                "pre_train_accuracy": pre_results["accuracy"],
                "accuracy_gain": kd_results["accuracy"] - pre_results["accuracy"],
                "kd_config": kd_config,
                "timestamp": time.time(),
            }
        )

        all_results["phases"]["kd_training"] = dict(kd_results)
        kd_key = f"{config.experiment_name}_kd"
        all_results[kd_key] = kd_results
        self.results[kd_key] = kd_results
        self._save_results(kd_key, kd_results)

        print("\nKD Training Results:")
        print(f"  Accuracy: {kd_results['accuracy']:.4f}")
        print(f"  Accuracy Gain: {kd_results['accuracy_gain']:+.4f}")
        print(f"  Top-2 Accuracy: {kd_results['top2_accuracy']:.4f}")

        all_results["timestamp"] = time.time()
        unified_key = f"{config.experiment_name}_unified"
        self.results[unified_key] = all_results
        self._save_results(unified_key, all_results)

        self._cleanup_model(moe_model_kd, preserve_teacher=True)
        self.current_model = None

        return all_results

    def _run_experiment_single_mode(
        self, config: MoEExperimentConfig, max_samples, train, train_steps, training_mode, kd_config, skip_training_eval
    ):
        print(f"\n{'=' * 70}\nRUNNING EXPERIMENT: {config.experiment_name}\n{'=' * 70}\n")
        print(f"Description: {config.description}")
        print("Configuration:")
        print(f"  Experts: {config.num_experts}")
        print(f"  Experts per token: {config.num_experts_per_tok}")
        print(f"  Layer placement: {config.expert_layers}")
        if train:
            print(f"  Training mode: {training_mode}")
            print(f"  Training steps: {train_steps}")
        print()

        print("Cleaning up memory before experiment...")
        self._clear_all_models_except_teacher()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory before model creation: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            if allocated > 10.0:
                print(f"  WARNING: High GPU memory usage detected ({allocated:.2f} GB).")
                print("  This may cause OOM. Teacher model may be using this memory.")

        moe_model = self._create_moe_model(config)
        self.current_model = moe_model

        print(f"\nPhase 1: Pre-training Evaluation (n={max_samples})...")
        pre_results = self._evaluate_comprehensive(model=moe_model, config=config, max_samples=max_samples)

        print(f"\n{'=' * 70}\nCOMPREHENSIVE PRE-TRAINING METRICS\n{'=' * 70}\n")
        print("Accuracy Metrics:")
        print(f"  MMLU Accuracy: {pre_results['accuracy']:.4f}")
        print(f"  Top-2 Accuracy: {pre_results['top2_accuracy']:.4f}")
        print(f"  ECE: {pre_results['ece']:.4f}")

        final_results = dict(pre_results)
        final_results["phase"] = "pre_train_only"
        final_results["training_mode"] = training_mode if train else None

        if train and self.train_dataset:
            print(f"\nPhase 2: Training for {train_steps} steps ({training_mode} mode)...")

            if training_mode == "kd":
                if kd_config is None:
                    raise ValueError("kd_config must be provided when training_mode='kd'")
                if self.teacher_model is None:
                    raise ValueError("teacher_model must be set on the runner for KD training")

            moe_model = self._apply_lora(moe_model)
            self.current_model = moe_model
            self._train_model(moe_model, config, train_steps, training_mode=training_mode, kd_config=kd_config, skip_eval=skip_training_eval)

            print(f"\nPhase 3: Post-training Evaluation (n={max_samples})...")
            post_results = self._evaluate_comprehensive(model=moe_model, config=config, max_samples=max_samples)

            print(f"\n{'=' * 70}\nCOMPREHENSIVE POST-TRAINING METRICS\n{'=' * 70}\n")
            print("Accuracy Metrics:")
            print(f"  MMLU Accuracy: {post_results['accuracy']:.4f}")
            print(f"  Top-2 Accuracy: {post_results['top2_accuracy']:.4f}")
            print(f"  ECE: {post_results['ece']:.4f}")

            final_results = dict(post_results)
            final_results["phase"] = "trained"
            final_results["training_mode"] = training_mode
            final_results["pre_train_accuracy"] = pre_results["accuracy"]
            final_results["accuracy_gain"] = post_results["accuracy"] - pre_results["accuracy"]
            final_results["pre_train_results"] = pre_results

        final_results["config"] = config.to_dict()
        final_results["timestamp"] = time.time()

        self.results[config.experiment_name] = final_results
        self._save_results(config.experiment_name, final_results)

        print(f"\nCleaning up {config.experiment_name} model...")
        self._cleanup_model(moe_model, preserve_teacher=True)
        self.current_model = None

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  GPU Memory after cleanup: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            print(f"  Teacher model preserved: {self.teacher_model is not None}")

        return final_results

    def compare_experiments(self, exp_names=None, include_baseline=True):
        """Build a pandas comparison table across experiments (accuracy, FLOPs, throughput, ...)."""
        import pandas as pd

        if exp_names is None:
            exp_names = list(self.results.keys())

        if include_baseline and "moe_baseline" in self.results and "moe_baseline" not in exp_names:
            exp_names = ["moe_baseline"] + [e for e in exp_names if e != "moe_baseline"]
            print("  Note: Including 'moe_baseline' in comparison")

        comparison_data = []
        baseline_data = None

        for name in exp_names:
            if name not in self.results:
                continue
            exp = self.results[name]

            if name == "moe_baseline":
                baseline_data = exp

            gain_str = "-"
            if "accuracy_gain" in exp:
                gain_str = f"{exp['accuracy_gain'] * 100:+.2f}%"
            elif baseline_data and name != "moe_baseline":
                baseline_acc = baseline_data.get("accuracy", 0)
                current_acc = exp.get("accuracy", 0)
                if baseline_acc > 0:
                    gain_str = f"{((current_acc - baseline_acc) / baseline_acc) * 100:+.2f}%"

            comparison_data.append(
                {
                    "Experiment": name,
                    "Pre Acc": f"{exp.get('pre_train_accuracy', exp.get('accuracy', 0)):.4f}",
                    "Post Acc": f"{exp.get('accuracy', 0):.4f}",
                    "Gain": gain_str,
                    "vs Baseline": (
                        f"{((exp.get('accuracy', 0) - baseline_data.get('accuracy', 0)) * 100):+.2f}%"
                        if baseline_data and name != "moe_baseline"
                        else "-"
                    ),
                    "Top-2": f"{exp.get('top2_accuracy', 0):.4f}",
                    "ECE": f"{exp.get('ece', 0):.4f}",
                    "FLOPs (G)": f"{exp.get('flops', 0) / 1e9:.2f}",
                    "Tokens/sec": f"{exp.get('tokens_per_second', 0):.1f}",
                    "ms/token": f"{exp.get('ms_per_token', 0):.2f}",
                    "Samples/sec": f"{exp.get('samples_per_second', 0):.2f}",
                    "Total Params (B)": f"{exp.get('total_params', 0) / 1e9:.2f}",
                    "Active Params (B)": f"{exp.get('active_params', 0) / 1e9:.2f}",
                    "Trainable Params (M)": f"{exp.get('trainable_params', 0) / 1e6:.2f}",
                    "Sparsity": f"{exp.get('sparsity_ratio', 0):.2%}",
                    "Model Size (MB)": f"{exp.get('model_size_mb', 0):.2f}",
                    "GPU Alloc (GB)": f"{exp.get('gpu_memory_allocated_gb', 0):.2f}",
                    "GPU Reserved (GB)": f"{exp.get('gpu_memory_reserved_gb', 0):.2f}",
                    "Training Mode": exp.get("training_mode", "N/A"),
                }
            )

        if not comparison_data:
            print("No results to compare yet.")
            return pd.DataFrame()

        df = pd.DataFrame(comparison_data)
        print("\n" + "=" * 180)
        print("EXPERIMENT COMPARISON")
        if baseline_data:
            print(f"Baseline: {baseline_data.get('accuracy', 0):.4f} accuracy")
        print("=" * 180 + "\n")
        print(df.to_string(index=False))

        return df

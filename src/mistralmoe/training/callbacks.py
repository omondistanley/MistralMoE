"""Trainer callback for periodic MMLU/loss/throughput evaluation during training.

Ported from moe_complete.ipynb lines 1113-1347.
"""

from __future__ import annotations

import gc
import time

import torch
from torch.utils.data import DataLoader
from transformers import TrainerCallback

from ..evaluation.metrics import evaluate_mmlu_comprehensive


class ComprehensiveEvalCallback(TrainerCallback):
    """Evaluates MMLU accuracy/ECE/throughput and eval loss every `eval_steps` steps."""

    def __init__(
        self,
        eval_dataset_for_accuracy,
        eval_tokenized_for_loss,
        tokenizer,
        answer_tokens,
        eval_steps=50,
        accuracy_samples=1000,
        device="cuda",
    ):
        super().__init__()
        self.eval_dataset_for_accuracy = eval_dataset_for_accuracy
        self.eval_tokenized_for_loss = eval_tokenized_for_loss
        self.tokenizer = tokenizer
        self.answer_tokens = answer_tokens
        self.eval_steps = eval_steps
        self.accuracy_samples = accuracy_samples
        self.device = device

        self.metrics_history = []
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"Evaluating every {self.eval_steps} steps")
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            self._evaluate_and_log(args, state, model)
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"END OF EPOCH {state.epoch:.0f}")
        self._evaluate_and_log(args, state, model, is_epoch_end=True)
        return control

    def _evaluate_and_log(self, args, state, model, is_epoch_end=False):
        """Compute and log all metrics."""
        train_loss = None
        if len(state.log_history) > 0:
            for log in reversed(state.log_history):
                if "loss" in log:
                    train_loss = log["loss"]
                    break

        learning_rate = None
        if len(state.log_history) > 0:
            for log in reversed(state.log_history):
                if "learning_rate" in log:
                    learning_rate = log["learning_rate"]
                    break

        eval_loss, eval_throughput, eval_latency = self._compute_eval_loss(model)
        perplexity = torch.exp(torch.tensor(eval_loss)).item()

        mmlu_results = evaluate_mmlu_comprehensive(
            model=model,
            tokenizer=self.tokenizer,
            eval_dataset=self.eval_dataset_for_accuracy,
            answer_tokens=self.answer_tokens,
            device=self.device,
            max_samples=self.accuracy_samples,
            show_progress=False,
        )

        peak_gpu_memory = self._get_peak_gpu_memory()
        elapsed = time.time() - self.start_time

        grad_norm = None
        if len(state.log_history) > 0:
            for log in reversed(state.log_history):
                if "grad_norm" in log:
                    grad_norm = log["grad_norm"]
                    break

        metrics = {
            "step": state.global_step,
            "epoch": state.epoch,
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "grad_norm": grad_norm,
            "mmlu_accuracy": mmlu_results["accuracy"],
            "mmlu_top2_accuracy": mmlu_results["top2_accuracy"],
            "mmlu_ece": mmlu_results["ece"],
            "mmlu_correct": mmlu_results["correct"],
            "mmlu_total": mmlu_results["total"],
            "throughput": mmlu_results["throughput"],
            "avg_latency": mmlu_results["avg_latency"],
            "eval_throughput": eval_throughput,
            "eval_latency": eval_latency,
            "peak_gpu_memory_gb": peak_gpu_memory,
            "elapsed_time": elapsed,
        }
        self.metrics_history.append(metrics)

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        "eval/step": state.global_step,
                        "eval/train_loss": train_loss if train_loss else 0,
                        "eval/eval_loss": eval_loss,
                        "eval/perplexity": perplexity,
                        "eval/mmlu_accuracy": mmlu_results["accuracy"],
                        "eval/mmlu_top2_accuracy": mmlu_results["top2_accuracy"],
                        "eval/mmlu_ece": mmlu_results["ece"],
                        "eval/throughput": mmlu_results["throughput"],
                        "eval/avg_latency": mmlu_results["avg_latency"],
                        "eval/peak_gpu_memory_gb": peak_gpu_memory,
                    }
                )
        except Exception:
            pass

        step_info = f"Epoch {state.epoch:.2f}" if is_epoch_end else f"Step {state.global_step}"

        print(f"EVALUATION at {step_info}")
        print(f"  Train Loss:      {train_loss:.4f}" if train_loss else "  Train Loss:      N/A")
        print(f"  Eval Loss:       {eval_loss:.4f}")
        print(f"  MMLU Accuracy:   {mmlu_results['accuracy']:.4f} ({mmlu_results['correct']}/{mmlu_results['total']})")
        print(f"  Perplexity:      {perplexity:.2f}")
        print(f"  Throughput:      {mmlu_results['throughput']:.2f} samples/sec")
        print(f"  Avg Latency:     {mmlu_results['avg_latency']:.4f} sec/sample")
        print(f"  Peak GPU Memory: {peak_gpu_memory:.2f} GB")
        print(f"  ECE (Calibration): {mmlu_results['ece']:.4f}")
        print(f"  Top-2 Accuracy:  {mmlu_results['top2_accuracy']:.4f}")
        if learning_rate:
            print(f"  Learning Rate:   {learning_rate:.2e}")
        if grad_norm:
            print(f"  Gradient Norm:   {grad_norm:.4f}")
        print(f" Elapsed Time:    {elapsed / 60:.1f} min")

        torch.cuda.empty_cache()
        gc.collect()
        model.train()

    def _compute_eval_loss(self, model, num_batches=50):
        """Compute average loss on evaluation set."""
        model.eval()
        total_loss = 0
        num_samples = 0

        eval_dataloader = DataLoader(self.eval_tokenized_for_loss, batch_size=1, shuffle=False)

        start_time = time.time()

        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= num_batches:
                    break

                processed_batch = {}
                for k, v in batch.items():
                    if isinstance(v, list):
                        v = torch.tensor(v)
                    if v.dim() == 1:
                        v = v.unsqueeze(0)
                    processed_batch[k] = v.to(self.device)

                if "labels" not in processed_batch and "input_ids" in processed_batch:
                    processed_batch["labels"] = processed_batch["input_ids"].clone()

                outputs = model(**processed_batch)

                if outputs.loss is None:
                    continue

                total_loss += outputs.loss.item()
                num_samples += 1

        duration = time.time() - start_time

        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        throughput = num_samples / duration if duration > 0 else 0.0
        latency = duration / num_samples if num_samples > 0 else 0.0

        return avg_loss, throughput, latency

    def _get_peak_gpu_memory(self):
        """Get peak GPU memory and reset stats."""
        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()
            return max_mem
        return 0.0

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        print(f"TRAINING COMPLETED in {elapsed / 60:.1f} minutes")

        if self.metrics_history:
            print("TRAINING METRICS SUMMARY:")
            print(f"{'Step':<8} {'Epoch':<7} {'Train':<8} {'Eval':<8} {'Perp':<7} {'MMLU':<8} {'Top-2':<8} {'ECE':<7} {'Latency':<9}")
            print(f"{'':8} {'':7} {'Loss':<8} {'Loss':<8} {'':7} {'Acc':<8} {'Acc':<8} {'':7} {'(sec)':<9}")
            print("-" * 80)
            for m in self.metrics_history:
                train_loss_str = f"{m['train_loss']:.4f}" if m["train_loss"] else "N/A"
                print(
                    f"{m['step']:<8} {m['epoch']:<7.2f} {train_loss_str:<8} {m['eval_loss']:<8.4f} "
                    f"{m['perplexity']:<7.2f} {m['mmlu_accuracy']:<8.4f} {m['mmlu_top2_accuracy']:<8.4f} "
                    f"{m['mmlu_ece']:<7.4f} {m['avg_latency']:<9.4f}"
                )

        return control

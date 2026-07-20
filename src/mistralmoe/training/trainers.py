"""Custom HuggingFace Trainer subclasses: dense KD, MoE standard, MoE + KD.

Ported from moe_complete.ipynb:
- DenseKDTrainer: lines 1494-1601 (dense-to-dense KD control experiment)
- MoETrainer: lines 2799-2930 (standard MoE fine-tuning: NTP + aux loss)
- IntegratedMoEKDTrainer: lines 2943-3103 (MoE + KD: NTP + KD + aux + routing KD)

Loss formula (paper eq. 1): L_total = (1-a)*L_NTP + a*L_KD + lambda*L_aux + beta*L_routing_KD
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer


class DenseKDTrainer(Trainer):
    """Knowledge Distillation Trainer for dense models.

    Loss: L_total = (1-alpha)*L_NTP + alpha*L_KD
    """

    def __init__(self, teacher_model=None, kd_config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kd_config = kd_config or {}
        self.kd_alpha = self.kd_config.get("kd_alpha", 0.5)
        self.temperature = self.kd_config.get("temperature", 4.0)

        self.training_metrics_history = []

        if self.teacher_model is not None:
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            print(f"Teacher model loaded (frozen, KD alpha={self.kd_alpha}, T={self.temperature})")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with Knowledge Distillation."""
        outputs = model(**inputs)
        ntp_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        device = ntp_loss.device if hasattr(ntp_loss, "device") else next(model.parameters()).device

        kd_loss = torch.tensor(0.0, device=device)
        kd_loss_value = 0.0

        if self.teacher_model is not None and self.kd_alpha > 0:
            tdev = next(self.teacher_model.parameters()).device
            tinputs = {k: (v.to(tdev) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                t_out = self.teacher_model(**tinputs)

            s_logits = outputs.logits / self.temperature
            t_logits = t_out.logits / self.temperature

            kd_loss = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            ) * (self.temperature**2)

            kd_loss_value = kd_loss.item()

        total_loss = (1 - self.kd_alpha) * ntp_loss + self.kd_alpha * kd_loss

        if self.state.global_step % self.args.logging_steps == 0:
            ntp_loss_value = ntp_loss.item() if hasattr(ntp_loss, "item") else float(ntp_loss)
            total_loss_value = total_loss.item() if hasattr(total_loss, "item") else float(total_loss)

            kd_ntp_ratio = kd_loss_value / ntp_loss_value if ntp_loss_value > 0 else 0.0
            kd_total_ratio = kd_loss_value / total_loss_value if total_loss_value > 0 else 0.0

            log_dict = {
                "train/ntp_loss": ntp_loss_value,
                "train/kd_loss": kd_loss_value,
                "train/total_loss": total_loss_value,
                "train/kd_alpha": self.kd_alpha,
                "train/temperature": self.temperature,
                "train/kd_ntp_ratio": kd_ntp_ratio,
                "train/kd_total_ratio": kd_total_ratio,
            }
            self.log(log_dict)

            self.training_metrics_history.append({"step": self.state.global_step, "epoch": self.state.epoch, **log_dict})

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(log_dict, step=self.state.global_step)
            except Exception:
                pass

        return (total_loss, outputs) if return_outputs else total_loss

    def get_training_metrics_summary(self):
        """Extract summary statistics from training metrics history."""
        if not self.training_metrics_history:
            return None

        return {
            "final_ntp_loss": self.training_metrics_history[-1]["train/ntp_loss"],
            "final_kd_loss": self.training_metrics_history[-1]["train/kd_loss"],
            "final_total_loss": self.training_metrics_history[-1]["train/total_loss"],
            "avg_ntp_loss": np.mean([m["train/ntp_loss"] for m in self.training_metrics_history]),
            "avg_kd_loss": np.mean([m["train/kd_loss"] for m in self.training_metrics_history]),
            "avg_total_loss": np.mean([m["train/total_loss"] for m in self.training_metrics_history]),
            "avg_kd_ntp_ratio": np.mean([m["train/kd_ntp_ratio"] for m in self.training_metrics_history]),
            "avg_kd_total_ratio": np.mean([m["train/kd_total_ratio"] for m in self.training_metrics_history]),
            "kd_alpha": self.kd_alpha,
            "temperature": self.temperature,
            "total_steps": len(self.training_metrics_history),
        }


def _resolve_moe_layers(model):
    """Get MoE decoder layers from a model, handling the PEFT/LoRA wrapper."""
    candidates = [
        lambda m: m.model.layers if hasattr(m, "model") and hasattr(m.model, "layers") else None,
        lambda m: m.base_model.model.layers
        if hasattr(m, "base_model") and hasattr(m.base_model, "model") and hasattr(m.base_model.model, "layers")
        else None,
        lambda m: m.base_model.model.model.layers
        if hasattr(m, "base_model") and hasattr(m.base_model, "model") and hasattr(m.base_model.model, "model")
        else None,
    ]

    for get_layers in candidates:
        try:
            layers = get_layers(model)
            if layers is not None:
                return layers
        except Exception:
            continue
    return []


class MoETrainer(Trainer):
    """Trainer for MoE models with standard fine-tuning (no KD).

    Total loss formula: L_total = L_NTP + lambda * L_aux
    """

    def __init__(self, router_aux_loss_coef=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_aux_loss_coef = router_aux_loss_coef
        self.training_metrics_history = []

    def _moe_layers(self, model):
        return _resolve_moe_layers(model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        for layer in self._moe_layers(model):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "forward"):
                mlp._collect_router_logits = True

        outputs = model(**inputs)
        ntp_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        aux_loss = self._compute_moe_auxiliary_loss(model)
        total_loss = ntp_loss + self.router_aux_loss_coef * aux_loss

        should_log = False
        current_step = 0
        current_epoch = 0

        if hasattr(self, "state") and self.state is not None:
            if hasattr(self.state, "global_step") and hasattr(self.args, "logging_steps"):
                current_step = self.state.global_step
                current_epoch = getattr(self.state, "epoch", 0)
                should_log = current_step % self.args.logging_steps == 0

        if should_log:
            ntp_loss_value = ntp_loss.item()
            aux_loss_value = aux_loss.item()
            total_loss_value = total_loss.item()

            log_dict = {
                "train/ntp_loss": ntp_loss_value,
                "train/kd_loss": 0.0,
                "train/aux_loss": aux_loss_value,
                "train/routing_kd_loss": 0.0,
                "train/total_loss": total_loss_value,
                "train/aux_loss_weight": self.router_aux_loss_coef,
                "train/kd_alpha": 0.0,
                "train/temperature": 0.0,
                "train/kd_ntp_ratio": 0.0,
                "train/kd_total_ratio": 0.0,
            }
            self.log(log_dict)

            self.training_metrics_history.append({"step": current_step, "epoch": current_epoch, **log_dict})

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(log_dict, step=current_step)
            except Exception:
                pass

        for layer in self._moe_layers(model):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "_collect_router_logits"):
                mlp._collect_router_logits = False

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_moe_auxiliary_loss(self, model):
        """Sum the load-balancing auxiliary loss from all MoE layers."""
        device = next(model.parameters()).device
        aux_loss = torch.tensor(0.0, device=device)

        for layer in self._moe_layers(model):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "compute_auxiliary_loss"):
                aux_loss = aux_loss + mlp.compute_auxiliary_loss()
        return aux_loss

    def get_training_metrics_summary(self):
        """Extract summary statistics from training metrics history."""
        if not self.training_metrics_history:
            return None

        return {
            "final_ntp_loss": self.training_metrics_history[-1]["train/ntp_loss"],
            "final_kd_loss": 0.0,
            "final_aux_loss": self.training_metrics_history[-1]["train/aux_loss"],
            "final_routing_kd_loss": 0.0,
            "final_total_loss": self.training_metrics_history[-1]["train/total_loss"],
            "avg_ntp_loss": np.mean([m["train/ntp_loss"] for m in self.training_metrics_history]),
            "avg_kd_loss": 0.0,
            "avg_aux_loss": np.mean([m["train/aux_loss"] for m in self.training_metrics_history]),
            "avg_routing_kd_loss": 0.0,
            "avg_total_loss": np.mean([m["train/total_loss"] for m in self.training_metrics_history]),
            "avg_kd_ntp_ratio": 0.0,
            "avg_kd_total_ratio": 0.0,
            "kd_alpha": 0.0,
            "temperature": 0.0,
            "routing_kd_weight": 0.0,
            "total_steps": len(self.training_metrics_history),
        }


class IntegratedMoEKDTrainer(Trainer):
    """Trainer for MoE models with Knowledge Distillation.

    Total loss formula: L_total = (1-alpha)*L_NTP + alpha*L_KD + lambda*L_aux + beta*L_routing_KD
    """

    def __init__(self, teacher_model=None, kd_config=None, router_aux_loss_coef=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kd_config = kd_config or {}
        self.router_aux_loss_coef = router_aux_loss_coef
        self.kd_alpha = self.kd_config.get("kd_alpha", 0.5)
        self.temperature = self.kd_config.get("temperature", 4.0)
        self.routing_kd_weight = self.kd_config.get("routing_kd_weight", 0.0)
        self.enable_routing_kd = self.kd_config.get("enable_routing_kd", False)
        self.training_metrics_history = []

    def _moe_layers(self, model):
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "base_model") and hasattr(model.base_model.model, "layers"):
            return model.base_model.model.layers
        return []

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with KD."""
        for layer in self._moe_layers(model):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "forward"):
                mlp._collect_router_logits = True

        outputs = model(**inputs)
        ntp_loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        device = ntp_loss.device if hasattr(ntp_loss, "device") else next(model.parameters()).device

        kd_loss = torch.tensor(0.0, device=device)
        routing_kd_loss = torch.tensor(0.0, device=device)

        if self.teacher_model is not None and self.kd_alpha > 0:
            tdev = next(self.teacher_model.parameters()).device
            tinputs = {k: (v.to(tdev) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                t_out = self.teacher_model(**tinputs)

            s_logits = outputs.logits / self.temperature
            t_logits = t_out.logits / self.temperature

            kd_loss = F.kl_div(
                F.log_softmax(s_logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean",
            ) * (self.temperature**2)

            if self.enable_routing_kd and self.routing_kd_weight > 0:
                ent = torch.tensor(0.0, device=device)
                layers_count = 0
                for layer in self._moe_layers(model):
                    mlp = getattr(layer, "mlp", None)
                    if mlp is not None and hasattr(mlp, "_last_router_probs") and mlp._last_router_probs is not None:
                        probs = mlp._last_router_probs
                        ent_layer = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                        ent = ent + ent_layer
                        layers_count += 1
                if layers_count > 0:
                    routing_kd_loss = ent / layers_count

        aux_loss = torch.tensor(0.0, device=device)
        for layer in self._moe_layers(model):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "compute_auxiliary_loss"):
                aux_loss = aux_loss + mlp.compute_auxiliary_loss()

        total_loss = (
            (1 - self.kd_alpha) * ntp_loss
            + self.kd_alpha * kd_loss
            + self.router_aux_loss_coef * aux_loss
            + self.routing_kd_weight * routing_kd_loss
        )

        should_log = False
        current_step = 0
        current_epoch = 0

        if hasattr(self, "state") and self.state is not None:
            if hasattr(self.state, "global_step") and hasattr(self.args, "logging_steps"):
                current_step = self.state.global_step
                current_epoch = getattr(self.state, "epoch", 0)
                should_log = current_step % self.args.logging_steps == 0

        if should_log:
            ntp_loss_value = ntp_loss.item()
            kd_loss_value = kd_loss.item() if kd_loss.item() > 0 else 0.0
            aux_loss_value = aux_loss.item()
            routing_kd_loss_value = routing_kd_loss.item() if routing_kd_loss.item() > 0 else 0.0
            total_loss_value = total_loss.item()

            kd_ntp_ratio = kd_loss_value / ntp_loss_value if ntp_loss_value > 0 else 0.0
            kd_total_ratio = kd_loss_value / total_loss_value if total_loss_value > 0 else 0.0

            log_dict = {
                "train/ntp_loss": ntp_loss_value,
                "train/kd_loss": kd_loss_value,
                "train/aux_loss": aux_loss_value,
                "train/routing_kd_loss": routing_kd_loss_value,
                "train/total_loss": total_loss_value,
                "train/kd_alpha": self.kd_alpha,
                "train/temperature": self.temperature,
                "train/routing_kd_weight": self.routing_kd_weight,
                "train/aux_loss_weight": self.router_aux_loss_coef,
                "train/kd_ntp_ratio": kd_ntp_ratio,
                "train/kd_total_ratio": kd_total_ratio,
            }
            self.log(log_dict)

            self.training_metrics_history.append({"step": current_step, "epoch": current_epoch, **log_dict})

            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(log_dict, step=current_step)
            except Exception:
                pass

        for layer in self._moe_layers(model):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and hasattr(mlp, "_collect_router_logits"):
                mlp._collect_router_logits = False

        return (total_loss, outputs) if return_outputs else total_loss

    def get_training_metrics_summary(self):
        """Extract summary statistics from training metrics history."""
        if not self.training_metrics_history:
            return None

        return {
            "final_ntp_loss": self.training_metrics_history[-1]["train/ntp_loss"],
            "final_kd_loss": self.training_metrics_history[-1]["train/kd_loss"],
            "final_aux_loss": self.training_metrics_history[-1]["train/aux_loss"],
            "final_routing_kd_loss": self.training_metrics_history[-1]["train/routing_kd_loss"],
            "final_total_loss": self.training_metrics_history[-1]["train/total_loss"],
            "avg_ntp_loss": np.mean([m["train/ntp_loss"] for m in self.training_metrics_history]),
            "avg_kd_loss": np.mean([m["train/kd_loss"] for m in self.training_metrics_history]),
            "avg_aux_loss": np.mean([m["train/aux_loss"] for m in self.training_metrics_history]),
            "avg_routing_kd_loss": np.mean([m["train/routing_kd_loss"] for m in self.training_metrics_history]),
            "avg_total_loss": np.mean([m["train/total_loss"] for m in self.training_metrics_history]),
            "avg_kd_ntp_ratio": np.mean([m["train/kd_ntp_ratio"] for m in self.training_metrics_history]),
            "avg_kd_total_ratio": np.mean([m["train/kd_total_ratio"] for m in self.training_metrics_history]),
            "kd_alpha": self.kd_alpha,
            "temperature": self.temperature,
            "routing_kd_weight": self.routing_kd_weight,
            "total_steps": len(self.training_metrics_history),
        }

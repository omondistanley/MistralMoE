"""Sparse upcycling: replace dense FFN layers with MoELayer, weights copied from the FFN.

Ported verbatim (behavior-preserving) from moe_complete.ipynb lines 2277-2513.
Each expert starts as an exact copy of the original dense FFN's weights, which
is why the untrained MoE model preserves dense accuracy (see README/paper).
"""

from __future__ import annotations

import ctypes
import gc
import os

import psutil
import torch

from .moe_layer import MoELayer


def replace_ffn_with_moe(
    model,
    num_experts=8,
    num_experts_per_tok=2,
    router_jitter_noise=0.0,
    router_aux_loss_coef=0.001,
    bnb_config=None,
    ram_threshold=50.0,
    use_disk_offload=True,
    layer_indices=None,
    half_width=False,
    enable_cpu_offload=True,
):
    """Replace dense FFN layers with MoE layers.

    `layer_indices=None` replaces every layer (the `expert_layers="all"`
    variants); pass a subset for placement variants (early/middle/late/mixed/
    sparse).
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    config = model.config
    hidden_size = config.hidden_size

    if half_width:
        intermediate_size = config.intermediate_size // 2
        print("   Using HALF-WIDTH experts (intermediate_size // 2)")
    else:
        intermediate_size = config.intermediate_size

    print("Model configuration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Original intermediate size: {config.intermediate_size}")
    print(f"  MoE intermediate size: {intermediate_size}")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Experts per layer: {num_experts}")
    print(f"  Experts per token: {num_experts_per_tok}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")

    device = torch.device("cuda")
    if bnb_config and hasattr(bnb_config, "bnb_4bit_compute_dtype"):
        compute_dtype = bnb_config.bnb_4bit_compute_dtype
    elif bnb_config and hasattr(bnb_config, "llm_int8_threshold"):
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.bfloat16

    print("\n Using GPU for weight processing")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"  Compute dtype: {compute_dtype}")

    def check_ram():
        return psutil.virtual_memory().percent

    def cleanup(aggressive=False):
        gc.collect()
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        gc.collect()

    def print_memory_stats(label=""):
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"    [{label}] GPU: {allocated:.2f}GB alloc / {reserved:.2f}GB reserved")

    def extract_weight(linear_layer, expected_shape, keep_on_cpu=True):
        """Extract a linear layer's weight, reshaping/transposing if needed."""
        expected_numel = (
            torch.Size(expected_shape).numel() if isinstance(expected_shape, (tuple, list)) else expected_shape.numel()
        )

        weight = linear_layer.weight.data.to(compute_dtype)

        if weight.shape != expected_shape:
            if weight.numel() == expected_numel:
                weight = weight.reshape(expected_shape)
            elif len(weight.shape) == 2 and len(expected_shape) == 2:
                if weight.shape[0] == expected_shape[1] and weight.shape[1] == expected_shape[0]:
                    weight = weight.t()
                elif weight.numel() == expected_numel:
                    weight = weight.reshape(expected_shape)

        return weight.cpu() if keep_on_cpu else weight.cuda()

    total_layers = len(model.model.layers)
    target_layers = list(range(total_layers)) if layer_indices is None else layer_indices
    print(f"\n  Processing {len(target_layers)} layers: {target_layers[:5]}{'...' if len(target_layers) > 5 else ''}\n")

    for i in target_layers:
        layer = model.model.layers[i]
        original_mlp = layer.mlp

        ram = check_ram()
        print(f"  Layer {i + 1}/{total_layers} (RAM: {ram:.1f}%):")

        print("    Extracting weights to CPU...", end=" ", flush=True)
        with torch.no_grad():
            gate_w_full = extract_weight(original_mlp.gate_proj, (config.intermediate_size, hidden_size), keep_on_cpu=True)
            up_w_full = extract_weight(original_mlp.up_proj, (config.intermediate_size, hidden_size), keep_on_cpu=True)
            down_w_full = extract_weight(original_mlp.down_proj, (hidden_size, config.intermediate_size), keep_on_cpu=True)

            gate_w = gate_w_full[:intermediate_size, :].clone()
            up_w = up_w_full[:intermediate_size, :].clone()
            down_w = down_w_full[:, :intermediate_size].clone()

            del gate_w_full, up_w_full, down_w_full
            gc.collect()
        print("Done")

        print("    Deleting original MLP...", end=" ", flush=True)
        del original_mlp
        layer.mlp = None
        cleanup(aggressive=True)
        print("Done")

        print("    Creating MoE layer...", end=" ", flush=True)
        moe_layer = MoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_jitter_noise=router_jitter_noise,
            router_aux_loss_coef=router_aux_loss_coef,
            bnb_config=bnb_config,
            device="cpu" if bnb_config is None else device,
            init_on_cpu=(bnb_config is None),
            dtype=compute_dtype,
            enable_cpu_offload=enable_cpu_offload,
        )
        print("Done")

        print(f"    Copying weights to {num_experts} experts ...", end=" ", flush=True)
        with torch.no_grad():
            gate_target_shape = moe_layer.gate_proj[0].weight.shape
            up_target_shape = moe_layer.up_proj[0].weight.shape
            down_target_shape = moe_layer.down_proj[0].weight.shape

            if gate_w.shape != gate_target_shape:
                if gate_w.numel() == gate_target_shape.numel():
                    gate_w = gate_w.reshape(gate_target_shape)
                else:
                    gate_w = gate_w[: gate_target_shape[0], : gate_target_shape[1]]

            if up_w.shape != up_target_shape:
                if up_w.numel() == up_target_shape.numel():
                    up_w = up_w.reshape(up_target_shape)
                else:
                    up_w = up_w[: up_target_shape[0], : up_target_shape[1]]

            if down_w.shape != down_target_shape:
                if down_w.numel() == down_target_shape.numel():
                    down_w = down_w.reshape(down_target_shape)
                elif len(down_w.shape) == 1 or (len(down_w.shape) == 2 and down_w.shape[1] == 1):
                    if down_w.numel() == down_target_shape.numel():
                        down_w = down_w.reshape(down_target_shape)
                    else:
                        transposed_shape = (down_target_shape[1], down_target_shape[0])
                        if down_w.numel() == torch.Size(transposed_shape).numel():
                            down_w = down_w.reshape(transposed_shape).t()
                        else:
                            raise RuntimeError(
                                f"Cannot reshape down_w from {down_w.shape} to {down_target_shape}. "
                                f"down_w.numel()={down_w.numel()}, target.numel()={down_target_shape.numel()}."
                            )
                elif down_w.shape[0] == down_target_shape[0] and down_w.shape[1] >= down_target_shape[1]:
                    down_w = down_w[:, : down_target_shape[1]]
                elif down_w.shape[1] == down_target_shape[1] and down_w.shape[0] >= down_target_shape[0]:
                    down_w = down_w[: down_target_shape[0], :]
                elif down_w.shape[0] == down_target_shape[1] and down_w.shape[1] == down_target_shape[0]:
                    down_w = down_w.t()
                else:
                    raise RuntimeError(
                        f"Cannot reshape down_w from {down_w.shape} to {down_target_shape}. "
                        f"down_w.numel()={down_w.numel()}, target.numel()={down_target_shape.numel()}."
                    )

            # All experts start with identical pretrained weights (sparse upcycling).
            for idx in range(num_experts):
                moe_layer.gate_proj[idx].weight.copy_(gate_w)
                moe_layer.up_proj[idx].weight.copy_(up_w)
                moe_layer.down_proj[idx].weight.copy_(down_w)

        print("Done")

        del gate_w, up_w, down_w
        gc.collect()

        print("    Moving to GPU...", end=" ", flush=True)
        moe_layer.gate_proj = moe_layer.gate_proj.to(device)
        moe_layer.up_proj = moe_layer.up_proj.to(device)
        moe_layer.down_proj = moe_layer.down_proj.to(device)
        moe_layer.router = moe_layer.router.to(device)
        print("Done")

        layer.mlp = moe_layer
        cleanup(aggressive=True)

        if (i + 1) % 4 == 0:
            print_memory_stats(f"Layer {i + 1}")

    cleanup(aggressive=True)

    print(f"\n Successfully replaced {len(target_layers)} FFN layers with MoE")
    print(f"  Expert dispatch: Efficient sparse routing (top-{num_experts_per_tok})")
    print(f"  Params per expert: ~{(intermediate_size * hidden_size * 3) / 1e6:.1f}M")
    print(f"  Active params per token: ~{(intermediate_size * hidden_size * 3 * num_experts_per_tok) / 1e6:.1f}M")

    gpu_final = torch.cuda.memory_allocated() / 1e9
    ram_final = check_ram()
    print(f"  Final: GPU {gpu_final:.2f}GB | RAM {ram_final:.1f}%")
    return model

from .moe_layer import MoELayer, compute_moe_auxiliary_loss, compute_moe_loss
from .upcycle import replace_ffn_with_moe

__all__ = [
    "MoELayer",
    "compute_moe_auxiliary_loss",
    "compute_moe_loss",
    "replace_ffn_with_moe",
]

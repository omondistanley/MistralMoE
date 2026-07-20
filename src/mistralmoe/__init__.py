"""mistralmoe: Sparse Upcycling and Knowledge Distillation for Mixture-of-Experts.

A modular extraction of moe_complete.ipynb into an importable package. See
that notebook for the original end-to-end pipeline; this package is a
parallel, faithful port for programmatic/CLI use (moe_complete.ipynb itself
is left untouched).
"""

from .config import EXPERIMENT_CONFIGS, MoEExperimentConfig, configure_environment

__all__ = ["EXPERIMENT_CONFIGS", "MoEExperimentConfig", "configure_environment"]

__version__ = "0.1.0"

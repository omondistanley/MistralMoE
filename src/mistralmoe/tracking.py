"""Weights & Biases setup and crash-resistant monkeypatches.

Ported from moe_complete.ipynb cells 3-5 (lines 45-155). In the notebook these
ran as import-time side effects in a Jupyter kernel (to strip wandb's IPython
cell hooks, which caused BrokenPipeError under long-running training cells).
Here they're explicit functions so importing this module has no side effects.
"""

from __future__ import annotations

import os


def disable_wandb_notebook_hooks() -> None:
    """Strip wandb's IPython pre/post-run-cell callbacks, if running under IPython.

    No-op outside a Jupyter/IPython kernel (e.g. plain scripts), which is the
    common case for this package.
    """
    os.environ["WANDB_DISABLE_SERVICE"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython:
            events = ipython.events
            for event_name in ["pre_run_cell", "post_run_cell"]:
                if hasattr(events, "callbacks") and event_name in events.callbacks:
                    callbacks_to_remove = [
                        cb
                        for cb in events.callbacks[event_name]
                        if hasattr(cb, "__self__") and "wandb" in str(type(cb.__self__)).lower()
                    ]
                    for cb in callbacks_to_remove:
                        try:
                            events.callbacks[event_name].remove(cb)
                        except Exception:
                            pass
    except Exception:
        pass


def install_safe_wandb_patches() -> None:
    """Monkeypatch wandb.init/wandb.log to swallow connection errors instead of crashing training."""
    import wandb

    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_SILENT"] = "true"

    original_init = wandb.init
    original_log = wandb.log

    def safe_init(*args, **kwargs):
        try:
            if wandb.run is not None:
                try:
                    wandb.finish()
                except Exception:
                    pass

            kwargs.setdefault("settings", wandb.Settings())
            if isinstance(kwargs["settings"], wandb.Settings):
                kwargs["settings"]._disable_stats = True
                kwargs["settings"]._disable_meta = True

            return original_init(*args, **kwargs)
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"Warning: wandb connection error (continuing without tracking): {type(e).__name__}")
            return None
        except Exception as e:
            print(f"Warning: wandb initialization error: {e}")
            return None

    def safe_log(*args, **kwargs):
        try:
            if wandb.run is not None:
                original_log(*args, **kwargs)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception:
            pass

    wandb.init = safe_init
    wandb.log = safe_log
    print("Wandb error handling enabled. Connection errors will be handled gracefully.")


def init_wandb_run(project: str = "MoE-variants", config: dict | None = None):
    """Log in and start a wandb run. Call install_safe_wandb_patches() first."""
    import wandb

    default_config = {
        "model": "mistralai/Mistral-7B-v0.1",
        "max_length": 512,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "epochs": 2,
        "learning_rate": 2e-5,
        "quantization": "4-bit",
        "lora_r": 8,
        "lora_alpha": 16,
    }

    wandb.login()
    run = wandb.init(project=project, config=config or default_config)
    print("WandB initialized successfully")
    return run

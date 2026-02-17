"""Tracking/authentication helpers (Hugging Face + Weights & Biases)."""

from __future__ import annotations

import os
from typing import Any, Dict

from config_utils import get_nested


def maybe_login_hf(config: Dict[str, Any]) -> None:
    """Authenticate with Hugging Face only when a token env var is configured.

    This keeps credentials out of source code while still supporting private
    models/datasets and hub operations.
    """

    token_env = get_nested(config, "tracking.hf_token_env")
    if not token_env:
        return

    token = os.getenv(token_env)
    if not token:
        return

    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def setup_wandb(config: Dict[str, Any]) -> bool:
    """Configure W&B and return whether tracking is enabled for this run."""

    use_wandb = bool(get_nested(config, "tracking.use_wandb", False))
    if not use_wandb:
        return False

    # Optional explicit login. If the key env var is absent, W&B may still
    # work if the user already authenticated on the machine.
    api_key_env = get_nested(config, "tracking.wandb_api_key_env", "WANDB_API_KEY")
    api_key = os.getenv(api_key_env)
    if api_key:
        import wandb

        wandb.login(key=api_key, relogin=False)

    # Set standard W&B environment variables so Trainer can auto-pick them up.
    project = get_nested(config, "tracking.wandb_project", "Indic-SLID")
    run_name = get_nested(config, "tracking.wandb_run_name")
    os.environ["WANDB_PROJECT"] = str(project)
    if run_name:
        os.environ["WANDB_NAME"] = str(run_name)

    return True

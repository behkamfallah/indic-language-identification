"""Tracking/authentication helpers (Hugging Face + Weights & Biases)."""

from __future__ import annotations

import os
from typing import Any, Dict

from huggingface_hub import login
from huggingface_hub import whoami
import wandb

from config_utils import get_nested


def setup_hf(config: Dict[str, Any]) -> bool:
    """Authenticate with Hugging Face only when a token env var is configured."""

    token_env = get_nested(config, "tracking.hf_token_env")
    if not token_env:
        print("Hugging Face login: not configured.")
        return False

    token = os.getenv(token_env)
    if not token:
        print(f"Hugging Face login: env var '{token_env}' not set.")
        return False

    try:
        login(token=token)
        whoami(token=token)  # verify token works
        print("Hugging Face login: SUCCESS.")
        return True
    except Exception as e:
        print(f"Hugging Face login: FAILED ({e}).")
        return False


def setup_wandb(config: Dict[str, Any]) -> bool:
    """Configure W&B and return whether tracking is enabled for this run."""

    use_wandb = bool(get_nested(config, "tracking.use_wandb"))
    if not isinstance(use_wandb, bool):
        raise ValueError("tracking.use_wandb must be boolean")
    if not use_wandb:
        print("W&B: disabled.")
        return False

    # Optional explicit login. If the key env var is absent, W&B may still
    # work if the user already authenticated on the machine.
    api_key_env = get_nested(config, "tracking.wandb_api_key_env")
    api_key = os.getenv(api_key_env)
    if api_key:
        try:
            wandb.login(key=api_key, relogin=False)
            print("W&B login: SUCCESS.")
        except Exception as e:
            print(f"W&B login: FAILED ({e}).")
            return False
    else:
        print(f"W&B login: no API key in env '{api_key_env}'. Using existing session if available.")

    # Set standard W&B environment variables so Trainer can auto-pick them up.
    project = get_nested(config, "tracking.wandb_project")
    run_name = get_nested(config, "tracking.wandb_run_name")

    os.environ["WANDB_PROJECT"] = str(project)
    if run_name:
        os.environ["WANDB_NAME"] = str(run_name)

    print(f"W&B: enabled (project='{project}', run='{run_name}').")
    return True

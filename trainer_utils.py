"""Trainer-side utilities: collator, TrainingArguments, metrics, Trainer."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List

import evaluate
import numpy as np
import torch
from transformers import AutoFeatureExtractor, Trainer, TrainingArguments

from config_utils import get_nested


class AudioDataCollator:
    """Pad variable-length audio features and attach class labels.

    We rely on the model's feature extractor to pad inputs so behavior remains
    consistent with the underlying architecture.
    """

    def __init__(self, feature_extractor: AutoFeatureExtractor, model_input_name: str):
        self.feature_extractor = feature_extractor
        self.model_input_name = model_input_name

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch: Dict[str, Any] = {
            self.model_input_name: [item[self.model_input_name] for item in features]
        }
        if "attention_mask" in features[0]:
            batch["attention_mask"] = [item["attention_mask"] for item in features]

        padded = self.feature_extractor.pad(
            batch,
            padding=True,
            return_tensors="pt",
        )
        padded["labels"] = torch.tensor([item["label"] for item in features], dtype=torch.long)
        return padded


def build_training_arguments(config: Dict[str, Any], output_dir: Path, run_name: str) -> TrainingArguments:
    """Create `TrainingArguments` from YAML config with sane defaults."""

    train_cfg = get_nested(config, "training", {})
    eval_strategy = str(train_cfg.get("eval_strategy", "steps"))
    use_wandb = bool(get_nested(config, "tracking.use_wandb", False))
    cuda_available = torch.cuda.is_available()
    fp16_requested = bool(train_cfg.get("fp16", cuda_available))
    bf16_requested = bool(train_cfg.get("bf16", False))

    kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": run_name,
        "group_by_length": bool(train_cfg.get("group_by_length", False)),
        "report_to": "wandb" if use_wandb else "none",
        "logging_steps": int(train_cfg.get("logging_steps", 10)),
        "per_device_train_batch_size": int(train_cfg.get("batch_size", 8)),
        "per_device_eval_batch_size": int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 8))),
        "save_strategy": str(train_cfg.get("save_strategy", "steps")),
        "learning_rate": float(train_cfg.get("learning_rate", 1e-5)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.1)),
        "load_best_model_at_end": bool(train_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": str(train_cfg.get("metric_for_best_model", "accuracy")),
        "greater_is_better": bool(train_cfg.get("greater_is_better", True)),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "fp16": fp16_requested and cuda_available,
        "bf16": bf16_requested and cuda_available,
        "push_to_hub": bool(train_cfg.get("push_to_hub", False)),
        "lr_scheduler_type": str(train_cfg.get("lr_scheduler_type", "linear")),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 0)),
        "seed": int(config.get("seed", 42)),
        "gradient_checkpointing": bool(train_cfg.get("gradient_checkpointing", False)),
        "save_steps": int(train_cfg.get("save_steps", 100)),
        # Keep custom fields (e.g. "length") available in batches.
        "remove_unused_columns": False,
    }

    if eval_strategy == "steps":
        kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 100))

    # Transformers renamed `evaluation_strategy` -> `eval_strategy`; keep
    # compatibility with both by introspecting the installed version.
    training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_signature:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy

    return TrainingArguments(**kwargs)


def build_trainer(
    model: Any,
    training_args: TrainingArguments,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: AudioDataCollator,
    feature_extractor: AutoFeatureExtractor,
) -> Trainer:
    """Create a Hugging Face Trainer with accuracy metric wiring."""

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = np.argmax(logits, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }

    # `processing_class` is used by newer Trainer versions.
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = feature_extractor
    else:
        trainer_kwargs["tokenizer"] = feature_extractor

    return Trainer(**trainer_kwargs)

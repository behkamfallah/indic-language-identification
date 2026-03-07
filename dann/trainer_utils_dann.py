"""Trainer utilities for DANN task-2 training."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from transformers import AutoFeatureExtractor, Trainer, TrainingArguments

try:
    from .config_utils_dann import get_nested
except ImportError:  # Script-mode execution
    from config_utils_dann import get_nested


@dataclass
class DANNConfig:
    alpha: float = 0.1
    grl_max_lambda: float = 1.0
    grl_schedule: str = "dann"  # "dann" or "constant"


class AudioDataCollatorWithSpeaker:
    """Pad variable-length audio and attach language + speaker labels."""

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
        padded["speaker_labels"] = torch.tensor([item["speaker_label"] for item in features], dtype=torch.long)
        return padded


class DANNTrainer(Trainer):
    """Trainer with custom combined loss for DANN."""

    def __init__(self, *args, dann_cfg: Optional[DANNConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dann_cfg = dann_cfg or DANNConfig()
        self.ce = nn.CrossEntropyLoss()
        # Keep evaluation labels tied to the language task only.
        self.label_names = ["labels"]

    def _max_steps(self) -> int:
        if getattr(self.state, "max_steps", 0) and self.state.max_steps > 0:
            return int(self.state.max_steps)
        if getattr(self.args, "max_steps", 0) and self.args.max_steps > 0:
            return int(self.args.max_steps)
        return 1

    def _grl_lambda(self) -> float:
        schedule = str(self.dann_cfg.grl_schedule).lower()
        if schedule == "constant":
            return float(self.dann_cfg.grl_max_lambda)
        if schedule != "dann":
            raise ValueError("dann.grl_schedule must be 'dann' or 'constant'.")

        step = int(getattr(self.state, "global_step", 0))
        max_steps = max(1, self._max_steps())
        p = min(1.0, step / max_steps)
        return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0) * float(self.dann_cfg.grl_max_lambda)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch

        labels = inputs.pop("labels")
        speaker_labels = inputs.pop("speaker_labels")

        lambd = self._grl_lambda()
        if hasattr(model, "set_grl_lambda"):
            model.set_grl_lambda(lambd)

        outputs = model(**inputs)
        lang_logits = outputs.logits
        speaker_logits = outputs.speaker_logits

        loss_lang = self.ce(lang_logits, labels)
        loss_spk = self.ce(speaker_logits, speaker_labels)
        # The gradient reversal layer already flips the speaker-loss gradient on
        # the path back into the shared encoder, so the forward loss stays additive.
        loss = loss_lang + float(self.dann_cfg.alpha) * loss_spk

        outputs.loss = loss

        if return_outputs:
            return loss, outputs
        return loss


def build_training_arguments(config: Dict[str, Any], output_dir: Path, run_name: str) -> TrainingArguments:
    """Create TrainingArguments from YAML config."""

    train_cfg = get_nested(config, "training")
    eval_strategy = str(train_cfg.get("eval_strategy", "steps"))
    use_wandb = bool(get_nested(config, "tracking.use_wandb"))
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
        "save_only_model": bool(train_cfg.get("save_only_model", True)),
        "save_safetensors": bool(train_cfg.get("save_safetensors", True)),
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
        "remove_unused_columns": False,
    }

    if eval_strategy == "steps":
        kwargs["eval_steps"] = int(train_cfg.get("eval_steps", 100))

    training_args_signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in training_args_signature:
        kwargs["eval_strategy"] = eval_strategy
    else:
        kwargs["evaluation_strategy"] = eval_strategy

    return TrainingArguments(**kwargs)


def build_dann_trainer(
    model: Any,
    training_args: TrainingArguments,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: AudioDataCollatorWithSpeaker,
    feature_extractor: AutoFeatureExtractor,
    dann_cfg: Optional[DANNConfig] = None,
) -> DANNTrainer:
    """Create DANNTrainer with language metrics on main logits."""

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred: Any) -> Dict[str, float]:
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            logits = predictions[0]
        elif isinstance(predictions, dict):
            logits = predictions["logits"]
        else:
            logits = predictions

        pred_ids = np.argmax(logits, axis=1)
        labels = eval_pred.label_ids

        metrics = accuracy_metric.compute(predictions=pred_ids, references=labels)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            pred_ids,
            average="weighted",
            zero_division=0,
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1
        return metrics

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "dann_cfg": dann_cfg,
    }

    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = feature_extractor
    else:
        trainer_kwargs["tokenizer"] = feature_extractor

    return DANNTrainer(**trainer_kwargs)

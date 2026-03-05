from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForAudioClassification
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from transformers.utils import ModelOutput

try:
    from .dann_grl import GradientReversal
except ImportError:  # Script-mode execution
    from dann_grl import GradientReversal


@dataclass
class DANNOutputs(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    speaker_logits: Optional[torch.Tensor] = None


class DANNForAudioClassification(nn.Module):
    """HF audio classifier with an added speaker-adversarial branch."""

    def __init__(
        self,
        model_id: str,
        num_labels: int,
        num_speakers: int,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        speaker_head_hidden: int = 256,
        grl_lambda: float = 1.0,
        pooling: str = "mean",
        apply_dropout: bool = False,
        dropout_value: float = 0.1,
        freeze_feature_encoder: bool = False,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")

        self.pooling = pooling

        cfg = AutoConfig.from_pretrained(
            model_id,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            output_hidden_states=True,
        )

        if apply_dropout:
            for attribute in [
                "hidden_dropout",
                "attention_dropout",
                "activation_dropout",
                "feat_proj_dropout",
                "final_dropout",
            ]:
                if hasattr(cfg, attribute):
                    setattr(cfg, attribute, float(dropout_value))

        self.base = AutoModelForAudioClassification.from_pretrained(
            model_id,
            config=cfg,
            ignore_mismatched_sizes=True,
        )
        self.main_input_name = str(getattr(self.base, "main_input_name", "input_values"))

        if freeze_feature_encoder:
            if hasattr(self.base, "freeze_feature_encoder"):
                self.base.freeze_feature_encoder()
            elif hasattr(self.base, "freeze_feature_extractor"):
                self.base.freeze_feature_extractor()

        self.grl = GradientReversal(grl_lambda)

        hidden_size = self._infer_hidden_size(cfg)
        self.speaker_head = nn.Sequential(
            nn.Linear(hidden_size, int(speaker_head_hidden)),
            nn.ReLU(),
            nn.Linear(int(speaker_head_hidden), int(num_speakers)),
        )
        self._dann_meta = {
            "pooling": pooling,
            "speaker_head_hidden": int(speaker_head_hidden),
            "num_speakers": int(num_speakers),
            "grl_lambda": float(grl_lambda),
        }

    @staticmethod
    def _infer_hidden_size(cfg: Any) -> int:
        for key in ("hidden_size", "hidden_dim", "classifier_proj_size", "d_model"):
            value = getattr(cfg, key, None)
            if value is not None:
                return int(value)
        raise ValueError("Could not infer hidden size from model config.")

    def set_grl_lambda(self, lambd: float) -> None:
        self.grl.set_lambda(lambd)

    @property
    def config(self):
        return self.base.config

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        del exclude_embeddings
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(
        self,
        save_directory: str | Path,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
        safe_serialization: bool = True,
        **kwargs: Any,
    ) -> None:
        """HF-compatible save used by W&B callback and external tooling."""
        del kwargs
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        if state_dict is None:
            state_dict = self.state_dict()

        self.base.config.save_pretrained(str(save_directory))
        with (save_directory / "dann_config.json").open("w", encoding="utf-8") as handle:
            json.dump(self._dann_meta, handle, indent=2, sort_keys=True)
            handle.write("\n")

        if safe_serialization:
            try:
                from safetensors.torch import save_file as safe_save_file

                safe_save_file(state_dict, str(save_directory / SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
                return
            except Exception:
                # Fall back to PyTorch serialization if safetensors is unavailable.
                pass

        torch.save(state_dict, save_directory / WEIGHTS_NAME)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Proxy HF Trainer gradient-checkpointing calls to the wrapped model."""
        if not hasattr(self.base, "gradient_checkpointing_enable"):
            raise AttributeError(
                f"Wrapped model {type(self.base).__name__} does not support gradient checkpointing."
            )

        try:
            self.base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            # Backward compatibility with transformers versions that do not take kwargs.
            self.base.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.base, "is_gradient_checkpointing", False))

    def _build_feature_mask(
        self,
        feature_length: int,
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        if attention_mask.ndim != 2:
            return None

        if attention_mask.shape[1] == feature_length:
            return attention_mask

        if hasattr(self.base, "_get_feature_vector_attention_mask"):
            return self.base._get_feature_vector_attention_mask(feature_length, attention_mask)

        return None

    def _pool(self, hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0, :]

        feature_mask = self._build_feature_mask(hidden.shape[1], attention_mask)
        if feature_mask is None:
            return hidden.mean(dim=1)

        mask = feature_mask.to(hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom

    @staticmethod
    def _extract_last_hidden(outputs: Any) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            return hidden_states[-1]

        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is not None:
            return last_hidden_state

        raise RuntimeError("Model outputs do not include hidden states required for DANN.")

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> DANNOutputs:
        del labels, speaker_labels

        # Accept both waveform (`input_values`) and feature (`input_features`) models.
        primary_input = input_values
        primary_key = "input_values"
        if primary_input is None and input_features is not None:
            primary_input = input_features
            primary_key = "input_features"

        if primary_input is None:
            if self.main_input_name in kwargs:
                primary_input = kwargs.pop(self.main_input_name)
                primary_key = self.main_input_name
            elif "input_values" in kwargs:
                primary_input = kwargs.pop("input_values")
                primary_key = "input_values"
            elif "input_features" in kwargs:
                primary_input = kwargs.pop("input_features")
                primary_key = "input_features"

        if primary_input is None:
            raise TypeError(
                "DANNForAudioClassification.forward expected one of "
                f"'{self.main_input_name}', 'input_values', or 'input_features'."
            )

        model_inputs = {primary_key: primary_input}

        outputs = self.base(
            **model_inputs,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        last_hidden = self._extract_last_hidden(outputs)
        pooled_rep = self._pool(last_hidden, attention_mask)
        speaker_logits = self.speaker_head(self.grl(pooled_rep))

        return DANNOutputs(
            loss=None,
            logits=outputs.logits,
            speaker_logits=speaker_logits,
        )

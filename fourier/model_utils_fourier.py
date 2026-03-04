"""Model builders for Fourier-domain audio classification experiments."""

from __future__ import annotations

from typing import Any, Dict

from transformers import AutoConfig, AutoModelForAudioClassification

try:
    from .config_utils_fourier import get_nested
except ImportError:  # Script-mode execution
    from config_utils_fourier import get_nested


def build_fourier_audio_classification_model(
    config: Dict[str, Any],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> AutoModelForAudioClassification:
    """Create and configure the model from a pre-trained checkpoint."""

    model_id = str(get_nested(config, "model.id"))
    apply_dropout = bool(get_nested(config, "model.apply_dropout"))
    dropout_value = float(get_nested(config, "model.dropout"))
    freeze_feature_encoder = bool(get_nested(config, "model.freeze_feature_encoder"))

    model_config = AutoConfig.from_pretrained(
        model_id,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    if apply_dropout:
        for attribute in [
            "hidden_dropout",
            "attention_dropout",
            "activation_dropout",
            "feat_proj_dropout",
            "final_dropout",
        ]:
            if hasattr(model_config, attribute):
                setattr(model_config, attribute, dropout_value)

    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    if freeze_feature_encoder:
        if hasattr(model, "freeze_feature_encoder"):
            model.freeze_feature_encoder()
        elif hasattr(model, "freeze_feature_extractor"):
            model.freeze_feature_extractor()

    return model


def count_parameters(model: Any) -> int:
    """Return total number of trainable + frozen parameters."""

    return sum(param.numel() for param in model.parameters())

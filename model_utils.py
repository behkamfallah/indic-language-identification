"""Model construction helpers for audio classification experiments."""

from __future__ import annotations

from typing import Any, Dict

from transformers import AutoConfig, AutoModelForAudioClassification

from config_utils import get_nested


def build_audio_classification_model(
    config: Dict[str, Any],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> AutoModelForAudioClassification:
    """Create and configure the model from a pre-trained checkpoint.

    Steps:
    1. Load backbone config and inject label mappings.
    2. Optionally adjust dropout values (for regularization experiments).
    3. Load pre-trained weights for classification fine-tuning.
    4. Optionally freeze front-end feature encoder layers.
    """

    model_id = str(get_nested(config, "model.id", "facebook/mms-300m"))
    apply_dropout = bool(get_nested(config, "model.apply_dropout", False))
    dropout_value = float(get_nested(config, "model.dropout", 0.1))
    freeze_feature_encoder = bool(get_nested(config, "model.freeze_feature_encoder", False))

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

    # SpecAugment: time + frequency masking applied inside the model during
    # training (model.training=True). Supported natively by wav2vec2/MMS
    # configs via mask_time_prob / mask_feature_prob attributes.
    spec_aug_cfg = config.get("augmentation", {}).get("spec_augment", {})
    if spec_aug_cfg.get("enabled", False):
        _spec_map = {
            "mask_time_prob": float(spec_aug_cfg.get("mask_time_prob", 0.05)),
            "mask_time_length": int(spec_aug_cfg.get("mask_time_length", 10)),
            "mask_feature_prob": float(spec_aug_cfg.get("mask_feature_prob", 0.004)),
            "mask_feature_length": int(spec_aug_cfg.get("mask_feature_length", 10)),
        }
        for attr, val in _spec_map.items():
            if hasattr(model_config, attr):
                setattr(model_config, attr, val)
        print(
            f"SpecAugment enabled: mask_time_prob={_spec_map['mask_time_prob']}, "
            f"mask_feature_prob={_spec_map['mask_feature_prob']}"
        )

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


def count_parameters(model: AutoModelForAudioClassification) -> int:
    """Return total number of trainable + frozen parameters."""

    return sum(param.numel() for param in model.parameters())

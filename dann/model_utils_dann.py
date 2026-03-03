"""Model builders for DANN audio classification."""

from __future__ import annotations

from typing import Any, Dict

try:
    from .config_utils_dann import get_nested
    from .dann_model import DANNForAudioClassification
except ImportError:  # Script-mode execution
    from config_utils_dann import get_nested
    from dann_model import DANNForAudioClassification


def build_dann_audio_classification_model(
    config: Dict[str, Any],
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    num_speakers: int,
) -> DANNForAudioClassification:
    model_id = str(get_nested(config, "model.id"))
    apply_dropout = bool(get_nested(config, "model.apply_dropout"))
    dropout_value = float(get_nested(config, "model.dropout"))
    freeze_feature_encoder = bool(get_nested(config, "model.freeze_feature_encoder"))

    dann_cfg = config.get("dann", {}) if isinstance(config.get("dann", {}), dict) else {}
    speaker_head_hidden = int(dann_cfg.get("speaker_head_hidden", 256))
    grl_max_lambda = float(dann_cfg.get("grl_max_lambda", 1.0))
    pooling = str(dann_cfg.get("pooling", "mean"))

    return DANNForAudioClassification(
        model_id=model_id,
        num_labels=len(label2id),
        num_speakers=int(num_speakers),
        label2id=label2id,
        id2label=id2label,
        speaker_head_hidden=speaker_head_hidden,
        grl_lambda=grl_max_lambda,
        pooling=pooling,
        apply_dropout=apply_dropout,
        dropout_value=dropout_value,
        freeze_feature_encoder=freeze_feature_encoder,
    )


def count_parameters(model: Any) -> int:
    return sum(param.numel() for param in model.parameters())

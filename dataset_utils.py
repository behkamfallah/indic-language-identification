"""Dataset preparation helpers for spoken language identification.

This module contains all data-side logic:
- loading dataset splits
- validating expected columns
- resampling audio
- encoding labels
- mapping raw audio to model-ready features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor

from augmentation_utils import build_augmentor
from config_utils import get_nested


@dataclass
class PreparedDatasets:
    """Container for fully encoded train/eval datasets plus label maps."""

    train_dataset: Any
    eval_dataset: Any
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    model_input_name: str


def ensure_column_exists(split_columns: list[str], column_name: str, column_role: str) -> None:
    """Raise a clear error when a required dataset column is missing."""

    if column_name not in split_columns:
        columns = ", ".join(split_columns)
        raise KeyError(
            f"{column_role} column '{column_name}' was not found. "
            f"Available columns: {columns}"
        )


def prepare_encoded_datasets(
    config: Dict[str, Any],
    feature_extractor: AutoFeatureExtractor,
    seed: int,
) -> PreparedDatasets:
    """Load and encode dataset splits based on YAML configuration.

    The output datasets are already tokenized/featurized for Trainer, with a
    numeric `label` field aligned to `label2id`.
    """

    dataset_id = str(get_nested(config, "data.dataset_id", "badrex/nnti-dataset-full"))
    train_split_name = str(get_nested(config, "data.train_split", "train"))
    eval_split_name = str(get_nested(config, "data.eval_split", "validation"))
    audio_column = str(get_nested(config, "data.audio_column", "audio_filepath"))
    label_column = str(get_nested(config, "data.label_column", "language"))
    speaker_column = str(get_nested(config, "data.speaker_column", "speaker_id"))
    sampling_rate = int(get_nested(config, "data.sampling_rate", 16000))
    max_duration_seconds = float(get_nested(config, "data.max_duration_seconds", 7.0))
    map_batch_size = int(get_nested(config, "data.preprocessing_batch_size", 32))

    dataset = load_dataset(dataset_id)
    train_ds = dataset[train_split_name].shuffle(seed=seed)
    eval_ds = dataset[eval_split_name].shuffle(seed=seed)

    # Fail fast if the YAML points to incorrect column names.
    ensure_column_exists(train_ds.column_names, audio_column, "Audio")
    ensure_column_exists(train_ds.column_names, label_column, "Label")

    # Cast audio to a fixed sampling rate so all examples are homogeneous.
    train_ds = train_ds.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
    eval_ds = eval_ds.cast_column(audio_column, Audio(sampling_rate=sampling_rate))

    # Build deterministic label ordering for reproducibility.
    labels = sorted(train_ds.unique(label_column))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Build waveform augmentor (None when augmentation is disabled in config).
    augmentor = build_augmentor(config, seed=seed)
    if augmentor is not None:
        print("Waveform augmentation enabled for training split.")

    # Keep only lightweight metadata columns after mapping to reduce memory.
    keep_cols = [col for col in [speaker_column, label_column] if col in train_ds.column_names]

    model_input_name = feature_extractor.model_input_names[0]
    max_length = int(sampling_rate * max_duration_seconds)

    # Whisper's feature extractor (WhisperFeatureExtractor) always needs input
    # padded to exactly n_samples (480 000 for 16 kHz = 30 s) to produce the
    # fixed-length mel spectrogram (3000 frames) the model expects.
    # For all other models we keep the existing truncation-only behaviour and
    # leave padding to the AudioDataCollator.
    _whisper_style = hasattr(feature_extractor, "n_samples")
    _fe_pad_to = feature_extractor.n_samples if _whisper_style else None

    def preprocess(examples: Dict[str, Any], apply_augmentation: bool = False) -> Dict[str, Any]:
        """Convert raw audio arrays into padded/truncated model features.

        Args:
            apply_augmentation: When True and an augmentor is configured,
                waveform augmentations are applied before feature extraction.
                Always False for the eval split.
        """

        # Always truncate to the configured max duration first.
        audio_arrays = [
            sample["array"][:max_length] for sample in examples[audio_column]
        ]

        # Apply waveform augmentations to the training split only.
        if apply_augmentation and augmentor is not None:
            audio_arrays = augmentor.augment_batch(audio_arrays)

        fe_kwargs: Dict[str, Any] = {
            "sampling_rate": sampling_rate,
            "truncation": True,
            "return_attention_mask": True,
        }
        if _fe_pad_to is not None:
            # Pad to the model's native fixed length (e.g. 30 s for Whisper).
            fe_kwargs["padding"] = "max_length"
            fe_kwargs["max_length"] = _fe_pad_to
        else:
            fe_kwargs["max_length"] = max_length

        encoded = feature_extractor(audio_arrays, **fe_kwargs)

        # Convert string labels into numeric class IDs.
        encoded["label"] = [label2id[label] for label in examples[label_column]]

        # Convert sequences to numpy arrays so downstream padding is stable.
        encoded[model_input_name] = [np.asarray(item) for item in encoded[model_input_name]]
        encoded["length"] = [len(item) for item in encoded[model_input_name]]
        return encoded

    train_encoded = train_ds.map(
        preprocess,
        remove_columns=[col for col in train_ds.column_names if col not in keep_cols],
        batched=True,
        batch_size=map_batch_size,
        fn_kwargs={"apply_augmentation": augmentor is not None},
    )
    eval_encoded = eval_ds.map(
        preprocess,
        remove_columns=[col for col in eval_ds.column_names if col not in keep_cols],
        batched=True,
        batch_size=map_batch_size,
        fn_kwargs={"apply_augmentation": False},
    )

    return PreparedDatasets(
        train_dataset=train_encoded,
        eval_dataset=eval_encoded,
        label2id=label2id,
        id2label=id2label,
        model_input_name=model_input_name,
    )

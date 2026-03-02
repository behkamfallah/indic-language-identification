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

from audio_augment_utils import build_speaker_obfuscation_augmenter
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
    """Raise error when a required dataset column is missing."""

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

    dataset_id = str(get_nested(config, "data.dataset_id"))
    train_split_name = str(get_nested(config, "data.train_split"))
    eval_split_name = str(get_nested(config, "data.eval_split"))
    audio_column = str(get_nested(config, "data.audio_column"))
    label_column = str(get_nested(config, "data.label_column"))
    speaker_column = str(get_nested(config, "data.speaker_column"))
    sampling_rate = int(get_nested(config, "data.sampling_rate"))
    max_duration_seconds = float(get_nested(config, "data.max_duration_seconds"))
    map_batch_size = int(get_nested(config, "data.preprocessing_batch_size"))
    augment_train_only = bool(get_nested(config, "augmentation.train_only"))

    dataset = load_dataset(dataset_id)
    train_ds = dataset[train_split_name].shuffle(seed=seed)
    eval_ds = dataset[eval_split_name].shuffle(seed=seed)
    print("Train columns:", train_ds.column_names)
    print("Eval columns:", eval_ds.column_names)
    print(f"Dataset {dataset_id} has {len(train_ds)} train samples.")
    print(f"Dataset {dataset_id} has {len(eval_ds)} eval samples.")

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

    # Keep only lightweight metadata columns after mapping to reduce memory.
    keep_cols = [col for col in [speaker_column, label_column] if col in train_ds.column_names]

    model_input_name = feature_extractor.model_input_names[0]
    max_length = int(sampling_rate * max_duration_seconds)
    augmenter = build_speaker_obfuscation_augmenter(
        config=config,
        sampling_rate=sampling_rate,
        seed=seed,
    )

    def preprocess(
        examples: Dict[str, Any],
        apply_augmentation: bool = False,
    ) -> Dict[str, Any]:
        """Convert raw audio arrays into padded/truncated model features."""

        audio_arrays = []
        for sample in examples[audio_column]:
            audio = np.asarray(sample["array"], dtype=np.float32)
            if apply_augmentation and augmenter is not None:
                audio = augmenter(audio)
            audio_arrays.append(audio)

        encoded = feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )

        # Convert string labels into numeric class IDs.
        encoded["label"] = [label2id[label] for label in examples[label_column]]

        # Convert sequences to numpy arrays so downstream padding is stable.
        encoded[model_input_name] = [np.asarray(item) for item in encoded[model_input_name]]
        encoded["length"] = [len(item) for item in encoded[model_input_name]]
        return encoded

    train_encoded = train_ds.map(
        lambda batch: preprocess(batch, apply_augmentation=augmenter is not None),
        remove_columns=[col for col in train_ds.column_names if col not in keep_cols],
        batched=True,
        batch_size=map_batch_size,
    )
    eval_encoded = eval_ds.map(
        lambda batch: preprocess(
            batch,
            apply_augmentation=augmenter is not None and not augment_train_only,
        ),
        remove_columns=[col for col in eval_ds.column_names if col not in keep_cols],
        batched=True,
        batch_size=map_batch_size,
    )

    return PreparedDatasets(
        train_dataset=train_encoded,
        eval_dataset=eval_encoded,
        label2id=label2id,
        id2label=id2label,
        model_input_name=model_input_name,
    )

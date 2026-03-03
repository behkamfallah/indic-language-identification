"""Dataset preparation for DANN task-2 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor

try:
    from .audio_augment_utils_dann import build_speaker_obfuscation_augmenter
    from .config_utils_dann import get_nested
except ImportError:  # Script-mode execution
    from audio_augment_utils_dann import build_speaker_obfuscation_augmenter
    from config_utils_dann import get_nested


@dataclass
class PreparedDatasetsDANN:
    train_dataset: Any
    eval_dataset: Any
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    speaker2id: Dict[str, int]
    id2speaker: Dict[int, str]
    model_input_name: str


def ensure_column_exists(split_columns: list[str], column_name: str, column_role: str) -> None:
    if column_name not in split_columns:
        columns = ", ".join(split_columns)
        raise KeyError(
            f"{column_role} column '{column_name}' was not found. "
            f"Available columns: {columns}"
        )


def get_optional_nested(config: Dict[str, Any], dotted_key: str, default: Any) -> Any:
    try:
        return get_nested(config, dotted_key)
    except KeyError:
        return default


def prepare_encoded_datasets_dann(
    config: Dict[str, Any],
    feature_extractor: AutoFeatureExtractor,
    seed: int,
) -> PreparedDatasetsDANN:
    dataset_id = str(get_nested(config, "data.dataset_id"))
    train_split_name = str(get_nested(config, "data.train_split"))
    eval_split_name = str(get_nested(config, "data.eval_split"))
    audio_column = str(get_nested(config, "data.audio_column"))
    label_column = str(get_nested(config, "data.label_column"))
    speaker_column = str(get_nested(config, "data.speaker_column"))
    sampling_rate = int(get_nested(config, "data.sampling_rate"))
    max_duration_seconds = float(get_nested(config, "data.max_duration_seconds"))
    map_batch_size = int(get_nested(config, "data.preprocessing_batch_size"))
    keep_in_memory = bool(get_optional_nested(config, "data.preprocessing_keep_in_memory", True))
    load_from_cache_file = bool(get_optional_nested(config, "data.preprocessing_load_from_cache_file", True))
    writer_batch_size = int(get_optional_nested(config, "data.preprocessing_writer_batch_size", map_batch_size))

    augment_train_only = bool(get_nested(config, "augmentation.train_only"))

    dann_cfg = config.get("dann", {}) if isinstance(config.get("dann", {}), dict) else {}
    unknown_speaker_bucket = bool(dann_cfg.get("unknown_speaker_bucket", True))
    unknown_speaker_token = str(dann_cfg.get("unknown_speaker_token", "<unk_speaker>"))

    dataset = load_dataset(dataset_id)
    train_ds = dataset[train_split_name].shuffle(seed=seed)
    eval_ds = dataset[eval_split_name].shuffle(seed=seed)
    print("Train columns:", train_ds.column_names)
    print("Eval columns:", eval_ds.column_names)
    print(f"Dataset {dataset_id} has {len(train_ds)} train samples.")
    print(f"Dataset {dataset_id} has {len(eval_ds)} eval samples.")

    ensure_column_exists(train_ds.column_names, audio_column, "Audio")
    ensure_column_exists(train_ds.column_names, label_column, "Label")
    ensure_column_exists(train_ds.column_names, speaker_column, "Speaker")

    train_ds = train_ds.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
    eval_ds = eval_ds.cast_column(audio_column, Audio(sampling_rate=sampling_rate))

    labels = sorted(str(label) for label in train_ds.unique(label_column))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    speakers = sorted(str(spk) for spk in train_ds.unique(speaker_column))
    speaker2id = {spk: idx for idx, spk in enumerate(speakers)}

    unknown_speaker_id = None
    if unknown_speaker_bucket:
        while unknown_speaker_token in speaker2id:
            unknown_speaker_token = f"{unknown_speaker_token}_x"
        unknown_speaker_id = len(speaker2id)
        speaker2id[unknown_speaker_token] = unknown_speaker_id

    id2speaker = {idx: spk for spk, idx in speaker2id.items()}

    keep_cols = [col for col in [speaker_column, label_column] if col in train_ds.column_names]

    model_input_name = feature_extractor.model_input_names[0]
    max_length = int(sampling_rate * max_duration_seconds)
    augmenter = build_speaker_obfuscation_augmenter(
        config=config,
        sampling_rate=sampling_rate,
        seed=seed,
    )

    def preprocess(examples: Dict[str, Any], apply_augmentation: bool = False) -> Dict[str, Any]:
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

        encoded["label"] = [label2id[str(label)] for label in examples[label_column]]

        speaker_values = [str(spk) for spk in examples[speaker_column]]
        if unknown_speaker_id is not None:
            encoded["speaker_label"] = [speaker2id.get(spk, unknown_speaker_id) for spk in speaker_values]
        else:
            missing = [spk for spk in speaker_values if spk not in speaker2id]
            if missing:
                unseen = sorted(set(missing))[:5]
                raise KeyError(
                    "Found eval speakers that do not exist in train and unknown_speaker_bucket=false. "
                    f"Examples: {unseen}"
                )
            encoded["speaker_label"] = [speaker2id[spk] for spk in speaker_values]

        encoded[model_input_name] = [np.asarray(item) for item in encoded[model_input_name]]
        encoded["length"] = [len(item) for item in encoded[model_input_name]]
        return encoded

    train_encoded = train_ds.map(
        lambda batch: preprocess(batch, apply_augmentation=augmenter is not None),
        remove_columns=[col for col in train_ds.column_names if col not in keep_cols],
        batched=True,
        batch_size=map_batch_size,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=load_from_cache_file,
        writer_batch_size=writer_batch_size,
    )
    eval_encoded = eval_ds.map(
        lambda batch: preprocess(
            batch,
            apply_augmentation=augmenter is not None and not augment_train_only,
        ),
        remove_columns=[col for col in eval_ds.column_names if col not in keep_cols],
        batched=True,
        batch_size=map_batch_size,
        keep_in_memory=keep_in_memory,
        load_from_cache_file=load_from_cache_file,
        writer_batch_size=writer_batch_size,
    )

    return PreparedDatasetsDANN(
        train_dataset=train_encoded,
        eval_dataset=eval_encoded,
        label2id=label2id,
        id2label=id2label,
        speaker2id=speaker2id,
        id2speaker=id2speaker,
        model_input_name=model_input_name,
    )

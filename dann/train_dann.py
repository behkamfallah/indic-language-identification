"""Standalone DANN training entrypoint for Task 2."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoFeatureExtractor, set_seed

try:
    from .config_utils_dann import get_nested, load_yaml, merge_dicts, parse_overrides, save_json
    from .dataset_utils_dann import prepare_encoded_datasets_dann
    from .model_utils_dann import build_dann_audio_classification_model, count_parameters
    from .tracking_utils_dann import setup_hf, setup_wandb
    from .trainer_utils_dann import (
        AudioDataCollatorWithSpeaker,
        DANNConfig,
        build_dann_trainer,
        build_training_arguments,
    )
except ImportError:  # Script-mode execution
    from config_utils_dann import get_nested, load_yaml, merge_dicts, parse_overrides, save_json
    from dataset_utils_dann import prepare_encoded_datasets_dann
    from model_utils_dann import build_dann_audio_classification_model, count_parameters
    from tracking_utils_dann import setup_hf, setup_wandb
    from trainer_utils_dann import (
        AudioDataCollatorWithSpeaker,
        DANNConfig,
        build_dann_trainer,
        build_training_arguments,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DANN training for Indic language identification (Task 2).")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional config override in key=value format, e.g. training.learning_rate=2e-5",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip training loop")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_yaml(args.config)
    overrides = parse_overrides(args.override)
    config = merge_dicts(config, overrides)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    use_hf = setup_hf(config)
    use_wandb = setup_wandb(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_run_name = str(get_nested(config, "run_name"))
    if not base_run_name:
        raise ValueError("Config file must specify 'run_name'.")
    run_id = f"{base_run_name}_{timestamp}"

    base_output_dir = str(get_nested(config, "output_dir"))
    if not base_output_dir:
        raise ValueError("Config file must specify 'output_dir'.")
    output_dir = Path(base_output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    base_save_dir = str(get_nested(config, "save_dir"))
    if not base_save_dir:
        raise ValueError("Config file must specify 'save_dir'.")
    config["save_dir"] = str(Path(base_save_dir) / run_id)

    wandb_name = get_nested(config, "tracking.wandb_run_name")
    if wandb_name:
        config["tracking"]["wandb_run_name"] = f"{wandb_name}_{timestamp}"

    save_json(output_dir / "resolved_config.json", config)

    model_id = str(get_nested(config, "model.id"))
    dataset_id = str(get_nested(config, "data.dataset_id"))

    print(f"Running DANN experiment {run_id}:")
    print(f"- Using config: {args.config}")
    print(f"- Model: {model_id}")
    print(f"- Dataset: {dataset_id}")
    print(f"- HF enabled: {use_hf}")
    print(f"- W&B enabled: {use_wandb}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id,
        do_normalize=True,
        return_attention_mask=True,
    )

    prepared_data = prepare_encoded_datasets_dann(
        config=config,
        feature_extractor=feature_extractor,
        seed=seed,
    )

    model = build_dann_audio_classification_model(
        config=config,
        label2id=prepared_data.label2id,
        id2label=prepared_data.id2label,
        num_speakers=len(prepared_data.speaker2id),
    )

    n_params = count_parameters(model)
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    if n_params > 600_000_000:
        print("WARNING: model exceeds the 600M parameter limit.")

    training_args = build_training_arguments(config, output_dir=output_dir, run_name=run_id)
    data_collator = AudioDataCollatorWithSpeaker(
        feature_extractor=feature_extractor,
        model_input_name=prepared_data.model_input_name,
    )

    dann_cfg_raw = config.get("dann", {}) if isinstance(config.get("dann", {}), dict) else {}
    dann_cfg = DANNConfig(
        alpha=float(dann_cfg_raw.get("alpha", 0.1)),
        grl_max_lambda=float(dann_cfg_raw.get("grl_max_lambda", 1.0)),
        grl_schedule=str(dann_cfg_raw.get("grl_schedule", "dann")),
    )

    trainer = build_dann_trainer(
        model=model,
        training_args=training_args,
        train_dataset=prepared_data.train_dataset,
        eval_dataset=prepared_data.eval_dataset,
        data_collator=data_collator,
        feature_extractor=feature_extractor,
        dann_cfg=dann_cfg,
    )

    if not args.skip_train:
        print("Training started...")
        train_result = trainer.train()
        save_json(output_dir / "train_metrics.json", train_result.metrics)

    if not args.skip_eval:
        print("Evaluation started...")
        eval_output = trainer.predict(prepared_data.eval_dataset)
        eval_metrics = eval_output.metrics
        save_json(output_dir / "eval_metrics.json", eval_metrics)

        predictions = eval_output.predictions
        if isinstance(predictions, tuple):
            lang_logits = predictions[0]
        else:
            lang_logits = predictions

        label_ids = eval_output.label_ids
        if isinstance(label_ids, tuple):
            labels = label_ids[0]
        else:
            labels = label_ids

        pred_ids = np.argmax(lang_logits, axis=1)
        cm = confusion_matrix(labels, pred_ids)
        cm_data = {
            "matrix": cm.tolist(),
            "labels": [prepared_data.id2label[i] for i in range(len(prepared_data.id2label))],
        }
        save_json(output_dir / "confusion_matrix.json", cm_data)
        print(f"Confusion matrix saved to: {output_dir / 'confusion_matrix.json'}")

    model_dir = Path(str(get_nested(config, "save_dir")))
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    if hasattr(trainer.model, "save_pretrained"):
        trainer.model.save_pretrained(str(model_dir))
    feature_extractor.save_pretrained(str(model_dir))

    save_json(
        output_dir / "label_mapping.json",
        {
            "label2id": prepared_data.label2id,
            "id2label": prepared_data.id2label,
            "speaker2id": prepared_data.speaker2id,
            "id2speaker": prepared_data.id2speaker,
        },
    )
    print(f"Model saved to: {model_dir}")


if __name__ == "__main__":
    main()

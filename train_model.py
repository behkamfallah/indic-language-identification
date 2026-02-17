"""Main training entrypoint for Indic spoken language identification.

The script is intentionally thin and readable:
1. Parse config + CLI overrides.
2. Build data/model/trainer via dedicated helper modules.
3. Run train/eval.
4. Persist artifacts and metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, set_seed

from config_utils import get_nested, load_yaml, merge_dicts, parse_overrides, save_json
from dataset_utils import prepare_encoded_datasets
from model_utils import build_audio_classification_model, count_parameters
from tracking_utils import maybe_login_hf, setup_wandb
from trainer_utils import AudioDataCollator, build_trainer, build_training_arguments


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for the training script."""

    parser = argparse.ArgumentParser(description="Task 1 training script for Indic language ID")
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
    """Execute one full experiment run from config to saved artifacts."""

    parser = build_arg_parser()
    args = parser.parse_args()

    # Load base YAML, then merge runtime overrides (if any).
    config = load_yaml(args.config)
    overrides = parse_overrides(args.override)
    config = merge_dicts(config, overrides)

    # Make all stochastic components reproducible where possible.
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Optional authentication setup (environment-variable driven).
    maybe_login_hf(config)
    use_wandb = setup_wandb(config)

    # Build run identity and output directory for this execution.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = str(get_nested(config, "run_name", f"task1_{timestamp}"))
    output_dir = Path(str(get_nested(config, "output_dir", f"./outputs/{run_name}")))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the fully resolved config for exact reproducibility.
    save_json(output_dir / "resolved_config.json", config)

    model_id = str(get_nested(config, "model.id", "facebook/mms-300m"))
    dataset_id = str(get_nested(config, "data.dataset_id", "badrex/nnti-dataset-full"))

    print(f"Using config: {args.config}")
    print(f"Model: {model_id}")
    print(f"Dataset: {dataset_id}")
    print(f"W&B enabled: {use_wandb}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    elif bool(get_nested(config, "training.fp16", False)) or bool(get_nested(config, "training.bf16", False)):
        print("Mixed precision requested but CUDA is unavailable. fp16/bf16 will be disabled automatically.")

    # Load feature extractor first; data preprocessing depends on its expected input.
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id,
        do_normalize=True,
        return_attention_mask=True,
    )

    # Prepare encoded train/eval datasets and label mappings.
    prepared_data = prepare_encoded_datasets(
        config=config,
        feature_extractor=feature_extractor,
        seed=seed,
    )

    # Build model configured for the dataset's label space.
    model = build_audio_classification_model(
        config=config,
        label2id=prepared_data.label2id,
        id2label=prepared_data.id2label,
    )

    # Print size and enforce awareness of project constraints.
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    if n_params > 600_000_000:
        print("WARNING: model exceeds the 600M parameter limit in the project instructions.")

    # Build Trainer components.
    training_args = build_training_arguments(config, output_dir=output_dir, run_name=run_name)
    data_collator = AudioDataCollator(
        feature_extractor=feature_extractor,
        model_input_name=prepared_data.model_input_name,
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=prepared_data.train_dataset,
        eval_dataset=prepared_data.eval_dataset,
        data_collator=data_collator,
        feature_extractor=feature_extractor,
    )

    # Launch training and save training metrics.
    if not args.skip_train:
        print("Training started...")
        train_result = trainer.train()
        save_json(output_dir / "train_metrics.json", train_result.metrics)

    # Run evaluation and save eval metrics.
    if not args.skip_eval:
        print("Evaluation started...")
        eval_metrics = trainer.evaluate()
        save_json(output_dir / "eval_metrics.json", eval_metrics)

    # Save model weights and feature extractor for inference/reuse.
    model_dir = Path(str(get_nested(config, "save_dir", output_dir / "model")))
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    feature_extractor.save_pretrained(str(model_dir))

    # Persist label mappings to keep post-training predictions interpretable.
    save_json(
        output_dir / "label_mapping.json",
        {"label2id": prepared_data.label2id, "id2label": prepared_data.id2label},
    )
    print(f"Model saved to: {model_dir}")


if __name__ == "__main__":
    main()

"""
Main training entrypoint for Indic spoken language identification.

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
import numpy as np
from sklearn.metrics import confusion_matrix
from transformers import AutoFeatureExtractor, set_seed

from config_utils import get_nested, load_yaml, merge_dicts, parse_overrides, save_json
from dataset_utils import prepare_encoded_datasets
from model_utils import build_audio_classification_model, count_parameters
from tracking_utils import maybe_login_hf, setup_wandb
from trainer_utils import AudioDataCollator, build_trainer, build_training_arguments


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for the training script."""

    parser = argparse.ArgumentParser(description="Training script for Indic language identification.")
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

    # Load YAML, then merge command-line overrides (if any).
    config = load_yaml(args.config)
    overrides = parse_overrides(args.override)
    config = merge_dicts(config, overrides)

    # Make all stochastic components reproducible where possible.
    seed = int(config.get("seed", 42))
    set_seed(seed)

    # Optional authentication setup.
    maybe_login_hf(config)
    use_wandb = setup_wandb(config)

    # Build run identity and output directory for this execution.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Strict configuration: Keys MUST exist in the YAML.
    # We fetch the exact values provided in the config without arbitrary fallbacks.
    base_run_name = str(get_nested(config, "run_name"))
    if not base_run_name:
         raise ValueError("Config file must specify 'run_name'.")

    # Create unique run ID by appending timestamp to the configured name
    run_id = f"{base_run_name}_{timestamp}"
    
    # 1. Output Directory
    base_output_dir = str(get_nested(config, "output_dir"))
    if not base_output_dir:
        raise ValueError("Config file must specify 'output_dir'.")
    output_dir = Path(base_output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Model Save Directory
    base_save_dir = str(get_nested(config, "save_dir"))
    if not base_save_dir:
        raise ValueError("Config file must specify 'save_dir'.")
    # Update config so Trainer uses this timestamped path
    config["save_dir"] = str(Path(base_save_dir) / run_id)

    # 3. WandB Run Name
    wandb_name = get_nested(config, "tracking.wandb_run_name")
    if wandb_name:
        config["tracking"]["wandb_run_name"] = f"{wandb_name}_{timestamp}"

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
        print("WARNING: model exceeds the 600M parameter limit.")

    # Build Trainer components.
    training_args = build_training_arguments(config, output_dir=output_dir, run_name=run_id)
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
        # Use predict() instead of evaluate() to get raw predictions for confusion matrix
        eval_output = trainer.predict(prepared_data.eval_dataset)
        eval_metrics = eval_output.metrics

        # Save standard metrics (now includes precision, recall, f1 from trainer_utils)
        save_json(output_dir / "eval_metrics.json", eval_metrics)

        # Compute and save confusion matrix
        logits = eval_output.predictions[0] if isinstance(eval_output.predictions, tuple) else eval_output.predictions
        predictions = np.argmax(logits, axis=1)
        labels = eval_output.label_ids
        
        cm = confusion_matrix(labels, predictions)
        
        # Save confusion matrix with class labels for easy visualization
        cm_data = {
            "matrix": cm.tolist(),
            "labels": [prepared_data.id2label[i] for i in range(len(prepared_data.id2label))]
        }
        save_json(output_dir / "confusion_matrix.json", cm_data)
        print(f"Confusion matrix saved to: {output_dir / 'confusion_matrix.json'}")

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

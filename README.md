# Indic Spoken Language Identification

## Environment

- Recommended Python: `3.11`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_cuda.txt
```

This installs the standard cross-platform `torch` and `torchaudio` wheels for local development. The Docker image upgrades them to CUDA 11.8 builds during image creation.

## Dataset

The default configs target the Hugging Face dataset `badrex/nnti-dataset-full` with:

- training split: `train`
- evaluation split: `validation`
- audio column: `audio_filepath`
- language label column: `language`
- speaker metadata column: `speaker_id`

All main pipelines resample audio to `16 kHz` and truncate clips to a configured maximum duration before feature extraction.

## Repository Layout

- `train_model.py`: main Task 1 / augmentation training entrypoint
- `config_utils.py`: YAML loading, override parsing, config merging, JSON export
- `dataset_utils.py`: dataset loading, resampling, label mapping, preprocessing
- `model_utils.py`: Hugging Face audio-classification model construction
- `trainer_utils.py`: collator, `TrainingArguments`, metrics, `Trainer`
- `audio_augment_utils.py`: Task 2 speaker-obfuscation augmentation (pitch, spectral, timbre masking)
- `tracking_utils.py`: Hugging Face / W&B login helpers
- `configs/`: root YAML configs for Task 1 and augmentation-based Task 2 runs
- `fourier/`: standalone Fourier-domain augmentation pipeline
- `dann/`: standalone domain-adversarial training pipeline
- `tsne_utils.py`, `analyze_tsne.py`, `tsne_run.py`: embedding extraction and visualization utilities
- `conf_mat_visualizer.py`: plots `confusion_matrix.json` outputs as a heatmap
- `notebooks/`: exploratory notebooks used during development and analysis

## Main Training Pipelines

### Task 1: core fine-tuning pipeline

Run the baseline MMS configuration:

```bash
python3 train_model.py --config configs/task1_baseline_mms300m.yaml
```

Other Task 1 configs:

- `configs/task1_baseline_30.yaml`
- `configs/task1_old_baseline_mms300m.yaml`
- `configs/task1_w2v_bert.yaml`

Example commands:

```bash
python3 train_model.py --config configs/task1_w2v_bert.yaml
```

Runtime overrides are supported for any YAML field:

```bash
python3 train_model.py \
  --config configs/task1_baseline_mms300m.yaml \
  --override training.learning_rate=1e-5 \
  --override training.num_train_epochs=8
```

### Task 2: augmentation-based speaker-bias mitigation

Root configs under `configs/` run the standard training pipeline with augmentation enabled.

MMS-based runs:

- `configs/task2_ablation1_pitch.yaml`
- `configs/task2_ablation1_spectral.yaml`
- `configs/task2_ablation1_timbre.yaml`
- `configs/task2_ablation2_p_s.yaml`

W2v-BERT-based runs:

- `configs/task2_bert_a1_p.yaml`
- `configs/task2_bert_a1_s.yaml`
- `configs/task2_bert_a1_t.yaml`

Example:

```bash
python3 train_model.py --config configs/task2_ablation1_pitch.yaml
python3 train_model.py --config configs/task2_bert_a1_p.yaml
```

### Task 2: Fourier-domain augmentation

Standalone entrypoint:

```bash
python3 fourier/train_fourier.py --config fourier/task2_fourier_mms300m.yaml
python3 fourier/train_fourier.py --config fourier/task2_fourier_bert.yaml
```

### Task 2: DANN

Standalone entrypoint:

```bash
python3 dann/train_dann.py --config dann/task2_dann_mms300m.yaml
python3 dann/train_dann.py --config dann/task2_dann_bert.yaml
```

## Analysis Utilities

Run the modular t-SNE analysis on a saved model:

```bash
python3 analyze_tsne.py \
  --config configs/task1_baseline_mms300m.yaml \
  --model_dir models/<run_id> \
  --tag baseline
```

Plot a saved confusion matrix:

```bash
python3 conf_mat_visualizer.py \
  --file outputs/<run_id>/confusion_matrix.json
```

## Outputs

For the main `train_model.py` pipeline, each run creates:

- `outputs/<run_id>/resolved_config.json`
- `outputs/<run_id>/train_metrics.json`
- `outputs/<run_id>/eval_metrics.json`
- `outputs/<run_id>/confusion_matrix.json`
- `outputs/<run_id>/label_mapping.json`
- `models/<run_id>/...` with the saved model weights and feature extractor

The DANN and Fourier entrypoints follow the same pattern inside their configured output and save directories.

## Tracking and Credentials

Some example YAML configs enable Weights & Biases by default. You can either:

- export the expected environment variables
- rely on an existing local W&B login session
- set `tracking.use_wandb: false` in the YAML

Environment variables used by the repo:

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
```

`HF_TOKEN` is only needed if you want authenticated Hugging Face access or `push_to_hub`.

## Notes

- Pitch and spectral augmentation depend on `torchaudio.sox_effects` and broken local SoX linkage will fail the pipeline.
- `configs/task1_commented_reference.yaml` is the starting point if you want to understand the configuration surface.

# Indic Language Identification

## Python Version

- Recommended: Python 3.11+

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

- `train_model.py`: main orchestration script (CLI entrypoint)
- `config_utils.py`: YAML loading, override parsing, nested config helpers, JSON artifact writing
- `tracking_utils.py`: Hugging Face and W&B auth/tracking setup
- `dataset_utils.py`: dataset loading, validation, audio casting, preprocessing/encoding
- `model_utils.py`: model config + model initialization + parameter counting
- `trainer_utils.py`: collator, `TrainingArguments`, metrics function, `Trainer` construction
- `configs/task1_baseline_mms300m.yaml`: baseline reproduction config
- `configs/task1_mms_tuned.yaml`: tuned MMS experiment
- `configs/task1_w2v_bert.yaml`: alternate pre-trained model experiment
- `configs/task1_commented_reference.yaml`: fully documented config with inline explanations for each field
- `docs/file_function_relationships.md`: Mermaid diagram of file/function dependencies
- `docs/training_execution_flow.md`: Mermaid diagram of runtime execution flow

## Run Task 1 Experiments

### 1. Reproduce baseline (MMS-300m)

```bash
python train_model.py --config configs/task1_baseline_mms300m.yaml
```

### 2. Improved run (tuned MMS-300m)

```bash
python train_model.py --config configs/task1_mms_tuned.yaml
```

### 3. Alternate model run (w2v-BERT-2.0)

```bash
python train_model.py --config configs/task1_w2v_bert.yaml
```

## Optional Runtime Overrides

Override any YAML value from CLI:

```bash
python train_model.py \
  --config configs/task1_mms_tuned.yaml \
  --override training.learning_rate=1.5e-5 \
  --override training.num_train_epochs=10
```

## Weights & Biases / Hugging Face Tokens

No tokens are hardcoded. If needed, set environment variables:

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
```

Then enable W&B in the YAML:

```yaml
tracking:
  use_wandb: true
```

## Outputs

For each run, the script writes:

- resolved config JSON
- train/eval metrics JSON
- label mapping JSON
- saved model + feature extractor

under the configured `output_dir` / `save_dir`.

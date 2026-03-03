# DANN (Task 2)

This folder is a standalone DANN training path for Task 2.
It does **not** require edits to files outside `dann/`.

## Files

- `train_dann.py`: main training entrypoint
- `dataset_utils_dann.py`: dataset encoding with `speaker_label`
- `dann_grl.py`: gradient reversal layer
- `dann_model.py`: language classifier + speaker adversary branch
- `trainer_utils_dann.py`: custom collator + `DANNTrainer` combined loss
- `model_utils_dann.py`: model builder
- `config_utils_dann.py`: config helpers (copied for isolation)
- `tracking_utils_dann.py`: HF/W&B auth helpers (copied for isolation)
- `audio_augment_utils_dann.py`: augmentation helpers (copied for isolation)
- `task2_dann_mms300m.yaml`: example config for Task 2

## Run

```bash
python3 dann/train_dann.py --config dann/task2_dann_mms300m.yaml
```

This default config runs **DANN only** (no augmentation / no waveform modifications).
SoX is only needed if you explicitly enable pitch/spectral augmentation.

## Quick sanity checks (before long training)

```python
# In a notebook from repo root:
from pathlib import Path
from transformers import AutoFeatureExtractor
from dann.config_utils_dann import load_yaml
from dann.dataset_utils_dann import prepare_encoded_datasets_dann
from dann.trainer_utils_dann import AudioDataCollatorWithSpeaker

cfg = load_yaml(Path("dann/task2_dann_mms300m.yaml"))
feat = AutoFeatureExtractor.from_pretrained(cfg["model"]["id"], do_normalize=True, return_attention_mask=True)
prep = prepare_encoded_datasets_dann(cfg, feat, seed=cfg.get("seed", 42))
collator = AudioDataCollatorWithSpeaker(feat, prep.model_input_name)

batch = collator([prep.train_dataset[i] for i in range(8)])
assert "labels" in batch and "speaker_labels" in batch
assert prep.model_input_name in batch
```

```python
# Forward shape check:
from dann.model_utils_dann import build_dann_audio_classification_model

model = build_dann_audio_classification_model(
    config=cfg,
    label2id=prep.label2id,
    id2label=prep.id2label,
    num_speakers=len(prep.speaker2id),
)
out = model(**{k: v for k, v in batch.items() if k in [prep.model_input_name, "attention_mask"]})
print(out.logits.shape, out.speaker_logits.shape)
```

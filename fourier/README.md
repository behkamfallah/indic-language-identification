# Fourier-Domain Training (Task 2)

This folder is a standalone Fourier-domain training path for Task 2.
It does **not** require edits to files outside `fourier/`.

## Why this is legitimate

For speech, frequency-domain representations are standard and useful, but the
practical tool is FFT/STFT (not a global Fourier-series fit of the whole clip).
This pipeline keeps the same MMS classifier and adds optional FFT-domain
waveform perturbation during preprocessing.

## Files

- `train_fourier.py`: main training entrypoint
- `dataset_utils_fourier.py`: dataset encoding + Fourier augmentation hook
- `fourier_augment_utils.py`: FFT magnitude/phase perturbation
- `model_utils_fourier.py`: model builder
- `trainer_utils_fourier.py`: collator + metrics + trainer builder
- `config_utils_fourier.py`: config helpers (copied for isolation)
- `tracking_utils_fourier.py`: HF/W&B auth helpers (copied for isolation)
- `task2_fourier_mms300m.yaml`: example config for Task 2

## Run

```bash
python3 fourier/train_fourier.py --config fourier/task2_fourier_mms300m.yaml
```

## Quick sanity check

```python
from pathlib import Path
from transformers import AutoFeatureExtractor
from fourier.config_utils_fourier import load_yaml
from fourier.dataset_utils_fourier import prepare_encoded_datasets_fourier
from fourier.trainer_utils_fourier import AudioDataCollatorFourier

cfg = load_yaml(Path("fourier/task2_fourier_mms300m.yaml"))
feat = AutoFeatureExtractor.from_pretrained(cfg["model"]["id"], do_normalize=True, return_attention_mask=True)
prep = prepare_encoded_datasets_fourier(cfg, feat, seed=cfg.get("seed", 42))
collator = AudioDataCollatorFourier(feat, prep.model_input_name)

batch = collator([prep.train_dataset[i] for i in range(8)])
assert "labels" in batch
assert prep.model_input_name in batch
```

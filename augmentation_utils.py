"""Waveform-level audio augmentation utilities for training.

All functions operate on raw 1D float32 numpy arrays at a known sampling rate.
Each augmentation is optional and probability-gated so it can be toggled
independently via the YAML config ``augmentation`` section.

Augmentation is intentionally kept to the **waveform** domain so it is model-
agnostic and does not require any model-specific hooks.

SpecAugment (frequency / time masking in the latent domain) is handled
separately in model_utils.py via model config attributes because it is applied
by the model itself during the forward pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------


def add_gaussian_noise(
    audio: np.ndarray,
    min_snr_db: float = 10.0,
    max_snr_db: float = 40.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise at a random SNR sampled uniformly from the given range.

    A high SNR (e.g. 40 dB) means barely audible noise; a low SNR (e.g. 10 dB)
    means heavy noise.  We clamp to avoid silencing near-silent clips.
    """

    if rng is None:
        rng = np.random.default_rng()

    signal_power = float(np.mean(audio.astype(np.float64) ** 2))
    if signal_power < 1e-10:
        return audio  # silent clip — skip to avoid division by zero

    snr_db = rng.uniform(min_snr_db, max_snr_db)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(audio.dtype)
    return audio + noise


def volume_jitter(
    audio: np.ndarray,
    min_gain_db: float = -6.0,
    max_gain_db: float = 6.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Scale audio amplitude by a random gain factor (in dB).

    Keeps the clip from clipping by clamping to [-1, 1] if the input is
    normalised float audio.
    """

    if rng is None:
        rng = np.random.default_rng()

    gain_db = rng.uniform(min_gain_db, max_gain_db)
    gain = 10.0 ** (gain_db / 20.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(audio.dtype)


def speed_perturb(
    audio: np.ndarray,
    min_speed: float = 0.9,
    max_speed: float = 1.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Perturb playback speed by resampling the waveform to a different length.

    A speed factor > 1 compresses the signal (faster speech); < 1 stretches it.
    This implicitly changes pitch as well, which is acceptable as a data
    augmentation side-effect — the language identity is preserved.

    Requires scipy.  If not installed the clip is returned unchanged.
    """

    try:
        from scipy.signal import resample as scipy_resample
    except ImportError:
        return audio

    if rng is None:
        rng = np.random.default_rng()

    speed_factor = float(rng.uniform(min_speed, max_speed))
    if abs(speed_factor - 1.0) < 1e-3:
        return audio

    new_length = max(1, int(round(len(audio) / speed_factor)))
    resampled = scipy_resample(audio, new_length).astype(audio.dtype)
    return resampled


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------


class WaveformAugmentor:
    """Applies a configurable chain of waveform augmentations to a batch.

    Each augmentation is independently gated by a probability ``p``.  The same
    random state is used across a preprocessing worker so results are
    reproducible when a fixed ``seed`` is supplied.

    Args:
        aug_config: The ``augmentation`` sub-dict from the YAML config.
        seed: Optional integer seed for the internal numpy RNG.
    """

    def __init__(self, aug_config: Dict[str, Any], seed: Optional[int] = None) -> None:
        self.aug_config = aug_config
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def augment_batch(self, audio_arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Apply the enabled augmentations to each array in the list."""

        return [self._augment_one(arr) for arr in audio_arrays]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augment_one(self, audio: np.ndarray) -> np.ndarray:
        """Run every enabled augmentation gate on a single waveform."""

        audio = self._maybe_apply(
            audio,
            key="noise",
            fn=lambda a, cfg: add_gaussian_noise(
                a,
                min_snr_db=float(cfg.get("min_snr_db", 10.0)),
                max_snr_db=float(cfg.get("max_snr_db", 40.0)),
                rng=self.rng,
            ),
        )

        audio = self._maybe_apply(
            audio,
            key="volume_jitter",
            fn=lambda a, cfg: volume_jitter(
                a,
                min_gain_db=float(cfg.get("min_gain_db", -6.0)),
                max_gain_db=float(cfg.get("max_gain_db", 6.0)),
                rng=self.rng,
            ),
        )

        audio = self._maybe_apply(
            audio,
            key="speed_perturb",
            fn=lambda a, cfg: speed_perturb(
                a,
                min_speed=float(cfg.get("min_speed", 0.9)),
                max_speed=float(cfg.get("max_speed", 1.1)),
                rng=self.rng,
            ),
        )

        return audio

    def _maybe_apply(self, audio: np.ndarray, key: str, fn) -> np.ndarray:
        """Apply ``fn`` only if the named augmentation is enabled and fires."""

        sub = self.aug_config.get(key, {})
        if not sub.get("enabled", False):
            return audio
        p = float(sub.get("prob", 0.5))
        if self.rng.random() < p:
            return fn(audio, sub)
        return audio


def build_augmentor(config: Dict[str, Any], seed: Optional[int] = None) -> Optional[WaveformAugmentor]:
    """Construct a ``WaveformAugmentor`` from the top-level config dict.

    Returns ``None`` when augmentation is globally disabled or the section is
    absent so callers can cheaply skip the augmentation step.
    """

    aug_cfg = config.get("augmentation", {})
    if not aug_cfg.get("enabled", False):
        return None
    return WaveformAugmentor(aug_cfg, seed=seed)

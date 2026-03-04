"""Fourier-domain augmentation helpers.

This module applies lightweight FFT perturbations that preserve global content
while reducing over-reliance on narrow speaker-specific spectral cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

try:
    from .config_utils_fourier import get_nested
except ImportError:  # Script-mode execution
    from config_utils_fourier import get_nested


@dataclass
class FourierAugmenter:
    """Apply FFT-domain perturbations to a waveform array."""

    config: Dict[str, Any]
    sampling_rate: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.config, dict):
            raise TypeError(f"fourier config must be a dict, got {type(self.config)}")

        self.rng = np.random.default_rng(int(self.seed))

        self.apply_prob = float(self.config.get("apply_prob", 1.0))
        if not (0.0 <= self.apply_prob <= 1.0):
            raise ValueError("fourier.apply_prob must be in [0, 1]")

        self.magnitude_dropout_prob = float(self.config.get("magnitude_dropout_prob", 0.8))
        if not (0.0 <= self.magnitude_dropout_prob <= 1.0):
            raise ValueError("fourier.magnitude_dropout_prob must be in [0, 1]")

        self.magnitude_dropout_bands_min = int(self.config.get("magnitude_dropout_bands_min", 1))
        self.magnitude_dropout_bands_max = int(self.config.get("magnitude_dropout_bands_max", 3))
        if (
            self.magnitude_dropout_bands_min < 0
            or self.magnitude_dropout_bands_max < 0
            or self.magnitude_dropout_bands_min > self.magnitude_dropout_bands_max
        ):
            raise ValueError("fourier.magnitude_dropout_bands_min/max are invalid")

        self.magnitude_dropout_width_min = float(self.config.get("magnitude_dropout_width_min", 0.02))
        self.magnitude_dropout_width_max = float(self.config.get("magnitude_dropout_width_max", 0.12))
        if not (0.0 <= self.magnitude_dropout_width_min <= self.magnitude_dropout_width_max <= 1.0):
            raise ValueError("fourier.magnitude_dropout_width_min/max must be in [0, 1] with min<=max")

        self.magnitude_dropout_attenuation_min = float(self.config.get("magnitude_dropout_attenuation_min", 0.0))
        self.magnitude_dropout_attenuation_max = float(self.config.get("magnitude_dropout_attenuation_max", 0.4))
        if not (
            0.0
            <= self.magnitude_dropout_attenuation_min
            <= self.magnitude_dropout_attenuation_max
            <= 1.0
        ):
            raise ValueError(
                "fourier.magnitude_dropout_attenuation_min/max must be in [0, 1] with min<=max"
            )

        self.phase_noise_prob = float(self.config.get("phase_noise_prob", 0.4))
        if not (0.0 <= self.phase_noise_prob <= 1.0):
            raise ValueError("fourier.phase_noise_prob must be in [0, 1]")

        self.phase_noise_std = float(self.config.get("phase_noise_std", 0.2))
        if self.phase_noise_std < 0.0:
            raise ValueError("fourier.phase_noise_std must be >= 0")

        self.highpass_prob = float(self.config.get("highpass_prob", 0.2))
        self.lowpass_prob = float(self.config.get("lowpass_prob", 0.2))
        for name, value in (("highpass_prob", self.highpass_prob), ("lowpass_prob", self.lowpass_prob)):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"fourier.{name} must be in [0, 1]")

        self.highpass_min_hz = float(self.config.get("highpass_min_hz", 60.0))
        self.highpass_max_hz = float(self.config.get("highpass_max_hz", 240.0))
        if self.highpass_min_hz > self.highpass_max_hz:
            raise ValueError("fourier.highpass_min_hz must be <= highpass_max_hz")

        self.lowpass_min_hz = float(self.config.get("lowpass_min_hz", 2600.0))
        self.lowpass_max_hz = float(self.config.get("lowpass_max_hz", 7000.0))
        if self.lowpass_min_hz > self.lowpass_max_hz:
            raise ValueError("fourier.lowpass_min_hz must be <= lowpass_max_hz")

        self.filter_floor = float(self.config.get("filter_floor", 0.05))
        if not (0.0 <= self.filter_floor <= 1.0):
            raise ValueError("fourier.filter_floor must be in [0, 1]")

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Return an augmented waveform (float32 numpy array)."""

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        if waveform.size < 8:
            return waveform.astype(np.float32, copy=False)
        if self.rng.random() > self.apply_prob:
            return waveform.astype(np.float32, copy=False)

        n_samples = waveform.shape[0]
        spectrum = np.fft.rfft(waveform)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        freqs_hz = np.fft.rfftfreq(n_samples, d=1.0 / float(self.sampling_rate))

        changed = False

        if self.magnitude_dropout_bands_max > 0 and self.rng.random() < self.magnitude_dropout_prob:
            changed = self._apply_magnitude_dropout(magnitude) or changed

        if self.phase_noise_std > 0.0 and self.rng.random() < self.phase_noise_prob:
            changed = self._apply_phase_noise(phase) or changed

        if self.rng.random() < self.highpass_prob:
            cutoff = float(self.rng.uniform(self.highpass_min_hz, self.highpass_max_hz))
            magnitude[freqs_hz < cutoff] *= self.filter_floor
            changed = True

        if self.rng.random() < self.lowpass_prob:
            cutoff = float(self.rng.uniform(self.lowpass_min_hz, self.lowpass_max_hz))
            magnitude[freqs_hz > cutoff] *= self.filter_floor
            changed = True

        if not changed:
            return waveform.astype(np.float32, copy=False)

        modified = magnitude * np.exp(1j * phase)
        reconstructed = np.fft.irfft(modified, n=n_samples)
        if not np.all(np.isfinite(reconstructed)):
            return waveform.astype(np.float32, copy=False)

        reconstructed = reconstructed.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(reconstructed))) if reconstructed.size > 0 else 0.0
        if peak > 1.0:
            reconstructed = reconstructed / peak

        return reconstructed.astype(np.float32, copy=False)

    def _apply_magnitude_dropout(self, magnitude: np.ndarray) -> bool:
        n_bins = int(magnitude.shape[0])
        if n_bins <= 3:
            return False

        n_bands = int(
            self.rng.integers(
                self.magnitude_dropout_bands_min,
                self.magnitude_dropout_bands_max + 1,
            )
        )
        if n_bands <= 0:
            return False

        changed = False
        for _ in range(n_bands):
            width_ratio = float(
                self.rng.uniform(
                    self.magnitude_dropout_width_min,
                    self.magnitude_dropout_width_max,
                )
            )
            width_bins = max(1, int(width_ratio * n_bins))
            width_bins = min(width_bins, n_bins - 1)
            start = int(self.rng.integers(1, n_bins - width_bins + 1))
            end = start + width_bins

            attenuation = float(
                self.rng.uniform(
                    self.magnitude_dropout_attenuation_min,
                    self.magnitude_dropout_attenuation_max,
                )
            )
            magnitude[start:end] *= attenuation
            changed = True

        return changed

    def _apply_phase_noise(self, phase: np.ndarray) -> bool:
        n_bins = int(phase.shape[0])
        if n_bins <= 2:
            return False

        noise = self.rng.normal(loc=0.0, scale=self.phase_noise_std, size=n_bins - 2)
        phase[1:-1] += noise.astype(np.float64, copy=False)
        return True


def build_fourier_augmenter(
    config: Dict[str, Any],
    sampling_rate: int,
    seed: int,
) -> FourierAugmenter | None:
    """Return configured Fourier augmenter or None if disabled."""

    try:
        fourier_cfg = get_nested(config, "fourier")
    except KeyError:
        return None
    if fourier_cfg is None:
        return None
    if not isinstance(fourier_cfg, dict):
        raise TypeError(f"config.fourier must be dict, got {type(fourier_cfg)}")

    enabled = fourier_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"fourier.enabled must be boolean, got {type(enabled)}")

    if not enabled:
        return None

    return FourierAugmenter(
        config=fourier_cfg,
        sampling_rate=int(sampling_rate),
        seed=int(seed),
    )

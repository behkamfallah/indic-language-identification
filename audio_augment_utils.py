"""Speaker-obfuscation audio augmentation helpers for Task 2.

The augmentation pipeline targets three speaker-identifying cues:
1) Pitch (F0): random SoX pitch shift with fixed sample-rate recovery.
2) Prominent spectral signatures: random narrowband attenuation filters.
3) Timbre/formant envelope: STFT-domain frequency/time masking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import warnings

import numpy as np
import torch

from config_utils import get_nested

try:
    from torchaudio.sox_effects import apply_effects_tensor
    from torchaudio.transforms import FrequencyMasking, TimeMasking

    TORCHAUDIO_AVAILABLE = True
except Exception:
    apply_effects_tensor = None  # type: ignore[assignment]
    FrequencyMasking = None  # type: ignore[assignment]
    TimeMasking = None  # type: ignore[assignment]
    TORCHAUDIO_AVAILABLE = False


@dataclass
class SpeakerObfuscationAugmenter:
    """Apply speaker-obfuscation augmentations to a waveform array."""

    config: Dict[str, Any]
    sampling_rate: int
    seed: int

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.apply_prob = float(self.config.get("apply_prob", 1.0))
        self._sox_available = TORCHAUDIO_AVAILABLE

        pitch_cfg = self.config.get("pitch", {})
        self.pitch_enabled = bool(pitch_cfg.get("enabled", True))
        self.pitch_prob = float(pitch_cfg.get("prob", 0.8))
        self.pitch_cents_min = float(pitch_cfg.get("cents_min", -250.0))
        self.pitch_cents_max = float(pitch_cfg.get("cents_max", 250.0))

        spectral_cfg = self.config.get("spectral", {})
        self.spectral_enabled = bool(spectral_cfg.get("enabled", True))
        self.spectral_prob = float(spectral_cfg.get("prob", 0.8))
        self.notches_min = int(spectral_cfg.get("notches_min", 1))
        self.notches_max = int(spectral_cfg.get("notches_max", 2))
        self.center_freq_min = float(spectral_cfg.get("center_freq_min", 250.0))
        self.center_freq_max = float(spectral_cfg.get("center_freq_max", 3800.0))
        self.q_min = float(spectral_cfg.get("q_min", 0.7))
        self.q_max = float(spectral_cfg.get("q_max", 2.0))
        self.attenuation_db_min = float(spectral_cfg.get("attenuation_db_min", -18.0))
        self.attenuation_db_max = float(spectral_cfg.get("attenuation_db_max", -6.0))
        self.highpass_prob = float(spectral_cfg.get("highpass_prob", 0.2))
        self.highpass_min_hz = float(spectral_cfg.get("highpass_min_hz", 50.0))
        self.highpass_max_hz = float(spectral_cfg.get("highpass_max_hz", 220.0))
        self.lowpass_prob = float(spectral_cfg.get("lowpass_prob", 0.2))
        self.lowpass_min_hz = float(spectral_cfg.get("lowpass_min_hz", 3000.0))
        self.lowpass_max_hz = float(spectral_cfg.get("lowpass_max_hz", 7000.0))

        timbre_cfg = self.config.get("timbre_mask", {})
        self.timbre_enabled = bool(timbre_cfg.get("enabled", True))
        self.timbre_prob = float(timbre_cfg.get("prob", 0.6))
        self.n_fft = int(timbre_cfg.get("n_fft", 400))
        self.hop_length = int(timbre_cfg.get("hop_length", 160))
        self.win_length = int(timbre_cfg.get("win_length", 400))
        self.freq_mask_param = int(timbre_cfg.get("freq_mask_param", 24))
        self.time_mask_param = int(timbre_cfg.get("time_mask_param", 30))
        self._window = torch.hann_window(self.win_length)
        self._freq_mask = (
            FrequencyMasking(freq_mask_param=self.freq_mask_param)
            if TORCHAUDIO_AVAILABLE and self.freq_mask_param > 0
            else None
        )
        self._time_mask = (
            TimeMasking(time_mask_param=self.time_mask_param)
            if TORCHAUDIO_AVAILABLE and self.time_mask_param > 0
            else None
        )

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Return an augmented waveform (float32 numpy array)."""

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        output = torch.as_tensor(waveform, dtype=torch.float32).unsqueeze(0)
        if output.numel() == 0 or self.rng.random() > self.apply_prob:
            return output.squeeze(0).numpy().astype(np.float32, copy=False)

        if self.pitch_enabled and self.rng.random() < self.pitch_prob:
            cents = float(self.rng.uniform(self.pitch_cents_min, self.pitch_cents_max))
            output = self._apply_sox_effects(
                output,
                effects=[
                    ["pitch", f"{cents:.2f}"],
                    ["rate", str(self.sampling_rate)],
                ],
            )

        if self.spectral_enabled and self.rng.random() < self.spectral_prob:
            output = self._apply_sox_effects(output, effects=self._build_spectral_effects())

        output = output.squeeze(0)
        if self.timbre_enabled and self.rng.random() < self.timbre_prob:
            output = self._apply_timbre_mask(output)

        max_abs = output.abs().max()
        if torch.isfinite(max_abs).item() and max_abs.item() > 1.0:
            output = output / max_abs

        return output.detach().cpu().numpy().astype(np.float32, copy=False)

    def _build_spectral_effects(self) -> list[list[str]]:
        """Create random narrowband attenuation and optional cutoff effects."""

        effects: list[list[str]] = []

        n_notches = int(self.rng.integers(self.notches_min, self.notches_max + 1))
        for _ in range(max(0, n_notches)):
            center = float(self.rng.uniform(self.center_freq_min, self.center_freq_max))
            q_value = float(self.rng.uniform(self.q_min, self.q_max))
            attenuation = float(self.rng.uniform(self.attenuation_db_min, self.attenuation_db_max))
            effects.append(
                ["equalizer", f"{center:.2f}", f"{q_value:.2f}q", f"{attenuation:.2f}"]
            )

        if self.rng.random() < self.highpass_prob:
            hp_hz = float(self.rng.uniform(self.highpass_min_hz, self.highpass_max_hz))
            effects.append(["highpass", f"{hp_hz:.2f}"])

        if self.rng.random() < self.lowpass_prob:
            lp_hz = float(self.rng.uniform(self.lowpass_min_hz, self.lowpass_max_hz))
            effects.append(["lowpass", f"{lp_hz:.2f}"])

        effects.append(["rate", str(self.sampling_rate)])
        return effects

    def _apply_sox_effects(self, waveform: torch.Tensor, effects: list[list[str]]) -> torch.Tensor:
        """Apply a SoX chain; fall back silently if SoX backend is unavailable."""

        if not effects or not self._sox_available or apply_effects_tensor is None:
            return waveform

        try:
            augmented, _ = apply_effects_tensor(waveform, self.sampling_rate, effects)
        except Exception as exc:  # pragma: no cover - backend-dependent.
            self._sox_available = False
            warnings.warn(
                f"SoX effects are unavailable in this torchaudio build ({exc}). "
                "Continuing without SoX-based augmentation.",
                RuntimeWarning,
            )
            return waveform
        return augmented

    def _apply_timbre_mask(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mask spectro-temporal bands in STFT magnitude and reconstruct signal."""

        if waveform.numel() < self.n_fft:
            return waveform
        if self._freq_mask is None and self._time_mask is None:
            return waveform

        window = self._window.to(waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        magnitude = stft.abs().unsqueeze(0)

        if self._freq_mask is not None:
            magnitude = self._freq_mask(magnitude)
        if self._time_mask is not None:
            magnitude = self._time_mask(magnitude)

        masked_stft = magnitude.squeeze(0) * torch.exp(1j * torch.angle(stft))
        reconstructed = torch.istft(
            masked_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=waveform.shape[-1],
        )
        if torch.isnan(reconstructed).any():
            return waveform
        return reconstructed


def build_speaker_obfuscation_augmenter(
    config: Dict[str, Any],
    sampling_rate: int,
    seed: int,
) -> SpeakerObfuscationAugmenter | None:
    """Return configured speaker-obfuscation augmenter or `None` if disabled."""

    aug_cfg = dict(get_nested(config, "augmentation", {}) or {})
    if not bool(aug_cfg.get("enabled", False)):
        return None

    if not TORCHAUDIO_AVAILABLE:
        warnings.warn(
            "augmentation.enabled=true but torchaudio is unavailable. "
            "Install torchaudio to enable Task 2 audio augmentation.",
            RuntimeWarning,
        )
        return None

    return SpeakerObfuscationAugmenter(
        config=aug_cfg,
        sampling_rate=sampling_rate,
        seed=seed,
    )

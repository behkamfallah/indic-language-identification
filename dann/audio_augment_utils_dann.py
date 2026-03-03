"""Speaker-obfuscation audio augmentation helpers for Task 2.

- If augmentation is enabled, missing torchaudio dependencies raise errors.
- SoX runtime is required for SoX-based effects (pitch/spectral).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

try:
    from .config_utils_dann import get_nested
except ImportError:  # Script-mode execution
    from config_utils_dann import get_nested

try:
    from torchaudio.sox_effects import apply_effects_tensor as _apply_effects_tensor
    from torchaudio.transforms import FrequencyMasking as _FrequencyMasking
    from torchaudio.transforms import TimeMasking as _TimeMasking
    _TORCHAUDIO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on environment
    _apply_effects_tensor = None
    _FrequencyMasking = None
    _TimeMasking = None
    _TORCHAUDIO_IMPORT_ERROR = exc


@dataclass
class SpeakerObfuscationAugmenter:
    """Apply speaker-obfuscation augmentations to a waveform array."""

    config: Dict[str, Any]
    sampling_rate: int
    seed: int

    @staticmethod
    def _ensure_torchaudio_available() -> None:
        if _TORCHAUDIO_IMPORT_ERROR is not None:
            raise ImportError(
                "torchaudio is required for augmentation, but it is unavailable."
            ) from _TORCHAUDIO_IMPORT_ERROR

    def __post_init__(self) -> None:
        if not isinstance(self.config, dict):
            raise TypeError(f"Augmenter config must be dict, got {type(self.config)}")

        self.rng = np.random.default_rng(int(self.seed))

        self.apply_prob = float(self.config.get("apply_prob", 1.0))
        if not (0.0 <= self.apply_prob <= 1.0):
            raise ValueError("augmentation.apply_prob must be in [0, 1]")

        pitch_cfg = self.config.get("pitch", {})
        if pitch_cfg is None:
            pitch_cfg = {}
        if not isinstance(pitch_cfg, dict):
            raise TypeError(f"augmentation.pitch must be dict, got {type(pitch_cfg)}")

        raw_pitch_enabled = pitch_cfg.get("enabled", True)
        if not isinstance(raw_pitch_enabled, bool):
            raise TypeError("augmentation.pitch.enabled must be boolean")
        self.pitch_enabled = raw_pitch_enabled

        self.pitch_prob = float(pitch_cfg.get("prob", 0.8))
        if not (0.0 <= self.pitch_prob <= 1.0):
            raise ValueError("augmentation.pitch.prob must be in [0, 1]")
        self.pitch_cents_min = float(pitch_cfg.get("cents_min", -250.0))
        self.pitch_cents_max = float(pitch_cfg.get("cents_max", 250.0))
        if self.pitch_cents_min > self.pitch_cents_max:
            raise ValueError("augmentation.pitch.cents_min must be <= cents_max")

        spectral_cfg = self.config.get("spectral", {})
        if spectral_cfg is None:
            spectral_cfg = {}
        if not isinstance(spectral_cfg, dict):
            raise TypeError(f"augmentation.spectral must be dict, got {type(spectral_cfg)}")

        raw_spectral_enabled = spectral_cfg.get("enabled", True)
        if not isinstance(raw_spectral_enabled, bool):
            raise TypeError("augmentation.spectral.enabled must be boolean")
        self.spectral_enabled = raw_spectral_enabled

        self.spectral_prob = float(spectral_cfg.get("prob", 0.8))
        if not (0.0 <= self.spectral_prob <= 1.0):
            raise ValueError("augmentation.spectral.prob must be in [0, 1]")

        self.notches_min = int(spectral_cfg.get("notches_min", 1))
        self.notches_max = int(spectral_cfg.get("notches_max", 2))
        if self.notches_min < 0 or self.notches_max < 0 or self.notches_min > self.notches_max:
            raise ValueError("augmentation.spectral.notches_min/max invalid")

        self.center_freq_min = float(spectral_cfg.get("center_freq_min", 250.0))
        self.center_freq_max = float(spectral_cfg.get("center_freq_max", 3800.0))
        if self.center_freq_min > self.center_freq_max:
            raise ValueError("augmentation.spectral.center_freq_min must be <= center_freq_max")

        self.q_min = float(spectral_cfg.get("q_min", 0.7))
        self.q_max = float(spectral_cfg.get("q_max", 2.0))
        if self.q_min <= 0 or self.q_max <= 0 or self.q_min > self.q_max:
            raise ValueError("augmentation.spectral.q_min/q_max invalid (must be >0 and min<=max)")

        self.attenuation_db_min = float(spectral_cfg.get("attenuation_db_min", -18.0))
        self.attenuation_db_max = float(spectral_cfg.get("attenuation_db_max", -6.0))
        if self.attenuation_db_min > self.attenuation_db_max:
            raise ValueError("augmentation.spectral.attenuation_db_min must be <= attenuation_db_max")

        self.highpass_prob = float(spectral_cfg.get("highpass_prob", 0.2))
        self.lowpass_prob = float(spectral_cfg.get("lowpass_prob", 0.2))
        for name, p in [("highpass_prob", self.highpass_prob), ("lowpass_prob", self.lowpass_prob)]:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"augmentation.spectral.{name} must be in [0, 1]")

        self.highpass_min_hz = float(spectral_cfg.get("highpass_min_hz", 50.0))
        self.highpass_max_hz = float(spectral_cfg.get("highpass_max_hz", 220.0))
        if self.highpass_min_hz > self.highpass_max_hz:
            raise ValueError("augmentation.spectral.highpass_min_hz must be <= highpass_max_hz")

        self.lowpass_min_hz = float(spectral_cfg.get("lowpass_min_hz", 3000.0))
        self.lowpass_max_hz = float(spectral_cfg.get("lowpass_max_hz", 7000.0))
        if self.lowpass_min_hz > self.lowpass_max_hz:
            raise ValueError("augmentation.spectral.lowpass_min_hz must be <= lowpass_max_hz")

        timbre_cfg = self.config.get("timbre_mask", {})
        if timbre_cfg is None:
            timbre_cfg = {}
        if not isinstance(timbre_cfg, dict):
            raise TypeError(f"augmentation.timbre_mask must be dict, got {type(timbre_cfg)}")

        raw_timbre_enabled = timbre_cfg.get("enabled", True)
        if not isinstance(raw_timbre_enabled, bool):
            raise TypeError("augmentation.timbre_mask.enabled must be boolean")
        self.timbre_enabled = raw_timbre_enabled

        self.timbre_prob = float(timbre_cfg.get("prob", 0.6))
        if not (0.0 <= self.timbre_prob <= 1.0):
            raise ValueError("augmentation.timbre_mask.prob must be in [0, 1]")

        self.n_fft = int(timbre_cfg.get("n_fft", 400))
        self.hop_length = int(timbre_cfg.get("hop_length", 160))
        self.win_length = int(timbre_cfg.get("win_length", 400))
        if self.n_fft <= 0 or self.hop_length <= 0 or self.win_length <= 0:
            raise ValueError("augmentation.timbre_mask n_fft/hop_length/win_length must be > 0")
        if self.win_length > self.n_fft:
            raise ValueError("augmentation.timbre_mask.win_length must be <= n_fft")

        self.freq_mask_param = int(timbre_cfg.get("freq_mask_param", 24))
        self.time_mask_param = int(timbre_cfg.get("time_mask_param", 30))
        if self.freq_mask_param < 0 or self.time_mask_param < 0:
            raise ValueError("augmentation.timbre_mask mask params must be >= 0")

        if self.pitch_enabled or self.spectral_enabled or self.timbre_enabled:
            self._ensure_torchaudio_available()

        self._window = torch.hann_window(self.win_length)

        # STRICT: if timbre is enabled, build masks exactly as configured (no None surprises).
        self._freq_mask = _FrequencyMasking(freq_mask_param=self.freq_mask_param) if self.freq_mask_param > 0 else None
        self._time_mask = _TimeMasking(time_mask_param=self.time_mask_param) if self.time_mask_param > 0 else None

        if self.timbre_enabled and self.freq_mask_param == 0 and self.time_mask_param == 0:
            raise ValueError(
                    "timbre_mask.enabled=true but both freq_mask_param and time_mask_param are 0; "
                    "this would be a no-op."
            )

        # Check SoX availability once at startup if SoX-based effects are enabled.
        if self.pitch_enabled or self.spectral_enabled:
            try:
                x = torch.zeros(1, min(1600, int(self.sampling_rate // 10)), dtype=torch.float32)
                _y, _sr = _apply_effects_tensor(x, self.sampling_rate, [["rate", str(self.sampling_rate)]])
            except Exception as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "SoX effects are enabled but unavailable. On macOS/Homebrew, run with "
                    "DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib"
                ) from exc

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Return an augmented waveform (float32 numpy array)."""

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        output = torch.as_tensor(waveform, dtype=torch.float32).unsqueeze(0)
        if output.numel() == 0:
            return output.squeeze(0).numpy().astype(np.float32, copy=False)

        if self.rng.random() > self.apply_prob:
            return output.squeeze(0).numpy().astype(np.float32, copy=False)

        # Pitch shift (SoX)
        if self.pitch_enabled and self.rng.random() < self.pitch_prob:
            cents = float(self.rng.uniform(self.pitch_cents_min, self.pitch_cents_max))
            output = self._apply_sox_effects(
                output,
                effects=[["pitch", f"{cents:.2f}"], ["rate", str(self.sampling_rate)]],
            )

        # Spectral notches + optional HP/LP (SoX)
        if self.spectral_enabled and self.rng.random() < self.spectral_prob:
            output = self._apply_sox_effects(output, effects=self._build_spectral_effects())

        # Timbre masking in STFT domain
        output = output.squeeze(0)
        if self.timbre_enabled and self.rng.random() < self.timbre_prob:
            output = self._apply_timbre_mask(output)

        # Normalize if clipping
        max_abs = output.abs().max()
        if torch.isfinite(max_abs).item() and max_abs.item() > 1.0:
            output = output / max_abs

        return output.detach().cpu().numpy().astype(np.float32, copy=False)

    def _build_spectral_effects(self) -> list[list[str]]:
        effects: list[list[str]] = []

        n_notches = int(self.rng.integers(self.notches_min, self.notches_max + 1))
        for _ in range(n_notches):
            center = float(self.rng.uniform(self.center_freq_min, self.center_freq_max))
            q_value = float(self.rng.uniform(self.q_min, self.q_max))
            attenuation = float(self.rng.uniform(self.attenuation_db_min, self.attenuation_db_max))
            effects.append(["equalizer", f"{center:.2f}", f"{q_value:.2f}q", f"{attenuation:.2f}"])

        if self.rng.random() < self.highpass_prob:
            hp_hz = float(self.rng.uniform(self.highpass_min_hz, self.highpass_max_hz))
            effects.append(["highpass", f"{hp_hz:.2f}"])

        if self.rng.random() < self.lowpass_prob:
            lp_hz = float(self.rng.uniform(self.lowpass_min_hz, self.lowpass_max_hz))
            effects.append(["lowpass", f"{lp_hz:.2f}"])

        effects.append(["rate", str(self.sampling_rate)])
        return effects

    def _apply_sox_effects(self, waveform: torch.Tensor, effects: list[list[str]]) -> torch.Tensor:
        """STRICT: Apply SoX chain; raise on failure."""
        if not effects:
            return waveform
        if _apply_effects_tensor is None:
            raise RuntimeError("torchaudio SoX effects are unavailable in this environment.")
        augmented, _ = _apply_effects_tensor(waveform, self.sampling_rate, effects)
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
            # STRICT: raise instead of silently returning original
            raise RuntimeError("NaNs produced during ISTFT reconstruction in timbre masking.")
        return reconstructed


def build_speaker_obfuscation_augmenter(
    config: Dict[str, Any],
    sampling_rate: int,
    seed: int,
) -> SpeakerObfuscationAugmenter | None:
    """Return configured speaker-obfuscation augmenter or None if disabled.

    STRICT: if augmentation section exists but has wrong types, raise.
    """

    aug_cfg = get_nested(config, "augmentation")
    if aug_cfg is None:
        return None
    if not isinstance(aug_cfg, dict):
        raise TypeError(f"config.augmentation must be a dict, got {type(aug_cfg)}")

    enabled = aug_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"augmentation.enabled must be boolean, got {type(enabled)}")

    if not enabled:
        return None

    return SpeakerObfuscationAugmenter(
        config=aug_cfg,
        sampling_rate=int(sampling_rate),
        seed=int(seed),
    )

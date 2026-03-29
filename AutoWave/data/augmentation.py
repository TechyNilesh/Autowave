"""Audio augmentation pipeline using audiomentations."""

from __future__ import annotations

import numpy as np

try:
    import audiomentations as AM

    def build_augmentation_pipeline(sample_rate: int = 16000) -> AM.Compose:
        """Build a default audio augmentation pipeline.

        Applies a random subset of: Gaussian noise, time stretch,
        pitch shift, and time shift.

        Args:
            sample_rate: Sample rate of the audio in Hz.

        Returns:
            An audiomentations.Compose transform callable.
        """
        return AM.Compose([
            AM.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            AM.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.4),
            AM.PitchShift(min_semitones=-3, max_semitones=3, p=0.4),
            AM.Shift(min_shift=-0.1, max_shift=0.1, p=0.4),
        ])

    def augment_waveform(
        waveform: np.ndarray,
        sample_rate: int,
        pipeline: AM.Compose | None = None,
    ) -> np.ndarray:
        """Apply augmentation to a waveform array.

        Args:
            waveform: 1-D float32 numpy array.
            sample_rate: Sample rate in Hz.
            pipeline: Augmentation pipeline. Builds default if None.

        Returns:
            Augmented waveform as float32 numpy array.
        """
        if pipeline is None:
            pipeline = build_augmentation_pipeline(sample_rate)
        return pipeline(samples=waveform, sample_rate=sample_rate)

    AUDIOMENTATIONS_AVAILABLE = True

except ImportError:
    AUDIOMENTATIONS_AVAILABLE = False

    def build_augmentation_pipeline(sample_rate: int = 16000):  # type: ignore[misc]
        raise ImportError(
            "audiomentations is required for augmentation. "
            "Install it with: pip install audiomentations"
        )

    def augment_waveform(waveform, sample_rate, pipeline=None):  # type: ignore[misc]
        raise ImportError(
            "audiomentations is required for augmentation. "
            "Install it with: pip install audiomentations"
        )

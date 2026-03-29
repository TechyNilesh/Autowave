"""Tests for audio augmentation."""

import numpy as np
import pytest


def test_augment_waveform_returns_array():
    pytest.importorskip("audiomentations")
    from autowave.data.augmentation import augment_waveform

    sr = 16000
    waveform = np.zeros(sr, dtype=np.float32)
    result = augment_waveform(waveform, sr)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) > 0


def test_build_pipeline_is_callable():
    pytest.importorskip("audiomentations")
    from autowave.data.augmentation import build_augmentation_pipeline

    pipeline = build_augmentation_pipeline(sample_rate=16000)
    assert callable(pipeline)

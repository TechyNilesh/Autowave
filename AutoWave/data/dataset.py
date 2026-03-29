"""PyTorch Dataset for audio classification."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from autowave.data.loader import AudioDataInfo
from autowave.data.augmentation import augment_waveform, AUDIOMENTATIONS_AVAILABLE


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load an audio file and resample to target_sr."""
    import librosa
    waveform, _ = librosa.load(path, sr=target_sr, mono=True)
    return waveform.astype(np.float32)


class AudioDataset(Dataset):
    """PyTorch Dataset that loads audio files and applies a HuggingFace
    feature extractor.

    Args:
        data_info: AudioDataInfo from load_from_folder / load_from_csv.
        feature_extractor: HuggingFace AutoFeatureExtractor instance.
        augment: Whether to apply audio augmentation during loading.
        max_duration_s: Maximum audio duration in seconds. Longer files
            are truncated.
    """

    def __init__(
        self,
        data_info: AudioDataInfo,
        feature_extractor,
        augment: bool = False,
        max_duration_s: float = 10.0,
    ) -> None:
        self.files = data_info.files
        self.label_ids = [data_info.label2id[l] for l in data_info.labels]
        self.feature_extractor = feature_extractor
        self.augment = augment and AUDIOMENTATIONS_AVAILABLE
        self.sample_rate: int = feature_extractor.sampling_rate
        self.max_samples = int(max_duration_s * self.sample_rate)

        if augment and not AUDIOMENTATIONS_AVAILABLE:
            import warnings
            warnings.warn(
                "audiomentations not installed — augmentation disabled. "
                "Install with: pip install audiomentations",
                stacklevel=2,
            )

        self._aug_pipeline = None
        if self.augment:
            from autowave.data.augmentation import build_augmentation_pipeline
            self._aug_pipeline = build_augmentation_pipeline(self.sample_rate)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        waveform = _load_audio(self.files[idx], self.sample_rate)

        # Truncate to max duration
        if len(waveform) > self.max_samples:
            waveform = waveform[: self.max_samples]

        # Augment
        if self.augment and self._aug_pipeline is not None:
            waveform = augment_waveform(waveform, self.sample_rate, self._aug_pipeline)

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
        )

        # Remove batch dimension added by feature extractor
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(self.label_ids[idx], dtype=torch.long)
        return item

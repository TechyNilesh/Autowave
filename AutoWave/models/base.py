"""Abstract base class for AutoWave models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAudioModel(ABC):
    """Abstract base for all AutoWave model wrappers."""

    @abstractmethod
    def from_pretrained(self, model_id: str, num_labels: int, label2id: dict, id2label: dict) -> None:
        """Load pretrained weights and configure the classification head."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the model and feature extractor to disk."""

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load a previously saved model from disk."""

    @property
    @abstractmethod
    def hf_model(self):
        """The underlying HuggingFace model object."""

    @property
    @abstractmethod
    def feature_extractor(self):
        """The HuggingFace feature extractor associated with this model."""

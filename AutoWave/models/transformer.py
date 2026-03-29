"""HuggingFace transformer model wrapper for audio classification."""

from __future__ import annotations

from pathlib import Path

from autowave.models.base import BaseAudioModel
from autowave.models.registry import resolve_model_id


class TransformerModel(BaseAudioModel):
    """Wraps HuggingFace AutoModelForAudioClassification.

    Handles loading a pretrained model, replacing its classification head
    for a new set of labels, and saving/loading the fine-tuned model.

    Args:
        model_name: Short name (e.g. "ast") or full HuggingFace model ID.
    """

    def __init__(self, model_name: str = "default") -> None:
        self._model_id = resolve_model_id(model_name)
        self._model = None
        self._feature_extractor = None

    def from_pretrained(
        self,
        model_id: str | None = None,
        num_labels: int = 2,
        label2id: dict | None = None,
        id2label: dict | None = None,
    ) -> None:
        """Load pretrained weights and configure classification head.

        Args:
            model_id: HuggingFace model ID. Uses model_name from __init__ if None.
            num_labels: Number of output classes.
            label2id: Mapping from label string to integer ID.
            id2label: Mapping from integer ID to label string.
        """
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        resolved_id = resolve_model_id(model_id) if model_id else self._model_id

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(resolved_id)
        self._model = AutoModelForAudioClassification.from_pretrained(
            resolved_id,
            num_labels=num_labels,
            label2id=label2id or {},
            id2label=id2label or {},
            ignore_mismatched_sizes=True,
        )

    def save(self, path: str | Path) -> None:
        """Save model and feature extractor to a directory.

        Args:
            path: Directory path to save to (created if it doesn't exist).
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(save_dir))
        self._feature_extractor.save_pretrained(str(save_dir))

    def load(self, path: str | Path) -> None:
        """Load a saved model and feature extractor from a directory.

        Args:
            path: Directory path previously used with save().

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(str(load_dir))
        self._model = AutoModelForAudioClassification.from_pretrained(str(load_dir))

    @property
    def hf_model(self):
        """The underlying HuggingFace model."""
        return self._model

    @property
    def feature_extractor(self):
        """The HuggingFace feature extractor."""
        return self._feature_extractor

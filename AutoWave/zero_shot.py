"""Zero-shot audio classification using CLAP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class ZeroShotClassifier:
    """Classify audio without any training using CLAP (Contrastive
    Language-Audio Pretraining).

    Matches audio against text label descriptions — no dataset or
    fine-tuning required.

    Args:
        model_id: HuggingFace CLAP model ID.
        device: "auto", "cpu", "cuda", or "mps".

    Example:
        >>> clf = ZeroShotClassifier()
        >>> clf.predict("bark.wav", labels=["dog", "cat", "bird"])
        [
            {"label": "dog", "confidence": 0.91},
            {"label": "cat", "confidence": 0.06},
            {"label": "bird", "confidence": 0.03},
        ]
    """

    DEFAULT_MODEL = "laion/clap-htsat-unfused"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self._processor = None
        self._model = None

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import ClapModel, ClapProcessor
        print(f"[AutoWave] Loading CLAP model: {self.model_id}")
        self._processor = ClapProcessor.from_pretrained(self.model_id)
        self._model = ClapModel.from_pretrained(self.model_id).to(self.device)
        self._model.eval()

    def predict(
        self,
        audio_path: str | Path,
        labels: list[str],
    ) -> list[dict[str, Any]]:
        """Classify an audio file against a set of text labels.

        Args:
            audio_path: Path to the audio file.
            labels: List of class label strings (e.g. ["dog", "cat", "rain"]).

        Returns:
            List of dicts sorted by confidence descending, each with
            "label" (str) and "confidence" (float 0–1).
        """
        import librosa

        if not labels:
            raise ValueError("labels must be a non-empty list of strings.")

        self._load_model()

        waveform, sr = librosa.load(str(audio_path), sr=48000, mono=True)
        waveform = waveform.astype(np.float32)

        inputs = self._processor(
            text=labels,
            audios=[waveform],
            return_tensors="pt",
            padding=True,
            sampling_rate=48000,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        logits = outputs.logits_per_audio[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = [
            {"label": label, "confidence": float(prob)}
            for label, prob in zip(labels, probs)
        ]
        return sorted(results, key=lambda x: x["confidence"], reverse=True)

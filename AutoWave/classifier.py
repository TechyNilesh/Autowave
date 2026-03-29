"""Main AudioClassifier — the primary user-facing API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class AudioClassifier:
    """Classify audio files using pretrained transformer models.

    Provides a simple scikit-learn-style API over HuggingFace Transformers.
    Fine-tunes a pretrained model on your labeled audio dataset.

    Args:
        model_name: Short name ("ast", "wav2vec2", "hubert", "wavlm") or
            any HuggingFace model ID. Defaults to AST pretrained on AudioSet.
        epochs: Number of fine-tuning epochs.
        batch_size: Training batch size per device.
        learning_rate: Initial learning rate.
        augment: Apply audio augmentation during training.
        device: "auto", "cpu", "cuda", or "mps".
        output_dir: Directory for checkpoints and saved model.
        max_duration_s: Max audio duration in seconds (longer files truncated).

    Example:
        >>> model = AudioClassifier()
        >>> model.fit("data/train/")
        >>> model.predict("test.wav")
        {"label": "dog_bark", "confidence": 0.94}
    """

    def __init__(
        self,
        model_name: str = "default",
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        augment: bool = True,
        device: str = "auto",
        output_dir: str = "autowave_output",
        max_duration_s: float = 10.0,
    ) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.augment = augment
        self.device = _resolve_device(device)
        self.output_dir = Path(output_dir)
        self.max_duration_s = max_duration_s

        self._model_wrapper = None
        self._label2id: dict[str, int] = {}
        self._id2label: dict[int, str] = {}
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_folder: str | Path,
        val_folder: str | Path | None = None,
    ) -> "AudioClassifier":
        """Fine-tune the model on a labeled dataset.

        Dataset format: a folder of subfolders, one per class.
            train/
                class_a/  audio1.wav  audio2.wav
                class_b/  audio3.wav

        Args:
            train_folder: Path to labeled training folder.
            val_folder: Optional validation folder (same structure).
                If None, no validation is run during training.

        Returns:
            self (for chaining).
        """
        from autowave.data.loader import load_from_folder
        from autowave.data.dataset import AudioDataset
        from autowave.models.transformer import TransformerModel
        from autowave.training.trainer import AutoWaveTrainer

        print(f"[AutoWave] Loading training data from: {train_folder}")
        train_info = load_from_folder(train_folder)
        print(f"[AutoWave] Found {len(train_info.files)} files, {train_info.num_classes} classes: "
              f"{list(train_info.label2id.keys())}")

        self._label2id = train_info.label2id
        self._id2label = train_info.id2label

        print(f"[AutoWave] Loading model: {self.model_name}  (device: {self.device})")
        self._model_wrapper = TransformerModel(self.model_name)
        self._model_wrapper.from_pretrained(
            num_labels=train_info.num_classes,
            label2id={k: v for k, v in self._label2id.items()},
            id2label={k: v for k, v in self._id2label.items()},
        )
        self._model_wrapper.hf_model.to(self.device)

        train_ds = AudioDataset(
            train_info,
            self._model_wrapper.feature_extractor,
            augment=self.augment,
            max_duration_s=self.max_duration_s,
        )

        val_ds = None
        if val_folder is not None:
            val_info = load_from_folder(val_folder)
            val_ds = AudioDataset(
                val_info,
                self._model_wrapper.feature_extractor,
                augment=False,
                max_duration_s=self.max_duration_s,
            )

        trainer = AutoWaveTrainer(
            output_dir=self.output_dir / "checkpoints",
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )
        print("[AutoWave] Training...")
        trainer.train(self._model_wrapper.hf_model, train_ds, val_ds)

        self._is_fitted = True
        print("[AutoWave] Training complete.")
        return self

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """Classify a single audio file.

        Args:
            audio_path: Path to an audio file.

        Returns:
            Dict with keys "label" (str) and "confidence" (float 0–1).

        Raises:
            RuntimeError: If the model has not been fitted or loaded.
        """
        self._check_fitted()
        results = self.predict_batch([str(audio_path)])
        return results[0]

    def predict_batch(self, audio_paths: list[str | Path]) -> list[dict[str, Any]]:
        """Classify multiple audio files.

        Args:
            audio_paths: List of paths to audio files.

        Returns:
            List of dicts, each with "label" and "confidence".
        """
        self._check_fitted()
        import librosa

        model = self._model_wrapper.hf_model.eval()
        fe = self._model_wrapper.feature_extractor
        sr = fe.sampling_rate
        max_samples = int(self.max_duration_s * sr)

        results = []
        with torch.no_grad():
            for path in audio_paths:
                waveform, _ = librosa.load(str(path), sr=sr, mono=True)
                waveform = waveform[:max_samples].astype(np.float32)

                inputs = fe(
                    waveform,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_samples,
                    truncation=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                pred_id = int(np.argmax(probs))

                results.append({
                    "label": self._id2label[pred_id],
                    "confidence": float(probs[pred_id]),
                })
        return results

    def evaluate(self, test_folder: str | Path) -> dict[str, Any]:
        """Evaluate the model on a labeled test set.

        Args:
            test_folder: Folder of labeled subfolders (same structure as fit()).

        Returns:
            Dict with "accuracy", "f1", and "report" (classification report str).
        """
        self._check_fitted()
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        from autowave.data.loader import load_from_folder

        test_info = load_from_folder(test_folder)
        predictions = self.predict_batch(test_info.files)
        pred_labels = [p["label"] for p in predictions]
        true_labels = test_info.labels

        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        report = classification_report(true_labels, pred_labels, zero_division=0)

        print(f"[AutoWave] Accuracy: {acc:.4f}  |  F1 (macro): {f1:.4f}")
        print(report)
        return {"accuracy": acc, "f1": f1, "report": report}

    def save(self, path: str | Path) -> None:
        """Save the fine-tuned model to disk.

        Args:
            path: Directory to save the model to.
        """
        self._check_fitted()
        save_dir = Path(path)
        self._model_wrapper.save(save_dir / "model")

        meta = {
            "model_name": self.model_name,
            "label2id": self._label2id,
            "id2label": {str(k): v for k, v in self._id2label.items()},
            "max_duration_s": self.max_duration_s,
        }
        with open(save_dir / "autowave_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[AutoWave] Model saved to: {save_dir}")

    @classmethod
    def load(cls, path: str | Path) -> "AudioClassifier":
        """Load a previously saved AudioClassifier.

        Args:
            path: Directory previously used with save().

        Returns:
            Loaded AudioClassifier ready for prediction.

        Raises:
            FileNotFoundError: If the directory or config file is missing.
        """
        from autowave.models.transformer import TransformerModel

        load_dir = Path(path)
        config_path = load_dir / "autowave_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No autowave_config.json found in: {path}")

        with open(config_path) as f:
            meta = json.load(f)

        instance = cls(
            model_name=meta["model_name"],
            max_duration_s=meta.get("max_duration_s", 10.0),
        )
        instance._label2id = meta["label2id"]
        instance._id2label = {int(k): v for k, v in meta["id2label"].items()}

        instance._model_wrapper = TransformerModel(meta["model_name"])
        instance._model_wrapper.load(load_dir / "model")
        instance._model_wrapper.hf_model.to(instance.device)
        instance._is_fitted = True

        print(f"[AutoWave] Model loaded from: {load_dir}")
        return instance

    def export_onnx(self, output_path: str | Path) -> None:
        """Export the model to ONNX format for production deployment.

        Args:
            output_path: Path for the output .onnx file.
        """
        self._check_fitted()
        from autowave.utils.export import export_onnx
        export_onnx(
            self._model_wrapper.hf_model,
            self._model_wrapper.feature_extractor,
            output_path,
            max_duration_s=self.max_duration_s,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Model is not trained. Call fit() first, or load a saved model with AudioClassifier.load()."
            )

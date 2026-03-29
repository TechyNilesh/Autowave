"""Integration tests for AudioClassifier using a tiny synthetic dataset."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _write_wav(path: Path, freq: float = 440.0, duration_s: float = 1.0, sr: int = 16000) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), samples, sr)


@pytest.fixture()
def tiny_dataset(tmp_path):
    """3 classes, 3 files each — sine waves at different frequencies."""
    classes = {"low": 220.0, "mid": 440.0, "high": 880.0}
    train = tmp_path / "train"
    test = tmp_path / "test"
    for split in [train, test]:
        for cls, freq in classes.items():
            cls_dir = split / cls
            cls_dir.mkdir(parents=True)
            for i in range(3):
                _write_wav(cls_dir / f"{cls}_{i}.wav", freq=freq)
    return train, test


@pytest.mark.slow
def test_fit_and_predict(tiny_dataset):
    """End-to-end: fit on tiny dataset, predict a single file."""
    from autowave import AudioClassifier

    train_folder, test_folder = tiny_dataset
    model = AudioClassifier(model_name="ast", epochs=1, batch_size=2, augment=False)
    model.fit(str(train_folder))

    # Predict should return label + confidence
    result = model.predict(str(next((train_folder / "low").glob("*.wav"))))
    assert "label" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["label"] in {"low", "mid", "high"}


@pytest.mark.slow
def test_save_load_predict(tiny_dataset, tmp_path):
    """Save and reload a model, predictions should still work."""
    from autowave import AudioClassifier

    train_folder, _ = tiny_dataset
    model = AudioClassifier(model_name="ast", epochs=1, batch_size=2, augment=False)
    model.fit(str(train_folder))

    save_dir = tmp_path / "saved_model"
    model.save(str(save_dir))

    loaded = AudioClassifier.load(str(save_dir))
    result = loaded.predict(str(next((train_folder / "mid").glob("*.wav"))))
    assert result["label"] in {"low", "mid", "high"}


def test_predict_before_fit_raises():
    from autowave import AudioClassifier
    model = AudioClassifier()
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict("some_file.wav")


def test_load_missing_dir_raises(tmp_path):
    from autowave import AudioClassifier
    with pytest.raises(FileNotFoundError):
        AudioClassifier.load(str(tmp_path / "no_model"))

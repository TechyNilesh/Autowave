"""Tests for data loading utilities."""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autowave.data.loader import load_from_folder, load_from_csv


def _write_wav(path: Path, duration_s: float = 0.5, sr: int = 16000) -> None:
    samples = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(str(path), samples, sr)


@pytest.fixture()
def tmp_audio_folder(tmp_path):
    """Create a folder with 2 classes and 2 files each."""
    for cls in ["cat", "dog"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(2):
            _write_wav(cls_dir / f"{cls}_{i}.wav")
    return tmp_path


@pytest.fixture()
def tmp_csv(tmp_path):
    """Create a CSV with 4 audio files across 2 classes."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    rows = []
    for cls in ["cat", "dog"]:
        for i in range(2):
            p = audio_dir / f"{cls}_{i}.wav"
            _write_wav(p)
            rows.append({"file": str(p), "label": cls})

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


class TestLoadFromFolder:
    def test_basic(self, tmp_audio_folder):
        info = load_from_folder(tmp_audio_folder)
        assert len(info.files) == 4
        assert info.num_classes == 2
        assert "cat" in info.label2id
        assert "dog" in info.label2id

    def test_label_mapping_consistent(self, tmp_audio_folder):
        info = load_from_folder(tmp_audio_folder)
        for label, lid in info.label2id.items():
            assert info.id2label[lid] == label

    def test_missing_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_from_folder(tmp_path / "nonexistent")

    def test_empty_folder(self, tmp_path):
        with pytest.raises(ValueError):
            load_from_folder(tmp_path)


class TestLoadFromCSV:
    def test_basic(self, tmp_csv):
        info = load_from_csv(tmp_csv, file_col="file", label_col="label")
        assert len(info.files) == 4
        assert info.num_classes == 2

    def test_missing_column(self, tmp_csv):
        with pytest.raises(ValueError, match="missing columns"):
            load_from_csv(tmp_csv, file_col="wrong_col", label_col="label")

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_from_csv(tmp_path / "no.csv", file_col="file", label_col="label")

"""Dataset loading utilities — folder-of-folders and CSV formats."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}


@dataclass
class AudioDataInfo:
    files: list[str]
    labels: list[str]
    label2id: dict[str, int]
    id2label: dict[int, str]
    num_classes: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_classes = len(self.label2id)


def load_from_folder(folder_path: str | Path) -> AudioDataInfo:
    """Load a labeled audio dataset from a folder of subfolders.

    Expected structure:
        folder/
            class_a/
                audio1.wav
                audio2.wav
            class_b/
                audio3.wav

    Args:
        folder_path: Path to the root folder containing class subfolders.

    Returns:
        AudioDataInfo with files, labels, and label mappings.

    Raises:
        FileNotFoundError: If folder_path does not exist.
        ValueError: If no audio files are found.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files: list[str] = []
    labels: list[str] = []

    class_names = sorted(
        d.name for d in folder.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if not class_names:
        raise ValueError(f"No class subfolders found in: {folder_path}")

    for class_name in class_names:
        class_dir = folder / class_name
        for audio_file in sorted(class_dir.iterdir()):
            if audio_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(str(audio_file))
                labels.append(class_name)

    if not files:
        raise ValueError(
            f"No audio files found in: {folder_path}. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for idx, name in enumerate(class_names)}

    return AudioDataInfo(
        files=files,
        labels=labels,
        label2id=label2id,
        id2label=id2label,
    )


def load_from_csv(
    csv_path: str | Path,
    file_col: str,
    label_col: str,
) -> AudioDataInfo:
    """Load a labeled audio dataset from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        file_col: Column name containing audio file paths.
        label_col: Column name containing class labels.

    Returns:
        AudioDataInfo with files, labels, and label mappings.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If columns are missing or no valid rows found.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    files: list[str] = []
    raw_labels: list[str] = []

    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file appears to be empty.")
        missing = {file_col, label_col} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}. Found: {reader.fieldnames}")

        for row in reader:
            path = row[file_col].strip()
            label = row[label_col].strip()
            if path and label:
                files.append(path)
                raw_labels.append(label)

    if not files:
        raise ValueError(f"No valid rows found in CSV: {csv_path}")

    class_names = sorted(set(raw_labels))
    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for idx, name in enumerate(class_names)}

    return AudioDataInfo(
        files=files,
        labels=raw_labels,
        label2id=label2id,
        id2label=id2label,
    )

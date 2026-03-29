"""Audio utility functions: resampling, conversion, metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AudioProperties:
    path: str
    channels: int
    sample_rate: int
    duration_s: float
    num_samples: int
    format: str


def read_properties(audio_path: str | Path) -> AudioProperties:
    """Read metadata from an audio file without fully loading it.

    Args:
        audio_path: Path to the audio file.

    Returns:
        AudioProperties dataclass.
    """
    import soundfile as sf

    path = Path(audio_path)
    info = sf.info(str(path))
    return AudioProperties(
        path=str(path),
        channels=info.channels,
        sample_rate=info.samplerate,
        duration_s=info.duration,
        num_samples=info.frames,
        format=info.format,
    )


def read_properties_bulk(audio_paths: list[str | Path]) -> list[AudioProperties]:
    """Read metadata for multiple audio files.

    Args:
        audio_paths: List of paths to audio files.

    Returns:
        List of AudioProperties.
    """
    return [read_properties(p) for p in audio_paths]


def resample(
    audio_path: str | Path,
    target_sr: int,
    output_path: str | Path | None = None,
) -> str:
    """Resample an audio file to a target sample rate.

    Args:
        audio_path: Path to the input audio file.
        target_sr: Target sample rate in Hz.
        output_path: Path for the resampled file. If None, overwrites input.

    Returns:
        Path to the resampled file as a string.
    """
    import librosa
    import soundfile as sf

    waveform, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    out = output_path or audio_path
    sf.write(str(out), waveform, target_sr)
    return str(out)


def convert_format(
    audio_path: str | Path,
    output_format: str,
    output_path: str | Path | None = None,
) -> str:
    """Convert an audio file to a different format.

    Args:
        audio_path: Path to the input audio file.
        output_format: Target format extension (e.g. "wav", "mp3", "flac").
        output_path: Output file path. If None, uses same name with new extension.

    Returns:
        Path to the converted file as a string.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(audio_path))
    src = Path(audio_path)
    out = Path(output_path) if output_path else src.with_suffix(f".{output_format.lower()}")
    audio.export(str(out), format=output_format.lower())
    return str(out)

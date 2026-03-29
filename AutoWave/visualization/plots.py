"""Audio visualization utilities."""

from __future__ import annotations

from pathlib import Path


def _load(audio_path: str | Path, sr: int | None = None):
    import librosa
    return librosa.load(str(audio_path), sr=sr, mono=True)


def waveform(audio_path: str | Path, sr: int | None = None, title: str | None = None) -> None:
    """Plot the waveform (time domain) of an audio file."""
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    y, sample_rate = _load(audio_path, sr)
    plt.figure(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sample_rate)
    plt.title(title or Path(audio_path).name)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def spectrogram(audio_path: str | Path, sr: int | None = None, title: str | None = None) -> None:
    """Plot a mel spectrogram of an audio file."""
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    y, sample_rate = _load(audio_path, sr)
    S = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S_db, sr=sample_rate, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title or f"Mel Spectrogram — {Path(audio_path).name}")
    plt.tight_layout()
    plt.show()


def mfcc(
    audio_path: str | Path,
    sr: int | None = None,
    n_mfcc: int = 40,
    title: str | None = None,
) -> None:
    """Plot MFCC features of an audio file."""
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt

    y, sample_rate = _load(audio_path, sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis="time")
    plt.colorbar()
    plt.title(title or f"MFCC — {Path(audio_path).name}")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()


def spectral_centroid(audio_path: str | Path, sr: int | None = None) -> None:
    """Plot spectral centroid over time."""
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np

    y, sample_rate = _load(audio_path, sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
    frames = np.arange(len(centroid))
    times = librosa.frames_to_time(frames, sr=sample_rate)

    plt.figure(figsize=(12, 3))
    plt.plot(times, centroid, label="Spectral Centroid")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.title(f"Spectral Centroid — {Path(audio_path).name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def spectral_rolloff(audio_path: str | Path, sr: int | None = None) -> None:
    """Plot spectral rolloff over time."""
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np

    y, sample_rate = _load(audio_path, sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)[0]
    frames = np.arange(len(rolloff))
    times = librosa.frames_to_time(frames, sr=sample_rate)

    plt.figure(figsize=(12, 3))
    plt.plot(times, rolloff, label="Spectral Rolloff", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.title(f"Spectral Rolloff — {Path(audio_path).name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def spectral_bandwidth(audio_path: str | Path, sr: int | None = None) -> None:
    """Plot spectral bandwidth over time."""
    import librosa
    import matplotlib.pyplot as plt
    import numpy as np

    y, sample_rate = _load(audio_path, sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)[0]
    frames = np.arange(len(bandwidth))
    times = librosa.frames_to_time(frames, sr=sample_rate)

    plt.figure(figsize=(12, 3))
    plt.plot(times, bandwidth, label="Spectral Bandwidth", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Hz")
    plt.title(f"Spectral Bandwidth — {Path(audio_path).name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def time_freq_overview(audio_path: str | Path, sr: int | None = None) -> None:
    """Plot waveform and mel spectrogram side by side."""
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np

    y, sample_rate = _load(audio_path, sr)
    S = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    librosa.display.waveshow(y, sr=sample_rate, ax=axes[0])
    axes[0].set_title("Waveform")

    img = librosa.display.specshow(S_db, sr=sample_rate, x_axis="time", y_axis="mel", ax=axes[1])
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("Mel Spectrogram")

    fig.suptitle(Path(audio_path).name)
    plt.tight_layout()
    plt.show()

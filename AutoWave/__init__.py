"""
AutoWave v2 — Audio classification powered by pretrained transformers.

Quick start:
    from autowave import AudioClassifier

    model = AudioClassifier()
    model.fit("data/train/")
    model.predict("audio.wav")  # {"label": "dog_bark", "confidence": 0.94}

Zero-shot (no training):
    from autowave import ZeroShotClassifier

    clf = ZeroShotClassifier()
    clf.predict("audio.wav", labels=["dog", "cat", "bird"])
"""

from autowave.classifier import AudioClassifier
from autowave.zero_shot import ZeroShotClassifier

__version__ = "2.0.0"
__all__ = ["AudioClassifier", "ZeroShotClassifier"]

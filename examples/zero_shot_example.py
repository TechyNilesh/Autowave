"""
AutoWave v2 — Zero-Shot Classification (no training needed)

Uses CLAP (Contrastive Language-Audio Pretraining) to match audio
against text labels without any fine-tuning.
"""

from autowave import ZeroShotClassifier

clf = ZeroShotClassifier()

results = clf.predict("bark.wav", labels=["dog barking", "cat meowing", "rain"])
for r in results:
    print(f"{r['label']}: {r['confidence']:.2%}")
# → dog barking: 91.3%
# → rain: 5.2%
# → cat meowing: 3.5%

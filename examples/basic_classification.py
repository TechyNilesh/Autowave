"""
AutoWave v2 — Basic Audio Classification Example

Folder structure expected:
    data/
        train/
            class_a/  audio1.wav  audio2.wav ...
            class_b/  audio3.wav ...
        test/
            class_a/  ...
            class_b/  ...
"""

from autowave import AudioClassifier

# --- 3-line usage ---
model = AudioClassifier()
model.fit("data/train/")
print(model.predict("data/test/class_a/sample.wav"))
# → {"label": "class_a", "confidence": 0.93}

# --- Advanced usage ---
model = AudioClassifier(
    model_name="ast",       # or "wav2vec2", "hubert", any HF model ID
    epochs=10,
    batch_size=8,
    augment=True,
    device="auto",          # picks CUDA > MPS > CPU automatically
)
model.fit("data/train/", val_folder="data/val/")

# Evaluate on a test set
results = model.evaluate("data/test/")
print(f"Accuracy: {results['accuracy']:.2%}")

# Save and reload
model.save("my_audio_model/")
loaded = AudioClassifier.load("my_audio_model/")
print(loaded.predict("new_audio.wav"))

# Export to ONNX for production
model.export_onnx("model.onnx")

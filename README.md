# AutoWave v2

**The simplest way to classify audio in Python.**

Powered by pretrained transformer models (AST, Wav2Vec2, HuBERT) via HuggingFace — fine-tune a state-of-the-art audio classifier on your own dataset in 3 lines of code.

```python
from autowave import AudioClassifier

model = AudioClassifier()
model.fit("data/train/")
model.predict("test.wav")  # {"label": "dog_bark", "confidence": 0.94}
```

![Generic badge](https://img.shields.io/badge/AutoWave-v2-orange.svg) ![Generic badge](https://img.shields.io/badge/Python-3.10+-blue.svg) [![Downloads](https://static.pepy.tech/personalized-badge/autowave?period=total&units=none&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/autowave)

**Creators:** [Nilesh Verma](https://nileshverma.com) · [Satyajit Pattnaik](https://github.com/pik1989) · [Kalash Jindal](https://github.com/erickeagle)

---

## Installation

```bash
pip install AutoWave
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0

---

## Quick Start

### 1. Prepare your dataset

Organize audio files into class subfolders:

```
data/
  train/
    dog/     bark1.wav  bark2.wav  ...
    cat/     meow1.wav  meow2.wav  ...
    bird/    chirp1.wav chirp2.wav ...
  test/
    dog/     ...
    cat/     ...
```

### 2. Train and predict

```python
from autowave import AudioClassifier

model = AudioClassifier()
model.fit("data/train/")
model.predict("data/test/dog/bark_test.wav")
# → {"label": "dog", "confidence": 0.97}
```

### 3. Evaluate

```python
results = model.evaluate("data/test/")
print(f"Accuracy: {results['accuracy']:.2%}")
print(results["report"])
```

### 4. Save and reload

```python
model.save("my_model/")
loaded = AudioClassifier.load("my_model/")
loaded.predict("new_audio.wav")
```

---

## Zero-Shot Classification (no training)

Classify audio against any text labels — no dataset or fine-tuning required:

```python
from autowave import ZeroShotClassifier

clf = ZeroShotClassifier()
clf.predict("audio.wav", labels=["dog barking", "cat meowing", "rain", "music"])
# → [{"label": "dog barking", "confidence": 0.91}, ...]
```

---

## Advanced Options

```python
model = AudioClassifier(
    model_name="ast",          # "ast" | "wav2vec2" | "hubert" | "wavlm" | any HF model ID
    epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    augment=True,              # noise, pitch shift, time stretch, shift
    device="auto",             # "auto" | "cuda" | "mps" | "cpu"
    output_dir="checkpoints/",
    max_duration_s=10.0,
)
model.fit("data/train/", val_folder="data/val/")
```

### Available models

| Short name | HuggingFace model | Best for |
|---|---|---|
| `ast` (default) | MIT/ast-finetuned-audioset-10-10-0.4593 | All audio types |
| `wav2vec2` | facebook/wav2vec2-base | Speech tasks |
| `hubert` | facebook/hubert-base-ls960 | Speech tasks |
| `wavlm` | microsoft/wavlm-base | Speech benchmarks |

Any HuggingFace `AutoModelForAudioClassification`-compatible model ID also works.

---

## Export to ONNX

```python
model.export_onnx("model.onnx")
```

---

## Visualization

```python
from autowave.visualization import plots

plots.waveform("audio.wav")
plots.spectrogram("audio.wav")
plots.mfcc("audio.wav")
plots.spectral_centroid("audio.wav")
plots.time_freq_overview("audio.wav")
```

---

## Audio Utilities

```python
from autowave.utils.audio import read_properties, resample, convert_format

# Metadata
props = read_properties("audio.wav")
print(props.sample_rate, props.duration_s, props.channels)

# Resample to 16 kHz
resample("audio.mp3", target_sr=16000, output_path="audio_16k.wav")

# Convert format
convert_format("audio.wav", output_format="mp3")
```

---

## Supported Audio Formats

`.wav` · `.mp3` · `.flac` · `.ogg` · `.m4a` · `.aiff`

---

## Core Contributors

| Name | Role | Links |
|---|---|---|
| [Nilesh Verma](https://nileshverma.com) | Creator & Lead Developer | [GitHub](https://github.com/TechyNilesh) |
| [Satyajit Pattnaik](https://github.com/pik1989) | Co-Creator & Researcher | [GitHub](https://github.com/pik1989) |
| [Kalash Jindal](https://github.com/erickeagle) | Co-Creator & Developer | [GitHub](https://github.com/erickeagle) |

---

## Citation

If you use AutoWave in your research or project, please cite:

```bibtex
@software{autowave2024,
  author       = {Verma, Nilesh and Pattnaik, Satyajit and Jindal, Kalash},
  title        = {{AutoWave}: Automatic Audio Classification with Pretrained Transformers},
  year         = {2024},
  version      = {2.0.0},
  url          = {https://github.com/TechyNilesh/Autowave},
  note         = {Python library for audio classification using AST, Wav2Vec2, HuBERT, and WavLM}
}
```

"""ONNX export utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def export_onnx(
    model,
    feature_extractor,
    output_path: str | Path,
    max_duration_s: float = 10.0,
    device: str = "cpu",
) -> None:
    """Export a fine-tuned audio classification model to ONNX.

    Args:
        model: HuggingFace model (already fine-tuned).
        feature_extractor: Corresponding HuggingFace feature extractor.
        output_path: Destination .onnx file path.
        max_duration_s: Max audio duration used during training.
        device: Device the model is currently on.

    Raises:
        ImportError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime  # noqa: F401 — validate it's installed
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX export. "
            "Install with: pip install onnxruntime"
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sr = feature_extractor.sampling_rate
    max_samples = int(max_duration_s * sr)

    # Create a dummy waveform input
    dummy_waveform = np.zeros(max_samples, dtype=np.float32)
    inputs = feature_extractor(
        dummy_waveform,
        sampling_rate=sr,
        return_tensors="pt",
        padding="max_length",
        max_length=max_samples,
        truncation=True,
    )
    dummy_input = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    input_names = list(dummy_input.keys())
    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch_size"}

    torch.onnx.export(
        model,
        tuple(dummy_input.values()),
        str(out),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    print(f"[AutoWave] ONNX model exported to: {out}")

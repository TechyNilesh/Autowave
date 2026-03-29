"""Model registry — short name to HuggingFace model ID mapping."""

from __future__ import annotations

# Map user-friendly short names to HuggingFace model IDs
MODEL_REGISTRY: dict[str, str] = {
    # Audio Spectrogram Transformer — best general-purpose default
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    # Wav2Vec2 — great for speech-related tasks
    "wav2vec2": "facebook/wav2vec2-base",
    # HuBERT — strong speech representations
    "hubert": "facebook/hubert-base-ls960",
    # WavLM — best on SUPERB speech benchmarks
    "wavlm": "microsoft/wavlm-base",
    # Default alias
    "default": "MIT/ast-finetuned-audioset-10-10-0.4593",
}


def resolve_model_id(model_name: str) -> str:
    """Resolve a short name or HuggingFace model ID.

    If model_name is a known short name (e.g. "ast", "wav2vec2"), returns
    the corresponding HuggingFace model ID. Otherwise returns model_name
    as-is (assumes it is already a valid HF model ID).

    Args:
        model_name: Short name or HuggingFace model ID.

    Returns:
        HuggingFace model ID string.
    """
    return MODEL_REGISTRY.get(model_name.lower(), model_name)


def list_models() -> dict[str, str]:
    """Return all available short-name model aliases.

    Returns:
        Dict mapping short names to HuggingFace model IDs.
    """
    return {k: v for k, v in MODEL_REGISTRY.items() if k != "default"}

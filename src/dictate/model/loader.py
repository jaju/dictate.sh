"""Model loading: download, configure, quantize, and return ready-to-use models."""

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from dictate.config import make_model_config
from dictate.model.asr import Qwen3ASRModel
from dictate.protocols import FeatureExtractorLike, TokenizerLike


def load_qwen3_asr(
    model_path: str,
) -> tuple[Qwen3ASRModel, TokenizerLike, FeatureExtractorLike]:
    """Load ASR model with aligned weights and preprocessing.

    Downloads the model if not cached locally, loads weights with
    quantization support, and returns the model + tokenizer + feature
    extractor triple.
    """
    import os

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    import transformers

    logging.getLogger("transformers").setLevel(logging.ERROR)

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    # Ensure artifacts are local for offline-ready loading.
    local_path = Path(model_path)
    if not local_path.exists():
        local_path = Path(
            snapshot_download(
                model_path,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.model",
                    "*.txt",
                ],
            )
        )

    # Config drives architecture/quantization; keep as source of truth.
    with open(local_path / "config.json", encoding="utf-8") as f:
        config_dict = json.load(f)

    config = make_model_config(config_dict)

    # Instantiate structure before loading weights.
    model = Qwen3ASRModel(config)

    # Load all shards before sanitizing for layout differences.
    weight_files = list(local_path.glob("*.safetensors"))
    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)))
    weights = Qwen3ASRModel.sanitize(weights)

    # Respect model-provided quantization to match weights.
    quantization = config_dict.get("quantization")
    if quantization:

        def class_predicate(p: str, m: nn.Module) -> bool:
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            if p.startswith("audio_tower"):
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    model.eval()

    # Match preprocessing to the model artifacts.
    prev_verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(local_path), trust_remote_code=True
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            str(local_path)
        )
    finally:
        transformers.logging.set_verbosity(prev_verbosity)

    return model, tokenizer, feature_extractor

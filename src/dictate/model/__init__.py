"""Model subpackage: Qwen3-ASR architecture and loading."""

from dictate.model.asr import Qwen3ASRModel
from dictate.model.loader import load_qwen3_asr

__all__ = ["Qwen3ASRModel", "load_qwen3_asr"]

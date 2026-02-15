"""Core inference package — no UI dependencies.

Re-exports key symbols for convenience.
"""

from voiss.core.config import ModelConfig
from voiss.core.model import Qwen3ASRModel, load_qwen3_asr
from voiss.core.protocols import FeatureExtractorLike, TokenizerLike
from voiss.core.text import PostprocessResult, apply_vocab
from voiss.core.transcribe import build_logit_bias, is_meaningful, transcribe
from voiss.core.types import IntentResult

__all__ = [
    "FeatureExtractorLike",
    "IntentResult",
    "ModelConfig",
    "PostprocessResult",
    "Qwen3ASRModel",
    "TokenizerLike",
    "apply_vocab",
    "build_logit_bias",
    "is_meaningful",
    "load_qwen3_asr",
    "transcribe",
]

"""Structural type protocols for tokenizer and feature extractor interfaces."""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class TokenizerLike(Protocol):
    """Structural type for HuggingFace-compatible tokenizers."""

    def encode(self, text: str, return_tensors: str) -> Any: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


class FeatureExtractorLike(Protocol):
    """Structural type for Whisper-compatible feature extractors."""

    def __call__(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        return_attention_mask: bool,
        truncation: bool,
        padding: bool,
        return_tensors: str,
    ) -> dict[str, Any]: ...

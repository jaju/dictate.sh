"""Text processing utilities for post-ASR cleanup.

Contains vocabulary correction and post-processing result types
that are shared between core and apps layers.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PostprocessResult:
    """Immutable result of a post-processing operation."""

    original: str
    rewritten: str
    model: str
    error: str = ""


def apply_vocab(text: str, vocab: dict[str, str]) -> str:
    """Apply vocabulary corrections to text.

    Each key in *vocab* is matched as a case-insensitive whole word
    and replaced with the corresponding value.
    """
    for wrong, correct in vocab.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(correct, text)
    return text

"""LLM-based rewriting of speech transcripts into structured markdown.

Uses litellm for provider-agnostic LLM access (Ollama, OpenAI, Claude, etc.).
Supports a vocabulary correction dictionary applied before LLM rewriting.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from dictate.constants import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_REWRITE_MAX_TOKENS,
    DEFAULT_REWRITE_SYSTEM_PROMPT,
    DEFAULT_VOCAB_FILE,
)


@dataclass(frozen=True, slots=True)
class RewriteConfig:
    """Configuration for the rewrite pipeline."""

    model: str
    system_prompt: str = DEFAULT_REWRITE_SYSTEM_PROMPT
    max_tokens: int = DEFAULT_REWRITE_MAX_TOKENS
    vocab: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Immutable result of a rewrite operation."""

    original: str
    rewritten: str
    model: str
    error: str = ""


def load_vocab(path: str | None = None) -> dict[str, str]:
    """Load vocabulary corrections from a JSON file.

    The file maps ASR misrecognitions to correct terms, e.g.::

        {"Turing": "Tollring", "kube cuddle": "kubectl"}

    Keys are case-insensitive patterns (matched as whole words).
    If *path* is None, looks in ``~/.config/dictate/vocab.json``.
    Returns an empty dict if the file does not exist.
    """
    if path is None:
        path = str(
            Path(DEFAULT_CONFIG_DIR).expanduser() / DEFAULT_VOCAB_FILE
        )

    p = Path(path).expanduser()
    if not p.exists():
        return {}

    with open(p) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def apply_vocab(text: str, vocab: dict[str, str]) -> str:
    """Apply vocabulary corrections to text.

    Each key in *vocab* is matched as a case-insensitive whole word
    and replaced with the corresponding value.
    """
    for wrong, correct in vocab.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(correct, text)
    return text


def rewrite_transcript(text: str, config: RewriteConfig) -> RewriteResult:
    """Rewrite a transcript turn into structured markdown via litellm.

    Applies vocabulary corrections before sending to the LLM.

    This is a blocking call designed to be run via ``asyncio.to_thread``.
    The litellm import is deferred to avoid import-time overhead when
    the notes pipeline is not in use.

    Exceptions are captured in *result.error* rather than raised so that
    the notes pipeline continues writing even when a single rewrite fails.
    """
    from litellm import completion  # deferred import

    # Apply vocabulary corrections before LLM rewrite.
    if config.vocab:
        text = apply_vocab(text, config.vocab)

    try:
        response = completion(
            model=config.model,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=config.max_tokens,
        )
        content = response.choices[0].message.content or ""
        return RewriteResult(
            original=text,
            rewritten=content.strip(),
            model=config.model,
        )
    except Exception as exc:
        return RewriteResult(
            original=text,
            rewritten="",
            model=config.model,
            error=str(exc),
        )

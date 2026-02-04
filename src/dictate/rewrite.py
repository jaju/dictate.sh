"""LLM-based rewriting of speech transcripts into structured markdown.

Uses litellm for provider-agnostic LLM access (Ollama, OpenAI, Claude, etc.).
"""

from dataclasses import dataclass

from dictate.constants import (
    DEFAULT_REWRITE_MAX_TOKENS,
    DEFAULT_REWRITE_SYSTEM_PROMPT,
)


@dataclass(frozen=True, slots=True)
class RewriteConfig:
    """Configuration for the rewrite pipeline."""

    model: str
    system_prompt: str = DEFAULT_REWRITE_SYSTEM_PROMPT
    max_tokens: int = DEFAULT_REWRITE_MAX_TOKENS


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Immutable result of a rewrite operation."""

    original: str
    rewritten: str
    model: str
    error: str = ""


def rewrite_transcript(text: str, config: RewriteConfig) -> RewriteResult:
    """Rewrite a transcript turn into structured markdown via litellm.

    This is a blocking call designed to be run via ``asyncio.to_thread``.
    The litellm import is deferred to avoid import-time overhead when
    the notes pipeline is not in use.

    Exceptions are captured in *result.error* rather than raised so that
    the notes pipeline continues writing even when a single rewrite fails.
    """
    from litellm import completion  # deferred import

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

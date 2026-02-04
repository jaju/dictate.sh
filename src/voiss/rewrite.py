"""LLM-based rewriting of speech transcripts into structured markdown.

Uses litellm for provider-agnostic LLM access (Ollama, OpenAI, Claude, etc.).
Supports a vocabulary correction dictionary applied before LLM rewriting.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from voiss.constants import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_FILE,
    DEFAULT_CONTEXT_BIAS,
    DEFAULT_PROMPT_FILE,
    DEFAULT_REWRITE_MAX_TOKENS,
    DEFAULT_REWRITE_SYSTEM_PROMPT,
)


@dataclass(frozen=True, slots=True)
class VoissConfig:
    """Top-level configuration loaded from ~/.config/voiss/config.json."""

    context_terms: tuple[str, ...] = ()
    replacements: dict[str, str] = field(default_factory=dict)
    bias_terms: tuple[str, ...] = ()
    bias_scale: float = DEFAULT_CONTEXT_BIAS
    system_prompt: str | None = None
    prompt_file_content: str | None = None


def load_config(path: str | None = None) -> VoissConfig:
    """Load voiss configuration from a JSON file.

    Reads ``~/.config/voiss/config.json`` (or the path given).
    The config file supports three optional subtrees::

        {
          "context": ["Kubernetes", "kubectl", "etcd"],
          "replacements": {"kube cuddle": "kubectl"},
          "bias": {"terms": ["kubectl", "etcd"], "scale": 5.0}
        }

    Returns a default (empty) config if the file does not exist.
    """
    config_dir = Path(DEFAULT_CONFIG_DIR).expanduser()

    if path is not None:
        config_path = Path(path).expanduser()
    else:
        config_path = config_dir / DEFAULT_CONFIG_FILE

    # Check for default prompt file (independent of config.json)
    prompt_file = config_dir / DEFAULT_PROMPT_FILE
    prompt_file_content: str | None = None
    if prompt_file.exists():
        prompt_file_content = prompt_file.read_text().strip() or None

    if config_path.exists():
        with open(config_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return VoissConfig(prompt_file_content=prompt_file_content)

        # context — list of terms for ASR system prompt biasing
        context_terms: tuple[str, ...] = ()
        raw_context = data.get("context", [])
        if isinstance(raw_context, list):
            context_terms = tuple(str(t) for t in raw_context if t)

        replacements = {}
        raw_replacements = data.get("replacements", {})
        if isinstance(raw_replacements, dict):
            replacements = {str(k): str(v) for k, v in raw_replacements.items()}

        bias_terms: tuple[str, ...] = ()
        bias_scale = DEFAULT_CONTEXT_BIAS
        raw_bias = data.get("bias", {})
        if isinstance(raw_bias, dict):
            raw_terms = raw_bias.get("terms", [])
            if isinstance(raw_terms, list):
                bias_terms = tuple(str(t) for t in raw_terms)
            raw_scale = raw_bias.get("scale")
            if isinstance(raw_scale, (int, float)):
                bias_scale = float(raw_scale)

        raw_system_prompt = data.get("system_prompt")
        system_prompt: str | None = (
            str(raw_system_prompt) if isinstance(raw_system_prompt, str) else None
        )

        return VoissConfig(
            context_terms=context_terms,
            replacements=replacements,
            bias_terms=bias_terms,
            bias_scale=bias_scale,
            system_prompt=system_prompt,
            prompt_file_content=prompt_file_content,
        )

    return VoissConfig(prompt_file_content=prompt_file_content)


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

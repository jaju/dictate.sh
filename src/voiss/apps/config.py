"""Application-level configuration and LLM post-processing.

Uses litellm for provider-agnostic LLM access (Ollama, OpenAI, Claude, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voiss.core.constants import (
    DEFAULT_ASR_MODEL,
    DEFAULT_CONFIG_DIR,
    DEFAULT_CONFIG_DIR_ENV,
    DEFAULT_CONFIG_FILE,
    DEFAULT_CONTEXT_BIAS,
    DEFAULT_LLM_MODEL,
    DEFAULT_POSTPROCESS_MAX_TOKENS,
    DEFAULT_POSTPROCESS_PROMPT,
    DEFAULT_PROMPT_FILE,
)
from voiss.core.text import apply_vocab

_log = logging.getLogger("speech")


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AsrConfig:
    """ASR model settings and context biasing."""

    model: str = DEFAULT_ASR_MODEL
    context_terms: tuple[str, ...] = ()
    logit_bias_terms: tuple[str, ...] = ()
    logit_bias_scale: float = DEFAULT_CONTEXT_BIAS


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Intent-analysis LLM settings."""

    model: str = DEFAULT_LLM_MODEL
    prompt: str | None = None


@dataclass(frozen=True, slots=True)
class LitellmPostprocessConfig:
    """LLM post-processing settings for transcript cleanup."""

    model: str | None = None
    prompt: str | None = None
    max_tokens: int = DEFAULT_POSTPROCESS_MAX_TOKENS
    flags: dict[str, Any] = field(default_factory=lambda: {"think": False})


@dataclass(frozen=True, slots=True)
class VoissConfig:
    """Top-level configuration loaded from ~/.config/voiss/config.json."""

    asr: AsrConfig = field(default_factory=AsrConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    corrections: dict[str, str] = field(default_factory=dict)
    litellm_postprocess: LitellmPostprocessConfig = field(
        default_factory=LitellmPostprocessConfig,
    )


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def _resolve_config_path(config_dir: Path, path_str: str) -> Path:
    """Resolve a path relative to *config_dir*. Absolute paths used as-is."""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return config_dir / p


def _resolve_prompt(
    config_dir: Path, section: dict[str, Any], section_name: str,
) -> str | None:
    """Resolve ``prompt`` / ``prompt_file`` from a config section."""
    prompt = section.get("prompt")
    prompt_file = section.get("prompt_file")
    if prompt and prompt_file:
        _log.debug(
            "Both 'prompt' and 'prompt_file' in %s; using 'prompt_file'",
            section_name,
        )
    if prompt_file:
        path = _resolve_config_path(config_dir, str(prompt_file))
        return path.read_text().strip()
    if prompt:
        return str(prompt)
    return None


def _read_default_prompt_file(config_dir: Path) -> str | None:
    """Fallback: read ``prompt.md`` from *config_dir* if it exists.

    Falls back to ``rewrite_prompt.md`` for backward compatibility.
    """
    for name in (DEFAULT_PROMPT_FILE, "rewrite_prompt.md"):
        path = config_dir / name
        if path.exists():
            return path.read_text().strip() or None
    return None


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_config(path: str | None = None) -> VoissConfig:
    """Load voiss configuration from a JSON file.

    Reads ``~/.config/voiss/config.json`` (or *path*).  Supports the
    ``VOISS_CONFIG_DIR`` environment variable to override the config
    directory.  Relative ``prompt_file`` paths are resolved against the
    config directory.

    Returns a default (empty) config if the file does not exist.
    """
    config_dir = Path(
        os.environ.get(DEFAULT_CONFIG_DIR_ENV, "") or DEFAULT_CONFIG_DIR,
    ).expanduser()
    config_path = Path(path).expanduser() if path else config_dir / DEFAULT_CONFIG_FILE

    if not config_path.exists():
        prompt = _read_default_prompt_file(config_dir)
        if prompt:
            return VoissConfig(
                litellm_postprocess=LitellmPostprocessConfig(prompt=prompt),
            )
        return VoissConfig()

    with open(config_path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        prompt = _read_default_prompt_file(config_dir)
        if prompt:
            return VoissConfig(
                litellm_postprocess=LitellmPostprocessConfig(prompt=prompt),
            )
        return VoissConfig()

    # -- audio.asr ---------------------------------------------------------
    audio = data.get("audio", {})
    asr_raw = audio.get("asr", {})
    logit_bias_raw = asr_raw.get("logit_bias", {})
    asr = AsrConfig(
        model=asr_raw.get("model", DEFAULT_ASR_MODEL),
        context_terms=tuple(str(t) for t in asr_raw.get("context", []) if t),
        logit_bias_terms=tuple(str(t) for t in logit_bias_raw.get("terms", [])),
        logit_bias_scale=float(logit_bias_raw.get("scale", DEFAULT_CONTEXT_BIAS)),
    )

    # -- audio.analysis ----------------------------------------------------
    analysis_raw = audio.get("analysis", {})
    analysis = AnalysisConfig(
        model=analysis_raw.get("model", DEFAULT_LLM_MODEL),
        prompt=_resolve_prompt(config_dir, analysis_raw, "audio.analysis"),
    )

    # -- audio.corrections -------------------------------------------------
    corrections = dict(audio.get("corrections", {}))

    # -- litellm_postprocess -----------------------------------------------
    lpp_raw = data.get("litellm_postprocess", {})
    lpp_prompt = _resolve_prompt(config_dir, lpp_raw, "litellm_postprocess")
    if lpp_prompt is None:
        lpp_prompt = _read_default_prompt_file(config_dir)
    lpp = LitellmPostprocessConfig(
        model=lpp_raw.get("model"),
        prompt=lpp_prompt,
        max_tokens=int(lpp_raw.get("max_tokens", DEFAULT_POSTPROCESS_MAX_TOKENS)),
        flags=dict(lpp_raw.get("flags", {"think": False})),
    )

    return VoissConfig(
        asr=asr,
        analysis=analysis,
        corrections=corrections,
        litellm_postprocess=lpp,
    )


def postprocess_transcript(
    text: str,
    config: LitellmPostprocessConfig,
    vocab: dict[str, str] | None = None,
) -> "PostprocessResult":
    """Post-process a transcript turn into structured markdown via litellm.

    Applies vocabulary corrections before sending to the LLM.

    This is a blocking call designed to be run via ``asyncio.to_thread``.
    The litellm import is deferred to avoid import-time overhead when
    the notes pipeline is not in use.

    Exceptions are captured in *result.error* rather than raised so that
    the notes pipeline continues writing even when a single post-process fails.
    """
    from litellm import completion  # deferred import

    from voiss.core.text import PostprocessResult

    model = config.model or ""

    if vocab:
        text = apply_vocab(text, vocab)

    system_prompt = config.prompt or DEFAULT_POSTPROCESS_PROMPT

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=config.max_tokens,
            **config.flags,
        )
        content = response.choices[0].message.content or ""
        return PostprocessResult(
            original=text,
            rewritten=content.strip(),
            model=model,
        )
    except Exception as exc:
        return PostprocessResult(
            original=text,
            rewritten="",
            model=model,
            error=str(exc),
        )

"""Notes pipeline: transcribe speech, rewrite via LLM, save to markdown.

Orchestrates the STT pipeline with per-turn LLM rewriting and file output.
The ASR model is loaded before Textual starts to avoid subprocess/fd conflicts.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from voiss.constants import DEFAULT_NOTES_DIR, DEFAULT_NOTES_DIR_ENV
from voiss.env import LOGGER
from voiss.rewrite import RewriteConfig, RewriteResult


@dataclass(frozen=True, slots=True)
class NotesConfig:
    """Configuration for a notes session."""

    rewrite: RewriteConfig
    output_path: Path


def resolve_notes_path(notes_file: str | None) -> Path:
    """Determine the output file path for this session.

    Priority:
    1. ``--notes-file`` flag (absolute or relative to cwd)
    2. ``VOISS_NOTES_DIR`` env var / default dir, with timestamp filename
    """
    if notes_file:
        return Path(notes_file).resolve()

    notes_dir = Path(
        os.environ.get(DEFAULT_NOTES_DIR_ENV, "")
        or os.path.expanduser(DEFAULT_NOTES_DIR)
    )
    notes_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return notes_dir / f"{timestamp}.md"


def load_system_prompt(
    prompt: str | None,
    file: str | None,
) -> str | None:
    """Load system prompt from string or file, returning None if neither."""
    if prompt:
        return prompt
    if file:
        return Path(file).read_text().strip()
    return None


def write_session_header(path: Path) -> None:
    """Write a session header. Creates the file if new, appends a separator if existing."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    if path.exists() and path.stat().st_size > 0:
        with open(path, "a") as f:
            f.write(f"\n---\n\n# Session — {timestamp}\n\n")
    else:
        with open(path, "w") as f:
            f.write(f"# Notes — {timestamp}\n\n")


def append_raw(path: Path, text: str) -> None:
    """Append raw text to the notes file (no LLM rewrite)."""
    with open(path, "a") as f:
        f.write(f"{text}\n\n")
        f.flush()


def append_turn(path: Path, result: RewriteResult) -> None:
    """Append a rewritten turn to the notes file."""
    with open(path, "a") as f:
        if result.error:
            f.write(f"{result.original}\n\n")
            LOGGER.warning(
                "Rewrite failed: %s — raw transcript saved", result.error
            )
        else:
            f.write(f"{result.rewritten}\n\n")
        f.flush()


def run_notes_pipeline(
    model_path: str,
    language: str,
    context: str | None,
    transcribe_interval: float,
    vad_frame_ms: int,
    vad_mode: int,
    vad_silence_ms: int,
    min_words: int,
    device: int | None,
    notes_config: NotesConfig,
    energy_threshold: float = 300.0,
    bias_terms: tuple[str, ...] = (),
    context_bias: float | None = None,
) -> None:
    """Load ASR model, warm up, then launch the Textual TUI.

    Model loading and warmup happen *before* Textual starts because:
    - huggingface_hub may spawn subprocesses (git) that fail under Textual
    - suppress_output() uses os.dup2 which races with Textual's terminal
    """
    import numpy as np
    from rich.console import Console

    from voiss.env import suppress_output
    from voiss.model import load_qwen3_asr
    from voiss.transcribe import build_logit_bias, transcribe

    console = Console(stderr=True)

    # Phase 1: Load model (may spawn subprocesses — must be pre-Textual)
    console.print("[bold]Loading ASR model...[/bold]")
    model, tokenizer, feature_extractor = load_qwen3_asr(model_path)

    # Build logit bias dict from bias terms (requires tokenizer)
    logit_bias: dict[int, float] | None = None
    if bias_terms and context_bias:
        logit_bias = build_logit_bias(bias_terms, tokenizer, context_bias)

    # Phase 2: Warmup (fd-level suppress safe here — no Textual yet)
    console.print("[dim]Warming up...[/dim]")
    with suppress_output():
        dummy = np.random.randn(16_000).astype(np.float32) * 0.01
        list(
            transcribe(
                model, tokenizer, feature_extractor, dummy,
                language, context=context,
                logit_bias=logit_bias,
            )
        )

    console.print("[green]Ready.[/green]")

    # Phase 3: Launch Textual (owns terminal from here)
    from voiss.notes_app import VoissNotesApp

    app = VoissNotesApp(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        language=language,
        context=context,
        transcribe_interval=transcribe_interval,
        vad_frame_ms=vad_frame_ms,
        vad_mode=vad_mode,
        vad_silence_ms=vad_silence_ms,
        min_words=min_words,
        device=device,
        notes_config=notes_config,
        energy_threshold=energy_threshold,
        logit_bias=logit_bias,
    )
    app.run()

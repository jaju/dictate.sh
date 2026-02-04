"""Notes pipeline: transcribe speech, rewrite via LLM, save to markdown.

Orchestrates the STT pipeline with per-turn LLM rewriting and file output.
Uses RealtimeTranscriber's on_turn_complete callback for integration.
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dictate.constants import DEFAULT_NOTES_DIR, DEFAULT_NOTES_DIR_ENV
from dictate.env import LOGGER
from dictate.rewrite import RewriteConfig, RewriteResult, rewrite_transcript


@dataclass(frozen=True, slots=True)
class NotesConfig:
    """Configuration for a notes session."""

    rewrite: RewriteConfig
    output_path: Path


def resolve_notes_path(notes_file: str | None) -> Path:
    """Determine the output file path for this session.

    Priority:
    1. ``--notes-file`` flag (absolute or relative to cwd)
    2. ``DICTATE_NOTES_DIR`` env var / default dir, with timestamp filename
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
    """Write the markdown session header to a new file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "w") as f:
        f.write(f"# Notes — {timestamp}\n\n")


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


async def run_notes_pipeline(
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
) -> None:
    """Launch the STT pipeline with notes rewriting on turn completion."""
    from dictate.pipeline import RealtimeTranscriber

    write_session_header(notes_config.output_path)
    LOGGER.info("Notes file: %s", notes_config.output_path)

    turn_count = 0

    async def on_turn_complete(transcript: str) -> None:
        nonlocal turn_count
        turn_count += 1

        preview = transcript[:80] + ("..." if len(transcript) > 80 else "")
        LOGGER.info("Turn %d: %s", turn_count, preview)

        result = await asyncio.to_thread(
            rewrite_transcript, transcript, notes_config.rewrite
        )

        append_turn(notes_config.output_path, result)

        if not result.error:
            LOGGER.info(
                "Turn %d rewritten (%d chars)", turn_count, len(result.rewritten)
            )

    transcriber = RealtimeTranscriber(
        model_path=model_path,
        language=language,
        transcribe_interval=transcribe_interval,
        vad_frame_ms=vad_frame_ms,
        vad_mode=vad_mode,
        vad_silence_ms=vad_silence_ms,
        min_words=min_words,
        analyze=False,
        device=device,
        no_ui=True,
        context=context,
        on_turn_complete=on_turn_complete,
    )

    await transcriber.run()

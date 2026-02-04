"""CLI entry point for dictate.

Parses arguments, configures logging, and launches the appropriate pipeline.
setup_environment() is called before importing pipeline to ensure MLX
environment variables are set before any MLX module is loaded.

Subcommands:
    (none)  — live transcription (default, backward-compatible)
    notes   — transcribe + LLM rewrite → markdown notes file
"""

import argparse
import logging
import os

from dictate.constants import (
    DEFAULT_ASR_MODEL,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_WORDS,
    DEFAULT_TRANSCRIBE_INTERVAL,
    DEFAULT_VAD_FRAME_MS,
    DEFAULT_VAD_MODE,
    DEFAULT_VAD_SILENCE_MS,
)


def _add_shared_stt_args(parser: argparse.ArgumentParser) -> None:
    """Add STT arguments shared across subcommands."""
    parser.add_argument(
        "--model", default=DEFAULT_ASR_MODEL, help="ASR model"
    )
    parser.add_argument(
        "--language", default=DEFAULT_LANGUAGE, help="Language"
    )
    parser.add_argument(
        "--transcribe-interval",
        type=float,
        default=DEFAULT_TRANSCRIBE_INTERVAL,
        help=f"How often to update transcription (default: {DEFAULT_TRANSCRIBE_INTERVAL}s)",
    )
    parser.add_argument(
        "--vad-frame-ms",
        type=int,
        default=DEFAULT_VAD_FRAME_MS,
        choices=[10, 20, 30],
        help=f"VAD frame size in ms (10/20/30, default: {DEFAULT_VAD_FRAME_MS})",
    )
    parser.add_argument(
        "--vad-mode",
        type=int,
        default=DEFAULT_VAD_MODE,
        help=f"VAD aggressiveness 0-3 (default: {DEFAULT_VAD_MODE})",
    )
    parser.add_argument(
        "--vad-silence-ms",
        type=int,
        default=DEFAULT_VAD_SILENCE_MS,
        help=f"Silence required to finalize a turn (default: {DEFAULT_VAD_SILENCE_MS}ms)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=f"Minimum words to finalize a turn (default: {DEFAULT_MIN_WORDS})",
    )
    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument(
        "--context",
        default=None,
        help="Domain vocabulary for ASR context biasing (inline string)",
    )
    context_group.add_argument(
        "--context-file",
        default=None,
        help="File containing domain vocabulary for ASR context biasing",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=DEFAULT_ENERGY_THRESHOLD,
        help=f"RMS energy gate threshold for noise rejection (default: {DEFAULT_ENERGY_THRESHOLD})",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommand support."""
    parser = argparse.ArgumentParser(
        description="Real-time speech-to-text and voice-driven notes with Qwen3-ASR"
    )

    # Shared STT args on top-level parser (for bare `dictate` usage)
    _add_shared_stt_args(parser)

    # Transcribe-only args on top-level parser
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Enable LLM intent analysis on turn completion",
    )
    parser.add_argument(
        "--llm-model", default=None, help="LLM model for analysis"
    )
    parser.add_argument(
        "--no-ui", action="store_true", help="Disable the Rich live UI"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="subcommand")

    # `dictate notes`
    notes_parser = subparsers.add_parser(
        "notes",
        help="Notes mode: transcribe, rewrite via LLM, save to markdown",
    )
    _add_shared_stt_args(notes_parser)
    notes_parser.add_argument(
        "--rewrite-model",
        required=True,
        help="LLM model for rewriting (e.g., ollama/llama3.2, openai/gpt-4o-mini)",
    )
    prompt_group = notes_parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt to guide rewriting style",
    )
    prompt_group.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to file containing the system prompt",
    )
    notes_parser.add_argument(
        "--notes-file",
        default=None,
        help="Output file path (default: auto-named in notes directory)",
    )
    notes_parser.add_argument(
        "--vocab-file",
        default=None,
        help="JSON vocabulary corrections file (default: ~/.config/dictate/vocab.json)",
    )

    return parser


def list_audio_devices() -> None:
    """Display available audio input devices."""
    import sounddevice as sd
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Audio Input Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Device", style="white")
    table.add_column("Default", style="green")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            is_default = "Yes" if i == sd.default.device[0] else ""
            table.add_row(str(i), d["name"], is_default)
    console.print(table)


def _resolve_context(args: argparse.Namespace) -> str | None:
    """Read ASR context from --context or --context-file."""
    from pathlib import Path

    if args.context_file:
        return Path(args.context_file).read_text().strip()
    return args.context


def _run_transcribe(args: argparse.Namespace) -> int:
    """Run the original transcription pipeline."""
    import asyncio

    from dictate.pipeline import RealtimeTranscriber

    transcriber = RealtimeTranscriber(
        model_path=args.model,
        language=args.language,
        transcribe_interval=args.transcribe_interval,
        vad_frame_ms=args.vad_frame_ms,
        vad_mode=args.vad_mode,
        vad_silence_ms=args.vad_silence_ms,
        min_words=args.min_words,
        analyze=args.analyze,
        llm_model=args.llm_model,
        device=args.device,
        no_ui=args.no_ui,
        context=_resolve_context(args),
        energy_threshold=args.energy_threshold,
    )

    asyncio.run(transcriber.run())
    return 0


def _run_notes(args: argparse.Namespace) -> int:
    """Run the notes pipeline (Textual TUI)."""
    from dictate.constants import DEFAULT_REWRITE_SYSTEM_PROMPT
    from dictate.notes import (
        NotesConfig,
        load_system_prompt,
        resolve_notes_path,
        run_notes_pipeline,
    )
    from dictate.rewrite import RewriteConfig, load_vocab

    system_prompt = load_system_prompt(
        args.system_prompt,
        args.system_prompt_file,
    )
    vocab = load_vocab(args.vocab_file)

    rewrite_config = RewriteConfig(
        model=args.rewrite_model,
        system_prompt=system_prompt or DEFAULT_REWRITE_SYSTEM_PROMPT,
        vocab=vocab,
    )
    notes_config = NotesConfig(
        rewrite=rewrite_config,
        output_path=resolve_notes_path(args.notes_file),
    )

    run_notes_pipeline(
        model_path=args.model,
        language=args.language,
        context=_resolve_context(args),
        transcribe_interval=args.transcribe_interval,
        vad_frame_ms=args.vad_frame_ms,
        vad_mode=args.vad_mode,
        vad_silence_ms=args.vad_silence_ms,
        min_words=args.min_words,
        device=args.device,
        notes_config=notes_config,
        energy_threshold=args.energy_threshold,
    )
    return 0


def main() -> int:
    """CLI entry point. Returns exit code."""
    # Must run before any MLX imports.
    from dictate.env import setup_environment

    setup_environment()

    import mlx.core as mx
    from rich.console import Console
    from rich.logging import RichHandler

    if hasattr(mx, "set_warnings_enabled"):
        mx.set_warnings_enabled(False)

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                show_time=False,
                show_path=False,
                rich_tracebacks=False,
            )
        ],
    )
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return 0

    if args.subcommand == "notes":
        return _run_notes(args)

    return _run_transcribe(args)

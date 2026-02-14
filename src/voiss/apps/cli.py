"""CLI entry point for voiss.

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

from voiss.core.constants import (
    DEFAULT_ASR_MODEL,
    DEFAULT_ENERGY_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_BUFFER_SECONDS,
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
        "--context-bias",
        type=float,
        default=None,
        help="Additive logit bias scale for context terms during ASR decoding (default: from config or 5.0)",
    )
    parser.add_argument(
        "--max-buffer",
        type=int,
        default=DEFAULT_MAX_BUFFER_SECONDS,
        help=f"Maximum audio buffer in seconds (default: {DEFAULT_MAX_BUFFER_SECONDS})",
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
    parser.add_argument(
        "--config-file",
        default=None,
        help="JSON config file (default: ~/.config/voiss/config.json)",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommand support."""
    parser = argparse.ArgumentParser(
        description="Real-time speech-to-text and voice-driven notes with Qwen3-ASR"
    )

    # Shared STT args on top-level parser (for bare `voiss` usage)
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

    # `voiss notes`
    notes_parser = subparsers.add_parser(
        "notes",
        help="Notes mode: transcribe, rewrite via LLM, save to markdown",
    )
    _add_shared_stt_args(notes_parser)
    notes_parser.add_argument(
        "--rewrite-model",
        default=None,
        help="LLM model for rewriting (e.g., ollama/llama3.2, openai/gpt-4o-mini). "
        "Falls back to litellm_postprocess.model in config.json.",
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


def _merge_context(
    cli_context: str | None,
    config_context_terms: tuple[str, ...],
) -> str | None:
    """Merge config context terms with CLI --context into a single string.

    Config terms come first; CLI terms are appended. The combined string
    is injected into the Qwen3-ASR system prompt for native SFT-trained
    context biasing.
    """
    parts: list[str] = list(config_context_terms)
    if cli_context:
        parts.extend(t.strip() for t in cli_context.split(",") if t.strip())
    return ", ".join(parts) if parts else None


def _run_transcribe(args: argparse.Namespace) -> int:
    """Run the original transcription pipeline."""
    import asyncio

    from voiss.apps.config import load_config
    from voiss.apps.pipeline import RealtimeTranscriber

    voiss_config = load_config(args.config_file)
    cli_context = _resolve_context(args)
    context = _merge_context(cli_context, voiss_config.asr.context_terms)

    # Logit bias: merge config bias terms with CLI context terms
    bias_terms = list(voiss_config.asr.logit_bias_terms)
    if cli_context:
        bias_terms.extend(t.strip() for t in cli_context.split(",") if t.strip())
    bias_scale = args.context_bias if args.context_bias is not None else voiss_config.asr.logit_bias_scale

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
        analysis_prompt=voiss_config.analysis.prompt,
        device=args.device,
        no_ui=args.no_ui,
        context=context,
        energy_threshold=args.energy_threshold,
        bias_terms=tuple(bias_terms),
        context_bias=bias_scale,
        max_buffer_seconds=args.max_buffer,
    )

    asyncio.run(transcriber.run())
    return 0


def _run_notes(args: argparse.Namespace) -> int:
    """Run the notes pipeline (Textual TUI)."""
    import dataclasses

    from voiss.apps.config import load_config
    from voiss.apps.notes import (
        NotesConfig,
        load_system_prompt,
        resolve_notes_path,
        run_notes_pipeline,
    )

    system_prompt = load_system_prompt(
        args.system_prompt,
        args.system_prompt_file,
    )

    voiss_config = load_config(args.config_file)

    lpp = voiss_config.litellm_postprocess
    if system_prompt:
        lpp = dataclasses.replace(lpp, prompt=system_prompt)
    if args.rewrite_model:
        lpp = dataclasses.replace(lpp, model=args.rewrite_model)

    if not lpp.model:
        parser = build_arg_parser()
        parser.error(
            "no rewrite model specified — use --rewrite-model or set "
            "litellm_postprocess.model in config.json"
        )

    notes_config = NotesConfig(
        postprocess=lpp,
        output_path=resolve_notes_path(args.notes_file),
        vocab=voiss_config.corrections,
    )

    cli_context = _resolve_context(args)
    context = _merge_context(cli_context, voiss_config.asr.context_terms)

    # Logit bias: merge config bias terms with CLI context terms
    bias_terms = list(voiss_config.asr.logit_bias_terms)
    if cli_context:
        bias_terms.extend(t.strip() for t in cli_context.split(",") if t.strip())
    bias_scale = args.context_bias if args.context_bias is not None else voiss_config.asr.logit_bias_scale

    run_notes_pipeline(
        model_path=args.model,
        language=args.language,
        context=context,
        transcribe_interval=args.transcribe_interval,
        vad_frame_ms=args.vad_frame_ms,
        vad_mode=args.vad_mode,
        vad_silence_ms=args.vad_silence_ms,
        min_words=args.min_words,
        device=args.device,
        notes_config=notes_config,
        energy_threshold=args.energy_threshold,
        bias_terms=tuple(bias_terms),
        context_bias=bias_scale,
        max_buffer_seconds=args.max_buffer,
    )
    return 0


def main() -> int:
    """CLI entry point. Returns exit code."""
    # Must run before any MLX imports.
    from voiss.core.env import setup_environment

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

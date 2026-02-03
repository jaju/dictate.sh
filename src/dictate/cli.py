"""CLI entry point for dictate.

Parses arguments, configures logging, and launches the transcription pipeline.
setup_environment() is called before importing pipeline to ensure MLX
environment variables are set before any MLX module is loaded.
"""

import argparse
import logging
import os

from dictate.constants import (
    DEFAULT_ASR_MODEL,
    DEFAULT_LANGUAGE,
    DEFAULT_MIN_WORDS,
    DEFAULT_TRANSCRIBE_INTERVAL,
    DEFAULT_VAD_FRAME_MS,
    DEFAULT_VAD_MODE,
    DEFAULT_VAD_SILENCE_MS,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Always-on speech transcription with Qwen3-ASR"
    )
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
    parser.add_argument(
        "--list-devices", action="store_true", help="List audio devices"
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device"
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


def main() -> int:
    """CLI entry point. Returns exit code."""
    # Must run before any MLX imports.
    from dictate.env import setup_environment

    setup_environment()

    import asyncio

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
    )

    asyncio.run(transcriber.run())
    return 0

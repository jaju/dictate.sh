"""Default configuration values for voiss."""

from typing import Final

DEFAULT_ASR_MODEL: Final = "mlx-community/Qwen3-ASR-0.6B-8bit"
DEFAULT_LLM_MODEL: Final = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_LANGUAGE: Final = "English"
DEFAULT_SAMPLE_RATE: Final = 16_000
DEFAULT_TRANSCRIBE_INTERVAL: Final = 0.5
DEFAULT_VAD_FRAME_MS: Final = 30
DEFAULT_VAD_MODE: Final = 3
DEFAULT_VAD_SILENCE_MS: Final = 500
DEFAULT_MIN_WORDS: Final = 3
DEFAULT_MAX_BUFFER_SECONDS: Final = 30
DEFAULT_AUDIO_QUEUE_MAXSIZE: Final = 200
DEFAULT_ENERGY_THRESHOLD: Final = 300.0

# Notes pipeline
DEFAULT_NOTES_DIR: Final = "~/.local/share/voiss/notes"
DEFAULT_NOTES_DIR_ENV: Final = "VOISS_NOTES_DIR"
DEFAULT_REWRITE_MAX_TOKENS: Final = 2048
DEFAULT_REWRITE_SYSTEM_PROMPT: Final = (
    "Clean up the following speech transcript. Fix grammar, punctuation, and "
    "filler words. Keep the original meaning and wording intact — do not add "
    "commentary, interpretation, or new information. If the text contains a "
    "list of items or enumerated points, format them as a markdown list. "
    "Format as markdown. Reply with only the cleaned text, nothing else."
)
DEFAULT_CONFIG_DIR: Final = "~/.config/voiss"
DEFAULT_CONFIG_FILE: Final = "config.json"
DEFAULT_CONTEXT_BIAS: Final = 5.0

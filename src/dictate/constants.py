"""Default configuration values for dictate."""

from typing import Final

DEFAULT_ASR_MODEL: Final = "mlx-community/Qwen3-ASR-0.6B-8bit"
DEFAULT_LLM_MODEL: Final = "mlx-community/Qwen3-0.6B-4bit"
DEFAULT_LANGUAGE: Final = "English"
DEFAULT_SAMPLE_RATE: Final = 16_000
DEFAULT_TRANSCRIBE_INTERVAL: Final = 0.5
DEFAULT_VAD_FRAME_MS: Final = 30
DEFAULT_VAD_MODE: Final = 2
DEFAULT_VAD_SILENCE_MS: Final = 500
DEFAULT_MIN_WORDS: Final = 3
DEFAULT_MAX_BUFFER_SECONDS: Final = 30
DEFAULT_AUDIO_QUEUE_MAXSIZE: Final = 200

# Notes pipeline
DEFAULT_NOTES_DIR: Final = "~/.local/share/dictate/notes"
DEFAULT_NOTES_DIR_ENV: Final = "DICTATE_NOTES_DIR"
DEFAULT_REWRITE_MAX_TOKENS: Final = 2048
DEFAULT_REWRITE_SYSTEM_PROMPT: Final = (
    "You are a note-taking assistant. Rewrite the following spoken transcript "
    "into clean, structured markdown notes. Preserve all information but improve "
    "clarity, fix grammar, and organize with headings or bullet points as "
    "appropriate. Output only the markdown, no preamble."
)

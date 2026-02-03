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

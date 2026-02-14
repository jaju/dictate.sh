"""Public API for voiss ASR inference on Apple Silicon.

All MLX imports are deferred inside functions so that
``import voiss.api`` is safe without triggering GPU initialization.

Typical usage::

    from voiss.api import load_asr_model, transcribe_file, TranscribeOptions

    engine = load_asr_model()
    result = transcribe_file(engine, "meeting.wav")
    print(result.text)
"""

from __future__ import annotations

import struct
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Literal

from voiss.core.constants import DEFAULT_ASR_MODEL, DEFAULT_LANGUAGE


@dataclass(frozen=True, slots=True)
class TranscribeOptions:
    """Options controlling ASR transcription behavior.

    Attributes:
        language: Target language name (must match model's supported list).
        context: Domain vocabulary injected into the ASR system prompt for
            native SFT context biasing.
        max_tokens: Maximum tokens to generate.
        logit_bias: Pre-built logit bias dict (from ``build_logit_bias``).
        condition: ``"adapted"`` passes *context* and *logit_bias* through;
            ``"baseline"`` ignores both for A/B comparison.
    """

    language: str = DEFAULT_LANGUAGE
    context: str | None = None
    max_tokens: int = 8192
    logit_bias: dict[int, float] | None = None
    condition: Literal["baseline", "adapted"] = "adapted"


@dataclass(frozen=True, slots=True)
class TranscribeResult:
    """Immutable result of a transcription call.

    Attributes:
        text: Full transcription text.
        tokens: Individual token strings generated.
        latency_ms: Wall-clock transcription time in milliseconds.
        metadata: Arbitrary extra info (model path, audio duration, etc.).
    """

    text: str
    tokens: list[str]
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class AsrEngine:
    """Bundles a loaded ASR model with its tokenizer and feature extractor.

    Use :func:`load_asr_model` to create instances — do not instantiate
    directly.
    """

    __slots__ = ("model", "tokenizer", "feature_extractor", "model_path")

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        feature_extractor: Any,
        model_path: str,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.model_path = model_path


def load_asr_model(
    model_path: str = DEFAULT_ASR_MODEL,
    *,
    setup_env: bool = True,
) -> AsrEngine:
    """Download (if needed) and load a Qwen3-ASR model.

    Args:
        model_path: HuggingFace model ID or local path.
        setup_env: If True, call ``setup_environment()`` to suppress
            noisy library warnings before loading.

    Returns:
        An :class:`AsrEngine` ready for transcription.
    """
    if setup_env:
        from voiss.core.env import setup_environment
        setup_environment()

    from voiss.core.model.loader import load_qwen3_asr

    model, tokenizer, feature_extractor = load_qwen3_asr(model_path)
    return AsrEngine(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        model_path=model_path,
    )


def transcribe_file(
    engine: AsrEngine,
    audio_path: str,
    options: TranscribeOptions | None = None,
) -> TranscribeResult:
    """Transcribe a WAV file.

    Reads the file via stdlib :mod:`wave`, converts to float32, resamples
    to 16 kHz if needed (basic ``np.interp``), and runs ASR inference.

    Args:
        engine: A loaded :class:`AsrEngine`.
        audio_path: Path to a WAV file.
        options: Transcription options (defaults to :class:`TranscribeOptions`).

    Returns:
        A :class:`TranscribeResult` with transcription text, tokens, and timing.
    """
    import numpy as np

    opts = options or TranscribeOptions()

    with wave.open(audio_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # Decode raw bytes to numpy array
    if sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
    elif sampwidth == 4:
        fmt = f"<{n_frames * n_channels}i"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
    else:
        # 8-bit unsigned
        samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    # Mono mixdown
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16 kHz if needed
    target_rate = 16_000
    if framerate != target_rate:
        duration = len(samples) / framerate
        n_target = int(duration * target_rate)
        x_old = np.linspace(0, duration, len(samples), endpoint=False)
        x_new = np.linspace(0, duration, n_target, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)

    return transcribe_array(engine, samples, sample_rate=target_rate, options=opts)


def transcribe_array(
    engine: AsrEngine,
    audio: "np.ndarray",
    sample_rate: int = 16_000,
    options: TranscribeOptions | None = None,
) -> TranscribeResult:
    """Transcribe a numpy float32 audio array.

    Args:
        engine: A loaded :class:`AsrEngine`.
        audio: Float32 numpy array of audio samples (mono, any sample rate).
        sample_rate: Sample rate of *audio*. Resampled to 16 kHz if different.
        options: Transcription options (defaults to :class:`TranscribeOptions`).

    Returns:
        A :class:`TranscribeResult` with transcription text, tokens, and timing.
    """
    import numpy as np

    from voiss.core.transcribe import transcribe

    opts = options or TranscribeOptions()

    # Resample if needed
    target_rate = 16_000
    if sample_rate != target_rate:
        duration = len(audio) / sample_rate
        n_target = int(duration * target_rate)
        x_old = np.linspace(0, duration, len(audio), endpoint=False)
        x_new = np.linspace(0, duration, n_target, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)

    # Determine context/bias based on condition
    context = opts.context if opts.condition == "adapted" else None
    logit_bias = opts.logit_bias if opts.condition == "adapted" else None

    start = time.perf_counter()
    tokens: list[str] = []
    for token in transcribe(
        engine.model,
        engine.tokenizer,
        engine.feature_extractor,
        audio,
        language=opts.language,
        max_tokens=opts.max_tokens,
        context=context,
        logit_bias=logit_bias,
    ):
        tokens.append(token)
    elapsed_ms = (time.perf_counter() - start) * 1000

    text = "".join(tokens).strip()
    return TranscribeResult(
        text=text,
        tokens=tokens,
        latency_ms=elapsed_ms,
        metadata={
            "model_path": engine.model_path,
            "condition": opts.condition,
            "language": opts.language,
        },
    )


def build_logit_bias(
    terms: list[str] | tuple[str, ...],
    tokenizer: Any,
    scale: float = 5.0,
) -> dict[int, float]:
    """Build a logit bias dict from domain vocabulary terms.

    Convenience re-export of :func:`voiss.core.transcribe.build_logit_bias`.

    Args:
        terms: Domain vocabulary terms to bias toward.
        tokenizer: The engine's tokenizer (``engine.tokenizer``).
        scale: Additive logit bias strength.

    Returns:
        Dict mapping token IDs to bias values.
    """
    from voiss.core.transcribe import build_logit_bias as _build

    return _build(terms, tokenizer, scale)

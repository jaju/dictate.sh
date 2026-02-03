"""Voice Activity Detection using WebRTC VAD.

Tracks speech/silence boundaries to detect when a speaker's turn
has completed, based on consecutive silence frames exceeding a threshold.
"""

import math
from dataclasses import dataclass

import numpy as np
import webrtcvad


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Immutable VAD configuration."""

    frame_ms: int = 30
    mode: int = 2
    silence_ms: int = 500
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if self.frame_ms not in (10, 20, 30):
            raise ValueError("frame_ms must be one of: 10, 20, 30")
        if not (0 <= self.mode <= 3):
            raise ValueError("mode must be between 0 and 3")


class VoiceActivityDetector:
    """WebRTC VAD state machine for turn detection."""

    __slots__ = (
        "_vad",
        "_config",
        "_frame_samples",
        "_silence_threshold",
        "_residual",
        "_speech_detected",
        "_silence_count",
        "_state",
    )

    def __init__(self, config: VadConfig) -> None:
        self._config = config
        self._vad = webrtcvad.Vad(config.mode)
        self._frame_samples = int(config.sample_rate * config.frame_ms / 1000)
        self._silence_threshold = int(
            math.ceil(config.silence_ms / config.frame_ms)
        )
        self._residual = np.array([], dtype=np.int16)
        self._speech_detected = False
        self._silence_count = 0
        self._state = "silence"

    @property
    def state(self) -> str:
        """Current VAD state: 'speech' or 'silence'."""
        return self._state

    @property
    def speech_detected(self) -> bool:
        """Whether speech has been detected in the current turn."""
        return self._speech_detected

    @property
    def frame_samples(self) -> int:
        """Number of samples per VAD frame."""
        return self._frame_samples

    def process(self, frame: np.ndarray) -> bool:
        """Process audio frame, return True if turn is complete.

        A turn completes when speech was detected and then enough
        consecutive silence frames have accumulated.
        """
        if frame.size == 0:
            return False

        self._residual = (
            frame.copy()
            if self._residual.size == 0
            else np.concatenate([self._residual, frame])
        )

        turn_complete = False
        while self._residual.size >= self._frame_samples:
            chunk = self._residual[: self._frame_samples]
            self._residual = self._residual[self._frame_samples :]
            is_speech = self._vad.is_speech(
                chunk.tobytes(), self._config.sample_rate
            )

            if is_speech:
                self._state = "speech"
                self._speech_detected = True
                self._silence_count = 0
            elif self._speech_detected:
                self._state = "silence"
                self._silence_count += 1
                if self._silence_count >= self._silence_threshold:
                    turn_complete = True
                    self._speech_detected = False
                    self._silence_count = 0
                    self._residual = np.array([], dtype=np.int16)
                    break
            else:
                self._state = "silence"

        return turn_complete

    def reset(self) -> None:
        """Clear state for a new turn."""
        self._residual = np.array([], dtype=np.int16)
        self._speech_detected = False
        self._silence_count = 0
        self._state = "silence"

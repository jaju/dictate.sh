"""Fixed-size circular buffer for int16 audio samples.

Avoids reallocation by writing into a pre-allocated numpy array.
The buffer wraps around when full, always keeping the most recent audio.
"""

from typing import Self

import numpy as np


class RingBuffer:
    """Circular audio buffer with O(1) append and bounded memory."""

    __slots__ = (
        "_buffer",
        "_write_pos",
        "_filled",
        "_total_written",
        "_sample_rate",
    )

    def __init__(self, buffer: np.ndarray, sample_rate: int) -> None:
        self._buffer = buffer
        self._write_pos = 0
        self._filled = 0
        self._total_written = 0
        self._sample_rate = sample_rate

    @classmethod
    def create(cls, max_seconds: int, sample_rate: int) -> Self:
        """Create a ring buffer sized for the given duration."""
        size = max_seconds * sample_rate
        return cls(np.zeros(size, dtype=np.int16), sample_rate)

    @property
    def filled_seconds(self) -> float:
        return self._filled / self._sample_rate

    @property
    def total_samples_written(self) -> int:
        return self._total_written

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def max_samples(self) -> int:
        return len(self._buffer)

    def append(self, frame: np.ndarray) -> None:
        """Write audio frame into the ring buffer."""
        if frame.size == 0:
            return
        n = frame.size
        capacity = len(self._buffer)
        end = self._write_pos + n
        if end <= capacity:
            self._buffer[self._write_pos : end] = frame
        else:
            first = capacity - self._write_pos
            self._buffer[self._write_pos :] = frame[:first]
            self._buffer[: end % capacity] = frame[first:]
        self._write_pos = end % capacity
        self._filled = min(capacity, self._filled + n)
        self._total_written += n

    def get_recent(self, seconds: float) -> np.ndarray:
        """Return the most recent audio as a contiguous copy."""
        if self._filled == 0:
            return np.array([], dtype=np.int16)
        num = min(int(seconds * self._sample_rate), self._filled)
        if num <= 0:
            return np.array([], dtype=np.int16)
        capacity = len(self._buffer)
        start = (self._write_pos - num) % capacity
        end = start + num
        if end <= capacity:
            return self._buffer[start:end].copy()
        return np.concatenate(
            [self._buffer[start:], self._buffer[: end % capacity]]
        )

    def reset(self) -> None:
        """Clear all state for a new turn."""
        self._write_pos = 0
        self._filled = 0

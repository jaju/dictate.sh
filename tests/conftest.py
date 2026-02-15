"""Shared test fixtures — no real MLX needed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


@dataclass
class FakeTokenIds:
    """Mimics the object returned by tokenizer.encode(return_tensors='np')."""

    ids: np.ndarray

    def flatten(self) -> np.ndarray:
        return self.ids.flatten()

    def tolist(self) -> list[int]:
        return self.ids.flatten().tolist()


class FakeTokenizer:
    """Minimal tokenizer stub for testing."""

    def __init__(self, vocab: dict[str, list[int]] | None = None) -> None:
        self._vocab = vocab or {}

    def encode(self, text: str, return_tensors: str = "np") -> Any:
        ids = self._vocab.get(text, [100, 101, 102])
        return FakeTokenIds(np.array(ids, dtype=np.int64))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(t) for t in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        return messages[0]["content"]


class FakeFeatureExtractor:
    """Minimal feature extractor stub."""

    def __call__(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16_000,
        return_attention_mask: bool = True,
        truncation: bool = False,
        padding: bool = True,
        return_tensors: str = "np",
    ) -> dict[str, np.ndarray]:
        seq_len = max(1, len(audio) // 160)
        return {
            "input_features": np.zeros((1, 128, seq_len), dtype=np.float32),
            "attention_mask": np.ones((1, seq_len), dtype=np.float32),
        }


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fake_feature_extractor() -> FakeFeatureExtractor:
    return FakeFeatureExtractor()


@pytest.fixture
def fake_model() -> MagicMock:
    model = MagicMock()
    model.config.support_languages = ("English", "Chinese")
    model.config.audio_token_id = 151676
    return model


@pytest.fixture
def fake_engine(
    fake_model: MagicMock,
    fake_tokenizer: FakeTokenizer,
    fake_feature_extractor: FakeFeatureExtractor,
) -> Any:
    from voiss.api import AsrEngine

    return AsrEngine(
        model=fake_model,
        tokenizer=fake_tokenizer,
        feature_extractor=fake_feature_extractor,
        model_path="test-model",
    )

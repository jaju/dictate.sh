"""Tests for voiss.api — public API stability and behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voiss.api import (
    AsrEngine,
    TranscribeOptions,
    TranscribeResult,
    transcribe_array,
)


class TestTranscribeOptions:
    def test_defaults(self) -> None:
        opts = TranscribeOptions()
        assert opts.language == "English"
        assert opts.context is None
        assert opts.max_tokens == 8192
        assert opts.logit_bias is None
        assert opts.condition == "adapted"

    def test_custom_values(self) -> None:
        bias = {100: 5.0}
        opts = TranscribeOptions(
            language="Chinese",
            context="Kubernetes",
            max_tokens=4096,
            logit_bias=bias,
            condition="baseline",
        )
        assert opts.language == "Chinese"
        assert opts.context == "Kubernetes"
        assert opts.condition == "baseline"

    def test_frozen(self) -> None:
        opts = TranscribeOptions()
        with pytest.raises(AttributeError):
            opts.language = "French"


class TestTranscribeResult:
    def test_fields(self) -> None:
        result = TranscribeResult(
            text="hello world",
            tokens=["hello", " world"],
            latency_ms=42.5,
        )
        assert result.text == "hello world"
        assert result.tokens == ["hello", " world"]
        assert result.latency_ms == 42.5
        assert result.metadata == {}

    def test_frozen(self) -> None:
        result = TranscribeResult(text="x", tokens=[], latency_ms=0.0)
        with pytest.raises(AttributeError):
            result.text = "y"


class TestAsrEngine:
    def test_slots(self, fake_engine: AsrEngine) -> None:
        assert fake_engine.model_path == "test-model"
        assert fake_engine.model is not None
        assert fake_engine.tokenizer is not None
        assert fake_engine.feature_extractor is not None


class TestConditionRouting:
    """Verify that condition='baseline' vs 'adapted' routes correctly."""

    @patch("voiss.core.transcribe.transcribe")
    def test_baseline_nullifies_context_and_bias(
        self, mock_transcribe: MagicMock, fake_engine: AsrEngine
    ) -> None:
        mock_transcribe.return_value = iter(["hello"])
        audio = np.zeros(16_000, dtype=np.float32)
        opts = TranscribeOptions(
            context="Kubernetes",
            logit_bias={100: 5.0},
            condition="baseline",
        )
        result = transcribe_array(fake_engine, audio, options=opts)
        assert result.text == "hello"
        call_kwargs = mock_transcribe.call_args
        assert call_kwargs.kwargs["context"] is None
        assert call_kwargs.kwargs["logit_bias"] is None

    @patch("voiss.core.transcribe.transcribe")
    def test_adapted_passes_context_and_bias(
        self, mock_transcribe: MagicMock, fake_engine: AsrEngine
    ) -> None:
        mock_transcribe.return_value = iter(["hello"])
        audio = np.zeros(16_000, dtype=np.float32)
        bias = {100: 5.0}
        opts = TranscribeOptions(
            context="Kubernetes",
            logit_bias=bias,
            condition="adapted",
        )
        result = transcribe_array(fake_engine, audio, options=opts)
        assert result.text == "hello"
        call_kwargs = mock_transcribe.call_args
        assert call_kwargs.kwargs["context"] == "Kubernetes"
        assert call_kwargs.kwargs["logit_bias"] == bias

"""Tests for voiss.core.text — vocabulary correction and PostprocessResult."""

from __future__ import annotations

import pytest

from voiss.core.text import PostprocessResult, apply_vocab


class TestApplyVocab:
    def test_case_insensitive_replacement(self) -> None:
        result = apply_vocab("use KUBECTL to deploy", {"kubectl": "kubectl"})
        assert "kubectl" in result

    def test_multiple_replacements(self) -> None:
        vocab = {"kube cuddle": "kubectl", "docker": "Docker"}
        result = apply_vocab("use kube cuddle and docker", vocab)
        assert "kubectl" in result
        assert "Docker" in result

    def test_empty_vocab_no_change(self) -> None:
        text = "hello world"
        assert apply_vocab(text, {}) == text

    def test_no_match_no_change(self) -> None:
        text = "hello world"
        vocab = {"kubernetes": "Kubernetes"}
        assert apply_vocab(text, vocab) == text


class TestPostprocessResult:
    def test_fields(self) -> None:
        result = PostprocessResult(
            original="raw text",
            rewritten="clean text",
            model="test-model",
        )
        assert result.original == "raw text"
        assert result.rewritten == "clean text"
        assert result.model == "test-model"
        assert result.error == ""

    def test_error_field(self) -> None:
        result = PostprocessResult(
            original="text",
            rewritten="",
            model="m",
            error="connection failed",
        )
        assert result.error == "connection failed"

    def test_frozen(self) -> None:
        result = PostprocessResult(original="x", rewritten="y", model="m")
        with pytest.raises(AttributeError):
            result.original = "z"

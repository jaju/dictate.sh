"""Tests for logit bias building from voiss.core.transcribe."""

from __future__ import annotations

import numpy as np

from voiss.core.transcribe import _SPECIAL_TOKEN_IDS, build_logit_bias

from .conftest import FakeTokenizer


class TestBuildLogitBias:
    def test_empty_terms_returns_empty(self) -> None:
        tok = FakeTokenizer()
        result = build_logit_bias([], tok, scale=5.0)
        assert result == {}

    def test_basic_mapping(self) -> None:
        tok = FakeTokenizer(vocab={"kubectl": [200, 201]})
        result = build_logit_bias(["kubectl"], tok, scale=3.0)
        assert result == {200: 3.0, 201: 3.0}

    def test_special_tokens_excluded(self) -> None:
        # Include a special token ID in the fake vocab
        special_id = next(iter(_SPECIAL_TOKEN_IDS))
        tok = FakeTokenizer(vocab={"test": [special_id, 999]})
        result = build_logit_bias(["test"], tok, scale=5.0)
        assert special_id not in result
        assert 999 in result

    def test_scale_applied(self) -> None:
        tok = FakeTokenizer(vocab={"word": [300]})
        result = build_logit_bias(["word"], tok, scale=7.5)
        assert result[300] == 7.5

    def test_duplicate_terms_deduplicated(self) -> None:
        tok = FakeTokenizer(vocab={"kube": [400, 401]})
        result = build_logit_bias(["kube", "kube"], tok, scale=5.0)
        assert result == {400: 5.0, 401: 5.0}

"""Import compatibility tests — verify all canonical paths resolve."""

from __future__ import annotations

import importlib

import pytest


# All public import paths that must work
_IMPORT_PATHS = [
    # Top-level
    "voiss",
    "voiss.api",
    # Core
    "voiss.core",
    "voiss.core.constants",
    "voiss.core.env",
    "voiss.core.protocols",
    "voiss.core.config",
    "voiss.core.text",
    "voiss.core.types",
    # Audio (stays at voiss level)
    "voiss.audio",
    "voiss.audio.ring_buffer",
]

# Paths that need MLX — test separately with skip
_MLX_PATHS = [
    "voiss.core.model",
    "voiss.core.model._utils",
    "voiss.core.model.encoder",
    "voiss.core.model.decoder",
    "voiss.core.model.asr",
    "voiss.core.model.loader",
    "voiss.core.transcribe",
]


@pytest.mark.parametrize("path", _IMPORT_PATHS)
def test_import_resolves(path: str) -> None:
    """Each core import path should resolve without error."""
    mod = importlib.import_module(path)
    assert mod is not None


@pytest.mark.parametrize("path", _MLX_PATHS)
def test_mlx_import_resolves(path: str) -> None:
    """MLX-dependent paths should resolve (skipped if MLX unavailable)."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.skip("MLX not available")
    mod = importlib.import_module(path)
    assert mod is not None


def test_lazy_api_reexports() -> None:
    """voiss.X should resolve for all API names via __getattr__."""
    import voiss

    for name in (
        "AsrEngine",
        "TranscribeOptions",
        "TranscribeResult",
        "load_asr_model",
        "transcribe_file",
        "transcribe_array",
        "build_logit_bias",
    ):
        assert hasattr(voiss, name), f"voiss.{name} not accessible"


def test_invalid_attr_raises() -> None:
    """Accessing a non-existent attribute should raise AttributeError."""
    import voiss

    with pytest.raises(AttributeError):
        _ = voiss.nonexistent_thing

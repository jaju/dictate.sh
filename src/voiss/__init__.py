__version__ = "0.3.0"


def __getattr__(name: str):
    """Lazy re-exports from voiss.api for convenience."""
    _api_names = {
        "AsrEngine",
        "TranscribeOptions",
        "TranscribeResult",
        "load_asr_model",
        "transcribe_file",
        "transcribe_array",
        "build_logit_bias",
    }
    if name in _api_names:
        from voiss import api

        return getattr(api, name)
    raise AttributeError(f"module 'voiss' has no attribute {name!r}")

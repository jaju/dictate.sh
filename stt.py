# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "dictate-stt",
# ]
#
# [tool.uv.sources]
# dictate-stt = { path = "." }
# ///
"""Standalone, low-latency transcription for Apple Silicon."""

from dictate.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

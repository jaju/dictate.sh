# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "voissistant",
# ]
#
# [tool.uv.sources]
# voissistant = { path = "." }
# ///
"""Standalone, low-latency transcription for Apple Silicon."""

from voiss.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

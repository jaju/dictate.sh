# CLAUDE.md — Agent Memory

## Project

**dictate.sh** — Real-time speech-to-text for Apple Silicon using MLX.
Single-file Python script (`stt.py`, ~1688 lines) being refactored into a `src/dictate/` package.

## Quick Start

- Python 3.12+, `uv` for all commands (`uv sync`, `uv run`, `uv pip`)
- Apple Silicon Mac required (MLX framework)
- **Virtual env**: Use `.venv/bin/python` for running Python directly (NOT `uv run python`)
- Run: `uv run stt.py` (current) → `uv run dictate` (after refactor)
- Sanity checks: `.venv/bin/python -c "from dictate.xxx import ..."` to validate modules

## Work Protocol

1. **Start of session**: Run `bd prime` for workflow context. Run `bd ready` to find unblocked work.
2. **Before starting a task**: `bd show <id>` for full context. Update task descriptions of downstream tasks if current work clarifies details.
3. **Working on a task**: Set state with `bd update <id> --assignee claude`. Keep changes small and incremental.
4. **Completing a task**: `bd close <id>`. Then `bd ready` for next task.
5. **End of session**: `bd sync` to persist state.

## Issue Tracking

This project uses **bd (beads)** for issue tracking.
Run `bd prime` for workflow context.

**Quick reference:**
- `bd ready` — find unblocked work
- `bd create "Title" --type task --priority 2` — create issue
- `bd show <id>` — full task details
- `bd close <id>` — complete work
- `bd dep <blocker> --blocks <blocked>` — set dependency
- `bd sync` — sync with git (run at session end)

## Architecture Reference

See `ARCHITECTURE.md` for full technical details.

## Key Decisions

- **Python 3.12**: Use `X | None`, `list[str]`, `type` aliases, `@override`, `slots=True`, `match/case`
- **No deprecated patterns**: No `typing.List/Dict/Tuple/Optional/Sequence`. Use builtins + `collections.abc`.
- **Frozen dataclasses**: All config types are `@dataclass(frozen=True, slots=True)`
- **Functional over OO**: Pure functions where state is not needed. Module-level functions over methods.
- **Immutable over mutable**: `tuple` over `list` for fixed collections. Factory functions over `__post_init__` mutation.
- **Dependencies**: mlx>=0.30.0, mlx-lm>=0.30.0, numpy>=2.0, rich>=14.0, webrtcvad-wheels>=2.0.14
- **Root `stt.py`**: Thin shim that imports from `dictate.cli`. No longer the full script.

## File Layout (Target)

```
src/dictate/
  __init__.py          — version only, no eager MLX imports
  constants.py         — Final-annotated defaults
  env.py               — setup_environment(), suppress_output(), LOGGER
  protocols.py         — TokenizerLike, FeatureExtractorLike
  config.py            — frozen dataclasses + make_model_config() factory
  model/
    __init__.py         — re-exports
    _utils.py           — tensor math helpers
    encoder.py          — audio encoder nn.Modules
    decoder.py          — text decoder nn.Modules
    asr.py              — Qwen3ASRModel composite
    loader.py           — load_qwen3_asr()
  transcribe.py        — transcribe() generator
  audio/
    __init__.py         — re-exports
    ring_buffer.py      — RingBuffer (mutable, perf-critical)
    vad.py              — VadConfig + VoiceActivityDetector
  analysis.py          — IntentResult + analyze_intent()
  ui.py                — UiState + pure render functions
  pipeline.py          — RealtimeTranscriber (orchestration)
  cli.py               — argparse + main()
```

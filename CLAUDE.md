# CLAUDE.md — Agent Memory

## Project

**voissistant** — Real-time speech-to-text and voice-driven notes for Apple Silicon using MLX.
Packaged as `src/voiss/` with subcommands: bare `voiss` for live transcription, `voiss notes` for LLM-rewritten markdown notes.

## Quick Start

- Python 3.12+, `uv` for all commands (`uv sync`, `uv run`, `uv pip`)
- Apple Silicon Mac required (MLX framework)
- **Virtual env**: Use `.venv/bin/python` for running Python directly (NOT `uv run python`)
- Run: `uv run voiss` (transcription), `uv run voiss notes --rewrite-model ollama/llama3.2` (notes)
- Sanity checks: `.venv/bin/python -c "from voiss.xxx import ..."` to validate modules

## Work Protocol

1. **Start of session**: Run `bd prime` for workflow context. Run `bd ready` to find unblocked work.
2. **Before starting a task**: `bd show <id>` for full context.
3. **Working on a task**: Set state with `bd update <id> --assignee claude`. Keep changes small and incremental.
4. **Completing a task**: `bd close <id>`. Then update descriptions of downstream/dependent tasks with any new context gained during implementation (concrete line numbers, API signatures, gotchas discovered). Then `bd ready` for next task.
5. **End of session**: `bd sync` to persist state.

**Dependency context propagation**: When completing a task, always check which tasks it blocks (`bd show <id>` lists blockers). Update those dependent tasks' descriptions with implementation details that will help the next session pick up work without re-exploring. This is critical for multi-session work.

**Fresh context over assumptions**: Task descriptions should reference core documentation files (`CLAUDE.md`, `ARCHITECTURE.md`, `README.md`) rather than hardcoding architectural assumptions. When picking up a task, always re-read these files for current state — the codebase evolves and earlier task descriptions may be stale. The docs are the source of truth.

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
- **Dependencies**: mlx>=0.30.0, mlx-lm>=0.30.0, numpy>=2.0, rich>=14.0, webrtcvad-wheels>=2.0.14, litellm>=1.40, textual>=1.0
- **Root `stt.py`**: Thin shim that imports from `voiss.cli`. No longer the full script.

## File Layout (Target)

```
src/voiss/
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
  transcribe.py        — transcribe() generator + is_meaningful() filter
  audio/
    __init__.py         — re-exports
    ring_buffer.py      — RingBuffer (mutable, perf-critical)
    vad.py              — VadConfig + VoiceActivityDetector
  analysis.py          — IntentResult + analyze_intent()
  rewrite.py           — RewriteConfig + RewriteResult + rewrite_transcript()
  notes.py             — NotesConfig + notes pipeline orchestrator + helpers
  notes_app.py         — Textual TUI for notes mode (manual commit workflow)
  ui.py                — UiState + pure render functions (transcription mode)
  pipeline.py          — RealtimeTranscriber (orchestration, on_turn_complete callback)
  cli.py               — subcommands (bare transcribe + notes) + main()
```

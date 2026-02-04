# dictate.sh

Real-time speech-to-text and voice-driven notes for Apple Silicon.

`dictate.sh` uses MLX for fast, local ASR (Qwen3-ASR) with VAD-based turn detection.
Beyond live transcription, it supports a **notes mode** that pipes each spoken turn
through an LLM (via [litellm](https://github.com/BerriAI/litellm)) to produce
clean, structured markdown — useful for meeting notes, voice journaling,
and domain-specific dictation.

## Features

- Low-latency, streaming ASR on Apple Silicon (MLX)
- Voice activity detection (VAD) for automatic turn boundaries
- **ASR context biasing** — supply domain vocabulary to improve transcription accuracy
- **Notes mode** — LLM-rewritten markdown notes saved to file, per session
- Configurable rewrite system prompts for domain-specific style and structure
- Supports any litellm-compatible LLM backend (Ollama, OpenAI, Claude, etc.)
- Optional intent analysis with a local LLM
- Live terminal UI (Rich) with clean stdout for piping
- Fully offline after models are downloaded (with local LLM backends)

## Requirements

- macOS on Apple Silicon (MLX)
- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) installed
- Microphone permission granted to your terminal
- For notes mode: an LLM backend (e.g., [Ollama](https://ollama.com/) running locally)

## Quick Start

Live transcription (default mode):

```bash
uv run dictate
```

Notes mode — transcribe and rewrite into markdown:

```bash
uv run dictate notes --rewrite-model ollama/llama3.2
```

With domain vocabulary for better transcription accuracy:

```bash
uv run dictate notes --rewrite-model ollama/llama3.2 \
    --context "Kubernetes, kubectl, etcd, CoreDNS, Istio"
```

With a custom system prompt for the rewriter:

```bash
uv run dictate notes --rewrite-model ollama/llama3.2 \
    --system-prompt "You are a medical scribe. Format as SOAP notes."
```

Or load the system prompt from a file:

```bash
uv run dictate notes --rewrite-model ollama/llama3.2 \
    --system-prompt-file ~/prompts/meeting-notes.txt
```

Save notes to a specific file:

```bash
uv run dictate notes --rewrite-model ollama/llama3.2 \
    --notes-file ./meeting-2026-02-04.md
```

## CLI Reference

### Shared Options (all modes)

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `mlx-community/Qwen3-ASR-0.6B-8bit` | ASR model |
| `--language` | `English` | Transcription language |
| `--context` | — | Domain vocabulary for ASR context biasing |
| `--context-file` | — | File containing domain vocabulary |
| `--transcribe-interval` | `0.5` | Seconds between ASR updates |
| `--vad-frame-ms` | `30` | VAD frame size (10/20/30) |
| `--vad-mode` | `2` | VAD aggressiveness (0-3) |
| `--vad-silence-ms` | `500` | Silence to finalize a turn (ms) |
| `--min-words` | `3` | Minimum words to finalize a turn |
| `--device` | — | Audio input device index |
| `--list-devices` | — | List audio input devices |

### Transcription Mode (`dictate`)

| Option | Description |
|--------|-------------|
| `--analyze` | Enable LLM intent analysis on turn completion |
| `--llm-model` | LLM for intent analysis (default: `mlx-community/Qwen3-0.6B-4bit`) |
| `--no-ui` | Disable the Rich live UI |

### Notes Mode (`dictate notes`)

| Option | Description |
|--------|-------------|
| `--rewrite-model` | **(required)** LLM model for rewriting (e.g., `ollama/llama3.2`) |
| `--system-prompt` | System prompt to guide rewriting style |
| `--system-prompt-file` | Path to file containing the system prompt |
| `--notes-file` | Output file path (default: auto-named in notes directory) |
| `--vocab-file` | JSON vocabulary corrections file (default: `~/.config/dictate/vocab.json`) |

Notes are saved to `$DICTATE_NOTES_DIR` (default: `~/.local/share/dictate/notes/`) as
timestamped markdown files. Use `--notes-file` to write to a specific path instead.

### Notes Mode Key Bindings

| Key | Action |
|-----|--------|
| `Enter` | Commit accumulated speech through LLM rewrite |
| `Space` | Pause / resume recording |
| `q` | Quit (saves uncommitted text raw to file) |

The notes TUI has two panels: the left panel (40%) shows accumulated speech turns
with vocab corrections, and the right panel (60%) shows rewritten markdown output.

## ASR Context Biasing

Qwen3-ASR supports context biasing — you can supply domain-specific terms, acronyms,
or names that the model will prefer during decoding. This is built into the model's
architecture (trained during SFT), not a post-processing hack.

Use `--context` for inline terms or `--context-file` to load from a file:

```bash
# Inline domain vocabulary
uv run dictate --context "MLX, Qwen, WebRTC, VAD, litellm"

# From a file (one term per line, or freeform text)
uv run dictate --context-file ~/vocab/medical-terms.txt
```

Context biasing works in both transcription and notes modes.

## Recommended Models

### ASR (MLX Qwen3-ASR)

| Model | Notes |
|-------|-------|
| `mlx-community/Qwen3-ASR-0.6B-4bit` | Fastest, lowest quality |
| `mlx-community/Qwen3-ASR-0.6B-8bit` | Good balance (default) |
| `mlx-community/Qwen3-ASR-0.6B-bf16` | Higher quality, more RAM |
| `mlx-community/Qwen3-ASR-1.7B-8bit` | Higher quality, slower |

### LLM for Notes Rewriting (via litellm)

| Model | Backend | Notes |
|-------|---------|-------|
| `ollama/llama3.2` | Ollama | Good local default |
| `ollama/mistral` | Ollama | Strong instruction following |
| `openai/gpt-4o-mini` | OpenAI | Cloud, fast and cheap |

Any model supported by [litellm](https://docs.litellm.ai/docs/providers) works.

## UI and Piping

- The Rich live UI renders on `stderr`; `stdout` carries clean transcript lines.
- When `stdout` is not a TTY, the UI is automatically suppressed.
- Notes mode uses a full-screen Textual TUI (separate from the Rich live UI).

```bash
# Pipe raw transcripts to another tool
uv run dictate | grep "important"

# Watch notes being written in real-time
uv run dictate notes --rewrite-model ollama/llama3.2 &
tail -f ~/.local/share/dictate/notes/*.md
```

## Troubleshooting

- **Too many short turns**: increase `--vad-silence-ms` or lower `--vad-mode`
- **No audio**: check mic permissions or try `--list-devices` + `--device`
- **Laggy output**: reduce `--transcribe-interval`
- **Notes rewrite failures**: check that your LLM backend is running (e.g., `ollama serve`). Raw transcripts are saved on rewrite failure.
- **Domain words misrecognized**: use `--context` or `--context-file` with your vocabulary

## Logging

- Set `LOG_LEVEL=DEBUG` for verbose logs.
- Hugging Face and httpx request logs are suppressed by default.

## Acknowledgements

This project is built upon [dictate.sh by Marc Puig](https://github.com/mpuig/dictate.sh),
which provided the original implementation of real-time speech transcription using MLX
on Apple Silicon. The foundational architecture — audio capture pipeline, VAD-based turn
detection, streaming ASR with Qwen3, and the Rich terminal UI — originates from that work.

## License

MIT. See `LICENSE`.

# dictate.sh

Voice-driven notes for Apple Silicon — speak, review, commit to markdown.

```
┌─────────── Dictate Notes ───────────┐
│                                     │
│  ┌─ Speech ──────┐ ┌─ Notes ──────┐ │
│  │               │ │              │ │
│  │ You speak     │ │ LLM-cleaned  │ │
│  │ here. VAD     │ │ markdown     │ │
│  │ detects turn  │ │ appears here │ │
│  │ boundaries.   │ │ after you    │ │
│  │               │ │ press Enter. │ │
│  │ Press Enter   │ │              │ │
│  │ when ready.   │ │ Saved to     │ │
│  │               │ │ file auto.   │ │
│  └───────────────┘ └──────────────┘ │
│                                     │
│  ␣ Record/Stop  ⏎ Commit  ⎋ Discard│
│  q Quit                            │
└─────────────────────────────────────┘
```

Press **Space** to record. Speak naturally — VAD detects when you pause.
Press **Enter** to send accumulated speech through an LLM for cleanup.
Clean markdown appears in the right panel and is saved to disk.

All processing runs locally on your Mac's GPU via MLX. No cloud required
(unless you choose a cloud LLM for rewriting).

```bash
uv run dictate notes --rewrite-model ollama/llama3.2
```

## Features

- **Full-screen notes TUI** — two-panel Textual interface with push-to-record workflow
- Local, streaming ASR on Apple Silicon (MLX, Qwen3-ASR)
- Voice activity detection (VAD) for automatic turn boundaries
- **LLM rewriting** — each commit cleaned up via any [litellm](https://docs.litellm.ai/docs/providers)-compatible model
- **ASR context biasing** — supply domain vocabulary to improve transcription accuracy
- Configurable system prompts for domain-specific output (SOAP notes, meeting minutes, etc.)
- Also works as a **live transcription** pipe (`uv run dictate`)
- Fully offline after models are downloaded (with local LLM backends)

## Requirements

- macOS on Apple Silicon (MLX)
- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/) installed
- Microphone permission granted to your terminal
- For notes mode: an LLM backend (e.g., [Ollama](https://ollama.com/) running locally)

## Quick Start

Notes mode (the main event):

```bash
uv run dictate notes --rewrite-model ollama/llama3.2
```

Live transcription (simpler, no TUI):

```bash
uv run dictate
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
| `--vad-mode` | `3` | VAD aggressiveness (0-3) |
| `--vad-silence-ms` | `500` | Silence to finalize a turn (ms) |
| `--min-words` | `3` | Minimum words to finalize a turn |
| `--energy-threshold` | `300.0` | RMS energy gate for noise rejection |
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
| `Space` | Start / stop recording |
| `Enter` | Commit accumulated speech through LLM rewrite |
| `Escape` | Discard accumulated text (with confirmation) |
| `q` | Quit (saves uncommitted text raw to file) |

The left panel header shows the current state — "Paused" or "● Listening" — so you
always know whether audio is being captured. Press Space to record, speak naturally,
press Space to stop. Repeat to accumulate multiple turns. Press Enter when ready —
the LLM rewrites your speech into clean markdown on the right panel and saves it to
disk. Press Escape to discard and start over.

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
- **Background noise triggering VAD**: increase `--energy-threshold` (default 300, try 500-800)
- **Quiet speech being dropped**: lower `--energy-threshold` (try 100-200)
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

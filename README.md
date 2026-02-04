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
| `--context-bias` | `5.0` | Additive logit bias scale for context terms |
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
| `--config-file` | JSON config file for context, replacements, and bias (default: `~/.config/dictate/config.json`) |

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

## Domain Vocabulary and Context Biasing

Qwen3-ASR has a **native context biasing capability trained during supervised fine-tuning
(SFT)** — the model was explicitly taught to attend to vocabulary hints in its system
prompt during decoding. This is the primary mechanism for improving transcription accuracy
on domain-specific terms, and it works well out of the box.

dictate builds on this with two additional layers for cases where native biasing isn't
enough:

1. **Prompt context biasing (native SFT)** — domain terms are injected into the Qwen3-ASR
   system prompt. The model was trained to prefer these terms during decoding. This is the
   recommended starting point and usually sufficient.

2. **Logit biasing (mechanical supplement)** — optionally, `--context-bias` applies an
   additive logit bias to the subword tokens of context terms during decoding. This directly
   increases token probability regardless of acoustic evidence. Useful as a nudge for
   stubborn misrecognitions, but too high a value can cause hallucination. Use sparingly.

3. **Post-ASR replacements** — the `replacements` section of `config.json` applies regex
   find-and-replace after transcription. This is a last resort for deterministic ASR failure
   patterns that neither prompt nor logit biasing can fix.

### Via CLI

```bash
# Inline domain vocabulary — feeds native SFT context biasing
uv run dictate --context "Kubernetes, kubectl, etcd, CoreDNS"

# With supplementary logit biasing (default scale: 5.0)
uv run dictate --context "Kubernetes, kubectl, etcd" --context-bias 4.0

# From a file (one term per line, or freeform text)
uv run dictate --context-file ~/vocab/medical-terms.txt
```

### Via Configuration File

dictate loads configuration from `~/.config/dictate/config.json`. All sections are
optional. A sample config for clinical notes is provided in
[`examples/config.json`](examples/config.json).

```json
{
  "context": [
    "acetaminophen", "ibuprofen", "metformin", "lisinopril",
    "hypertension", "tachycardia", "echocardiogram",
    "SOAP", "CBC", "MRI", "EKG"
  ],
  "replacements": {
    "echo cardiogram": "echocardiogram",
    "die a beat ease": "diabetes"
  },
  "bias": {
    "terms": ["acetaminophen", "echocardiogram", "metformin"],
    "scale": 5.0
  }
}
```

| Section | Purpose |
|---------|---------|
| **`context`** | Terms injected into the ASR system prompt for native SFT context biasing. This is the primary accuracy lever. |
| **`replacements`** | Post-ASR regex find-and-replace (case-insensitive). Catches deterministic misrecognitions. |
| **`bias.terms`** | Subset of terms whose subword tokens get additive logit bias during decoding. Reserve for the hardest words. |
| **`bias.scale`** | Logit bias strength (default 5.0). Overridden by `--context-bias` on the CLI. |

Config context terms and CLI `--context` terms are merged (config first, CLI appended).

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
- **Domain words misrecognized**: use `--context` or `--context-file` with your vocabulary. If prompt biasing alone isn't enough, the logit bias (`--context-bias`) will increase token probabilities directly. Try values between 3.0-8.0. Too high (>10) may cause hallucination of biased terms.
- **Bias causing wrong words**: lower `--context-bias` (try 2.0-3.0) or remove overly broad terms from config

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

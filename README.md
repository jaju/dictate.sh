# voissistant

Voice-driven notes for Apple Silicon — speak, review, commit to markdown.

```
┌─────────── Voiss Notes ────────────┐
│                                     │
│  ┌─ Speech ──────┐ ┌─ Notes ──────┐ │
│  │               │ │              │ │
│  │ You speak     │ │ Rendered     │ │
│  │ here. VAD     │ │ markdown     │ │
│  │ detects turn  │ │ appears here │ │
│  │ boundaries.   │ │ after you    │ │
│  │               │ │ commit.      │ │
│  │ Tab/click to  │ │              │ │
│  │ focus, then   │ │ Tab/click to │ │
│  │ e to edit.    │ │ focus, then  │ │
│  │ (in-memory)   │ │ e to edit.   │ │
│  └───────────────┘ └──────────────┘ │
│                                     │
│  ␣ Rec  e Edit  ⏎ Raw  r Rewrite  │
│  ⎋ Discard  q Quit                 │
└─────────────────────────────────────┘
```

Press **Space** to record. Speak naturally — VAD detects when you pause.
Press **Enter** to commit accumulated speech raw (with vocab corrections).
Press **r** to send through an LLM for cleanup before committing.
Clean markdown is rendered in the right panel and saved to disk.

Both panels support modal editing. Panels have three visual states: **blue**
(default), **orange** (selected via Tab/click — no cursor, no interaction),
and **red** with an "EDIT" label (editing — cursor visible, full typing).
Press **e** on a selected panel to enter edit mode. In edit mode, all keys
type normally (Space, Enter, q, etc. insert text instead of triggering
actions). Press **Ctrl+S** to save your edit, or **Escape** to cancel and revert.
Left panel edits update the in-memory accumulator (for correcting speech before
commit). Right panel edits write back to the notes file.

All processing runs locally on your Mac's GPU via MLX. No cloud required
(unless you choose a cloud LLM for rewriting).

```bash
uv run voiss notes --rewrite-model ollama/llama3.2
```

## Features

- **Full-screen notes TUI** — two-panel Textual interface with push-to-record workflow and modal editing
- **Dual commit modes** — Enter for raw commit (fast), `r` for LLM-rewritten commit
- **Rendered markdown** — right panel displays notes as rendered markdown (switches to TextArea for editing)
- Local, streaming ASR on Apple Silicon (MLX, Qwen3-ASR)
- Voice activity detection (VAD) for automatic turn boundaries
- **LLM rewriting** — each `r` commit cleaned up via any [litellm](https://docs.litellm.ai/docs/providers)-compatible model
- **ASR context biasing** — supply domain vocabulary to improve transcription accuracy
- Configurable system prompts for domain-specific output (SOAP notes, meeting minutes, etc.)
- Also works as a **live transcription** pipe (`uv run voiss`)
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
uv run voiss notes --rewrite-model ollama/llama3.2
```

Live transcription (simpler, no TUI):

```bash
uv run voiss
```

With domain vocabulary for better transcription accuracy:

```bash
uv run voiss notes --rewrite-model ollama/llama3.2 \
    --context "Kubernetes, kubectl, etcd, CoreDNS, Istio"
```

With a custom system prompt for the rewriter:

```bash
uv run voiss notes --rewrite-model ollama/llama3.2 \
    --system-prompt "You are a medical scribe. Format as SOAP notes."
```

Or load the system prompt from a file:

```bash
uv run voiss notes --rewrite-model ollama/llama3.2 \
    --system-prompt-file ~/prompts/meeting-notes.txt
```

Save notes to a specific file:

```bash
uv run voiss notes --rewrite-model ollama/llama3.2 \
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

### Transcription Mode (`voiss`)

| Option | Description |
|--------|-------------|
| `--analyze` | Enable LLM intent analysis on turn completion |
| `--llm-model` | LLM for intent analysis (default: `mlx-community/Qwen3-0.6B-4bit`) |
| `--no-ui` | Disable the Rich live UI |

### Notes Mode (`voiss notes`)

| Option | Description |
|--------|-------------|
| `--rewrite-model` | **(required)** LLM model for rewriting (e.g., `ollama/llama3.2`) |
| `--system-prompt` | System prompt to guide rewriting style |
| `--system-prompt-file` | Path to file containing the system prompt |
| `--notes-file` | Output file path (default: auto-named in notes directory) |
| `--config-file` | JSON config file for context, replacements, and bias (default: `~/.config/voiss/config.json`) |

Notes are saved to `$VOISS_NOTES_DIR` (default: `~/.local/share/voiss/notes/`) as
timestamped markdown files. Use `--notes-file` to write to a specific path instead.

### Notes Mode Key Bindings

| Key | Action |
|-----|--------|
| `Space` | Start / stop recording |
| `Enter` | Commit accumulated speech raw (with vocab corrections, no LLM) |
| `r` | Commit with LLM rewrite |
| `e` | Enter edit mode on focused panel |
| `Escape` | Cancel edit (in edit mode) / Discard accumulated text (normal mode) |
| `Ctrl+S` | Save edit and exit edit mode |
| `q` | Quit with confirmation (saves uncommitted text raw to file) |

**Normal mode:** All keys above trigger their actions. Tab or click to focus a panel
(accent border appears). The left panel header shows "Paused" or "● Listening" so
you always know whether audio is being captured. The right panel displays rendered
markdown; it switches to a TextArea editor when you press `e`.

**Edit mode:** Press `e` on a focused panel to start editing. Space, Enter, q, and
e all type normally into the editor. Press `Ctrl+S` to save or `Escape` to cancel
and revert changes. Left panel edits update the in-memory accumulator; right panel
edits write to the notes file.

Press Space to record, speak naturally, press Space to stop. Repeat to accumulate
multiple turns. Press Enter for a quick raw commit, or `r` when you want the
LLM to clean up your speech into polished markdown. Press Escape to discard and start
over.

## Domain Vocabulary and Context Biasing

Qwen3-ASR has a **native context biasing capability trained during supervised fine-tuning
(SFT)** — the model was explicitly taught to attend to vocabulary hints in its system
prompt during decoding. This is the primary mechanism for improving transcription accuracy
on domain-specific terms, and it works well out of the box.

voiss builds on this with two additional layers for cases where native biasing isn't
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
uv run voiss --context "Kubernetes, kubectl, etcd, CoreDNS"

# With supplementary logit biasing (default scale: 5.0)
uv run voiss --context "Kubernetes, kubectl, etcd" --context-bias 4.0

# From a file (one term per line, or freeform text)
uv run voiss --context-file ~/vocab/medical-terms.txt
```

### Via Configuration File

voiss loads configuration from `~/.config/voiss/config.json`. All sections are
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
uv run voiss | grep "important"

# Watch notes being written in real-time
uv run voiss notes --rewrite-model ollama/llama3.2 &
tail -f ~/.local/share/voiss/notes/*.md
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

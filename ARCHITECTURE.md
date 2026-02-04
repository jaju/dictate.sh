# ARCHITECTURE.md — Technical Reference

## Overview

voissistant is a real-time speech-to-text tool and voice-driven notes system for Apple Silicon Macs. It captures microphone audio, detects speech boundaries via VAD, transcribes using a local ASR model (Qwen3-ASR) running on MLX, and supports two modes: live transcription (with optional intent analysis) and notes mode (LLM-rewritten markdown notes via litellm).

## Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| ML Runtime | MLX | >=0.30.0 | Apple Silicon GPU inference |
| LM Inference | mlx-lm | >=0.30.0 | Token generation, model loading |
| ASR Model | Qwen3-ASR | 0.6B/1.7B | Speech recognition (52 languages) |
| Audio Capture | sounddevice | >=0.5 | Low-latency mic input stream |
| VAD | webrtcvad-wheels | >=2.0.14 | Voice activity detection |
| Feature Extraction | transformers (Whisper) | >=4.47 | Mel-spectrogram (128 bins) |
| Model Hub | huggingface-hub | >=0.27 | Model download + caching |
| LLM Gateway | litellm | >=1.40 | Provider-agnostic LLM access (Ollama, OpenAI, etc.) |
| Terminal UI | Rich | >=14.0 | Live panels, tables, styled text |
| Notes TUI | Textual | >=1.0 | Full-screen TUI for notes mode |
| Numerics | NumPy | >=2.0 | Audio buffers, array ops |
| Runtime | Python | >=3.12 | Modern type syntax, slots |

## Data Flow

```
Microphone (16kHz, mono, int16)
    │
    ▼
Audio Callback (sounddevice thread)
    │  call_soon_threadsafe → asyncio.Queue
    ▼
Processor Loop (async)
    ├─ RingBuffer.append()     — circular int16 buffer, 30s max
    ├─ VAD.process()           — WebRTC frame analysis (10/20/30ms)
    │   └─ turn_complete?      — silence >= threshold
    └─ Periodic ASR            — every 0.5s if new audio
        ├─ int16 → float32     — normalize by 32768
        ├─ WhisperFeatureExtractor → mel spectrogram
        ├─ AudioEncoder        — conv downsample + transformer
        ├─ Token generation    — streaming via generate_step
        │   └─ logit_bias?     — additive bias on domain token logits
        └─ Update transcript
    │
    ▼ (on turn complete)
Handle Turn
    ├─ Validate (min words, meaningful content)
    ├─ Optional: analyze_intent() via LLM
    └─ Emit to UI + stdout
    │
    ▼
Display Loop (async)
    ├─ Rich Live UI → stderr
    └─ Clean text → stdout (pipe-friendly)
```

## Model Architecture (Qwen3-ASR)

### Audio Encoder
- 3x Conv2d layers: downsample 4x in time, 4x in frequency
- Linear projection to d_model (1024)
- Sinusoidal position embeddings (fixed, not learned)
- 24 transformer layers with multi-head self-attention (16 heads)
- Block attention masks: attention limited to chunk boundaries
- Chunked processing: n_window=50, n_window_infer=800 for bounded memory
- Output projection: d_model → output_dim (2048) via GELU + linear

### Text Decoder
- Qwen3 architecture: 28 transformer layers
- GQA: 16 query heads, 8 KV heads, head_dim=128
- RoPE positional encoding (theta=1M)
- SwiGLU MLP (gate + up projection, SiLU activation)
- RMSNorm (eps=1e-6)
- Tied embeddings (embed_tokens weight shared with lm_head)

### Audio-Text Fusion
- Audio pad tokens (`<|audio_pad|>`) in the chat template are replaced with audio encoder features
- Features injected into embedding space via scatter-add + mask

### Tokenizer
- Qwen3 tokenizer with special tokens:
  - `<|audio_start|>` / `<|audio_end|>` — audio boundary markers
  - `<|audio_pad|>` (id=151676) — placeholder for audio features
  - EOS tokens: 151645, 151643
- Chat template format: `<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>`
- **Context biasing (native SFT)**: The system prompt accepts domain vocabulary text that biases the decoder toward specific terms. This is a **native Qwen3-ASR capability trained during supervised fine-tuning** — the model was explicitly taught to attend to system prompt vocabulary during decoding. It is not a post-processing hack. Exposed via `--context` / `--context-file` CLI flags and the `context` list in `config.json`.
- **Logit biasing (mechanical)**: Optionally, `--context-bias` enables additive logit biasing on subword tokens of context terms via `mlx_lm.sample_utils.make_logits_processors()`. This is a separate, cruder mechanism that directly increases token probabilities regardless of acoustic evidence. Useful as a supplementary nudge for stubborn misrecognitions, but not a substitute for the native SFT biasing.

## Concurrency Model

- **Single async event loop** coordinates all work
- **GPU lock** (`asyncio.Lock`): MLX is not re-entrant; serialize all GPU calls
- **Buffer lock** (`asyncio.Lock`): protects ring buffer during concurrent read/write
- **Audio callback**: runs on sounddevice thread, uses `call_soon_threadsafe` to enqueue
- **`asyncio.to_thread`**: offloads blocking model calls (transcribe, analyze, rewrite) to thread pool
- **Turn callback**: `on_turn_complete` async callback on RealtimeTranscriber fires after each finalized turn (live transcription mode)
- **Notes TUI isolation**: `notes_app.py` does not depend on `pipeline.py` — it directly uses the same building blocks (RingBuffer, VAD, transcribe) with Textual managing the lifecycle

## Audio Processing Details

- **Sample rate**: 16,000 Hz, mono, int16
- **Ring buffer**: pre-allocated numpy array, circular write, avoids reallocation
- **VAD**: WebRTC VAD, configurable aggressiveness (0-3), frame sizes 10/20/30ms
- **Turn detection**: speech detected → silence frames counted → threshold → turn complete
- **Minimum content filter**: regex-based meaningful check + min word count

## Module Dependency Graph (Target)

```
constants          (no deps)
env                (no deps)
protocols          (numpy types)
config             (no deps)
model/_utils       (mlx)
model/encoder      (mlx, numpy, config, model/_utils)
model/decoder      (mlx, config, model/_utils)
model/asr          (mlx, numpy, config, encoder, decoder)
model/loader       (mlx, json, pathlib, hf_hub, transformers, config, asr, protocols)
transcribe         (mlx, numpy, mlx_lm, model, protocols)
audio/ring_buffer  (numpy)
audio/vad          (numpy, webrtcvad)
analysis           (re, mlx_lm, env, protocols)
rewrite            (litellm, json, pathlib, constants)  — also VoissConfig + load_config()
notes              (pathlib, numpy, rich, constants, env, model, transcribe, notes_app)
notes_app          (textual, rich, asyncio, numpy, sounddevice, notes, rewrite, transcribe, audio)
ui                 (rich, analysis)
pipeline           (asyncio, numpy, sounddevice, rich, all voiss modules)
cli                (argparse, constants, env, pipeline, notes, rewrite)
```

No circular dependencies. Strictly bottom-up.

## Notes Pipeline

The notes mode (`voiss notes`) uses a four-layer pipeline for domain-accurate structured notes:

```
Layer 1: ASR Context Biasing — native SFT (primary)
    config.json → "context": ["Kubernetes", "etcd", "CoreDNS"]
    --context "kubectl, Istio"  (merged with config terms)
    → injected into Qwen3-ASR system prompt as vocabulary hint
    → model was trained during SFT to attend to these terms
    → the primary mechanism for domain vocabulary accuracy

Layer 2: ASR Logit Biasing — mechanical (supplementary)
    config.json → "bias": {"terms": ["kubectl"], "scale": 5.0}
    --context-bias 4.0  (overrides config scale)
    → context terms tokenized → subword token IDs → additive logit bias
    → mlx_lm.sample_utils.make_logits_processors() → generate_step()
    → blunt force: increases token probability regardless of acoustics
    → use sparingly for terms that SFT biasing alone doesn't fix

Layer 3: Post-ASR Replacements
    config.json → "replacements": {"kube cuddle": "kubectl"}
    → regex find-and-replace after transcription, before LLM rewrite
    → catches deterministic ASR failure patterns

Layer 4: LLM Rewriting
    --rewrite-model ollama/llama3.2
    --system-prompt "Format as SOAP notes"
    → each finalized turn sent to external LLM via litellm
    → returns structured markdown appended to session file
```

### Textual TUI (notes_app.py)

Notes mode uses a Textual full-screen TUI with manual commit workflow:

- **Left panel** (40%): debounced, vocab-corrected transcript. VAD turns accumulate here across silence boundaries.
- **Right panel** (60%): Markdown widget showing rendered notes in display mode, TextArea for editing. CSS toggles visibility based on `-editing` class.
- **Footer**: live pipeline state (VAD, buffer, ASR latency, turn count) + key bindings.

**Key bindings**: Space starts/stops recording (push-to-record), Enter commits raw (with vocab corrections), `r` commits with LLM rewrite, q quits (saving uncommitted text raw).

**Dual commit modes**: Enter appends vocab-corrected text directly to the notes file (fast, no LLM). `r` runs the text through the configured LLM for cleanup before appending. Both modes clear the left panel, reset audio state, and reload the right panel.

**VAD + manual commit coexistence**: `_handle_turn_complete()` still fires on VAD silence (required for ASR buffer management), but the `on_turn_complete` callback only appends to an accumulator list — it does not auto-trigger rewrite. The user presses Enter or `r` when ready to commit.

**Architecture**: The Textual app does NOT use `RealtimeTranscriber`. Instead, `notes.run_notes_pipeline()` loads the ASR model and runs warmup *before* Textual starts (subprocess-safe, fd-safe). Then Textual owns the terminal and event loop. The app directly manages `RingBuffer`, `VoiceActivityDetector`, `sd.InputStream`, and an `asyncio.Queue` — with a `_processor()` task running on Textual's own event loop. ASR inference uses Python-level `redirect_stdout`/`redirect_stderr` only (no `os.dup2`) to avoid fd conflicts with Textual's terminal rendering. Rewrite runs in a separate `@work(thread=True)` worker via `rewrite_transcript()`.

Output files are saved to `$VOISS_NOTES_DIR` (default `~/.local/share/voiss/notes/`) as timestamped markdown, or to a path specified by `--notes-file`. On rewrite failure, the raw transcript is preserved.

## Key Patterns

1. **Deferred imports**: MLX and mlx_lm imported inside functions, not at module level, to control import ordering (env vars must be set first)
2. **Output suppression**: fd-level stderr redirect + StringIO to silence noisy library output during inference
3. **Streaming generation**: `generate_step` yields tokens one at a time for low-latency display
4. **Warmup**: dummy audio transcription on startup primes JIT caches and triggers one-time warnings off-screen
5. **TTY detection**: auto-disable Rich UI when stdout is piped; clean transcript lines only
6. **Pre-load before TUI**: ASR model loaded + warmed up before Textual starts, avoiding subprocess/fd conflicts
7. **Callback-based extension**: `on_turn_complete` callback on pipeline allows new features (webhooks) without subclassing
7. **Error resilience in rewrite**: exceptions captured in `RewriteResult.error`, raw transcript preserved on failure

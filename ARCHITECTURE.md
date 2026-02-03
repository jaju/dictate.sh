# ARCHITECTURE.md — Technical Reference

## Overview

dictate.sh is a real-time speech-to-text transcription tool for Apple Silicon Macs. It captures microphone audio, detects speech boundaries via VAD, transcribes using a local ASR model (Qwen3-ASR) running on MLX, and optionally analyzes intent with a small local LLM.

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
| Terminal UI | Rich | >=14.0 | Live panels, tables, styled text |
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
- Chat template format: `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage English<asr_text>`

## Concurrency Model

- **Single async event loop** coordinates all work
- **GPU lock** (`asyncio.Lock`): MLX is not re-entrant; serialize all GPU calls
- **Buffer lock** (`asyncio.Lock`): protects ring buffer during concurrent read/write
- **Audio callback**: runs on sounddevice thread, uses `call_soon_threadsafe` to enqueue
- **`asyncio.to_thread`**: offloads blocking model calls (transcribe, analyze) to thread pool

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
ui                 (rich, analysis)
pipeline           (asyncio, numpy, sounddevice, rich, all dictate modules)
cli                (argparse, asyncio, sounddevice, rich, constants, env, pipeline)
```

No circular dependencies. Strictly bottom-up.

## Key Patterns

1. **Deferred imports**: MLX and mlx_lm imported inside functions, not at module level, to control import ordering (env vars must be set first)
2. **Output suppression**: fd-level stderr redirect + StringIO to silence noisy library output during inference
3. **Streaming generation**: `generate_step` yields tokens one at a time for low-latency display
4. **Warmup**: dummy audio transcription on startup primes JIT caches and triggers one-time warnings off-screen
5. **TTY detection**: auto-disable Rich UI when stdout is piped; clean transcript lines only

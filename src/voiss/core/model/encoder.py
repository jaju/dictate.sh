"""Audio encoder: conv downsampling + transformer layers.

Compresses raw mel-spectrogram features into contextual embeddings
via 2D convolutions and self-attention with block masking.
"""

import math
from typing import override

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from voiss.core.config import AudioEncoderConfig
from voiss.core.model._utils import get_feat_extract_output_lengths


# ---------------------------------------------------------------------------
# Stateless helpers (extracted from AudioEncoder methods for testability)
# ---------------------------------------------------------------------------


def compute_chunk_layout(
    feature_lens: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Define chunking so long inputs stay bounded in memory/latency."""
    chunk_counts = np.ceil(feature_lens / chunk_size).astype(np.int32)
    chunk_lengths: list[int] = []
    for feat_len, num_chunks in zip(feature_lens, chunk_counts):
        feat_len = int(feat_len)
        num_chunks = int(num_chunks)
        for j in range(num_chunks):
            if j == num_chunks - 1:
                remainder = feat_len % chunk_size
                chunk_lengths.append(chunk_size if remainder == 0 else remainder)
            else:
                chunk_lengths.append(chunk_size)
    return chunk_counts, np.array(chunk_lengths, dtype=np.int32)


def slice_feature_chunks(
    input_features: mx.array,
    feature_lens: np.ndarray,
    chunk_counts: np.ndarray,
    chunk_size: int,
) -> list[mx.array]:
    """Cut features into chunks so conv/attention operate on windows."""
    chunks: list[mx.array] = []
    for feat, feat_len, num_chunks in zip(
        input_features, feature_lens, chunk_counts
    ):
        feat_len = int(feat_len)
        num_chunks = int(num_chunks)
        pos = 0
        remainder = feat_len % chunk_size
        for j in range(num_chunks):
            clen = (
                chunk_size if (j < num_chunks - 1 or remainder == 0) else remainder
            )
            chunks.append(feat[:, pos : pos + clen])
            pos += clen
    return chunks


def pad_chunks(
    chunks: list[mx.array], chunk_lengths: np.ndarray
) -> tuple[mx.array, int]:
    """Pad for batching so convs run as a single dense tensor."""
    max_chunk_len = int(chunk_lengths.max())
    padded: list[mx.array] = []
    for chunk, clen in zip(chunks, chunk_lengths):
        clen = int(clen)
        if clen < max_chunk_len:
            chunk = mx.pad(chunk, [(0, 0), (0, max_chunk_len - clen)])
        padded.append(chunk)
    return mx.stack(padded, axis=0), max_chunk_len


def build_cu_seqlens(
    aftercnn_lens: np.ndarray, window_aftercnn: int
) -> list[int]:
    """Provide segment boundaries so attention stays inside windows."""
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len = int(cnn_len)
        full_windows = cnn_len // window_aftercnn
        if full_windows:
            cu_chunk_lens.extend([window_aftercnn] * full_windows)
        remainder = cnn_len % window_aftercnn
        if remainder:
            cu_chunk_lens.append(remainder)
    return np.cumsum(cu_chunk_lens).tolist()


def create_block_attention_mask(
    seq_len: int, cu_seqlens: list[int], dtype: mx.Dtype
) -> mx.array:
    """Limit attention to chunk boundaries for stability and speed."""
    mask = mx.full((seq_len, seq_len), -1e9, dtype=dtype)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        mask[start:end, start:end] = 0.0
    return mask


# ---------------------------------------------------------------------------
# nn.Module classes
# ---------------------------------------------------------------------------


class SinusoidalPositionEmbedding(nn.Module):
    """Fixed positions so timing is known without extra learned parameters."""

    def __init__(self, length: int, channels: int, max_timescale: float = 10000.0):
        super().__init__()
        log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = mx.exp(
            -log_timescale_increment * mx.arange(channels // 2, dtype=mx.float32)
        )
        positions = mx.arange(length, dtype=mx.float32)[:, None]
        scaled_time = positions * inv_timescales[None, :]
        self._positional_embedding = mx.concatenate(
            [mx.sin(scaled_time), mx.cos(scaled_time)], axis=1
        )

    @override
    def __call__(self, seqlen: int) -> mx.array:
        return self._positional_embedding[:seqlen, :]


class AudioAttention(nn.Module):
    """Self-attention to relate distant audio frames for context."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    @override
    def __call__(
        self, hidden_states: mx.array, mask: mx.array | None = None
    ) -> mx.array:
        bsz, seq_len, _ = hidden_states.shape
        queries = self.q_proj(hidden_states) * self.scaling
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        attn_output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=1.0, mask=mask
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, seq_len, self.embed_dim
        )
        return self.out_proj(attn_output)


class AudioEncoderLayer(nn.Module):
    """Transformer block to mix local and global audio features."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    @override
    def __call__(
        self, hidden_states: mx.array, mask: mx.array | None = None
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask=mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.gelu(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class AudioEncoder(nn.Module):
    """Audio encoder that compresses time then builds contextual features."""

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer

        self.conv2d1 = nn.Conv2d(
            1, config.downsample_hidden_size, kernel_size=3, stride=2, padding=1
        )
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        freq_after_conv = ((((config.num_mel_bins + 1) // 2) + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * freq_after_conv, embed_dim, bias=False
        )
        self.positional_embedding = SinusoidalPositionEmbedding(
            config.max_source_positions, embed_dim
        )
        self.layers = [AudioEncoderLayer(config) for _ in range(config.encoder_layers)]
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, config.output_dim)

    @override
    def __call__(
        self,
        input_features: mx.array,
        feature_attention_mask: mx.array | None = None,
    ) -> mx.array:
        if feature_attention_mask is not None:
            feature_lens = feature_attention_mask.sum(axis=-1).astype(mx.int32)
        else:
            feature_lens = mx.array(
                [input_features.shape[-1]] * input_features.shape[0], dtype=mx.int32
            )

        feature_lens_np = np.array(feature_lens)
        aftercnn_lens = get_feat_extract_output_lengths(feature_lens)
        chunk_size = self.n_window * 2
        chunk_counts, chunk_lengths = compute_chunk_layout(
            feature_lens_np, chunk_size
        )
        chunks = slice_feature_chunks(
            input_features, feature_lens_np, chunk_counts, chunk_size
        )
        padded_feature, _ = pad_chunks(chunks, chunk_lengths)
        feature_lens_after_cnn = get_feat_extract_output_lengths(
            mx.array(chunk_lengths)
        )
        feature_lens_after_cnn_np = np.array(feature_lens_after_cnn)
        max_len_after_cnn = int(feature_lens_after_cnn_np.max())

        x = padded_feature[:, :, :, None]
        x = nn.gelu(self.conv2d1(x))
        x = nn.gelu(self.conv2d2(x))
        x = nn.gelu(self.conv2d3(x))

        b, f, t, c = x.shape
        x = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
        x = self.conv_out(x)
        x = x + self.positional_embedding(x.shape[1])[None, :, :]

        hidden_list = [
            x[i, : int(feature_lens_after_cnn_np[i])] for i in range(x.shape[0])
        ]
        hidden_states = mx.concatenate(hidden_list, axis=0)

        aftercnn_lens_np = np.array(aftercnn_lens)
        window_aftercnn = max_len_after_cnn * (
            self.n_window_infer // (self.n_window * 2)
        )
        cu_seqlens = build_cu_seqlens(aftercnn_lens_np, window_aftercnn)
        attention_mask = create_block_attention_mask(
            hidden_states.shape[0], cu_seqlens, hidden_states.dtype
        )
        attention_mask = attention_mask[None, None, :, :]
        hidden_states = hidden_states[None, :, :]

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=attention_mask)

        hidden_states = self.ln_post(hidden_states[0])
        hidden_states = nn.gelu(self.proj1(hidden_states))
        return self.proj2(hidden_states)

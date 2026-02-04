"""Text decoder: Qwen3-style transformer with GQA and RoPE.

Autoregressively generates text tokens conditioned on audio embeddings.
"""

from typing import Any, override

import mlx.core as mx
import mlx.nn as nn

from voiss.config import TextConfig
from voiss.model._utils import create_additive_causal_mask


class TextAttention(nn.Module):
    """Self-attention so text tokens can condition on prior context."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    @override
    def __call__(
        self, hidden_states: mx.array, cache: Any | None = None
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            B, L, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        offset = cache.offset if cache else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)

        if cache:
            keys, values = cache.update_and_fetch(keys, values)

        mask = create_additive_causal_mask(queries.shape[2], offset=offset).astype(
            queries.dtype
        )
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        return self.o_proj(
            output.transpose(0, 2, 1, 3).reshape(B, -1, self.num_heads * self.head_dim)
        )


class TextMLP(nn.Module):
    """SwiGLU MLP: nonlinear mixing to expand and compress token features."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    @override
    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TextDecoderLayer(nn.Module):
    """Pre-norm decoder block: attention + MLP with residuals."""

    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = TextAttention(config, layer_idx)
        self.mlp = TextMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    @override
    def __call__(
        self, hidden_states: mx.array, cache: Any | None = None
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states), cache=cache
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )


class TextModel(nn.Module):
    """Full text decoder: embeddings + N transformer layers + final norm."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            TextDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @override
    def __call__(
        self,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: list[Any] | None = None,
    ) -> mx.array:
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_tokens(input_ids)
        )
        cache = cache or [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, cache=cache[i])
        return self.norm(hidden_states)

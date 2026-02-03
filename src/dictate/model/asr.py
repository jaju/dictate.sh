"""Qwen3-ASR composite model: audio encoder + text decoder.

Fuses audio features into the text embedding space and generates
transcription tokens autoregressively.
"""

from typing import Any, override

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dictate.config import ModelConfig
from dictate.model.decoder import TextModel
from dictate.model.encoder import AudioEncoder

type Weights = dict[str, mx.array]


class Qwen3ASRModel(nn.Module):
    """Composite ASR model combining audio encoder and text decoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.audio_tower = AudioEncoder(config.audio_config)
        self.model = TextModel(config.text_config)
        self.lm_head = (
            None
            if config.text_config.tie_word_embeddings
            else nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
        )

    def get_audio_features(
        self,
        input_features: mx.array,
        feature_attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Encode raw audio features into contextual embeddings."""
        return self.audio_tower(input_features, feature_attention_mask)

    @override
    def __call__(
        self,
        input_ids: mx.array,
        input_embeddings: mx.array | None = None,
        input_features: mx.array | None = None,
        feature_attention_mask: mx.array | None = None,
        cache: list[Any] | None = None,
    ) -> mx.array:
        inputs_embeds = (
            input_embeddings
            if input_embeddings is not None
            else self.model.embed_tokens(input_ids)
        )

        if input_features is not None and (
            cache is None or cache[0] is None or cache[0].offset == 0
        ):
            audio_features = self.get_audio_features(
                input_features, feature_attention_mask
            ).astype(inputs_embeds.dtype)
            audio_token_mask = input_ids == self.config.audio_token_id

            if audio_token_mask.any():
                batch_size, seq_len, hidden_dim = inputs_embeds.shape
                flat_mask_np = np.array(audio_token_mask.reshape(-1))
                audio_indices = np.nonzero(flat_mask_np)[0]
                if len(audio_indices) > 0 and audio_features.shape[0] > 0:
                    num_to_replace = min(
                        len(audio_indices), audio_features.shape[0]
                    )
                    flat_embeds = inputs_embeds.reshape(-1, hidden_dim)
                    indices = mx.array(audio_indices[:num_to_replace])
                    replacement = (
                        mx.zeros_like(flat_embeds)
                        .at[indices]
                        .add(audio_features[:num_to_replace])
                    )
                    mask = (
                        mx.zeros(
                            (flat_embeds.shape[0],), dtype=flat_embeds.dtype
                        )
                        .at[indices]
                        .add(1)
                    )
                    flat_embeds = mx.where(
                        mask[:, None] > 0, replacement, flat_embeds
                    )
                    inputs_embeds = flat_embeds.reshape(
                        batch_size, seq_len, hidden_dim
                    )

        hidden_states = self.model(inputs_embeds=inputs_embeds, cache=cache)
        return (
            self.model.embed_tokens.as_linear(hidden_states)
            if self.lm_head is None
            else self.lm_head(hidden_states)
        )

    @property
    def layers(self) -> list:
        return self.model.layers

    @property
    def sample_rate(self) -> int:
        return 16_000

    def make_cache(self) -> list[Any]:
        """Create a fresh KV cache for autoregressive generation."""
        from mlx_lm.models.cache import KVCache

        return [
            KVCache()
            for _ in range(self.config.text_config.num_hidden_layers)
        ]

    @staticmethod
    def sanitize(weights: Weights) -> Weights:
        """Clean up weight keys for loading into this model structure."""
        sanitized: Weights = {}
        is_formatted = not any(k.startswith("thinker.") for k in weights)
        for k, v in weights.items():
            if k.startswith("thinker."):
                k = k[len("thinker."):]
            if k == "lm_head.weight":
                continue
            if (
                not is_formatted
                and "conv2d" in k
                and "weight" in k
                and len(v.shape) == 4
            ):
                v = v.transpose(0, 2, 3, 1)
            sanitized[k] = v
        return sanitized

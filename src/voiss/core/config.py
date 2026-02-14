"""Frozen configuration dataclasses for model architecture and runtime settings."""

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class AudioEncoderConfig:
    """Configuration for the audio encoder (conv + transformer)."""

    num_mel_bins: int = 128
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    scale_embedding: bool = False
    max_source_positions: int = 1500
    n_window: int = 50
    output_dim: int = 2048
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480


@dataclass(frozen=True, slots=True)
class TextConfig:
    """Configuration for the text decoder (Qwen3 transformer)."""

    vocab_size: int = 151_936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 1_000_000.0


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Top-level model configuration combining audio and text configs."""

    audio_config: AudioEncoderConfig = AudioEncoderConfig()
    text_config: TextConfig = TextConfig()
    audio_token_id: int = 151_676
    support_languages: tuple[str, ...] = ()


def _filter_fields(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only keys that match dataclass fields."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def make_model_config(raw: dict[str, Any]) -> ModelConfig:
    """Factory: resolve nested dicts into frozen sub-configs.

    Handles both flat configs and configs wrapped in a 'thinker_config' key
    (as found in some Qwen3-ASR model repos).
    """
    if "thinker_config" in raw:
        thinker = raw["thinker_config"]
        raw = {
            "audio_config": thinker.get("audio_config", {}),
            "text_config": thinker.get("text_config", {}),
            "audio_token_id": thinker.get("audio_token_id", 151_676),
            "support_languages": raw.get("support_languages", []),
        }

    audio_raw = raw.get("audio_config")
    audio_config = (
        AudioEncoderConfig(**_filter_fields(AudioEncoderConfig, audio_raw))
        if isinstance(audio_raw, dict)
        else audio_raw
        if isinstance(audio_raw, AudioEncoderConfig)
        else AudioEncoderConfig()
    )

    text_raw = raw.get("text_config")
    text_config = (
        TextConfig(**_filter_fields(TextConfig, text_raw))
        if isinstance(text_raw, dict)
        else text_raw
        if isinstance(text_raw, TextConfig)
        else TextConfig()
    )

    languages = raw.get("support_languages", ())
    if isinstance(languages, list):
        languages = tuple(languages)

    return ModelConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_id=raw.get("audio_token_id", 151_676),
        support_languages=languages,
    )

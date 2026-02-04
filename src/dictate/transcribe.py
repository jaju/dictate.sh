"""Streaming ASR transcription using Qwen3-ASR.

The transcribe() generator yields tokens one at a time for low-latency
display during real-time speech processing.
"""

from collections.abc import Generator

import mlx.core as mx
import numpy as np

from dictate.model._utils import get_feat_extract_output_lengths
from dictate.model.asr import Qwen3ASRModel
from dictate.protocols import FeatureExtractorLike, TokenizerLike


def transcribe(
    model: Qwen3ASRModel,
    tokenizer: TokenizerLike,
    feature_extractor: FeatureExtractorLike,
    audio: np.ndarray,
    language: str = "English",
    max_tokens: int = 8192,
    context: str | None = None,
) -> Generator[str, None, None]:
    """Stream transcription tokens from audio input.

    Encodes audio features, replaces audio pad tokens in the prompt
    with encoder outputs, then generates text tokens autoregressively.

    When *context* is provided, it is injected into the Qwen3-ASR system
    prompt to bias the decoder toward domain-specific vocabulary.
    """
    from mlx_lm.generate import generate_step

    # Match the model's expected feature pipeline.
    audio_inputs = feature_extractor(
        audio,
        sampling_rate=16_000,
        return_attention_mask=True,
        truncation=False,
        padding=True,
        return_tensors="np",
    )
    input_features = mx.array(audio_inputs["input_features"])
    feature_attention_mask = mx.array(audio_inputs["attention_mask"])

    # Size the audio pad tokens in the prompt.
    audio_lengths = feature_attention_mask.sum(axis=-1)
    aftercnn_lens = get_feat_extract_output_lengths(audio_lengths)
    num_audio_tokens = int(aftercnn_lens[0].item())

    # Qwen3-ASR expects audio pads inside the chat template.
    supported = model.config.support_languages or ()
    supported_lower = {lang.lower(): lang for lang in supported}
    lang_name = supported_lower.get(language.lower(), language)

    system_content = context or ""
    prompt = (
        f"<|im_start|>system\n{system_content}<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|>"
        f"{'<|audio_pad|>' * num_audio_tokens}"
        f"<|audio_end|><|im_end|>\n"
        f"<|im_start|>assistant\nlanguage {lang_name}<asr_text>"
    )
    input_ids = mx.array(tokenizer.encode(prompt, return_tensors="np"))

    # Compute audio features once for embedding replacement.
    audio_features = model.get_audio_features(
        input_features, feature_attention_mask
    )
    mx.eval(audio_features)

    # Replace audio token embeddings with audio features.
    inputs_embeds = model.model.embed_tokens(input_ids)
    audio_features = audio_features.astype(inputs_embeds.dtype)
    audio_token_mask = input_ids == model.config.audio_token_id

    if audio_token_mask.any():
        batch_size, seq_len, hidden_dim = inputs_embeds.shape
        flat_mask_np = np.array(audio_token_mask.reshape(-1))
        audio_indices = np.nonzero(flat_mask_np)[0]

        if len(audio_indices) > 0:
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

    mx.eval(inputs_embeds)
    input_embeddings = inputs_embeds[0]
    prompt_ids = input_ids[0] if input_ids.ndim > 1 else input_ids

    eos_token_ids = [151645, 151643]

    for token, _ in generate_step(
        prompt=prompt_ids,
        input_embeddings=input_embeddings,
        model=model,
        max_tokens=max_tokens,
    ):
        if token in eos_token_ids:
            break
        yield tokenizer.decode([int(token)])

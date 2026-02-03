"""Tensor math helpers for the model subpackage.

Pure functions for causal masks and feature length calculations.
"""

import mlx.core as mx

type MaskArray = mx.array


def create_additive_causal_mask(n: int, offset: int = 0) -> MaskArray:
    """Return an additive causal mask to prevent attention to future tokens."""
    rinds = mx.arange(offset + n)
    linds = mx.arange(offset, offset + n) if offset else rinds
    mask = linds[:, None] < rinds[None]
    return mask * -1e9


def floor_div(a: mx.array, b: int) -> mx.array:
    """Floor-divide while keeping MLX tensors, avoiding host/device sync."""
    return mx.floor(a.astype(mx.float32) / b).astype(mx.int32)


def get_feat_extract_output_lengths(input_lengths: mx.array) -> mx.array:
    """Track time-downsampling so chunk masks align with conv output."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = floor_div(input_lengths_leave - 1, 2) + 1
    output_lengths = (
        floor_div(floor_div(feat_lengths - 1, 2) + 1 - 1, 2)
        + 1
        + (input_lengths // 100) * 13
    )
    return output_lengths

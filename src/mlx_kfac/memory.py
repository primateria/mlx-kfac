"""K-FAC optimizer-state memory estimation."""

from __future__ import annotations

import math

import mlx.nn as nn
from mlx.utils import tree_flatten


_FLOAT32_BYTES = 4
# Three uint64 counters, one bool refresh flag, and four float32 scalar leaves
# shared by K-FAC and its AdamW fallback. This is kept in bytes because those
# leaves are intentionally not all float32.
_GLOBAL_STATE_BYTES = 3 * 8 + 1 + 4 * _FLOAT32_BYTES


def _decomposition_elements(sizes, decomposition):
    matrix_elements = sum(size * size for size in sizes)
    if decomposition == "cholesky":
        return matrix_elements
    return matrix_elements + sum(sizes)


def _affine_state_elements(
    optimizer,
    out_dims,
    in_dims,
    out_sizes,
    in_sizes,
):
    factor_elements = sum(size * size for size in out_sizes)
    factor_elements += sum(size * size for size in in_sizes)
    decomposition_elements = _decomposition_elements(
        out_sizes, optimizer.decomposition
    )
    decomposition_elements += _decomposition_elements(
        in_sizes, optimizer.decomposition
    )
    combined_elements = 0
    if optimizer.damping_strategy == "pi" and (
        out_dims == 1 or in_dims == 1
    ):
        other_sizes = out_sizes if in_dims == 1 else in_sizes
        combined_elements = _decomposition_elements(
            other_sizes, optimizer.decomposition
        )
    # A/G damping, current/cached repetition scale, and velocity.
    return (
        factor_elements
        + decomposition_elements
        + combined_elements
        + out_dims * in_dims
        + 4
    )


def _embedding_state_elements(optimizer, vocab_size, dims):
    feature_sizes = []
    remaining = dims
    while remaining:
        width = (
            remaining
            if optimizer.max_factor_size is None
            else min(remaining, optimizer.max_factor_size)
        )
        feature_sizes.append(width)
        remaining -= width
    factor_elements = sum(size * size for size in feature_sizes)
    decomposition_elements = _decomposition_elements(
        feature_sizes, optimizer.decomposition
    )
    combined_elements = 0
    if optimizer.damping_strategy == "pi":
        if dims == 1:
            combined_elements = vocab_size
        elif vocab_size == 1:
            combined_elements = _decomposition_elements(
                feature_sizes, optimizer.decomposition
            )
    # Current/cached token diagonals, velocity, damping, and repetition scale.
    return (
        2 * vocab_size
        + factor_elements
        + decomposition_elements
        + combined_elements
        + vocab_size * dims
        + 4
    )


def estimate_kfac_state_bytes(optimizer, model, parameters=None) -> int:
    """Estimate persistent float32 optimizer state before allocating factors."""
    parameters = parameters or model.trainable_parameters()
    flat_parameters = dict(tree_flatten(parameters))
    supported = set()
    elements = 0

    for path, layer in optimizer._find_layers(model).items():
        prefix = f"{path}." if path else ""
        weight_path = prefix + "weight"
        if weight_path not in flat_parameters:
            continue
        weight = flat_parameters[weight_path]
        if isinstance(layer, nn.Embedding):
            elements += _embedding_state_elements(
                optimizer, weight.shape[0], weight.shape[1]
            )
            supported.add(weight_path)
            continue

        out_dims = weight.shape[0]
        in_dims = math.prod(weight.shape[1:])
        has_bias = "bias" in layer and prefix + "bias" in flat_parameters
        in_dims += int(has_bias)
        out_sizes, in_sizes = optimizer._partition_sizes(
            path, out_dims, in_dims
        )
        elements += _affine_state_elements(
            optimizer, out_dims, in_dims, out_sizes, in_sizes
        )
        supported.add(weight_path)
        if has_bias:
            supported.add(prefix + "bias")

    # AdamW fallback keeps two float32 moments for every unsupported leaf.
    for path, parameter in flat_parameters.items():
        if path not in supported:
            elements += 2 * parameter.size
    return int(_GLOBAL_STATE_BYTES + elements * _FLOAT32_BYTES)

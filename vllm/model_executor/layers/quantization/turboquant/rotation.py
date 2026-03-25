# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Randomized Hadamard Transform for TurboQuant.

Implements Π = D·H where H is the Walsh-Hadamard matrix and D is a
random diagonal of ±1. This gives the same distributional properties
as a full random orthogonal rotation while being O(d log d) instead
of O(d²).
"""

import math

import torch


def generate_random_signs(
    dim: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a deterministic random ±1 sign vector.

    Args:
        dim: Dimension of the sign vector.
        seed: Random seed for reproducibility.
        device: Target device.

    Returns:
        Tensor of shape (dim,) with values ±1, dtype float32.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    # Generate random bits and map {0, 1} -> {-1, +1}
    signs = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32)
    signs = signs * 2.0 - 1.0
    return signs.to(device)


def fast_walsh_hadamard_transform(
    x: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """In-place Fast Walsh-Hadamard Transform along the last dimension.

    The last dimension must be a power of 2.

    Args:
        x: Input tensor of shape (..., d) where d is a power of 2.
        normalize: If True, divide by sqrt(d) for orthonormal transform.

    Returns:
        Transformed tensor of same shape.
    """
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Last dim must be power of 2, got {d}"

    # Butterfly stages of the Hadamard transform
    h = 1
    result = x.clone()
    while h < d:
        # Split into pairs separated by h
        result_view = result.view(*result.shape[:-1], d // (2 * h), 2, h)
        a = result_view[..., 0, :].clone()
        b = result_view[..., 1, :].clone()
        result_view[..., 0, :] = a + b
        result_view[..., 1, :] = a - b
        h *= 2

    if normalize:
        result = result / math.sqrt(d)

    return result


def randomized_hadamard_transform(
    x: torch.Tensor,
    signs: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Apply Randomized Hadamard Transform: y = H @ diag(signs) @ x / sqrt(d).

    Args:
        x: Input tensor of shape (..., d).
        signs: Random ±1 vector of shape (d,).
        normalize: If True, divide by sqrt(d).

    Returns:
        Rotated tensor of same shape.
    """
    # Apply random sign flip
    y = x * signs
    # Apply Walsh-Hadamard transform
    return fast_walsh_hadamard_transform(y, normalize=normalize)


def inverse_randomized_hadamard_transform(
    y: torch.Tensor,
    signs: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Inverse Randomized Hadamard Transform.

    Since H is self-inverse (up to normalization) and D is self-inverse,
    the inverse is: x = diag(signs) @ H @ y / sqrt(d).

    Args:
        y: Rotated tensor of shape (..., d).
        signs: Same random ±1 vector used in the forward transform.
        normalize: If True, divide by sqrt(d).

    Returns:
        Original tensor of same shape.
    """
    # Apply Walsh-Hadamard transform (self-inverse up to scale)
    x = fast_walsh_hadamard_transform(y, normalize=normalize)
    # Apply inverse sign flip (D is self-inverse)
    return x * signs


def make_rotation_seed(
    layer_idx: int,
    head_idx: int,
    is_key: bool,
    base_seed: int = 42,
) -> int:
    """Generate a deterministic seed for a specific rotation matrix.

    Args:
        layer_idx: Layer index in the model.
        head_idx: Attention head index.
        is_key: True for key, False for value.
        base_seed: Base seed from config.

    Returns:
        Deterministic seed for this (layer, head, key/value) combination.
    """
    # Use a simple hash combining scheme
    kv_flag = 1 if is_key else 0
    return base_seed + layer_idx * 100000 + head_idx * 100 + kv_flag

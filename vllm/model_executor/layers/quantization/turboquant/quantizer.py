# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant quantizer implementations.

Provides TurboQuantMSE (Algorithm 1) and TurboQuantProd (Algorithm 2)
from the paper. Pure PyTorch implementation for Phase 1.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate" (Zandieh et al., 2025)
"""

import math

import torch

from vllm.model_executor.layers.quantization.turboquant.codebook import (
    LloydMaxCodebook,
)
from vllm.model_executor.layers.quantization.turboquant.rotation import (
    generate_random_signs,
    inverse_randomized_hadamard_transform,
    randomized_hadamard_transform,
)


def _scalar_quantize(
    x: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    """Quantize each element to the nearest centroid index.

    Uses boundary-based assignment: idx = sum(x > boundary[i]) for each i.
    This is equivalent to nearest-centroid for sorted centroids with
    midpoint boundaries.

    Args:
        x: Input tensor, any shape.
        centroids: 1D tensor of 2^b sorted centroid values.
        boundaries: 1D tensor of 2^b - 1 sorted boundary values.

    Returns:
        Integer tensor of same shape as x, with values in [0, 2^b - 1].
    """
    # Compare x against each boundary: shape (..., num_boundaries)
    # x > boundary gives True/False, sum gives the bin index
    x_expanded = x.unsqueeze(-1)
    boundaries_expanded = boundaries.view(*([1] * len(x.shape)), -1)
    indices = (x_expanded > boundaries_expanded).sum(dim=-1)
    return indices.to(torch.uint8)


def _scalar_dequantize(
    indices: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Look up centroid values from indices.

    Args:
        indices: Integer tensor with values in [0, 2^b - 1].
        centroids: 1D tensor of 2^b centroid values.

    Returns:
        Float tensor with centroid values at each index position.
    """
    return centroids[indices.long()]


def pack_indices(
    indices: torch.Tensor,
    bit_width: int,
) -> torch.Tensor:
    """Pack b-bit indices into uint8 bytes.

    Args:
        indices: Tensor of shape (..., d) with values in [0, 2^b - 1].
        bit_width: Bits per index (1-4).

    Returns:
        Packed uint8 tensor. Shape (..., ceil(d * bit_width / 8)).
    """
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]
    flat = indices.reshape(-1, d).to(torch.uint8)
    total_bits = d * bit_width
    packed_size = (total_bits + 7) // 8

    packed = torch.zeros(
        flat.shape[0], packed_size, dtype=torch.uint8, device=indices.device
    )

    bit_pos = 0
    for i in range(d):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        val = flat[:, i].to(torch.uint8)

        if bit_offset + bit_width <= 8:
            # Fits within one byte
            packed[:, byte_idx] |= val << bit_offset
        else:
            # Spans two bytes
            bits_in_first = 8 - bit_offset
            packed[:, byte_idx] |= (val & ((1 << bits_in_first) - 1)) << bit_offset
            packed[:, byte_idx + 1] |= val >> bits_in_first

        bit_pos += bit_width

    return packed.reshape(*batch_shape, packed_size)


def unpack_indices(
    packed: torch.Tensor,
    bit_width: int,
    dim: int,
) -> torch.Tensor:
    """Unpack uint8 bytes into b-bit indices.

    Args:
        packed: Packed uint8 tensor of shape (..., packed_size).
        bit_width: Bits per index (1-4).
        dim: Original dimension d (number of indices to extract).

    Returns:
        Tensor of shape (..., d) with values in [0, 2^b - 1], dtype uint8.
    """
    batch_shape = packed.shape[:-1]
    packed_flat = packed.reshape(-1, packed.shape[-1])
    mask = (1 << bit_width) - 1

    indices = torch.zeros(
        packed_flat.shape[0], dim, dtype=torch.uint8, device=packed.device
    )

    bit_pos = 0
    for i in range(dim):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8

        if bit_offset + bit_width <= 8:
            indices[:, i] = (packed_flat[:, byte_idx] >> bit_offset) & mask
        else:
            bits_in_first = 8 - bit_offset
            low = packed_flat[:, byte_idx] >> bit_offset
            high = packed_flat[:, byte_idx + 1] & (
                (1 << (bit_width - bits_in_first)) - 1
            )
            indices[:, i] = low | (high << bits_in_first)

        bit_pos += bit_width

    return indices.reshape(*batch_shape, dim)


class TurboQuantMSE:
    """TurboQuant MSE-optimal quantizer (Algorithm 1).

    Quantizes unit-norm vectors by:
    1. Applying a randomized Hadamard rotation.
    2. Scalar-quantizing each coordinate using Lloyd-Max codebook.

    Dequantizes by looking up centroids and applying inverse rotation.
    """

    def __init__(
        self,
        bit_width: int,
        head_dim: int,
        device: torch.device,
        seed: int = 42,
    ):
        self.bit_width = bit_width
        self.head_dim = head_dim
        self.device = device
        self.seed = seed

        # Get codebook scaled for this dimension
        self.centroids, self.boundaries = LloydMaxCodebook.get(
            bit_width, head_dim, device
        )
        # Generate rotation signs
        self.signs = generate_random_signs(head_dim, seed, device)

    def quantize(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize input vectors.

        Args:
            x: Input tensor of shape (..., head_dim). Need not be unit-norm;
               norms are extracted and stored separately.

        Returns:
            Tuple of (packed_indices, norms).
            packed_indices: uint8 tensor of packed b-bit indices.
            norms: float16 tensor of L2 norms, shape (...,).
        """
        # Compute and store norms
        norms = torch.norm(x, dim=-1, keepdim=True)
        # Avoid division by zero
        safe_norms = norms.clamp(min=1e-10)
        x_normalized = x / safe_norms

        # Apply randomized Hadamard rotation
        y = randomized_hadamard_transform(x_normalized.float(), self.signs)

        # Scalar quantize each coordinate
        indices = _scalar_quantize(y, self.centroids, self.boundaries)

        # Pack indices
        packed = pack_indices(indices, self.bit_width)

        return packed, norms.squeeze(-1).to(torch.float16)

    def dequantize(
        self,
        packed: torch.Tensor,
        norms: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize packed indices back to vectors.

        Args:
            packed: Packed uint8 tensor from quantize().
            norms: Float16 norms tensor from quantize().

        Returns:
            Reconstructed tensor of shape (..., head_dim), same dtype as norms
            promoted to float.
        """
        # Unpack indices
        indices = unpack_indices(packed, self.bit_width, self.head_dim)

        # Look up centroids
        y_hat = _scalar_dequantize(indices, self.centroids)

        # Apply inverse rotation
        x_hat = inverse_randomized_hadamard_transform(y_hat, self.signs)

        # Rescale by norms
        x_hat = x_hat * norms.unsqueeze(-1).float()

        return x_hat


class TurboQuantProd:
    """TurboQuant unbiased inner-product quantizer (Algorithm 2).

    Two-stage approach:
    1. MSE quantization at (b-1) bits per coordinate.
    2. QJL (1-bit sign sketch) on the residual.

    Total storage: b*d bits + O(1) per vector.
    """

    def __init__(
        self,
        bit_width: int,
        head_dim: int,
        device: torch.device,
        seed: int = 42,
    ):
        if bit_width < 2:
            raise ValueError(
                "TurboQuantProd requires bit_width >= 2 (1 bit for MSE + 1 bit for QJL)"
            )
        self.bit_width = bit_width
        self.head_dim = head_dim
        self.device = device

        # Stage 1: MSE quantizer at (b-1) bits
        self.mse_quantizer = TurboQuantMSE(
            bit_width=bit_width - 1,
            head_dim=head_dim,
            device=device,
            seed=seed,
        )

        # Stage 2: QJL random projection matrix
        # Generate deterministic random Gaussian matrix
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + 999999)
        self.S = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32).to(
            device
        )

        self._qjl_coeff = math.sqrt(math.pi / 2) / head_dim

    def quantize(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize with unbiased inner-product guarantee.

        Args:
            x: Input tensor of shape (..., head_dim).

        Returns:
            Tuple of (mse_packed, mse_norms, qjl_signs, residual_norms).
        """
        # Compute norms
        norms = torch.norm(x, dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-10)
        x_normalized = x / safe_norms

        # Stage 1: MSE quantize the normalized vector
        mse_packed, _ = self.mse_quantizer.quantize(x_normalized)

        # Dequantize to get MSE reconstruction (of unit-norm vector)
        x_mse = self.mse_quantizer.dequantize(
            mse_packed, torch.ones_like(norms.squeeze(-1))
        )

        # Compute residual
        residual = x_normalized - x_mse
        residual_norm = torch.norm(residual, dim=-1)

        # Stage 2: QJL sign sketch of residual
        # S @ residual^T -> shape (..., head_dim)
        projected = torch.matmul(residual.float(), self.S.t())
        qjl_signs = (projected >= 0).to(torch.uint8)

        # Pack QJL signs (1 bit each)
        qjl_packed = pack_indices(qjl_signs, 1)

        return (
            mse_packed,
            norms.squeeze(-1).to(torch.float16),
            qjl_packed,
            residual_norm.to(torch.float16),
        )

    def dequantize(
        self,
        mse_packed: torch.Tensor,
        norms: torch.Tensor,
        qjl_packed: torch.Tensor,
        residual_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize with unbiased inner-product reconstruction.

        Args:
            mse_packed: Packed MSE indices from quantize().
            norms: L2 norms from quantize().
            qjl_packed: Packed QJL sign bits from quantize().
            residual_norms: Residual L2 norms from quantize().

        Returns:
            Reconstructed tensor of shape (..., head_dim).
        """
        # Stage 1: MSE dequantization (unit-norm reconstruction)
        x_mse = self.mse_quantizer.dequantize(
            mse_packed,
            torch.ones(*norms.shape, dtype=torch.float16, device=norms.device),
        )

        # Stage 2: QJL dequantization
        qjl_signs = unpack_indices(qjl_packed, 1, self.head_dim)
        # Map {0, 1} -> {-1, +1}
        qjl_float = qjl_signs.float() * 2.0 - 1.0

        # x_qjl = (sqrt(pi/2) / d) * gamma * S^T @ signs
        x_qjl = (
            self._qjl_coeff
            * residual_norms.unsqueeze(-1).float()
            * (torch.matmul(qjl_float, self.S))
        )

        # Combine and rescale
        x_hat = (x_mse + x_qjl) * norms.unsqueeze(-1).float()

        return x_hat

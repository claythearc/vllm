# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant KV compressor for online quantize-dequantize of K/V tensors.

This module provides the TurboQuantKVCompressor class which applies
TurboQuant quantization followed by immediate dequantization to K and V
tensors before they are stored in the standard KV cache. This simulates
the lossy compression effect of TurboQuant while maintaining compatibility
with all attention backends (Phase 2).

Actual packed storage with memory savings is deferred to Phase 3
(custom CUDA kernels).
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.turboquant.codebook import (
    LloydMaxCodebook,
)
from vllm.model_executor.layers.quantization.turboquant.rotation import (
    fast_walsh_hadamard_transform,
    generate_random_signs,
)

logger = init_logger(__name__)


class TurboQuantKVCompressor:
    """Applies TurboQuant quantize-dequantize to K/V tensors in-place.

    For Phase 2, this operates on float16/bfloat16 tensors and returns
    float16/bfloat16 tensors. The quantization→dequantization round-trip
    introduces the same distortion as actual TurboQuant compression,
    allowing accuracy validation without custom CUDA kernels.

    Supports mixed-precision (outlier channels) for non-integer bit-widths
    like 2.5 and 3.5.

    Args:
        bit_width: Effective bit-width (e.g., 2.5, 3.0, 3.5, 4.0).
        head_dim: Dimension of each attention head (must be power of 2).
        num_kv_heads: Number of key-value heads.
        outlier_channels: Number of channels receiving +1 bit (for
            non-integer bit-widths like 2.5, 3.5).
        seed: Base seed for deterministic rotation matrices.
        device: Target device for codebook and rotation tensors.
    """

    def __init__(
        self,
        bit_width: float,
        head_dim: int,
        num_kv_heads: int,
        outlier_channels: int = 32,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ):
        self.bit_width = bit_width
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.outlier_channels = outlier_channels
        self.seed = seed
        self.device = device

        # Determine integer bit-widths for normal and outlier channels
        self.bits_normal = int(bit_width)
        self.bits_outlier = self.bits_normal + 1
        self.has_outliers = (
            bit_width != float(self.bits_normal) and outlier_channels > 0
        )

        # Number of normal vs outlier channels
        if self.has_outliers:
            self.n_outlier = min(outlier_channels, head_dim)
            self.n_normal = head_dim - self.n_outlier
        else:
            self.n_outlier = 0
            self.n_normal = head_dim

        # Preload codebooks (scaled for head_dim)
        self.centroids_normal, self.boundaries_normal = LloydMaxCodebook.get(
            self.bits_normal, head_dim, device
        )
        if self.has_outliers:
            self.centroids_outlier, self.boundaries_outlier = LloydMaxCodebook.get(
                self.bits_outlier, head_dim, device
            )

        # Generate rotation signs (shared across heads for simplicity).
        # Per-head seeds can be used for more randomness, but sharing
        # works well in practice and simplifies the implementation.
        self.signs_k = generate_random_signs(head_dim, seed, device)
        self.signs_v = generate_random_signs(head_dim, seed + 1, device)

        # Outlier channel indices (fixed: last n_outlier channels after
        # rotation, which is equivalent to random channels in the original
        # space due to the random rotation).
        if self.has_outliers:
            self.outlier_mask = torch.zeros(head_dim, dtype=torch.bool, device=device)
            self.outlier_mask[-self.n_outlier :] = True
            self.normal_mask = ~self.outlier_mask

    def _quantize_dequantize_1d(
        self,
        y: torch.Tensor,
        centroids: torch.Tensor,
        boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """Scalar quantize then dequantize a tensor.

        Args:
            y: Rotated coordinates, any shape.
            centroids: Sorted centroid values.
            boundaries: Sorted decision boundaries.

        Returns:
            Tensor of same shape with values snapped to nearest centroids.
        """
        # Quantize: assign each value to nearest centroid via boundary check
        y_expanded = y.unsqueeze(-1)
        boundaries_expanded = boundaries.view(*([1] * len(y.shape)), -1)
        indices = (y_expanded > boundaries_expanded).sum(dim=-1)
        # Dequantize: look up centroid values
        return centroids[indices.long()]

    def _apply_turboquant_mse(
        self,
        x: torch.Tensor,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply TurboQuant MSE quantize→dequantize round-trip.

        Args:
            x: Input tensor of shape [num_tokens, num_heads, head_dim].
            signs: Random ±1 signs for Hadamard rotation.

        Returns:
            Quantized-then-dequantized tensor of same shape and dtype.
        """
        orig_dtype = x.dtype
        x_float = x.float()

        # Compute per-token-per-head norms
        norms = torch.norm(x_float, dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-10)
        x_normalized = x_float / safe_norms

        # Apply randomized Hadamard rotation: y = H @ diag(signs) @ x
        y = x_normalized * signs
        y = fast_walsh_hadamard_transform(y, normalize=True)

        if self.has_outliers:
            # Mixed-precision: higher bits for outlier channels
            y_q = torch.empty_like(y)
            y_q[..., self.normal_mask] = self._quantize_dequantize_1d(
                y[..., self.normal_mask],
                self.centroids_normal,
                self.boundaries_normal,
            )
            y_q[..., self.outlier_mask] = self._quantize_dequantize_1d(
                y[..., self.outlier_mask],
                self.centroids_outlier,
                self.boundaries_outlier,
            )
        else:
            y_q = self._quantize_dequantize_1d(
                y, self.centroids_normal, self.boundaries_normal
            )

        # Inverse rotation: x_hat = diag(signs) @ H @ y_q
        x_hat = fast_walsh_hadamard_transform(y_q, normalize=True)
        x_hat = x_hat * signs

        # Rescale by original norms
        x_hat = x_hat * norms

        return x_hat.to(orig_dtype)

    def compress_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply TurboQuant compression to K and V tensors.

        This applies quantize→dequantize round-trip, producing float
        tensors with TurboQuant's lossy compression applied.

        Args:
            key: Key tensor of shape [num_tokens, num_kv_heads, head_dim].
            value: Value tensor of shape [num_tokens, num_kv_heads, head_dim].

        Returns:
            Tuple of (compressed_key, compressed_value) with same shape/dtype.
        """
        key_compressed = self._apply_turboquant_mse(key, self.signs_k)
        value_compressed = self._apply_turboquant_mse(value, self.signs_v)
        return key_compressed, value_compressed

    @property
    def effective_bit_width(self) -> float:
        """Actual average bits per channel including outliers."""
        if self.has_outliers:
            return (
                self.n_normal * self.bits_normal + self.n_outlier * self.bits_outlier
            ) / self.head_dim
        return float(self.bits_normal)

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs FP16."""
        return 16.0 / self.effective_bit_width

    def __repr__(self) -> str:
        return (
            f"TurboQuantKVCompressor("
            f"bit_width={self.bit_width}, "
            f"head_dim={self.head_dim}, "
            f"outlier_channels={self.n_outlier}, "
            f"compression={self.compression_ratio:.1f}x)"
        )

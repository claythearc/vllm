# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for TurboQuant Phase 3.

Provides fused GPU kernels for TurboQuant KV cache quantization:
1. Fused quantize-dequantize (Phase 2 compatible round-trip)
2. Quantize to raw indices + norms (for packed storage)
3. Dequantize from raw indices + norms

These kernels fuse normalize + rotation + scalar quantize/dequantize +
inverse rotation + rescale into single GPU kernel launches, eliminating
intermediate memory allocations and multiple kernel launches from the
Phase 1 pure-PyTorch implementation.

The Walsh-Hadamard rotation is implemented via precomputed rotation
matrices and tl.dot matrix multiplication, trading O(d log d) butterfly
for O(d^2) matmul. For typical head_dim (64-256), this is efficient on
GPU tensor/CUDA cores and avoids complex in-register permutation logic.
"""

import math

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON

logger = init_logger(__name__)

if HAS_TRITON:
    import triton
    import triton.language as tl


def _make_hadamard_matrix(d: int, device: torch.device) -> torch.Tensor:
    """Construct the d x d Hadamard matrix using Sylvester's construction.

    H_1 = [[1]], H_{2n} = [[H_n, H_n], [H_n, -H_n]].
    Result is NOT normalized (caller should divide by sqrt(d) if needed).
    """
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < d:
        H = torch.cat(
            [
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ],
            dim=0,
        )
    return H


def make_rotation_matrices(
    signs: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute forward and inverse rotation matrices for Triton kernels.

    Forward transform: y = (1/sqrt(d)) * H @ diag(signs) @ x
    Inverse transform: x = diag(signs) @ (1/sqrt(d)) * H @ y

    For batched matmul (Y = X @ M):
      Forward: Y = X_norm @ M_fwd^T  (where M_fwd = H_norm @ diag(signs))
      Inverse: X_hat = Y_hat @ M_inv^T (where M_inv = diag(signs) @ H_norm)

    Since H is symmetric and M_fwd^T = M_inv, we store:
      M_fwd = H_norm * signs[None, :]   (column-wise multiply)
      M_inv = M_fwd^T                   (= signs[:, None] * H_norm)

    Returns:
        (M_fwd, M_inv) as float32 contiguous tensors of shape (d, d).
    """
    d = signs.shape[0]
    H_norm = _make_hadamard_matrix(d, device) / math.sqrt(d)

    # M_fwd[i,j] = H_norm[i,j] * signs[j]
    M_fwd = H_norm * signs.unsqueeze(0)
    # M_inv = M_fwd^T = diag(signs) @ H_norm
    M_inv = M_fwd.t().contiguous()
    M_fwd = M_fwd.contiguous()

    return M_fwd.float(), M_inv.float()


if HAS_TRITON:

    @triton.jit
    def _turboquant_fused_qd_kernel(
        # Input/output pointers
        X_ptr,
        Out_ptr,
        # Rotation matrix pointers (head_dim x head_dim, float32)
        M_fwd_ptr,
        M_inv_ptr,
        # Codebook pointers
        Centroids_ptr,
        Boundaries_ptr,
        # Outlier codebook pointers (unused if HAS_OUTLIERS=False)
        Centroids_out_ptr,
        Boundaries_out_ptr,
        # Dimensions
        num_tokens,
        head_dim,
        n_normal,
        # Strides for X/Out: layout (num_tokens, num_heads, head_dim)
        stride_tok,
        stride_head,
        # Strides for rotation matrices: row-major (head_dim, head_dim)
        stride_rot_row,
        stride_rot_col,
        # Constexpr config
        NUM_BOUNDS: tl.constexpr,
        NUM_BOUNDS_OUT: tl.constexpr,
        HAS_OUTLIERS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused TurboQuant quantize-dequantize kernel.

        Each program handles BLOCK_B tokens for one head.
        Grid: (cdiv(num_tokens, BLOCK_B), num_heads)
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        token_start = pid_b * BLOCK_B
        b_offs = tl.arange(0, BLOCK_B)
        d_offs = tl.arange(0, BLOCK_D)
        b_mask = (token_start + b_offs) < num_tokens

        # --- Load input block: (BLOCK_B, BLOCK_D) ---
        x_ptrs = (
            X_ptr
            + (token_start + b_offs[:, None]) * stride_tok
            + pid_h * stride_head
            + d_offs[None, :]
        )
        X = tl.load(x_ptrs, mask=b_mask[:, None], other=0.0).to(tl.float32)

        # --- Compute per-row L2 norms and normalize ---
        norms_sq = tl.sum(X * X, axis=1)
        norms = tl.sqrt(norms_sq + 1e-20)
        X_norm = X / norms[:, None]

        # --- Forward rotation: Y = X_norm @ M_inv ---
        # M_inv = M_fwd^T, so Y = X_norm @ M_fwd^T = forward rotation
        rot_inv_ptrs = (
            M_inv_ptr
            + d_offs[:, None] * stride_rot_row
            + d_offs[None, :] * stride_rot_col
        )
        M_inv = tl.load(rot_inv_ptrs)
        Y = tl.dot(X_norm, M_inv)

        # --- Scalar quantize-dequantize ---
        if HAS_OUTLIERS:
            is_normal = (d_offs < n_normal)[None, :]

            # Quantize normal channels
            idx_n = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.int32)
            for i in tl.static_range(NUM_BOUNDS):
                bnd = tl.load(Boundaries_ptr + i)
                idx_n += (bnd < Y).to(tl.int32)

            # Quantize outlier channels
            idx_o = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.int32)
            for i in tl.static_range(NUM_BOUNDS_OUT):
                bnd = tl.load(Boundaries_out_ptr + i)
                idx_o += (bnd < Y).to(tl.int32)

            # Dequantize via centroid lookup
            Y_hat_n = tl.load(Centroids_ptr + idx_n)
            Y_hat_o = tl.load(Centroids_out_ptr + idx_o)
            Y_hat = tl.where(is_normal, Y_hat_n, Y_hat_o)
        else:
            indices = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.int32)
            for i in tl.static_range(NUM_BOUNDS):
                bnd = tl.load(Boundaries_ptr + i)
                indices += (bnd < Y).to(tl.int32)
            Y_hat = tl.load(Centroids_ptr + indices)

        # --- Inverse rotation: X_hat = Y_hat @ M_fwd ---
        rot_fwd_ptrs = (
            M_fwd_ptr
            + d_offs[:, None] * stride_rot_row
            + d_offs[None, :] * stride_rot_col
        )
        M_fwd = tl.load(rot_fwd_ptrs)
        X_hat = tl.dot(Y_hat, M_fwd)

        # --- Rescale by original norms ---
        X_hat = X_hat * norms[:, None]

        # --- Store output ---
        out_ptrs = (
            Out_ptr
            + (token_start + b_offs[:, None]) * stride_tok
            + pid_h * stride_head
            + d_offs[None, :]
        )
        tl.store(out_ptrs, X_hat, mask=b_mask[:, None])

    @triton.jit
    def _turboquant_quantize_kernel(
        # Input/output pointers
        X_ptr,
        Indices_ptr,
        Norms_ptr,
        # Rotation matrix
        M_inv_ptr,
        # Codebook
        Centroids_ptr,
        Boundaries_ptr,
        Centroids_out_ptr,
        Boundaries_out_ptr,
        # Dimensions
        num_tokens,
        num_heads,
        head_dim,
        n_normal,
        # Input strides
        stride_tok,
        stride_head,
        # Index output strides
        stride_idx_tok,
        stride_idx_head,
        # Rotation matrix strides
        stride_rot_row,
        stride_rot_col,
        # Config
        NUM_BOUNDS: tl.constexpr,
        NUM_BOUNDS_OUT: tl.constexpr,
        HAS_OUTLIERS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Quantize kernel: outputs raw uint8 indices and float16 norms."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        token_start = pid_b * BLOCK_B
        b_offs = tl.arange(0, BLOCK_B)
        d_offs = tl.arange(0, BLOCK_D)
        b_mask = (token_start + b_offs) < num_tokens

        # Load input
        x_ptrs = (
            X_ptr
            + (token_start + b_offs[:, None]) * stride_tok
            + pid_h * stride_head
            + d_offs[None, :]
        )
        X = tl.load(x_ptrs, mask=b_mask[:, None], other=0.0).to(tl.float32)

        # Norms
        norms_sq = tl.sum(X * X, axis=1)
        norms = tl.sqrt(norms_sq + 1e-20)
        X_norm = X / norms[:, None]

        # Forward rotation
        rot_ptrs = (
            M_inv_ptr
            + d_offs[:, None] * stride_rot_row
            + d_offs[None, :] * stride_rot_col
        )
        M = tl.load(rot_ptrs)
        Y = tl.dot(X_norm, M)

        # Scalar quantize
        if HAS_OUTLIERS:
            is_normal = (d_offs < n_normal)[None, :]
            idx_n = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.int32)
            for i in tl.static_range(NUM_BOUNDS):
                bnd = tl.load(Boundaries_ptr + i)
                idx_n += (bnd < Y).to(tl.int32)
            idx_o = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.int32)
            for i in tl.static_range(NUM_BOUNDS_OUT):
                bnd = tl.load(Boundaries_out_ptr + i)
                idx_o += (bnd < Y).to(tl.int32)
            indices = tl.where(is_normal, idx_n, idx_o)
        else:
            indices = tl.zeros([BLOCK_B, BLOCK_D], dtype=tl.int32)
            for i in tl.static_range(NUM_BOUNDS):
                bnd = tl.load(Boundaries_ptr + i)
                indices += (bnd < Y).to(tl.int32)

        # Store indices
        idx_ptrs = (
            Indices_ptr
            + (token_start + b_offs[:, None]) * stride_idx_tok
            + pid_h * stride_idx_head
            + d_offs[None, :]
        )
        tl.store(idx_ptrs, indices.to(tl.uint8), mask=b_mask[:, None])

        # Store norms
        norm_ptrs = Norms_ptr + (token_start + b_offs) * num_heads + pid_h
        tl.store(norm_ptrs, norms.to(tl.float16), mask=b_mask)

    @triton.jit
    def _turboquant_dequantize_kernel(
        # Input/output pointers
        Indices_ptr,
        Norms_ptr,
        Out_ptr,
        # Rotation matrix
        M_fwd_ptr,
        # Codebook
        Centroids_ptr,
        Centroids_out_ptr,
        # Dimensions
        num_tokens,
        num_heads,
        head_dim,
        n_normal,
        # Output strides
        stride_out_tok,
        stride_out_head,
        # Index input strides
        stride_idx_tok,
        stride_idx_head,
        # Rotation matrix strides
        stride_rot_row,
        stride_rot_col,
        # Config
        HAS_OUTLIERS: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Dequantize kernel: from raw uint8 indices + norms to float."""
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)

        token_start = pid_b * BLOCK_B
        b_offs = tl.arange(0, BLOCK_B)
        d_offs = tl.arange(0, BLOCK_D)
        b_mask = (token_start + b_offs) < num_tokens

        # Load indices
        idx_ptrs = (
            Indices_ptr
            + (token_start + b_offs[:, None]) * stride_idx_tok
            + pid_h * stride_idx_head
            + d_offs[None, :]
        )
        indices = tl.load(idx_ptrs, mask=b_mask[:, None], other=0).to(tl.int32)

        # Load norms
        norm_ptrs = Norms_ptr + (token_start + b_offs) * num_heads + pid_h
        norms = tl.load(norm_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Dequantize: centroid lookup
        if HAS_OUTLIERS:
            is_normal = (d_offs < n_normal)[None, :]
            Y_hat_n = tl.load(Centroids_ptr + indices)
            Y_hat_o = tl.load(Centroids_out_ptr + indices)
            Y_hat = tl.where(is_normal, Y_hat_n, Y_hat_o)
        else:
            Y_hat = tl.load(Centroids_ptr + indices)

        # Inverse rotation: X_hat = Y_hat @ M_fwd
        rot_ptrs = (
            M_fwd_ptr
            + d_offs[:, None] * stride_rot_row
            + d_offs[None, :] * stride_rot_col
        )
        M_fwd = tl.load(rot_ptrs)
        X_hat = tl.dot(Y_hat, M_fwd)

        # Rescale
        X_hat = X_hat * norms[:, None]

        # Store
        out_ptrs = (
            Out_ptr
            + (token_start + b_offs[:, None]) * stride_out_tok
            + pid_h * stride_out_head
            + d_offs[None, :]
        )
        tl.store(out_ptrs, X_hat, mask=b_mask[:, None])


# ---------------------------------------------------------------------------
# Python wrapper functions
# ---------------------------------------------------------------------------


def turboquant_fused_qd(
    x: torch.Tensor,
    M_fwd: torch.Tensor,
    M_inv: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    centroids_outlier: torch.Tensor | None = None,
    boundaries_outlier: torch.Tensor | None = None,
    n_normal: int = 0,
) -> torch.Tensor:
    """Fused quantize-dequantize round-trip using Triton kernel.

    Args:
        x: Input (num_tokens, num_heads, head_dim), float16/bfloat16/float32.
        M_fwd: Forward rotation matrix (head_dim, head_dim), float32.
        M_inv: Inverse rotation matrix (head_dim, head_dim), float32.
        centroids: Normal-channel centroids (num_centroids,), float32.
        boundaries: Normal-channel boundaries (num_boundaries,), float32.
        centroids_outlier: Outlier-channel centroids (optional).
        boundaries_outlier: Outlier-channel boundaries (optional).
        n_normal: Number of normal (non-outlier) channels. 0 = all normal.

    Returns:
        Quantized-dequantized tensor, same shape as input, float32.
    """
    assert HAS_TRITON, "Triton is required for TurboQuant Triton kernels"
    num_tokens, num_heads, head_dim = x.shape

    # Output in float32 (caller casts to desired dtype)
    out = torch.empty(
        num_tokens,
        num_heads,
        head_dim,
        dtype=torch.float32,
        device=x.device,
    )

    has_outliers = centroids_outlier is not None
    num_boundaries = boundaries.shape[0]

    if not has_outliers:
        # Provide dummy pointers (never accessed when HAS_OUTLIERS=False)
        centroids_outlier = centroids
        boundaries_outlier = boundaries
        n_normal = head_dim

    assert boundaries_outlier is not None
    num_boundaries_outlier = boundaries_outlier.shape[0] if has_outliers else 1

    BLOCK_B = 16
    BLOCK_D = head_dim

    grid = (triton.cdiv(num_tokens, BLOCK_B), num_heads)

    _turboquant_fused_qd_kernel[grid](
        x,
        out,
        M_fwd,
        M_inv,
        centroids,
        boundaries,
        centroids_outlier,
        boundaries_outlier,
        num_tokens,
        head_dim,
        n_normal,
        x.stride(0),
        x.stride(1),
        M_fwd.stride(0),
        M_fwd.stride(1),
        NUM_BOUNDS=num_boundaries,
        NUM_BOUNDS_OUT=num_boundaries_outlier,
        HAS_OUTLIERS=has_outliers,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
    )

    return out


def turboquant_quantize(
    x: torch.Tensor,
    M_inv: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
    centroids_outlier: torch.Tensor | None = None,
    boundaries_outlier: torch.Tensor | None = None,
    n_normal: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to raw indices + norms using Triton kernel.

    Args:
        x: Input (num_tokens, num_heads, head_dim).
        M_inv: Inverse rotation matrix (head_dim, head_dim), float32.
        centroids: Normal-channel centroids, float32.
        boundaries: Normal-channel boundaries, float32.
        centroids_outlier: Outlier centroids (optional).
        boundaries_outlier: Outlier boundaries (optional).
        n_normal: Number of normal channels. 0 = all normal.

    Returns:
        (indices, norms):
          indices: uint8 tensor (num_tokens, num_heads, head_dim)
          norms: float16 tensor (num_tokens, num_heads)
    """
    assert HAS_TRITON, "Triton is required for TurboQuant Triton kernels"
    num_tokens, num_heads, head_dim = x.shape

    indices = torch.empty(
        num_tokens,
        num_heads,
        head_dim,
        dtype=torch.uint8,
        device=x.device,
    )
    norms = torch.empty(
        num_tokens,
        num_heads,
        dtype=torch.float16,
        device=x.device,
    )

    has_outliers = centroids_outlier is not None
    num_boundaries = boundaries.shape[0]

    if not has_outliers:
        centroids_outlier = centroids
        boundaries_outlier = boundaries
        n_normal = head_dim

    assert boundaries_outlier is not None
    num_boundaries_outlier = boundaries_outlier.shape[0] if has_outliers else 1

    BLOCK_B = 16
    BLOCK_D = head_dim

    grid = (triton.cdiv(num_tokens, BLOCK_B), num_heads)

    _turboquant_quantize_kernel[grid](
        x,
        indices,
        norms,
        M_inv,
        centroids,
        boundaries,
        centroids_outlier,
        boundaries_outlier,
        num_tokens,
        num_heads,
        head_dim,
        n_normal,
        x.stride(0),
        x.stride(1),
        indices.stride(0),
        indices.stride(1),
        M_inv.stride(0),
        M_inv.stride(1),
        NUM_BOUNDS=num_boundaries,
        NUM_BOUNDS_OUT=num_boundaries_outlier,
        HAS_OUTLIERS=has_outliers,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
    )

    return indices, norms


def turboquant_dequantize(
    indices: torch.Tensor,
    norms: torch.Tensor,
    M_fwd: torch.Tensor,
    centroids: torch.Tensor,
    centroids_outlier: torch.Tensor | None = None,
    n_normal: int = 0,
) -> torch.Tensor:
    """Dequantize from raw indices + norms using Triton kernel.

    Args:
        indices: uint8 tensor (num_tokens, num_heads, head_dim).
        norms: float16 tensor (num_tokens, num_heads).
        M_fwd: Forward rotation matrix (head_dim, head_dim), float32.
        centroids: Normal-channel centroids, float32.
        centroids_outlier: Outlier centroids (optional).
        n_normal: Number of normal channels. 0 = all normal.

    Returns:
        Reconstructed tensor (num_tokens, num_heads, head_dim), float32.
    """
    assert HAS_TRITON, "Triton is required for TurboQuant Triton kernels"
    num_tokens, num_heads, head_dim = indices.shape

    out = torch.empty(
        num_tokens,
        num_heads,
        head_dim,
        dtype=torch.float32,
        device=indices.device,
    )

    has_outliers = centroids_outlier is not None
    if not has_outliers:
        centroids_outlier = centroids
        n_normal = head_dim

    BLOCK_B = 16
    BLOCK_D = head_dim

    grid = (triton.cdiv(num_tokens, BLOCK_B), num_heads)

    _turboquant_dequantize_kernel[grid](
        indices,
        norms,
        out,
        M_fwd,
        centroids,
        centroids_outlier,
        num_tokens,
        num_heads,
        head_dim,
        n_normal,
        out.stride(0),
        out.stride(1),
        indices.stride(0),
        indices.stride(1),
        M_fwd.stride(0),
        M_fwd.stride(1),
        HAS_OUTLIERS=has_outliers,
        BLOCK_B=BLOCK_B,
        BLOCK_D=BLOCK_D,
    )

    return out

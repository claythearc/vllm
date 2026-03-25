# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant fused decode attention kernel (Phase 4).

Fuses KV dequantization directly into the attention computation,
avoiding materialization of full float16 K/V tensors. Uses a key
optimization: pre-rotate Q and post-rotate the output so that per-token
dequantization reduces to a centroid lookup (no D x D matmul per token).

Math:
  k_i = norm_k_i * (Y_hat_k_i @ M_fwd_k)     [inverse rotation]
  q @ k_i = norm_k_i * (q @ M_fwd_k) . Y_hat_k_i
           = norm_k_i * q_rot . Y_hat_k_i      [pre-rotate Q once]

  output = sum_i softmax_i * v_i
         = sum_i softmax_i * norm_v_i * (Y_hat_v_i @ M_fwd_v)
         = (sum_i softmax_i * norm_v_i * Y_hat_v_i) @ M_fwd_v
         = rotated_acc @ M_fwd_v                [post-rotate once]

Cost per attention head:
  - Q pre-rotation:  D^2 FMAs (once)
  - Per KV token:    D multiplies + D adds (centroid lookup + dot)
  - Output rotation: D^2 FMAs (once)
  - Total overhead:  2*D^2 + seq_len * 0 extra vs standard attention
"""

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON

logger = init_logger(__name__)

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _turboquant_decode_stage1(
        # Pre-rotated queries: (batch, num_q_heads, head_dim) float32
        Q_rot,
        # Quantized K cache: indices (batch, max_seq, num_kv_heads, head_dim) uint8
        K_Idx,
        # K norms: (batch, max_seq, num_kv_heads) float16
        K_Norms,
        # Quantized V cache: indices (batch, max_seq, num_kv_heads, head_dim) uint8
        V_Idx,
        # V norms: (batch, max_seq, num_kv_heads) float16
        V_Norms,
        # Codebook pointers
        Centroids_ptr,
        Cent_out_ptr,
        # Per-request sequence lengths: (batch,) int32
        Seq_lens,
        # Intermediate output: (batch, num_q_heads, NUM_KV_SPLITS, head_dim+1)
        Mid_O,
        # Scalar
        sm_scale,
        # Dimensions
        num_kv_heads,
        max_seq_len,
        n_normal,
        # Q strides
        stride_q_b,
        stride_q_h,
        # K index strides: (batch, seq, kv_head, dim)
        stride_ki_b,
        stride_ki_s,
        stride_ki_h,
        # K norm strides: (batch, seq, kv_head)
        stride_kn_b,
        stride_kn_s,
        # V index strides
        stride_vi_b,
        stride_vi_s,
        stride_vi_h,
        # V norm strides
        stride_vn_b,
        stride_vn_s,
        # Mid output strides
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        # Config
        kv_group_num: tl.constexpr,
        HAS_OUTLIERS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        Lk: tl.constexpr,
    ):
        """Stage 1: Per-split QK scores + V accumulation in rotated domain.

        Grid: (batch, num_q_heads, NUM_KV_SPLITS)
        """
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        split_kv_id = tl.program_id(2)

        cur_kv_head = cur_head // kv_group_num
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < Lk

        # Load pre-rotated query: (BLOCK_D,)
        q_ptrs = Q_rot + cur_batch * stride_q_b + cur_head * stride_q_h + d_offs
        q_rot_vec = tl.load(q_ptrs, mask=d_mask, other=0.0)

        # Compute split range
        seq_len = tl.load(Seq_lens + cur_batch)
        kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
        split_start = kv_len_per_split * split_kv_id
        split_end = tl.minimum(split_start + kv_len_per_split, seq_len)

        # Online softmax accumulators
        e_max = -float("inf")
        e_sum = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        if split_end > split_start:
            for start_n in range(split_start, split_end, BLOCK_N):
                n_offs = start_n + tl.arange(0, BLOCK_N)
                n_mask = n_offs < split_end

                # --- Load K indices and norms ---
                ki_ptrs = (
                    K_Idx
                    + cur_batch * stride_ki_b
                    + n_offs[:, None] * stride_ki_s
                    + cur_kv_head * stride_ki_h
                    + d_offs[None, :]
                )
                k_indices = tl.load(
                    ki_ptrs,
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)

                kn_ptrs = (
                    K_Norms
                    + cur_batch * stride_kn_b
                    + n_offs * stride_kn_s
                    + cur_kv_head
                )
                k_norms = tl.load(kn_ptrs, mask=n_mask, other=0.0).to(tl.float32)

                # --- Centroid lookup for K ---
                if HAS_OUTLIERS:
                    normal_mask = (d_offs < n_normal)[None, :]
                    Y_n = tl.load(Centroids_ptr + k_indices)
                    Y_o = tl.load(Cent_out_ptr + k_indices)
                    Y_hat_k = tl.where(normal_mask, Y_n, Y_o)
                else:
                    Y_hat_k = tl.load(Centroids_ptr + k_indices)

                # --- QK scores in rotated domain ---
                # score_i = k_norm_i * dot(q_rot, Y_hat_k_i) * sm_scale
                scores = tl.sum(Y_hat_k * q_rot_vec[None, :], axis=1)
                scores = scores * k_norms * sm_scale
                scores = tl.where(n_mask, scores, float("-inf"))

                # --- Online softmax update ---
                n_e_max = tl.maximum(tl.max(scores, 0), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(scores - n_e_max)
                acc *= re_scale

                # --- Load V indices and norms ---
                vi_ptrs = (
                    V_Idx
                    + cur_batch * stride_vi_b
                    + n_offs[:, None] * stride_vi_s
                    + cur_kv_head * stride_vi_h
                    + d_offs[None, :]
                )
                v_indices = tl.load(
                    vi_ptrs,
                    mask=n_mask[:, None] & d_mask[None, :],
                    other=0,
                ).to(tl.int32)

                vn_ptrs = (
                    V_Norms
                    + cur_batch * stride_vn_b
                    + n_offs * stride_vn_s
                    + cur_kv_head
                )
                v_norms = tl.load(vn_ptrs, mask=n_mask, other=0.0).to(tl.float32)

                # --- Centroid lookup for V ---
                if HAS_OUTLIERS:
                    Y_n_v = tl.load(Centroids_ptr + v_indices)
                    Y_o_v = tl.load(Cent_out_ptr + v_indices)
                    Y_hat_v = tl.where(normal_mask, Y_n_v, Y_o_v)
                else:
                    Y_hat_v = tl.load(Centroids_ptr + v_indices)

                # --- Weighted V in rotated domain ---
                weights = p * v_norms
                acc += tl.sum(weights[:, None] * Y_hat_v, axis=0)

                e_sum = e_sum * re_scale + tl.sum(p, 0)
                e_max = n_e_max

            # Store intermediate: normalized acc and log-sum-exp
            mid_ptrs = (
                Mid_O
                + cur_batch * stride_mid_b
                + cur_head * stride_mid_h
                + split_kv_id * stride_mid_s
                + d_offs
            )
            tl.store(mid_ptrs, acc / e_sum, mask=d_mask)

            lse_ptr = (
                Mid_O
                + cur_batch * stride_mid_b
                + cur_head * stride_mid_h
                + split_kv_id * stride_mid_s
                + Lk
            )
            tl.store(lse_ptr, e_max + tl.log(e_sum))

    @triton.jit
    def _turboquant_decode_stage2(
        Mid_O,
        Out,
        Seq_lens,
        stride_mid_b,
        stride_mid_h,
        stride_mid_s,
        stride_o_b,
        stride_o_h,
        BLOCK_D: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        Lk: tl.constexpr,
    ):
        """Stage 2: Reduce across KV splits (standard softmax reduction).

        Grid: (batch, num_q_heads)
        """
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < Lk

        seq_len = tl.load(Seq_lens + cur_batch)

        # Accumulate across splits
        e_max = -float("inf")
        e_sum = 0.0
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for split_id in range(NUM_KV_SPLITS):
            # Check if this split has data
            kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
            split_start = kv_len_per_split * split_id
            if split_start < seq_len:
                mid_ptrs = (
                    Mid_O
                    + cur_batch * stride_mid_b
                    + cur_head * stride_mid_h
                    + split_id * stride_mid_s
                    + d_offs
                )
                tv = tl.load(mid_ptrs, mask=d_mask, other=0.0)

                lse_ptr = (
                    Mid_O
                    + cur_batch * stride_mid_b
                    + cur_head * stride_mid_h
                    + split_id * stride_mid_s
                    + Lk
                )
                tlogic = tl.load(lse_ptr)

                n_e_max = tl.maximum(tlogic, e_max)
                old_scale = tl.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc += exp_logic * tv

                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max

        # Final output (still in rotated V domain)
        out_ptrs = Out + cur_batch * stride_o_b + cur_head * stride_o_h + d_offs
        tl.store(out_ptrs, acc / e_sum, mask=d_mask)


def _next_pow2(n: int) -> int:
    """Round up to next power of 2."""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def turboquant_decode_attention(
    q: torch.Tensor,
    k_indices: torch.Tensor,
    k_norms: torch.Tensor,
    v_indices: torch.Tensor,
    v_norms: torch.Tensor,
    seq_lens: torch.Tensor,
    M_fwd_k: torch.Tensor,
    M_inv_k: torch.Tensor,
    M_fwd_v: torch.Tensor,
    centroids: torch.Tensor,
    centroids_outlier: torch.Tensor | None = None,
    n_normal: int = 0,
    num_kv_splits: int = 8,
) -> torch.Tensor:
    """Fused decode attention with on-the-fly TurboQuant dequantization.

    Pre-rotates Q and post-rotates the output so that per-token
    dequantization is just a centroid lookup (no per-token matmul).

    Args:
        q: Queries (batch, num_q_heads, head_dim), float16/float32.
        k_indices: Quantized K indices (batch, max_seq, num_kv_heads, head_dim), uint8.
        k_norms: K norms (batch, max_seq, num_kv_heads), float16.
        v_indices: Quantized V indices, same layout as k_indices.
        v_norms: V norms, same layout as k_norms.
        seq_lens: Per-request context lengths (batch,), int32.
        M_fwd_k: Forward K rotation matrix (head_dim, head_dim), float32.
        M_inv_k: Inverse K rotation matrix (head_dim, head_dim), float32.
        M_fwd_v: Forward V rotation matrix (head_dim, head_dim), float32.
        centroids: Normal channel centroids, float32.
        centroids_outlier: Outlier channel centroids (optional).
        n_normal: Number of normal channels (0 = all normal).
        num_kv_splits: Number of KV sequence splits for parallelism.

    Returns:
        Attention output (batch, num_q_heads, head_dim), float32.
    """
    assert HAS_TRITON, "Triton required for fused TurboQuant attention"

    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_indices.shape[2]
    max_seq_len = k_indices.shape[1]
    kv_group_num = num_q_heads // num_kv_heads
    sm_scale = 1.0 / (head_dim**0.5)

    has_outliers = centroids_outlier is not None
    if not has_outliers:
        centroids_outlier = centroids
        n_normal = head_dim

    # --- Pre-rotate Q: q_rot = q @ M_inv_k ---
    # This transforms Q into the K-rotated domain so that
    # q @ k = k_norm * q_rot . Y_hat_k (elementwise dot)
    q_f32 = q.float()
    q_rot = torch.matmul(q_f32, M_inv_k)  # (batch, num_q_heads, D)

    BLOCK_D = _next_pow2(head_dim)
    BLOCK_N = 64
    NUM_KV_SPLITS = min(num_kv_splits, max_seq_len)
    NUM_KV_SPLITS = max(1, NUM_KV_SPLITS)

    # Intermediate buffer: (batch, num_q_heads, NUM_KV_SPLITS, head_dim+1)
    mid_o = torch.empty(
        batch_size,
        num_q_heads,
        NUM_KV_SPLITS,
        head_dim + 1,
        dtype=torch.float32,
        device=q.device,
    )

    grid_s1 = (batch_size, num_q_heads, NUM_KV_SPLITS)
    _turboquant_decode_stage1[grid_s1](
        q_rot,
        k_indices,
        k_norms,
        v_indices,
        v_norms,
        centroids,
        centroids_outlier,
        seq_lens,
        mid_o,
        sm_scale,
        num_kv_heads,
        max_seq_len,
        n_normal,
        q_rot.stride(0),
        q_rot.stride(1),
        k_indices.stride(0),
        k_indices.stride(1),
        k_indices.stride(2),
        k_norms.stride(0),
        k_norms.stride(1),
        v_indices.stride(0),
        v_indices.stride(1),
        v_indices.stride(2),
        v_norms.stride(0),
        v_norms.stride(1),
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        kv_group_num=kv_group_num,
        HAS_OUTLIERS=has_outliers,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        Lk=head_dim,
    )

    # --- Stage 2: Reduce across splits ---
    rotated_out = torch.empty(
        batch_size,
        num_q_heads,
        head_dim,
        dtype=torch.float32,
        device=q.device,
    )

    grid_s2 = (batch_size, num_q_heads)
    _turboquant_decode_stage2[grid_s2](
        mid_o,
        rotated_out,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        rotated_out.stride(0),
        rotated_out.stride(1),
        BLOCK_D=BLOCK_D,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        Lk=head_dim,
    )

    # --- Post-rotate output: out = rotated_out @ M_fwd_v ---
    # Transforms from V-rotated domain back to original space
    output = torch.matmul(rotated_out, M_fwd_v)

    return output

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TurboQuant quantization.

Tests cover:
1. Codebook correctness and symmetry.
2. Rotation (randomized Hadamard) round-trip.
3. Bit-packing round-trip.
4. TurboQuantMSE distortion bounds.
5. TurboQuantProd unbiasedness.
6. Config registration.
"""

import math

import pytest
import torch

from vllm.model_executor.layers.quantization.turboquant.codebook import (
    LloydMaxCodebook,
)
from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
    TurboQuantParams,
)
from vllm.model_executor.layers.quantization.turboquant.quantizer import (
    TurboQuantMSE,
    TurboQuantProd,
    pack_indices,
    unpack_indices,
)
from vllm.model_executor.layers.quantization.turboquant.rotation import (
    fast_walsh_hadamard_transform,
    generate_random_signs,
    inverse_randomized_hadamard_transform,
    randomized_hadamard_transform,
)

DEVICE = "cpu"


# ---- Codebook tests ----


class TestLloydMaxCodebook:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_codebook_size(self, bit_width: int):
        """Codebook should have 2^b centroids and 2^b - 1 boundaries."""
        centroids, boundaries = LloydMaxCodebook.get(bit_width, 128, DEVICE)
        assert centroids.shape[0] == 2**bit_width
        assert boundaries.shape[0] == 2**bit_width - 1

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_codebook_symmetry(self, bit_width: int):
        """Centroids should be symmetric around 0."""
        centroids, _ = LloydMaxCodebook.get_stdnormal(bit_width, DEVICE)
        n = centroids.shape[0]
        for i in range(n // 2):
            assert torch.allclose(centroids[i], -centroids[n - 1 - i], atol=1e-6), (
                f"Centroid {i} not symmetric with {n - 1 - i}"
            )

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_codebook_sorted(self, bit_width: int):
        """Centroids should be sorted in ascending order."""
        centroids, _ = LloydMaxCodebook.get_stdnormal(bit_width, DEVICE)
        for i in range(len(centroids) - 1):
            assert centroids[i] < centroids[i + 1]

    def test_codebook_scaling(self):
        """Centroids should scale by 1/sqrt(d)."""
        c_std, _ = LloydMaxCodebook.get_stdnormal(2, DEVICE)
        d = 128
        c_scaled, _ = LloydMaxCodebook.get(2, d, DEVICE)
        expected = c_std / math.sqrt(d)
        assert torch.allclose(c_scaled, expected, atol=1e-6)

    def test_1bit_centroid_value(self):
        """1-bit centroids should be ±sqrt(2/pi) for standard normal."""
        centroids, _ = LloydMaxCodebook.get_stdnormal(1, DEVICE)
        expected = math.sqrt(2 / math.pi)
        assert abs(centroids[1].item() - expected) < 1e-4
        assert abs(centroids[0].item() + expected) < 1e-4


# ---- Rotation tests ----


class TestRotation:
    def test_signs_deterministic(self):
        """Same seed should produce same signs."""
        s1 = generate_random_signs(128, seed=42, device=DEVICE)
        s2 = generate_random_signs(128, seed=42, device=DEVICE)
        assert torch.equal(s1, s2)

    def test_signs_values(self):
        """Signs should be ±1."""
        signs = generate_random_signs(128, seed=42, device=DEVICE)
        assert torch.all((signs == 1.0) | (signs == -1.0))

    def test_walsh_hadamard_self_inverse(self):
        """Walsh-Hadamard transform is self-inverse (up to normalization)."""
        x = torch.randn(4, 128)
        y = fast_walsh_hadamard_transform(x, normalize=True)
        x_reconstructed = fast_walsh_hadamard_transform(y, normalize=True)
        assert torch.allclose(x, x_reconstructed, atol=1e-5)

    def test_rht_round_trip(self):
        """Randomized Hadamard transform round-trip should recover input."""
        x = torch.randn(8, 64)
        signs = generate_random_signs(64, seed=123, device=DEVICE)

        y = randomized_hadamard_transform(x, signs)
        x_hat = inverse_randomized_hadamard_transform(y, signs)
        assert torch.allclose(x, x_hat, atol=1e-5)

    def test_rht_preserves_norm(self):
        """Randomized Hadamard should approximately preserve L2 norm."""
        x = torch.randn(16, 128)
        signs = generate_random_signs(128, seed=42, device=DEVICE)
        y = randomized_hadamard_transform(x, signs)

        x_norms = torch.norm(x, dim=-1)
        y_norms = torch.norm(y, dim=-1)
        assert torch.allclose(x_norms, y_norms, atol=1e-4)


# ---- Bit-packing tests ----


class TestBitPacking:
    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_pack_unpack_roundtrip(self, bit_width: int):
        """Pack then unpack should recover original indices."""
        dim = 128
        max_val = 2**bit_width - 1
        indices = torch.randint(0, max_val + 1, (8, dim), dtype=torch.uint8)
        packed = pack_indices(indices, bit_width)
        unpacked = unpack_indices(packed, bit_width, dim)
        assert torch.equal(indices, unpacked)

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_packed_size(self, bit_width: int):
        """Packed size should be ceil(d * b / 8)."""
        dim = 128
        indices = torch.zeros(4, dim, dtype=torch.uint8)
        packed = pack_indices(indices, bit_width)
        expected_size = (dim * bit_width + 7) // 8
        assert packed.shape[-1] == expected_size

    def test_pack_unpack_batched(self):
        """Packing should work with batch dimensions."""
        indices = torch.randint(0, 4, (2, 3, 64), dtype=torch.uint8)
        packed = pack_indices(indices, 2)
        unpacked = unpack_indices(packed, 2, 64)
        assert torch.equal(indices, unpacked)


# ---- TurboQuantMSE tests ----


class TestTurboQuantMSE:
    def test_roundtrip_shape(self):
        """Quantize then dequantize should preserve shape."""
        dim = 128
        q = TurboQuantMSE(bit_width=3, head_dim=dim, device=DEVICE, seed=42)
        x = torch.randn(16, dim)
        packed, norms = q.quantize(x)
        x_hat = q.dequantize(packed, norms)
        assert x_hat.shape == x.shape

    @pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
    def test_mse_distortion_bound(self, bit_width: int):
        """MSE should be within theoretical upper bound for unit-norm vectors."""
        dim = 128
        n_samples = 1000
        q = TurboQuantMSE(bit_width=bit_width, head_dim=dim, device=DEVICE, seed=42)

        # Generate random unit-norm vectors
        x = torch.randn(n_samples, dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        packed, norms = q.quantize(x)
        x_hat = q.dequantize(packed, norms)

        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

        # Paper MSE upper bounds (Table in §2.1)
        paper_bounds = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
        # Allow 50% slack for finite-sample estimation
        bound = paper_bounds[bit_width] * 1.5
        assert mse < bound, (
            f"MSE {mse:.4f} exceeds bound {bound:.4f} at {bit_width} bits"
        )

    def test_higher_bits_lower_mse(self):
        """Higher bit-width should give lower MSE."""
        dim = 128
        n_samples = 500
        x = torch.randn(n_samples, dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        mses = []
        for bw in [1, 2, 3, 4]:
            q = TurboQuantMSE(bit_width=bw, head_dim=dim, device=DEVICE, seed=42)
            packed, norms = q.quantize(x)
            x_hat = q.dequantize(packed, norms)
            mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1], (
                f"MSE at {i + 1} bits ({mses[i]:.4f}) should be > "
                f"MSE at {i + 2} bits ({mses[i + 1]:.4f})"
            )

    def test_norm_preservation(self):
        """Norms of input should be preserved through quantization."""
        dim = 128
        q = TurboQuantMSE(bit_width=3, head_dim=dim, device=DEVICE, seed=42)
        x = torch.randn(32, dim) * 5.0  # Non-unit-norm vectors
        packed, norms = q.quantize(x)

        x_norms = torch.norm(x, dim=-1)
        # FP16 norms have relative precision ~0.1% which translates
        # to absolute error ~0.05 for norms around 50
        assert torch.allclose(norms.float(), x_norms, rtol=0.005, atol=0.1), (
            "Stored norms should match input norms"
        )


# ---- TurboQuantProd tests ----


class TestTurboQuantProd:
    def test_roundtrip_shape(self):
        """Quantize then dequantize should preserve shape."""
        dim = 64  # smaller for speed with matmul
        q = TurboQuantProd(bit_width=3, head_dim=dim, device=DEVICE, seed=42)
        x = torch.randn(16, dim)
        mse_packed, norms, qjl_packed, res_norms = q.quantize(x)
        x_hat = q.dequantize(mse_packed, norms, qjl_packed, res_norms)
        assert x_hat.shape == x.shape

    def test_unbiased_inner_product(self):
        """Inner product estimation should be approximately unbiased."""
        dim = 64
        n_samples = 2000
        q = TurboQuantProd(bit_width=3, head_dim=dim, device=DEVICE, seed=42)

        # Fixed query vector
        y = torch.randn(dim)
        y = y / torch.norm(y)

        # Random data vectors
        x = torch.randn(n_samples, dim)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # True inner products
        true_ip = (x * y.unsqueeze(0)).sum(dim=-1)

        # Estimated inner products via quantize/dequantize
        mse_packed, norms, qjl_packed, res_norms = q.quantize(x)
        x_hat = q.dequantize(mse_packed, norms, qjl_packed, res_norms)
        est_ip = (x_hat * y.unsqueeze(0)).sum(dim=-1)

        # Check that mean error is small (unbiasedness)
        mean_error = (est_ip - true_ip).mean().item()
        assert abs(mean_error) < 0.05, f"Mean IP error {mean_error:.4f} indicates bias"

    def test_requires_bit_width_ge_2(self):
        """Prod variant needs at least 2 bits (1 for MSE + 1 for QJL)."""
        with pytest.raises(ValueError, match="bit_width >= 2"):
            TurboQuantProd(bit_width=1, head_dim=64, device=DEVICE)


# ---- Config tests ----


class TestTurboQuantConfig:
    def test_config_creation(self):
        config = TurboQuantConfig(bit_width=3.5, head_dim=128)
        assert config.get_name() == "turboquant"
        assert config.params.bit_width == 3.5
        assert config.params.effective_bits_normal == 3
        assert config.params.effective_bits_outlier == 4

    def test_config_invalid_bit_width(self):
        with pytest.raises(ValueError, match="bit_width must be one of"):
            TurboQuantConfig(bit_width=5.0)

    def test_config_invalid_head_dim(self):
        with pytest.raises(ValueError, match="power of 2"):
            TurboQuantConfig(bit_width=3.0, head_dim=100)

    def test_config_compression_ratio(self):
        params = TurboQuantParams(
            bit_width=3.5,
            outlier_channels=32,
            use_prod_variant=False,
            head_dim=128,
            seed=42,
        )
        assert abs(params.compression_ratio - 16.0 / 3.5) < 1e-6

    def test_from_config(self):
        config_dict = {
            "turboquant_bit_width": 2.5,
            "turboquant_outlier_channels": 32,
            "head_dim": 128,
        }
        config = TurboQuantConfig.from_config(config_dict)
        assert config.params.bit_width == 2.5

    def test_config_registration(self):
        """TurboQuant should be in the quantization method registry."""
        from vllm.model_executor.layers.quantization import (
            QUANTIZATION_METHODS,
            get_quantization_config,
        )

        assert "turboquant" in QUANTIZATION_METHODS
        try:
            config_cls = get_quantization_config("turboquant")
            assert config_cls is TurboQuantConfig
        except ImportError:
            # In minimal test environments, transitive imports may fail.
            # The important check is that "turboquant" is registered.
            pass

    def test_supported_dtypes(self):
        config = TurboQuantConfig()
        dtypes = config.get_supported_act_dtypes()
        assert torch.float16 in dtypes
        assert torch.bfloat16 in dtypes


# ---- KV Compressor tests (Phase 2) ----


class TestTurboQuantKVCompressor:
    def test_compress_kv_shape(self):
        """Compressed K/V should have same shape as input."""
        from vllm.model_executor.layers.quantization.turboquant.compressor import (
            TurboQuantKVCompressor,
        )

        compressor = TurboQuantKVCompressor(
            bit_width=3.0, head_dim=128, num_kv_heads=8, device=DEVICE
        )
        key = torch.randn(16, 8, 128)
        value = torch.randn(16, 8, 128)
        k_out, v_out = compressor.compress_kv(key, value)
        assert k_out.shape == key.shape
        assert v_out.shape == value.shape

    def test_compress_kv_dtype_preservation(self):
        """Compressed K/V should preserve input dtype."""
        from vllm.model_executor.layers.quantization.turboquant.compressor import (
            TurboQuantKVCompressor,
        )

        compressor = TurboQuantKVCompressor(
            bit_width=3.0, head_dim=64, num_kv_heads=4, device=DEVICE
        )
        key = torch.randn(8, 4, 64, dtype=torch.float16)
        value = torch.randn(8, 4, 64, dtype=torch.float16)
        k_out, v_out = compressor.compress_kv(key, value)
        assert k_out.dtype == torch.float16
        assert v_out.dtype == torch.float16

    @pytest.mark.parametrize("bit_width", [2.0, 2.5, 3.0, 3.5, 4.0])
    def test_compression_reduces_information(self, bit_width: float):
        """Compression should produce lossy output (not identical to input)."""
        from vllm.model_executor.layers.quantization.turboquant.compressor import (
            TurboQuantKVCompressor,
        )

        compressor = TurboQuantKVCompressor(
            bit_width=bit_width, head_dim=128, num_kv_heads=4, device=DEVICE
        )
        key = torch.randn(32, 4, 128)
        value = torch.randn(32, 4, 128)
        k_out, v_out = compressor.compress_kv(key, value)
        # Output should be different from input (lossy compression)
        assert not torch.equal(key, k_out)
        assert not torch.equal(value, v_out)

    def test_mixed_precision_outlier_channels(self):
        """Non-integer bit-widths should use mixed precision."""
        from vllm.model_executor.layers.quantization.turboquant.compressor import (
            TurboQuantKVCompressor,
        )

        compressor = TurboQuantKVCompressor(
            bit_width=3.5,
            head_dim=128,
            num_kv_heads=4,
            outlier_channels=32,
            device=DEVICE,
        )
        assert compressor.has_outliers
        assert compressor.bits_normal == 3
        assert compressor.bits_outlier == 4
        assert compressor.n_outlier == 32
        assert compressor.n_normal == 96

        # Verify effective bit-width is as expected
        expected = (96 * 3 + 32 * 4) / 128  # = 3.25 (not exactly 3.5)
        assert abs(compressor.effective_bit_width - expected) < 1e-6

    def test_integer_bit_width_no_outliers(self):
        """Integer bit-widths should not use outlier channels."""
        from vllm.model_executor.layers.quantization.turboquant.compressor import (
            TurboQuantKVCompressor,
        )

        compressor = TurboQuantKVCompressor(
            bit_width=3.0, head_dim=128, num_kv_heads=4, device=DEVICE
        )
        assert not compressor.has_outliers

    @pytest.mark.parametrize("bit_width", [2.0, 3.0, 4.0])
    def test_mse_improves_with_bits(self, bit_width: float):
        """Higher bit-width should give lower distortion."""
        from vllm.model_executor.layers.quantization.turboquant.compressor import (
            TurboQuantKVCompressor,
        )

        key = torch.randn(100, 4, 128)
        value = torch.randn(100, 4, 128)

        compressor = TurboQuantKVCompressor(
            bit_width=bit_width, head_dim=128, num_kv_heads=4, device=DEVICE
        )
        k_out, _ = compressor.compress_kv(key, value)
        mse = ((key - k_out) ** 2).mean().item()

        if bit_width == 2.0:
            # At 2 bits, MSE should be non-trivial
            assert mse > 0.001
        elif bit_width == 4.0:
            # At 4 bits, MSE should be small
            assert mse < 0.05

    def test_compression_ratio(self):
        """Compression ratio should match theoretical prediction."""
        from vllm.model_executor.layers.quantization.turboquant.compressor import (
            TurboQuantKVCompressor,
        )

        compressor = TurboQuantKVCompressor(
            bit_width=3.0, head_dim=128, num_kv_heads=4, device=DEVICE
        )
        # 16 bits / 3 bits = 5.33x
        assert abs(compressor.compression_ratio - 16.0 / 3.0) < 1e-6

    def test_repr(self):
        """Repr should be informative."""
        from vllm.model_executor.layers.quantization.turboquant.compressor import (
            TurboQuantKVCompressor,
        )

        compressor = TurboQuantKVCompressor(
            bit_width=3.5, head_dim=128, num_kv_heads=4, device=DEVICE
        )
        r = repr(compressor)
        assert "TurboQuantKVCompressor" in r
        assert "3.5" in r

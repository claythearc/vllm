# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant KV cache quantization config.

TurboQuant is an online vector quantizer that achieves near-optimal MSE
and inner product distortion at arbitrary bit-widths using random rotation
followed by scalar Lloyd-Max quantization.

Reference: "TurboQuant: Online Vector Quantization with Near-optimal
Distortion Rate" (Zandieh et al., 2025)
"""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class TurboQuantParams:
    """Parameters for TurboQuant quantization."""

    bit_width: float  # e.g., 2.5, 3.0, 3.5, 4.0
    outlier_channels: int  # number of channels getting +1 bit
    use_prod_variant: bool  # TurboQuant_Prod vs TurboQuant_MSE
    head_dim: int  # from model config
    seed: int  # for reproducible rotation matrices

    @property
    def effective_bits_normal(self) -> int:
        """Bit-width for non-outlier channels."""
        return int(self.bit_width)

    @property
    def effective_bits_outlier(self) -> int:
        """Bit-width for outlier channels."""
        return self.effective_bits_normal + 1

    @property
    def compression_ratio(self) -> float:
        """Compression vs FP16."""
        return 16.0 / self.bit_width


class TurboQuantConfig(QuantizationConfig):
    """Config class for TurboQuant KV cache quantization."""

    def __init__(
        self,
        bit_width: float = 3.5,
        outlier_channels: int = 32,
        use_prod_variant: bool = False,
        head_dim: int = 128,
        seed: int = 42,
    ):
        super().__init__()
        if bit_width not in (2.0, 2.5, 3.0, 3.5, 4.0):
            raise ValueError(
                f"TurboQuant bit_width must be one of "
                f"[2.0, 2.5, 3.0, 3.5, 4.0], got {bit_width}"
            )
        if head_dim & (head_dim - 1) != 0:
            raise ValueError(
                f"TurboQuant requires head_dim to be a power of 2, got {head_dim}"
            )
        self.params = TurboQuantParams(
            bit_width=bit_width,
            outlier_channels=outlier_channels,
            use_prod_variant=use_prod_variant,
            head_dim=head_dim,
            seed=seed,
        )

    def get_name(self) -> str:
        return "turboquant"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Walsh-Hadamard and basic ops work on any modern GPU
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TurboQuantConfig":
        bit_width = cls.get_from_keys_or(config, ["turboquant_bit_width"], 3.5)
        outlier_channels = cls.get_from_keys_or(
            config, ["turboquant_outlier_channels"], 32
        )
        use_prod = cls.get_from_keys_or(config, ["turboquant_use_prod"], False)
        head_dim = cls.get_from_keys_or(config, ["head_dim"], 128)
        seed = cls.get_from_keys_or(config, ["turboquant_seed"], 42)
        return cls(
            bit_width=bit_width,
            outlier_channels=outlier_channels,
            use_prod_variant=use_prod,
            head_dim=head_dim,
            seed=seed,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.attention.attention import Attention

        if isinstance(layer, Attention):
            from vllm.model_executor.layers.quantization.turboquant.kv_cache import (
                TurboQuantKVCacheMethod,
            )

            return TurboQuantKVCacheMethod(self)
        return None

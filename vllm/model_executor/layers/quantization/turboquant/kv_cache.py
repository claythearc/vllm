# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant KV cache method for vLLM integration.

Provides TurboQuantKVCacheMethod which extends BaseKVCacheMethod to
support TurboQuant-compressed KV cache storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.kv_cache import (
    BaseKVCacheMethod,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

logger = init_logger(__name__)


class TurboQuantKVCacheMethod(BaseKVCacheMethod):
    """KV cache quantization method using TurboQuant.

    This method stores quantized KV cache entries as packed bit indices
    plus per-token norms, achieving significant compression (4-6x) over
    FP16 with minimal quality degradation.

    The quantization is applied online — no calibration data is needed.
    Each KV vector is:
    1. Normalized (norm stored as FP16 scalar).
    2. Rotated via Randomized Hadamard Transform.
    3. Scalar-quantized using precomputed Lloyd-Max codebook.
    4. Bit-packed for storage.

    Dequantization reverses these steps.
    """

    def __init__(self, quant_config: TurboQuantConfig):
        super().__init__(quant_config)
        self.turboquant_config = quant_config
        self._quantizers: dict[str, object] = {}

    def create_weights(self, layer: torch.nn.Module) -> None:
        """Create scale parameters for compatibility with attention backends.

        TurboQuant doesn't use FP8-style scales, but we register them
        for compatibility with the BaseKVCacheMethod interface.
        """
        super().create_weights(layer)

        # Store TurboQuant params on the layer for access during forward
        layer.turboquant_params = self.turboquant_config.params

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Initialize TurboQuant state after model loading.

        Sets default scales (1.0) since TurboQuant doesn't use FP8 scales,
        and stores the config params on the layer for runtime access.
        """
        # skip if there are no weights to process
        if not hasattr(layer, "q_scale"):
            return

        # Set all scales to 1.0 since TurboQuant doesn't use them
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

        layer._k_scale_float = 1.0
        layer._v_scale_float = 1.0
        layer._q_scale_float = 1.0

        # Clean up the Parameter versions
        if hasattr(layer, "k_scale"):
            del layer.k_scale
        if hasattr(layer, "v_scale"):
            del layer.v_scale
        if hasattr(layer, "q_scale"):
            del layer.q_scale
        if hasattr(layer, "prob_scale"):
            del layer.prob_scale

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(
            "TurboQuantKVCacheMethod.apply should not be called directly."
        )

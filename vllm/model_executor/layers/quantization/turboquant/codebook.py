# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precomputed Lloyd-Max codebooks for TurboQuant.

For dimension d >= 64, the coordinate distribution after random rotation
of unit-norm vectors is well-approximated by N(0, 1/d). The codebook
depends only on b (bit-width), with centroids scaled by 1/sqrt(d).

Reference centroids are for the standard normal N(0,1). At runtime they
are scaled by 1/sqrt(d) for actual use.
"""

import math
from typing import ClassVar

import torch


class LloydMaxCodebook:
    """Precomputed optimal scalar quantization codebooks.

    Uses Lloyd-Max algorithm to find optimal centroids for the distribution
    of coordinates after random rotation of unit-norm vectors.
    For d >= 64, this is well-approximated by N(0, 1/d).
    """

    # Precomputed centroids for standard normal N(0,1).
    # Scale by 1/sqrt(d) at runtime.
    # These are symmetric: for each positive centroid c, -c is also present.
    _CENTROIDS_STDNORMAL: ClassVar[dict[int, list[float]]] = {
        1: [-0.7978845608, 0.7978845608],  # ±sqrt(2/pi)
        2: [-1.5104176088, -0.4527800398, 0.4527800398, 1.5104176088],
        3: [
            -2.1519784335,
            -1.3439092613,
            -0.7560052489,
            -0.2451209536,
            0.2451209536,
            0.7560052489,
            1.3439092613,
            2.1519784335,
        ],
        4: [
            -2.7326368919,
            -2.0690799744,
            -1.6180334670,
            -1.2562067015,
            -0.9423402690,
            -0.6567589957,
            -0.3880823086,
            -0.1284185637,
            0.1284185637,
            0.3880823086,
            0.6567589957,
            0.9423402690,
            1.2562067015,
            1.6180334670,
            2.0690799744,
            2.7326368919,
        ],
    }

    # Precomputed boundaries (decision thresholds) for standard normal.
    # boundary[i] = (centroid[i] + centroid[i+1]) / 2
    _BOUNDARIES_STDNORMAL: ClassVar[dict[int, list[float]]] = {}

    @classmethod
    def _compute_boundaries(cls, centroids: list[float]) -> list[float]:
        """Compute decision boundaries as midpoints between centroids."""
        return [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(len(centroids) - 1)
        ]

    @classmethod
    def _ensure_boundaries(cls, bit_width: int) -> None:
        """Lazily compute and cache boundaries."""
        if bit_width not in cls._BOUNDARIES_STDNORMAL:
            centroids = cls._CENTROIDS_STDNORMAL[bit_width]
            cls._BOUNDARIES_STDNORMAL[bit_width] = cls._compute_boundaries(centroids)

    @classmethod
    def get(
        cls,
        bit_width: int,
        dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (centroids, boundaries) tensors on device.

        Args:
            bit_width: Number of bits per coordinate (1, 2, 3, or 4).
            dim: Head dimension (used to scale centroids by 1/sqrt(d)).
            device: Target device for tensors.

        Returns:
            Tuple of (centroids, boundaries) tensors, both float32.
            Centroids has 2^b elements, boundaries has 2^b - 1 elements.
        """
        if bit_width not in cls._CENTROIDS_STDNORMAL:
            raise ValueError(
                f"Unsupported bit_width {bit_width}. "
                f"Supported: {list(cls._CENTROIDS_STDNORMAL.keys())}"
            )

        cls._ensure_boundaries(bit_width)

        scale = 1.0 / math.sqrt(dim)
        centroids = torch.tensor(
            [c * scale for c in cls._CENTROIDS_STDNORMAL[bit_width]],
            dtype=torch.float32,
            device=device,
        )
        boundaries = torch.tensor(
            [b * scale for b in cls._BOUNDARIES_STDNORMAL[bit_width]],
            dtype=torch.float32,
            device=device,
        )
        return centroids, boundaries

    @classmethod
    def get_stdnormal(
        cls,
        bit_width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unscaled (standard normal) codebook tensors.

        Useful when scaling is done separately.
        """
        if bit_width not in cls._CENTROIDS_STDNORMAL:
            raise ValueError(
                f"Unsupported bit_width {bit_width}. "
                f"Supported: {list(cls._CENTROIDS_STDNORMAL.keys())}"
            )

        cls._ensure_boundaries(bit_width)

        centroids = torch.tensor(
            cls._CENTROIDS_STDNORMAL[bit_width],
            dtype=torch.float32,
            device=device,
        )
        boundaries = torch.tensor(
            cls._BOUNDARIES_STDNORMAL[bit_width],
            dtype=torch.float32,
            device=device,
        )
        return centroids, boundaries

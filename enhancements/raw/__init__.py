"""Raw passthrough enhancement — the no-op baseline.

The Δ mAP column in the final results table is computed against this baseline,
so this module is the most important one in the project (structurally, if
not intellectually).
"""

from __future__ import annotations

import numpy as np

from enhancements.base import Enhancement, register


@register
class RawEnhancement(Enhancement):
    """Pass-through baseline. Returns the input unchanged."""

    def name(self) -> str:
        return "raw"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        return image

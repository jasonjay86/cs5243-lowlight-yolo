"""CLAHE low-light enhancement, registered as ``"clahe"``.

Owner: Krutin Patel (see enhancements/clahe/README.md)

    from enhancements.base import get_enhancement
    enh = get_enhancement("clahe")
    out = enh.enhance(bgr_uint8_image)     # -> enhanced BGR, then hand to YOLO

The algorithm lives in ``image_enhance.py`` (CLAHE + gamma + top-hat + PCA
fusion on the HSV V channel). A plain LAB-CLAHE variant is kept as
``mode="lab"`` (name ``"clahe_lab"``) for comparison; it is opt-in and is not
the registered default.
"""

from __future__ import annotations

import time
from typing import Literal

import cv2
import numpy as np

from enhancements.base import Enhancement, register
from enhancements.clahe.image_enhance import low_light_enhance, make_clahe

Mode = Literal["fusion", "lab"]


@register
class CLAHEEnhancement(Enhancement):
    """CLAHE enhancement.

    Args:
        clip_limit: CLAHE histogram clip threshold (higher = more contrast and
            more noise amplification). Typical low-light range 2.0-4.0.
        tile_grid_size: (cols, rows) CLAHE tile grid.
        mode: ``"fusion"`` (default; image_enhance.py pipeline) or ``"lab"``
            (plain CLAHE on the L channel of LAB).
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        mode: Mode = "fusion",
    ) -> None:
        if mode not in ("fusion", "lab"):
            raise ValueError(f"mode must be 'fusion' or 'lab', got {mode!r}")
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.mode: Mode = mode
        self._clahe = make_clahe(clip_limit, tile_grid_size)
        self._total_ms = 0.0
        self._n_calls = 0

    def name(self) -> str:
        return "clahe" if self.mode == "fusion" else "clahe_lab"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance one HxWx3 uint8 BGR image; returns same shape/dtype."""
        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8 BGR input, got dtype={image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape={image.shape}")

        t0 = time.perf_counter()
        if self.mode == "fusion":
            out = low_light_enhance(image, self._clahe)
        else:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_chan, a_chan, b_chan = cv2.split(lab)
            out = cv2.cvtColor(
                cv2.merge((self._clahe.apply(l_chan), a_chan, b_chan)),
                cv2.COLOR_LAB2BGR,
            )
        self._total_ms += (time.perf_counter() - t0) * 1000.0
        self._n_calls += 1
        return out

    def flops_per_megapixel(self) -> float | None:
        """None on purpose: CLAHE is histogram/LUT/morphology arithmetic, not a
        network, so a FLOPs figure isn't meaningful. Report latency instead."""
        return None

    def avg_latency_ms_per_image(self) -> float | None:
        """Running mean CPU wall-clock ms per enhance() call (None if unused)."""
        return None if self._n_calls == 0 else self._total_ms / self._n_calls

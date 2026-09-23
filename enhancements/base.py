"""
Enhancement interface — implement this BEFORE writing any enhancement code.

Every enhancement method (raw, CLAHE, SNR-Aware) implements `Enhancement`.
The detection and evaluation code only ever talks to this interface.

Why: it keeps the three modules independent and makes the comparison
harness trivial (just iterate over a list of Enhancement instances).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Enhancement(ABC):
    """All low-light image enhancements implement this contract."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier used in result tables. Lowercase, hyphen-free.
        Examples: 'raw', 'clahe', 'snr-aware'."""
        ...

    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance a single image.

        Args:
            image: HxWx3 uint8 BGR image (OpenCV convention).

        Returns:
            HxWx3 uint8 BGR enhanced image. Same shape and dtype as input.
        """
        ...

    def enhance_path(self, image_path: str | Path) -> np.ndarray:
        """Convenience: read from disk, enhance, return. Default impl is fine
        for most modules; override only if you need a streaming approach."""
        import cv2  # local import — keeps base.py importable without cv2 installed
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.enhance(img)

    # ----- Optional metadata for the eval harness -----

    def flops_per_megapixel(self) -> float | None:
        """Reported GFLOPs per megapixel, at stated reference resolution.
        Return None if not measured yet."""
        return None

    def avg_latency_ms_per_image(self) -> float | None:
        """Average wall-clock latency per image, in ms, on reference GPU.
        Return None if not measured yet."""
        return None


# Registry pattern — each module registers itself so the eval harness can
# discover available methods without hardcoded imports.
_REGISTRY: dict[str, type[Enhancement]] = {}


def register(cls: type[Enhancement]) -> type[Enhancement]:
    """Class decorator to register an Enhancement subclass."""
    instance = cls()
    _REGISTRY[instance.name()] = cls
    return cls


def available_enhancements() -> dict[str, type[Enhancement]]:
    """Return all registered enhancement classes, keyed by name()."""
    return dict(_REGISTRY)


def get_enhancement(name: str, **kwargs) -> Enhancement:
    """Instantiate a registered enhancement by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown enhancement '{name}'. "
                       f"Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)

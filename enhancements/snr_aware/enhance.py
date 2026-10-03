"""
SNR-Aware inference wrapper — adapts the SNRAwareGenerator to the project's
Enhancement interface (enhancements/base.py).

Owner: Jason Johnson

The SNR-Aware network is a transformer-based low-light enhancer that requires
GPU memory roughly proportional to the number of 4x4 patches in the input.
For a 1280x720 input that's 320x180 = 57,600 tokens per frame; 6 transformer
layers with d_model=1024 at fp32 is ~1.5 GB of activations.

Loading is lazy: the first call to enhance() loads the .pth file via
SNRAwareGenerator.from_checkpoint(). Subsequent calls reuse the cached model.
A single SNRAwareEnhancement instance is therefore cheap; many are not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2  # noqa: F401  -- kept for downstream consumers that import from enhance.py
import numpy as np
import torch

from enhancements.snr_aware.model import SNRAwareGenerator

# Default weights path: relative to this file's project root.
# Overridable per-instance via the constructor.
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "LOLv1.pth"


def _pad_to_multiple(img: np.ndarray, multiple: int) -> tuple[np.ndarray, int, int]:
    """Pad the bottom-right of an HxWxC image so H and W are multiples of `multiple`.

    Returns the padded image plus the (h_pad, w_pad) amounts actually added so
    the caller can crop the result back to the original size.
    """
    h, w = img.shape[:2]
    h_pad = (multiple - h % multiple) % multiple
    w_pad = (multiple - w % multiple) % multiple
    if h_pad == 0 and w_pad == 0:
        return img, 0, 0
    padded = np.pad(
        img,
        ((0, h_pad), (0, w_pad), (0, 0)),
        mode="reflect",
    )
    return padded, h_pad, w_pad


class SNRAwareEnhancement:
    """Inference wrapper exposing the SNRAwareGenerator as an Enhancement.

    Usage:
        enhancer = SNRAwareEnhancement()              # uses default weights path
        out = enhancer.enhance(bgr_image_uint8)        # HxWx3 uint8 BGR -> HxWx3 uint8 BGR
        out = enhancer.enhance_path("frame.jpg")       # default base-class impl reads + enhances

    Args:
        weights_path: path to a vendored .pth file (default: weights/LOLv1.pth
                    inside this module's directory).
        device: 'cpu', 'cuda', 'cuda:0', or a torch.device. Default 'cpu'
                for portability — flip to 'cuda' on the eval host.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.weights_path = Path(weights_path) if weights_path is not None else _DEFAULT_WEIGHTS
        self.device = torch.device(device) if isinstance(device, str) else device
        self._model: Optional[SNRAwareGenerator] = None

    def name(self) -> str:
        """Short identifier used in result tables. Returns 'snr-aware'."""
        return "snr-aware"

    def _ensure_loaded(self) -> SNRAwareGenerator:
        """Lazy-load the model on first enhance()."""
        if self._model is None:
            self._model = SNRAwareGenerator.from_checkpoint(
                str(self.weights_path),
                device=self.device,
            )
        return self._model

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance a single BGR uint8 image.

        Args:
            image: HxWx3 uint8 BGR (OpenCV convention). Same shape & dtype
                   contract as the rest of the pipeline.

        Returns:
            HxWx3 uint8 BGR enhanced image, same shape as input.
        """
        if image.dtype != np.uint8:
            raise TypeError(f"Expected uint8 BGR input, got dtype={image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR image, got shape={image.shape}")

        model = self._ensure_loaded()

        h0, w0 = image.shape[:2]

        # 1. Pad to multiple of 4 — the network downsamples twice (conv_first_2
        # and conv_first_3 both have stride 2). Without padding, off-by-one
        # spatial dims produce wrong output sizes or runtime errors.
        padded, h_pad, w_pad = _pad_to_multiple(image, multiple=4)

        # 2. uint8 -> float32, normalised to [0, 1], BGR -> RGB.
        # The LOLv1.pth weights were trained on RGB-normalised input.
        x = padded.astype(np.float32) / 255.0
        x = x[:, :, ::-1].copy()  # BGR -> RGB (use copy() to kill the neg-stride view)
        x = np.ascontiguousarray(x.transpose(2, 0, 1))  # HWC -> CHW
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, 3, H', W')

        # 3. Forward pass. Default mask = all ones (full light trunk path),
        # which is the closest approximation to "uniformly well-lit input".
        with torch.no_grad():
            y_tensor = model(x_tensor)

        # 4. Float -> uint8, RGB -> BGR, drop batch dim, crop padding.
        y = y_tensor.squeeze(0).clamp(0.0, 1.0).cpu().numpy()
        y = (y * 255.0 + 0.5).astype(np.uint8)
        y = np.transpose(y, (1, 2, 0))  # CHW -> HWC
        y = y[:, :, ::-1].copy()  # RGB -> BGR
        y = y[:h0, :w0, :]  # crop back to original H, W
        return y

    # The default `enhance_path()` from Enhancement base class works fine —
    # it reads the file via cv2.imread and delegates to enhance(). No override
    # needed unless we want streaming behaviour later.
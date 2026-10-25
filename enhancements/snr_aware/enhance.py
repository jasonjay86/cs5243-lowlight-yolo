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


def compute_snr_mask(image: np.ndarray) -> np.ndarray:
    """Compute the per-pixel SNR prior the SNR-Aware network expects.

    This reproduces the recipe from the upstream reference implementation
    (JIA-Lab-research/SNR-Aware-Low-Light-Enhance, models/Video_base_model4_m.py
    ::test4 and data/dataset_LOLv1.py):

        1. nf  = 5x5 box blur of the low-light input  ("noise floor" proxy)
        2. dark  = grayscale-weighted input
           light = grayscale-weighted nf
           noise  = |dark - light|
        3. mask  = light / (noise + eps)               (per-pixel SNR)
        4. mask  = mask / (per-image max + eps)       (normalise to [0,1])
        5. mask  = clamp(mask, 0, 1)

    Returns:
        Float32 array of shape (H, W) in [0, 1], where 1.0 marks high-SNR
        (clean) regions that take the light reconstruction path and 0.0 marks
        low-SNR (noisy) regions that take the transformer path.

    Note: upstream computes `nf` from the low-light input itself, so this is a
    self-referential noise estimate. That is intentional: it matches how
    LOLv1.pth was trained, and it is the only SNR signal available at
    inference time for an unpaired image.
    """
    img = image.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    # 1. 5x5 blur as the noise-floor proxy (upstream: cv2.blur, box filter).
    nf = cv2.blur(img, (5, 5))

    # 2. Luma-weighted dark (input) and light (blurred) estimates.
    dark = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    light = nf[:, :, 0] * 0.299 + nf[:, :, 1] * 0.587 + nf[:, :, 2] * 0.114

    # 3. Per-pixel SNR.
    noise = np.abs(dark - light)
    mask = light / (noise + 0.0001)

    # 4. Normalise by the per-image maximum.
    mask_max = mask.max()
    if mask_max > 0:
        mask = mask / (mask_max + 0.0001)

    # 5. Clamp to [0, 1].
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


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

        # 3. Forward pass. The SNR prior is REQUIRED by the network: it decides
        # per-pixel whether a region takes the transformer path (low SNR) or
        # the light reconstruction path (high SNR). Upstream computes it in
        # the model wrapper from the input and its 5x5 blur; we reproduce
        # that recipe in compute_snr_mask().
        #
        # Previously we called model(x_tensor) with no mask, and model.py
        # defaulted to an all-ones mask. That routed the whole image down the
        # light path and zeroed the transformer output, so SNR-Aware was
        # never actually doing SNR-aware processing.
        mask_np = compute_snr_mask(padded)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_tensor = model(x_tensor, mask_tensor)

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
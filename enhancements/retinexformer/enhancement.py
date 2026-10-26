from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from enhancements.base import Enhancement, register

from .model import RetinexFormer


@register
class RetinexformerEnhancement(Enhancement):
    """Retinexformer low-light image enhancement."""

    def __init__(self, device=None):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)
        self.model = self._load_model()

    def name(self) -> str:
        return "retinexformer"

    def _load_model(self) -> RetinexFormer:
        weights_path = (
            Path(__file__).resolve().parent
            / "weights"
            / "LOL_v1.pth"
        )

        if not weights_path.is_file():
            raise FileNotFoundError(
                "Retinexformer checkpoint not found: "
                f"{weights_path}"
            )

        model = RetinexFormer(
            in_channels=3,
            out_channels=3,
            n_feat=40,
            stage=1,
            num_blocks=[1, 2, 2],
        )

        checkpoint = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["params"],
            strict=True,
        )

        model.to(self.device)
        model.eval()

        return model

    def enhance(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image cannot be None")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "Retinexformer expects an HxWx3 BGR image; "
                f"received shape {image.shape}"
            )

        # OpenCV BGR -> RGB
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # HWC uint8 -> NCHW float32 [0, 1]
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(
                device=self.device,
                dtype=torch.float32,
            )
            .div(255.0)
        )

        _, _, height, width = tensor.shape

        # Retinexformer expects dimensions divisible by 4.
        factor = 4

        pad_height = (
            factor - height % factor
        ) % factor

        pad_width = (
            factor - width % factor
        ) % factor

        if pad_height or pad_width:
            tensor = F.pad(
                tensor,
                (0, pad_width, 0, pad_height),
                mode="reflect",
            )

        with torch.inference_mode():
            restored = self.model(tensor)

        # Remove padding.
        restored = restored[
            :,
            :,
            :height,
            :width,
        ]

        # Retinexformer can output values outside [0, 1].
        restored = restored.clamp(0.0, 1.0)

        # NCHW -> HWC
        restored_rgb = (
            restored[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        # float32 [0, 1] -> uint8 [0, 255]
        restored_rgb = (
            restored_rgb * 255.0
        ).round().astype(np.uint8)

        # RGB -> OpenCV BGR
        return cv2.cvtColor(
            restored_rgb,
            cv2.COLOR_RGB2BGR,
        )

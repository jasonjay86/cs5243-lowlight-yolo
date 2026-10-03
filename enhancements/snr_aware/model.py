"""
SNR-Aware low-light image enhancement model definition.

Owner: Jason Johnson
Paper: Xu, Wang, Fu, Jia — "SNR-aware Low-Light Image Enhancement", CVPR 2022.
       https://jiaya.me/papers/cvpr22_xiaogang.pdf
Source: https://github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance
        (commit f0c23b9, "models/archs/low_light_transformer.py" and its
        dependencies — flattened into one file for this project)

The constructor arguments reproduce the upstream API: nf=64, nframes=5, groups=8,
front_RBs=1, back_RBs=1, center=None, predeblur=False, HR_in=False, w_TSA=True.

The vendored LOLv1.pth was trained with `front_RBs=1` and `back_RBs=1` (NOT the
upstream defaults of 5 and 10). The state_dict has exactly one residual block
under `feature_extraction.0` and one under `recon_trunk.0`; strict-matching
catches any extra layers as missing-key errors. If you ever swap to a different
checkpoint (e.g. LOLv2_real.pth), check its `len(state_dict)` and the
`feature_extraction.*` / `recon_trunk.*` key ranges before deciding which
constructor args to use.

Public surface:
    SNRAwareGenerator: nn.Module, the full network.
    from_checkpoint(path): classmethod, returns (model, device) ready for
                            inference. Performs strict load and proves the
                            module names match the vendored checkpoint.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks. Names are kept verbatim from upstream (models/archs/*) so
# load_state_dict(strict=True) matches the LOLv1.pth state_dict by construction.
# ---------------------------------------------------------------------------


def _make_layer(block, n_layers: int) -> nn.Sequential:
    """Stack `block()` `n_layers` times. From upstream arch_util.make_layer."""
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    """Conv-ReLU-Conv + identity skip. No batch norm.

    From upstream arch_util.ResidualBlock_noBN.
    """

    def __init__(self, nf: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention. From upstream Modules.py."""

    def __init__(self, temperature: float, attn_dropout: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention4(nn.Module):
    """Multi-head self-attention used inside the SNR-aware transformer.

    Pre-LN variant (LayerNorm applied to Q/K/V before projection). Bias-free
    on the four projection matrices — that matches the LOLv1.pth state_dict,
    which has weights but no bias terms for w_qs/w_ks/w_vs/fc.

    From upstream SubLayers.py.MultiHeadAttention4.
    """

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = q + residual
        return q, attn


class PositionwiseFeedForward4(nn.Module):
    """Two-Linear + ReLU FFN with pre-LN and residual. From upstream SubLayers.py."""

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x = x + residual
        return x


class EncoderLayer3(nn.Module):
    """Self-attention + position-wise FFN. From upstream Layers.py."""

    def __init__(self, d_model: int, d_inner: int, n_head: int, d_k: int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention4(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward4(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder_patch66(nn.Module):
    """6-layer SNR-aware transformer encoder. From upstream Models.py.

    Operates on tokens of dimension d_model=1024; the LOLv1 checkpoint was
    trained with n_head=8, d_k=64, d_v=64 (so each head projects to 64, and
    concatenated 8*64=512 across heads, then the FC projects 512 -> 1024).
    """

    def __init__(
        self,
        d_word_vec: int = 516,
        n_layers: int = 6,
        n_head: int = 8,
        d_k: int = 64,
        d_v: int = 64,
        d_model: int = 1024,
        d_inner: int = 2048,
        dropout: float = 0.0,
        n_position: int = 10,
        scale_emb: bool = False,
    ):
        super().__init__()
        self.n_position = n_position
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [EncoderLayer3(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]
        )
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_fea, src_location, return_attns: bool = False, src_mask=None):
        enc_output = src_fea
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
        return enc_output


# ---------------------------------------------------------------------------
# Top-level generator. Renamed from upstream `low_light_transformer` so the
# project's import path is obvious. Field names match LOLv1.pth exactly.
# ---------------------------------------------------------------------------


class SNRAwareGenerator(nn.Module):
    """SNR-aware low-light enhancement network (Xu et al., CVPR 2022).

    Args:
        nf: feature channel width (default 64 — matches LOLv1.pth).
        nframes: video frame count (default 5; the LOLv1 weights are
                 single-frame so this only matters at inference time if
                 you wrap this for temporal models — vendored as-is for
                 parameter parity with upstream).
        groups, front_RBs, back_RBs, center, predeblur, HR_in, w_TSA:
                 all forwarded verbatim from upstream constructor signature.
                 Defaults below are set to match LOLv1.pth (front_RBs=1,
                 back_RBs=1), NOT the upstream defaults (5 and 10). If you
                 train a new model or load a different checkpoint, set
                 these to match the training config.
    """

    def __init__(
        self,
        nf: int = 64,
        nframes: int = 5,
        groups: int = 8,
        front_RBs: int = 1,
        back_RBs: int = 1,
        center=None,
        predeblur: bool = False,
        HR_in: bool = False,
        w_TSA: bool = True,
    ):
        super().__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA

        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)

        if self.HR_in:
            # 3-stage downsample (stride 2 on stages 2 and 3) — LOLv1.pth trained
            # with this configuration. The /4 spatial reduction is what lets the
            # 6-layer transformer operate on a manageable token count.
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            # Single-conv shallow feature extractor (LR-input mode).
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)

        # Feature extractor stack (1 residual block by default — matches LOLv1).
        self.feature_extraction = _make_layer(ResidualBlock_noBN_f, front_RBs)
        # Heavy recon trunk (1 block by default — matches LOLv1) — used for
        # low-SNR regions.
        self.recon_trunk = _make_layer(ResidualBlock_noBN_f, back_RBs)
        # Light recon trunk (6 blocks) — used for high-SNR regions. NOT taken
        # from front_RBs/back_RBs; upstream hardcodes this to 6 and the
        # checkpoint confirms it.
        self.recon_trunk_light = _make_layer(ResidualBlock_noBN_f, 6)

        # Decoder: PixelShuffle upsample with concat-skip from L1_fea_{1,2,3}.
        # Channel arithmetic: nf*2 = 128 (cat of nf=64 output with nf=64 skip)
        # is projected to nf*4 = 256, then PixelShuffle(2) -> nf=64 spatial x2.
        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        # Note: upconv2's hardcoded 64*4 = 256 in upstream is intentional; the
        # first conv operates on cat(recon, L1_fea_3)=128ch -> 256ch then
        # PixelShuffle. The second conv has identical in/out (128/256) but
        # different skip sources. We preserve upstream's 64*4=256 literal here.
        self.upconv2 = nn.Conv2d(nf * 2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)

    # The forward() method is the only place where we accept a `mask` argument
    # — that's the SNR prior passed in at evaluation time. See enhance.py for
    # how we synthesize a uniform mask for inference when SNR info isn't
    # supplied.
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run the SNR-aware generator.

        Args:
            x: input image tensor of shape (B, 3, H, W), any dtype/float range.
               The vendored LOLv1 weights expect 0..1 float inputs.
            mask: SNR prior map of shape (B, 1, H, W), float in [0, 1].
                  1.0 = high-SNR (use light trunk), 0.0 = low-SNR (use heavy
                  trunk). If None, defaults to an all-ones mask (full
                  light-trunk path) which is the closest approximation to
                  "uniformly well-lit input".
        """
        x_center = x

        # Stage 1: shallow feature extraction. With HR_in=True this is a
        # /4 downsample via two stride-2 convs.
        L1_fea_1 = self.lrelu(self.conv_first_1(x_center))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)
        fea_light = self.recon_trunk_light(fea)

        # Default mask: all-ones — entire image is "high SNR", use light trunk.
        if mask is None:
            mask = torch.ones(
                (x_center.size(0), 1, x_center.size(2), x_center.size(3)),
                dtype=x_center.dtype,
                device=x_center.device,
            )

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode="nearest")

        # 2D positional encoding grid (normalised -1..1) — passed to
        # transformer as `src_location`. The encoder currently ignores it but
        # upstream reserves the API; preserved for parity.
        xs_coords = np.linspace(-1, 1, fea.size(3) // 4)
        ys_coords = np.linspace(-1, 1, fea.size(2) // 4)
        xs_grid = np.meshgrid(xs_coords, ys_coords)
        xs_grid = np.stack(xs_grid, 2)
        xs_grid = torch.Tensor(xs_grid).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).to(fea.device)
        xs_grid = xs_grid.view(fea.size(0), -1, 2)

        height = fea.shape[2]
        width = fea.shape[3]
        # Unfold 64-channel features into 4x4 patches -> token sequence.
        # Each token: 64ch * 4 * 4 = 1024 (matches d_model=1024).
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)

        # SNR-aware mask: which tokens are "low-SNR" (use transformer path)
        # vs "high-SNR" (use light trunk).
        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)
        mask_unfold = torch.where(mask_unfold <= 0.5, torch.zeros_like(mask_unfold), mask_unfold)

        fea_unfold = self.transformer(fea_unfold, xs_grid, src_mask=mask_unfold)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        fea_unfold = F.fold(
            fea_unfold,
            output_size=(height, width),
            kernel_size=(4, 4),
            stride=4,
            padding=0,
            dilation=1,
        )

        # Mix: low-SNR tokens get transformer output; high-SNR tokens bypass
        # the transformer and use the light trunk directly.
        channel = fea.shape[1]
        mask_broadcast = mask.repeat(1, channel, 1, 1)
        fea = fea_unfold * (1 - mask_broadcast) + fea_light * mask_broadcast

        # Decoder: heavy trunk + progressive upsampling with concat-skip.
        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        # Residual: predict a delta, add to input.
        out_noise = out_noise + x_center

        return out_noise

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        *,
        device: str | torch.device = "cpu",
        HR_in: bool = True,
        front_RBs: int = 1,
        back_RBs: int = 1,
    ) -> "SNRAwareGenerator":
        """Build an SNRAwareGenerator and load vendored weights strictly.

        Args:
            path: filesystem path to a vendored .pth file (e.g. weights/LOLv1.pth).
            device: torch device for the loaded model. Defaults to 'cpu' to
                    avoid surprise CUDA init in non-GPU hosts.
            HR_in: must match the training config of the weights file. The
                   LOLv1.pth we vendored was trained with HR_in=True, which
                   is the default.
            front_RBs, back_RBs: number of residual blocks in feature_extraction
                   and recon_trunk respectively. The vendored LOLv1.pth was
                   trained with front_RBs=1, back_RBs=1 (NOT the upstream
                   defaults of 5 and 10). Override only if you've trained a
                   different model.

        Returns:
            A `model.eval()`'d `SNRAwareGenerator` on `device`, with all
            parameters loaded and `requires_grad=False`.

        Raises:
            RuntimeError: if `load_state_dict(strict=True)` fails — the
                          module attribute names have drifted from the
                          checkpoint.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        model = cls(HR_in=HR_in, front_RBs=front_RBs, back_RBs=back_RBs)
        state_dict = torch.load(str(path), map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            # strict=True *should* raise before reaching here, but be defensive.
            raise RuntimeError(
                f"State dict mismatch for {path.name}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model.to(device)


__all__ = ["SNRAwareGenerator"]
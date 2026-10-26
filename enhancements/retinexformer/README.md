# Retinexformer Enhancement

Retinexformer low-light image enhancement integration for the
CS5243 low-light YOLO project.

## Source

The model architecture is adapted from:

https://github.com/caiyuanhao1998/Retinexformer

Retinexformer: One-stage Retinex-based Transformer for
Low-light Image Enhancement, ICCV 2023.

The upstream implementation is MIT licensed. A copy of the
license is included in:

    licenses/LICENSE.txt

## Pretrained Weights

This integration uses the pretrained LOL-v1 checkpoint:

    LOL_v1.pth

Download the pretrained Retinexformer models from the upstream
project and place the LOL-v1 checkpoint at:

    enhancements/retinexformer/weights/LOL_v1.pth

The weights directory is ignored by Git and pretrained weights
are not stored in this repository.

## Configuration

The LOL-v1 model is instantiated using the upstream configuration:

    in_channels = 3
    out_channels = 3
    n_feat = 40
    stage = 1
    num_blocks = [1, 2, 2]

## Usage

Run the smoke test from the repository root:

    python -m scripts.test_retinexformer \
        --input /path/to/low_light_image.png \
        --output /tmp/retinexformer_output.png

To explicitly select a device:

    python -m scripts.test_retinexformer \
        --input /path/to/low_light_image.png \
        --output /tmp/retinexformer_output.png \
        --device cuda

The enhancement accepts an OpenCV BGR uint8 image and returns
an enhanced BGR uint8 image with the same dimensions.

## Citation

Cai, Yuanhao, et al.
"Retinexformer: One-stage Retinex-based Transformer for
Low-light Image Enhancement."
ICCV 2023.

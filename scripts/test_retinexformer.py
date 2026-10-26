import argparse

import cv2

from enhancements.retinexformer import RetinexformerEnhancement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Device such as cpu, cuda, or cuda:0",
    )

    args = parser.parse_args()

    image = cv2.imread(args.input)

    if image is None:
        raise FileNotFoundError(
            f"Could not read input image: {args.input}"
        )

    enhancer = RetinexformerEnhancement(
        device=args.device,
    )

    enhanced = enhancer.enhance(image)

    if not cv2.imwrite(args.output, enhanced):
        raise RuntimeError(
            f"Could not write output image: {args.output}"
        )

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Shape:  {image.shape}")


if __name__ == "__main__":
    main()

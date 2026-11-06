"""Compare any number of reconstructions against an explicit reference image."""

import argparse
import csv
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--images",
        required=True,
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for candidate images in the same order",
    )
    parser.add_argument(
        "--protocol",
        choices=("standard", "legacy"),
        default="standard",
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--skip-lpips",
        action="store_true",
        help="Explicitly omit LPIPS; output field is empty",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/evaluation/metrics.csv"),
    )
    parser.add_argument(
        "--collage",
        type=Path,
        help="Optional comparison image, reference last",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.6,
        help="Display scale only; metrics use original inputs",
    )
    args = parser.parse_args(argv)
    if args.labels is not None and len(args.labels) != len(args.images):
        parser.error("--labels must have one label per candidate image")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    return args


def main():
    args = parse_args()
    import numpy as np
    from PIL import Image
    from gradcfg.evaluation.metrics import calculate_image_similarity
    from gradcfg.tools.images import compose

    with Image.open(args.reference) as image:
        reference = np.array(image.convert("RGB"))
    labels = args.labels or [path.stem for path in args.images]
    rows, images = [], []
    for path, label in zip(args.images, labels):
        with Image.open(path) as image:
            candidate = np.array(image.convert("RGB"))
        scores = calculate_image_similarity(
            candidate,
            reference,
            device=args.device,
            protocol=args.protocol,
            skip_lpips=args.skip_lpips,
        )
        row = {
            "label": label,
            "image": str(path),
            "reference": str(args.reference),
            "protocol": args.protocol,
            "alignment": "resize_min",
            **scores,
        }
        rows.append(row)
        images.append(Image.fromarray(candidate))
        print(row)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.collage:
        images.append(Image.fromarray(reference))
        images = [
            image.resize(
                (
                    max(1, round(image.width * args.scale)),
                    max(1, round(image.height * args.scale)),
                )
            )
            for image in images
        ]
        args.collage.parent.mkdir(parents=True, exist_ok=True)
        compose(images, "horizontal", gap=10).save(args.collage)
    print(f"Saved {args.out_csv} (protocol={args.protocol})")


if __name__ == "__main__":
    main()

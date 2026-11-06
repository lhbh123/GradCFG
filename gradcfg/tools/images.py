"""Image strip utilities replacing connect/move/rearrange/vertical5_to_horizontal."""

import argparse
from pathlib import Path

from PIL import Image


def compose(images, orientation="vertical", gap=0):
    if not images:
        raise ValueError("At least one image is required.")
    if orientation not in ("vertical", "horizontal") or gap < 0:
        raise ValueError("Invalid orientation or negative gap.")
    vertical = orientation == "vertical"
    width = (
        max(im.width for im in images)
        if vertical
        else sum(im.width for im in images) + gap * (len(images) - 1)
    )
    height = (
        sum(im.height for im in images) + gap * (len(images) - 1)
        if vertical
        else max(im.height for im in images)
    )
    output = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in images:
        output.paste(
            image,
            (
                ((width - image.width) // 2, offset)
                if vertical
                else (offset, (height - image.height) // 2)
            ),
        )
        offset += (image.height if vertical else image.width) + gap
    return output


def split_vertical(image, n=None, segment_height=None, allow_remainder=False):
    if n is not None and n <= 0:
        raise ValueError("n must be positive.")
    if segment_height is None:
        if n is None or image.height % n:
            raise ValueError(
                "Specify n that divides image height, or an explicit segment height."
            )
        segment_height = image.height // n
    if segment_height <= 0:
        raise ValueError("Segment height must be positive.")
    count = image.height // segment_height if n is None else n
    if count <= 0 or count * segment_height > image.height:
        raise ValueError("The image does not contain the requested segments.")
    if not allow_remainder and count * segment_height != image.height:
        raise ValueError(
            "Segments leave unused rows; pass --allow-remainder to crop explicitly."
        )
    return [
        image.crop((0, i * segment_height, image.width, (i + 1) * segment_height))
        for i in range(count)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    stack = sub.add_parser("stack", help="Stack images in the given order")
    stack.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        type=Path,
    )
    stack.add_argument(
        "--size",
        type=int,
        help="Resize each input to size x size, matching old connect.py",
    )
    stack.add_argument(
        "--orientation",
        choices=("vertical", "horizontal"),
        default="vertical",
    )
    for name in ("relayout", "reorder", "crop"):
        child = sub.add_parser(name)
        child.add_argument(
            "--input",
            required=True,
            type=Path,
        )
        child.add_argument(
            "--n",
            type=int,
        )
        child.add_argument(
            "--segment-height",
            type=int,
        )
        child.add_argument(
            "--allow-remainder",
            action="store_true",
        )
        if name == "relayout":
            child.add_argument(
                "--orientation",
                choices=("vertical", "horizontal"),
                default="horizontal",
            )
        if name == "reorder":
            child.add_argument(
                "--order",
                nargs="+",
                required=True,
                type=int,
                help="Zero-based permutation",
            )
        if name == "crop":
            child.add_argument(
                "--size",
                type=int,
                default=512,
            )
    for child in (
        stack,
        *[sub.choices[name] for name in ("relayout", "reorder", "crop")],
    ):
        child.add_argument(
            "--output",
            required=True,
            type=Path,
        )
    args = parser.parse_args()
    if args.output.resolve() in [
        path.resolve()
        for path in (args.inputs if args.command == "stack" else [args.input])
    ]:
        parser.error(
            "Output must differ from input; preserve the source experiment image."
        )
    if args.command == "stack":
        images = []
        for path in args.inputs:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if args.size is not None:
            if args.size <= 0:
                parser.error("--size must be positive")
            images = [
                im.resize((args.size, args.size), Image.Resampling.LANCZOS)
                for im in images
            ]
        result = compose(images, args.orientation)
    else:
        with Image.open(args.input) as source:
            image = source.convert("RGB")
        height = args.segment_height
        if args.command in ("crop", "reorder") and height is None and args.n is None:
            height = 512
        images = split_vertical(image, args.n, height, args.allow_remainder)
        if args.command == "reorder":
            if sorted(args.order) != list(range(len(images))):
                parser.error("--order must be a permutation of all segment indices")
            images = [images[index] for index in args.order]
        if args.command == "crop":
            if args.size <= 0:
                parser.error("--size must be positive")
            images = [
                im.resize((args.size, args.size), Image.Resampling.LANCZOS)
                for im in images
            ]
        result = compose(images, getattr(args, "orientation", "vertical"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Saved {args.output}: {result.size}")


if __name__ == "__main__":
    main()

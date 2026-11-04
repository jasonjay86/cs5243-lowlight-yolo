#!/usr/bin/env python3
"""Prepare a one-class YOLO dataset from the frozen CCT-20 split manifest.

The script:

1. Reads the committed image-level split manifest.
2. Reads the CCT-20 COCO-style bounding-box annotation JSON.
3. Rescales annotation-space boxes to the actual on-disk image dimensions.
4. Collapses every valid wildlife category to YOLO class 0 (``animal``).
5. Creates explicit empty label files for known-empty images.
6. Creates relative symlinks to source images by default.
7. Writes ``dataset.yaml`` and a machine-readable preparation summary.
8. Validates image counts, box counts, label ranges, and split isolation.

The split manifest is authoritative for train/validation/test membership. This
script never reassigns images to a different split.

Example
-------

    python scripts/prepare_yolo_dataset.py \
        --manifest data/cct20_splits/split_manifest.csv \
        --annotations data/cct20/caltech_bboxes_20200316.json \
        --output data/cct20_yolo

The default output layout is:

    data/cct20_yolo/
    +-- dataset.yaml
    +-- preparation_summary.json
    +-- images/
    |   +-- train/
    |   +-- val/
    |   +-- test/
    +-- labels/
        +-- train/
        +-- val/
        +-- test/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLITS = ("train", "val", "test")
CLASS_ID = 0
CLASS_NAME = "animal"

# Categories excluded from animal-detector labels. Car-containing frames were
# already excluded when the canonical split was created, but retaining this
# guard prevents accidental label contamination if inputs change.
NON_ANIMAL_CATEGORIES = {"empty", "car"}


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    file_name: str
    image_path: Path
    location: str
    seq_id: str
    split: str
    is_empty: bool
    expected_boxes: int
    species: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationImage:
    image_id: str
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class YoloBox:
    x_center: float
    y_center: float
    width: float
    height: float

    def line(self) -> str:
        return (
            f"{CLASS_ID} "
            f"{self.x_center:.8f} {self.y_center:.8f} "
            f"{self.width:.8f} {self.height:.8f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Frozen split_manifest.csv produced by create_cct_split.py.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="CCT-20 COCO-style bounding-box JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cct20_yolo"),
        help="YOLO dataset output directory.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How images are materialized in the YOLO tree. Default: symlink.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and recreate an existing output directory.",
    )
    parser.add_argument(
        "--allow-box-count-mismatch",
        action="store_true",
        help=(
            "Continue if manifest n_boxes differs from converted valid boxes. "
            "Use only after reviewing the reported mismatch reasons."
        ),
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=1.0,
        help="Minimum scaled box width and height in pixels. Default: 1.0.",
    )
    parser.add_argument(
        "--absolute-yaml-path",
        action="store_true",
        help="Write an absolute path in dataset.yaml instead of '.'.",
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_bool(value: str, field_name: str, row_number: int) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(
        f"manifest row {row_number}: invalid {field_name} value {value!r}"
    )


def read_manifest(path: Path) -> list[ManifestRow]:
    required = {
        "image_id",
        "file_name",
        "image_path",
        "location",
        "seq_id",
        "split",
        "is_empty",
        "n_boxes",
        "species",
    }
    rows: list[ManifestRow] = []
    #manifest_parent = path.resolve().parent

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                "manifest is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            split = row["split"].strip()
            if split not in SPLITS:
                raise ValueError(
                    f"manifest row {row_number}: invalid split {split!r}"
                )
            raw_path = Path(row["image_path"])

            if raw_path.is_absolute():
                image_path = raw_path.resolve()
            else:
                image_path = (REPO_ROOT / raw_path).resolve()
            #raw_path = Path(row["image_path"])
            #image_path = (
            #    raw_path if raw_path.is_absolute() else manifest_parent / raw_path
            #).resolve()
            try:
                expected_boxes = int(row["n_boxes"])
            except ValueError as exc:
                raise ValueError(
                    f"manifest row {row_number}: invalid n_boxes {row['n_boxes']!r}"
                ) from exc
            species = tuple(
                item for item in row["species"].split(";") if item
            )
            rows.append(
                ManifestRow(
                    image_id=row["image_id"].strip(),
                    file_name=row["file_name"].strip(),
                    image_path=image_path,
                    location=row["location"].strip(),
                    seq_id=row["seq_id"].strip(),
                    split=split,
                    is_empty=parse_bool(row["is_empty"], "is_empty", row_number),
                    expected_boxes=expected_boxes,
                    species=species,
                )
            )

    if not rows:
        raise ValueError(f"manifest contains no data rows: {path}")
    return rows


def read_annotations(path: Path) -> tuple[
    dict[str, AnnotationImage], dict[str, list[dict]], dict[object, str]
]:
    data = json.loads(path.read_text(encoding="utf-8"))
    categories = {
        item["id"]: str(item["name"])
        for item in data.get("categories", [])
    }
    images: dict[str, AnnotationImage] = {}
    for item in data.get("images", []):
        image_id = str(item["id"])
        images[image_id] = AnnotationImage(
            image_id=image_id,
            file_name=str(item.get("file_name", f"{image_id}.jpg")),
            width=int(item["width"]),
            height=int(item["height"]),
        )
    annotations: dict[str, list[dict]] = defaultdict(list)
    for item in data.get("annotations", []):
        annotations[str(item["image_id"])].append(item)
    return images, annotations, categories


def prepare_output(output: Path, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise SystemExit(
                f"ERROR: output exists: {output}; pass --overwrite deliberately"
            )
        resolved = output.resolve()
        if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
            raise SystemExit(f"ERROR: refusing to remove unsafe output path: {resolved}")
        shutil.rmtree(output)

    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def unique_output_name(row: ManifestRow) -> str:
    """Use image_id as the stem to avoid duplicate basenames across folders."""
    suffix = Path(row.file_name).suffix.lower() or row.image_path.suffix.lower() or ".jpg"
    return f"{row.image_id}{suffix}"


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"source image not found: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")

    if mode == "symlink":
        relative_source = os.path.relpath(source, start=destination.parent)
        destination.symlink_to(relative_source)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"unsupported link mode: {mode}")


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def convert_bbox(
    bbox: Iterable[float],
    annotation_width: int,
    annotation_height: int,
    disk_width: int,
    disk_height: int,
    min_box_size: float,
) -> tuple[YoloBox | None, str | None]:
    values = list(bbox)
    if len(values) != 4:
        return None, "bbox_not_length_4"
    try:
        x, y, width, height = (float(value) for value in values)
    except (TypeError, ValueError):
        return None, "bbox_non_numeric"

    if annotation_width <= 0 or annotation_height <= 0:
        return None, "invalid_annotation_dimensions"
    if disk_width <= 0 or disk_height <= 0:
        return None, "invalid_disk_dimensions"
    if width <= 0 or height <= 0:
        return None, "non_positive_bbox"

    scale_x = disk_width / annotation_width
    scale_y = disk_height / annotation_height

    x1 = x * scale_x
    y1 = y * scale_y
    x2 = (x + width) * scale_x
    y2 = (y + height) * scale_y

    # COCO annotations can occasionally extend slightly beyond an image.
    x1 = clip(x1, 0.0, float(disk_width))
    y1 = clip(y1, 0.0, float(disk_height))
    x2 = clip(x2, 0.0, float(disk_width))
    y2 = clip(y2, 0.0, float(disk_height))

    scaled_width = x2 - x1
    scaled_height = y2 - y1
    if scaled_width < min_box_size or scaled_height < min_box_size:
        return None, "scaled_bbox_too_small"

    result = YoloBox(
        x_center=((x1 + x2) / 2.0) / disk_width,
        y_center=((y1 + y2) / 2.0) / disk_height,
        width=scaled_width / disk_width,
        height=scaled_height / disk_height,
    )

    values_to_check = (
        result.x_center,
        result.y_center,
        result.width,
        result.height,
    )
    if not all(0.0 <= value <= 1.0 for value in values_to_check):
        return None, "normalized_bbox_out_of_range"
    if result.width <= 0.0 or result.height <= 0.0:
        return None, "normalized_bbox_non_positive"
    return result, None


def validate_manifest(rows: list[ManifestRow]) -> dict:
    image_ids = Counter(row.image_id for row in rows)
    duplicates = [image_id for image_id, count in image_ids.items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate manifest image_id, e.g. {duplicates[0]}")

    location_splits: dict[str, set[str]] = defaultdict(set)
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        location_splits[row.location].add(row.split)
        sequence_splits[row.seq_id].add(row.split)
        if row.is_empty and row.expected_boxes != 0:
            raise ValueError(
                f"empty image {row.image_id} has n_boxes={row.expected_boxes}"
            )
        if not row.is_empty and row.expected_boxes <= 0:
            raise ValueError(
                f"animal image {row.image_id} has n_boxes={row.expected_boxes}"
            )

    leaked_locations = {
        key: value for key, value in location_splits.items() if len(value) > 1
    }
    leaked_sequences = {
        key: value for key, value in sequence_splits.items() if len(value) > 1
    }
    if leaked_locations:
        raise ValueError(
            f"manifest location leakage, e.g. {next(iter(leaked_locations.items()))}"
        )
    if leaked_sequences:
        raise ValueError(
            f"manifest sequence leakage, e.g. {next(iter(leaked_sequences.items()))}"
        )

    return {
        "n_images": len(rows),
        "n_locations": len(location_splits),
        "n_sequences": len(sequence_splits),
        "location_counts": {
            split: len({row.location for row in rows if row.split == split})
            for split in SPLITS
        },
        "duplicate_images": len(duplicates),
        "leaked_locations": len(leaked_locations),
        "leaked_sequences": len(leaked_sequences),
    }


def write_dataset_yaml(output: Path, absolute_path: bool) -> None:
    root = str(output.resolve()) if absolute_path else "."
    content = (
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n"
    )
    (output / "dataset.yaml").write_text(content, encoding="utf-8")


def validate_generated_dataset(output: Path, expected: dict) -> dict:
    actual: dict[str, dict] = {}
    for split in SPLITS:
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        images = sorted(path for path in image_dir.iterdir() if path.is_file() or path.is_symlink())
        labels = sorted(label_dir.glob("*.txt"))
        if len(images) != len(labels):
            raise RuntimeError(
                f"{split}: {len(images)} images but {len(labels)} label files"
            )

        label_stems = {path.stem for path in labels}
        image_stems = {path.stem for path in images}
        if label_stems != image_stems:
            missing_labels = sorted(image_stems - label_stems)[:5]
            missing_images = sorted(label_stems - image_stems)[:5]
            raise RuntimeError(
                f"{split}: image/label stem mismatch; "
                f"missing labels={missing_labels}, missing images={missing_images}"
            )

        boxes = 0
        empty_labels = 0
        for label_path in labels:
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                if len(parts) != 5:
                    raise RuntimeError(
                        f"{label_path}:{line_number}: expected 5 fields"
                    )
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
                if class_id != CLASS_ID:
                    raise RuntimeError(
                        f"{label_path}:{line_number}: unexpected class {class_id}"
                    )
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    raise RuntimeError(
                        f"{label_path}:{line_number}: coordinate outside [0, 1]"
                    )
                if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                    raise RuntimeError(
                        f"{label_path}:{line_number}: non-positive width/height"
                    )
                boxes += 1

        actual[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": boxes,
            "empty_labels": empty_labels,
        }
        for key in ("images", "boxes", "empty_labels"):
            if actual[split][key] != expected[split][key]:
                raise RuntimeError(
                    f"{split}: generated {key}={actual[split][key]}, "
                    f"expected {expected[split][key]}"
                )
    return actual


def main() -> int:
    args = build_parser().parse_args()
    for label, path in (
        ("manifest", args.manifest),
        ("annotations", args.annotations),
    ):
        if not path.is_file():
            raise SystemExit(f"ERROR: {label} file not found: {path}")
    if args.min_box_size <= 0:
        raise SystemExit("ERROR: --min-box-size must be greater than zero")

    rows = read_manifest(args.manifest)
    manifest_validation = validate_manifest(rows)
    annotation_images, annotations, categories = read_annotations(args.annotations)
    prepare_output(args.output, args.overwrite)

    split_stats = {
        split: Counter(images=0, boxes=0, empty_labels=0, animal_images=0)
        for split in SPLITS
    }
    skipped_reasons = Counter()
    mismatch_examples: list[dict] = []
    rescaled_images = 0
    species_boxes = {split: Counter() for split in SPLITS}

    print(f"[prepare] manifest images: {len(rows)}")
    print(f"[prepare] output: {args.output}")
    print(f"[prepare] image mode: {args.link_mode}")

    for index, row in enumerate(rows, start=1):
        annotation_image = annotation_images.get(row.image_id)
        image_annotations = annotations.get(row.image_id, [])

        if not row.is_empty and annotation_image is None:
            raise RuntimeError(
                f"animal image {row.image_id} is missing annotation image metadata"
            )

        try:
            with Image.open(row.image_path) as image:
                disk_width, disk_height = image.size
        except Exception as exc:
            raise RuntimeError(f"could not read image {row.image_path}: {exc}") from exc

        output_name = unique_output_name(row)
        output_image = args.output / "images" / row.split / output_name
        output_label = args.output / "labels" / row.split / f"{row.image_id}.txt"
        materialize_image(row.image_path, output_image, args.link_mode)

        yolo_boxes: list[YoloBox] = []
        converted_species: list[str] = []

        if not row.is_empty:
            assert annotation_image is not None
            if (
                annotation_image.width != disk_width
                or annotation_image.height != disk_height
            ):
                rescaled_images += 1

            for annotation in image_annotations:
                category_name = categories.get(annotation.get("category_id"), "unknown")
                if category_name in NON_ANIMAL_CATEGORIES:
                    skipped_reasons[f"excluded_category:{category_name}"] += 1
                    continue
                if "bbox" not in annotation:
                    skipped_reasons["missing_bbox"] += 1
                    continue

                yolo_box, reason = convert_bbox(
                    annotation["bbox"],
                    annotation_image.width,
                    annotation_image.height,
                    disk_width,
                    disk_height,
                    args.min_box_size,
                )
                if yolo_box is None:
                    skipped_reasons[reason or "unknown_conversion_failure"] += 1
                    continue
                yolo_boxes.append(yolo_box)
                converted_species.append(category_name)
                species_boxes[row.split][category_name] += 1

        output_label.write_text(
            "".join(f"{box.line()}\n" for box in yolo_boxes),
            encoding="utf-8",
        )

        actual_boxes = len(yolo_boxes)
        if actual_boxes != row.expected_boxes:
            mismatch_examples.append(
                {
                    "image_id": row.image_id,
                    "split": row.split,
                    "manifest_boxes": row.expected_boxes,
                    "converted_boxes": actual_boxes,
                    "manifest_species": list(row.species),
                    "converted_species": converted_species,
                }
            )

        split_stats[row.split]["images"] += 1
        split_stats[row.split]["boxes"] += actual_boxes
        if actual_boxes == 0:
            split_stats[row.split]["empty_labels"] += 1
        else:
            split_stats[row.split]["animal_images"] += 1

        if index % 5000 == 0 or index == len(rows):
            print(f"[prepare] {index}/{len(rows)} images")

    if mismatch_examples and not args.allow_box_count_mismatch:
        example = mismatch_examples[0]
        raise RuntimeError(
            f"{len(mismatch_examples)} manifest/converted box-count mismatches; "
            f"first example={example}. The output is incomplete for experimental "
            "use. Review the mismatch, remove the output directory, and rerun. "
            "Pass --allow-box-count-mismatch only if the discrepancy is understood."
        )

    expected = {
        split: {
            "images": split_stats[split]["images"],
            "boxes": split_stats[split]["boxes"],
            "empty_labels": split_stats[split]["empty_labels"],
        }
        for split in SPLITS
    }
    generated_validation = validate_generated_dataset(args.output, expected)
    write_dataset_yaml(args.output, args.absolute_yaml_path)

    summary = {
        "version": 1,
        "class_map": {str(CLASS_ID): CLASS_NAME},
        "link_mode": args.link_mode,
        "min_box_size_pixels": args.min_box_size,
        "source": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "annotations": str(args.annotations),
            "annotations_sha256": sha256_file(args.annotations),
        },
        "manifest_validation": manifest_validation,
        "rescaled_images": rescaled_images,
        "skipped_annotations": dict(sorted(skipped_reasons.items())),
        "box_count_mismatches": {
            "count": len(mismatch_examples),
            "examples": mismatch_examples[:20],
        },
        "splits": {
            split: {
                **dict(split_stats[split]),
                "species_boxes": dict(sorted(species_boxes[split].items())),
            }
            for split in SPLITS
        },
        "generated_validation": generated_validation,
    }
    (args.output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n[prepare] completed")
    for split in SPLITS:
        stats = split_stats[split]
        print(
            f"  {split:<5} images={stats['images']:>6} "
            f"animal_images={stats['animal_images']:>6} "
            f"empty_labels={stats['empty_labels']:>6} "
            f"boxes={stats['boxes']:>6}"
        )
    print(f"  rescaled images: {rescaled_images}")
    print(f"  box-count mismatches: {len(mismatch_examples)}")
    print(f"  dataset YAML: {args.output / 'dataset.yaml'}")
    print(f"  summary:      {args.output / 'preparation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

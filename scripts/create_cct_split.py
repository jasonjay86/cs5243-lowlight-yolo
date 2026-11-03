#!/usr/bin/env python3
"""Create an exact 14/3/3 location-disjoint split for CCT-20.

The script enumerates every valid assignment of 20 camera locations into:

    train: 14 locations
    val:    3 locations
    test:   3 locations

For 20 locations this evaluates C(20, 3) * C(17, 3) = 775,200 labeled
validation/test assignments. The remaining 14 locations form training.
The globally lowest-cost assignment under the configured objective is selected.

The bounding-box JSON is authoritative for animal boxes/species. The image-level
JSON or JSON ZIP is authoritative for genuine empty images and supplies
location, sequence, and capture-time metadata. Unannotated images are not
assumed empty. Images containing a non-animal category such as "car" are
excluded, matching the existing evaluation workflow.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.dataset import (  # noqa: E402
    NON_ANIMAL_CATEGORIES,
    index_images_on_disk,
    read_json_or_zip,
)

SPLITS = ("train", "val", "test")

# Exact independent-group counts.
TRAIN_LOCATIONS = 14
VAL_LOCATIONS = 3
TEST_LOCATIONS = 3

# Objective weights for a one-class animal detector.
WEIGHT_IMAGES = 0.5
WEIGHT_BOXES = 3.0
WEIGHT_EMPTY = 3.0
WEIGHT_LIGHT = 1.0
WEIGHT_SPECIES_PROPORTION = 1.0

# A species constrains validation/test only when it is sufficiently abundant
# and occurs at enough independent locations to make grouped coverage feasible.
COMMON_SPECIES_MIN_TOTAL_BOXES = 100
COMMON_SPECIES_MIN_LOCATIONS = 3
COMMON_SPECIES_MIN_BOXES = 50
COMMON_SPECIES_SOFT_MIN_BOXES = 100

PENALTY_TRAIN_SPECIES_ZERO = 50.0
PENALTY_COMMON_SPECIES_ZERO = 30.0
PENALTY_COMMON_SPECIES_LT_50 = 12.0
PENALTY_COMMON_SPECIES_LT_100 = 4.0


@dataclass
class ImageRecord:
    image_id: str
    file_name: str
    image_path: str
    location: str
    seq_id: str
    frame_num: int | None
    seq_num_frames: int | None
    date_captured: str
    capture_hour: int | None
    is_night: bool | None
    is_empty: bool
    species: list[str] = field(default_factory=list)
    n_boxes: int = 0
    split: str = ""


@dataclass
class Totals:
    n_images: float = 0.0
    n_boxes: float = 0.0
    n_empty: float = 0.0
    n_night: float = 0.0
    n_day: float = 0.0
    species_boxes: Counter = field(default_factory=Counter)
    species_frames: Counter = field(default_factory=Counter)

    def copy(self) -> "Totals":
        return Totals(
            n_images=self.n_images,
            n_boxes=self.n_boxes,
            n_empty=self.n_empty,
            n_night=self.n_night,
            n_day=self.n_day,
            species_boxes=Counter(self.species_boxes),
            species_frames=Counter(self.species_frames),
        )

    def add(self, other: "Totals") -> None:
        self.n_images += other.n_images
        self.n_boxes += other.n_boxes
        self.n_empty += other.n_empty
        self.n_night += other.n_night
        self.n_day += other.n_day
        self.species_boxes.update(other.species_boxes)
        self.species_frames.update(other.species_frames)

    def subtract(self, other: "Totals") -> None:
        self.n_images -= other.n_images
        self.n_boxes -= other.n_boxes
        self.n_empty -= other.n_empty
        self.n_night -= other.n_night
        self.n_day -= other.n_day
        self.species_boxes.subtract(other.species_boxes)
        self.species_frames.subtract(other.species_frames)


@dataclass
class LocationGroup:
    location: str
    records: list[ImageRecord]
    totals: Totals
    sequences: set[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-labels", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cct20_splits/split_manifest.csv"),
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--night-start-hour", type=int, default=19)
    parser.add_argument("--night-end-hour", type=int, default=6)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print enumeration progress every N candidates; 0 disables it.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> Path:
    for label, path in (
        ("images", args.images),
        ("annotations", args.annotations),
        ("image labels", args.image_labels),
    ):
        if not path.exists():
            raise SystemExit(f"ERROR: {label} path does not exist: {path}")
    if not args.images.is_dir():
        raise SystemExit(f"ERROR: --images must be a directory: {args.images}")
    for name, hour in (
        ("night-start-hour", args.night_start_hour),
        ("night-end-hour", args.night_end_hour),
    ):
        if not 0 <= hour <= 23:
            raise SystemExit(f"ERROR: --{name} must be between 0 and 23")
    if args.progress_every < 0:
        raise SystemExit("ERROR: --progress-every cannot be negative")
    summary_path = args.summary_json or args.output.with_suffix(".summary.json")
    existing = [path for path in (args.output, summary_path) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "ERROR: output already exists: "
            + ", ".join(str(path) for path in existing)
            + "; use --overwrite only deliberately"
        )
    return summary_path


def category_map(data: dict) -> dict:
    return {item["id"]: str(item["name"]) for item in data.get("categories", [])}


def images_by_id(data: dict) -> dict[str, dict]:
    return {str(item["id"]): item for item in data.get("images", [])}


def annotations_by_image(data: dict) -> dict[str, list[dict]]:
    output: dict[str, list[dict]] = defaultdict(list)
    for item in data.get("annotations", []):
        output[str(item["image_id"])].append(item)
    return output


def labels_by_image(data: dict) -> dict[str, set[str]]:
    categories = category_map(data)
    output: dict[str, set[str]] = defaultdict(set)
    for item in data.get("annotations", []):
        name = categories.get(item["category_id"])
        if name is not None:
            output[str(item["image_id"])].add(name)
    return output


def parse_capture_time(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def hour_is_night(hour: int, start: int, end: int) -> bool:
    if start > end:
        return hour >= start or hour < end
    return start <= hour < end


def optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_records(
    images_dir: Path,
    bbox_data: dict,
    image_data: dict,
    night_start: int,
    night_end: int,
) -> tuple[list[ImageRecord], dict]:
    bbox_categories = category_map(bbox_data)
    bbox_annotations = annotations_by_image(bbox_data)
    bbox_images = images_by_id(bbox_data)
    metadata_images = images_by_id(image_data)
    metadata_labels = labels_by_image(image_data)

    print(f"[split] indexing images under {images_dir}")
    on_disk = index_images_on_disk(images_dir)
    print(f"[split] indexed {len(on_disk)} image files")

    image_ids = sorted(set(bbox_images) | set(metadata_images))
    diagnostics = Counter(metadata_image_ids=len(image_ids))
    records: list[ImageRecord] = []

    for image_id in image_ids:
        bbox_meta = bbox_images.get(image_id, {})
        image_meta = metadata_images.get(image_id, {})
        merged = {**image_meta, **bbox_meta}
        file_name = str(merged.get("file_name") or f"{image_id}.jpg")
        disk_path = on_disk.get(Path(file_name).name) or on_disk.get(f"{image_id}.jpg")
        if disk_path is None:
            diagnostics["missing_on_disk"] += 1
            continue

        annotations = bbox_annotations.get(image_id, [])
        category_names = {
            bbox_categories.get(item["category_id"]) for item in annotations
        }
        if category_names & NON_ANIMAL_CATEGORIES:
            diagnostics["skipped_car"] += 1
            continue

        boxed = [
            item
            for item in annotations
            if "bbox" in item
            and bbox_categories.get(item["category_id"]) != "empty"
        ]
        species = [
            str(bbox_categories.get(item["category_id"], "unknown"))
            for item in boxed
        ]
        is_animal = bool(boxed)
        is_empty = (
            not is_animal
            and metadata_labels.get(image_id, set()) == {"empty"}
        )
        if not is_animal and not is_empty:
            diagnostics["unannotated_or_ambiguous"] += 1
            continue

        location = str(
            image_meta.get("location", bbox_meta.get("location", ""))
        ).strip()
        seq_id = str(
            image_meta.get("seq_id", bbox_meta.get("seq_id", ""))
        ).strip()
        date_captured = str(
            image_meta.get(
                "date_captured",
                bbox_meta.get("date_captured", bbox_meta.get("datetime", "")),
            )
        ).strip()
        captured = parse_capture_time(date_captured)
        capture_hour = captured.hour if captured is not None else None
        is_night = (
            hour_is_night(capture_hour, night_start, night_end)
            if capture_hour is not None
            else None
        )

        if not location:
            diagnostics["missing_location"] += 1
        if not seq_id:
            diagnostics["missing_seq_id"] += 1
        if captured is None:
            diagnostics["missing_or_invalid_date"] += 1

        records.append(
            ImageRecord(
                image_id=image_id,
                file_name=Path(file_name).name,
                image_path=str(disk_path),
                location=location,
                seq_id=seq_id,
                frame_num=optional_int(image_meta.get("frame_num")),
                seq_num_frames=optional_int(image_meta.get("seq_num_frames")),
                date_captured=date_captured,
                capture_hour=capture_hour,
                is_night=is_night,
                is_empty=is_empty,
                species=species,
                n_boxes=len(boxed),
            )
        )
        diagnostics["usable"] += 1
        diagnostics["animal"] += int(is_animal)
        diagnostics["empty"] += int(is_empty)
        diagnostics["boxes"] += len(boxed)

    return records, dict(diagnostics)


def totals_for_records(records: Iterable[ImageRecord]) -> Totals:
    totals = Totals()
    for record in records:
        totals.n_images += 1
        totals.n_boxes += record.n_boxes
        totals.n_empty += int(record.is_empty)
        totals.n_night += int(record.is_night is True)
        totals.n_day += int(record.is_night is False)
        if record.is_empty:
            totals.species_frames["empty"] += 1
        else:
            totals.species_boxes.update(record.species)
            totals.species_frames.update(set(record.species))
    return totals


def build_location_groups(records: list[ImageRecord]) -> dict[str, LocationGroup]:
    missing_locations = [record.image_id for record in records if not record.location]
    missing_sequences = [record.image_id for record in records if not record.seq_id]
    if missing_locations:
        raise SystemExit(
            f"ERROR: {len(missing_locations)} usable images lack location; "
            f"examples: {', '.join(missing_locations[:5])}"
        )
    if missing_sequences:
        raise SystemExit(
            f"ERROR: {len(missing_sequences)} usable images lack seq_id; "
            f"examples: {', '.join(missing_sequences[:5])}"
        )

    members: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        members[record.location].append(record)

    return {
        location: LocationGroup(
            location=location,
            records=group_records,
            totals=totals_for_records(group_records),
            sequences={record.seq_id for record in group_records},
        )
        for location, group_records in members.items()
    }


def scaled_target(total: Totals, fraction: float) -> Totals:
    return Totals(
        n_images=total.n_images * fraction,
        n_boxes=total.n_boxes * fraction,
        n_empty=total.n_empty * fraction,
        n_night=total.n_night * fraction,
        n_day=total.n_day * fraction,
        species_boxes=Counter(
            {species: count * fraction for species, count in total.species_boxes.items()}
        ),
        species_frames=Counter(
            {species: count * fraction for species, count in total.species_frames.items()}
        ),
    )


def sum_location_totals(
    locations: Iterable[str], groups: dict[str, LocationGroup]
) -> Totals:
    output = Totals()
    for location in locations:
        output.add(groups[location].totals)
    return output


def relative_error(value: float, target: float) -> float:
    return abs(value - target) / max(target, 1.0)


def species_location_counts(groups: dict[str, LocationGroup]) -> Counter:
    output = Counter()
    for group in groups.values():
        for species, count in group.totals.species_boxes.items():
            if count > 0:
                output[species] += 1
    return output


def eligible_common_species(
    total: Totals, location_counts: Counter
) -> set[str]:
    return {
        species
        for species, count in total.species_boxes.items()
        if count >= COMMON_SPECIES_MIN_TOTAL_BOXES
        and location_counts[species] >= COMMON_SPECIES_MIN_LOCATIONS
    }


def split_cost(
    current: Totals,
    target: Totals,
    split: str,
    all_species: list[str],
    eligible_species: set[str],
) -> float:
    species_error = sum(
        relative_error(current.species_boxes[species], target.species_boxes[species])
        for species in all_species
    ) / max(len(all_species), 1)

    cost = (
        WEIGHT_IMAGES * relative_error(current.n_images, target.n_images)
        + WEIGHT_BOXES * relative_error(current.n_boxes, target.n_boxes)
        + WEIGHT_EMPTY * relative_error(current.n_empty, target.n_empty)
        + WEIGHT_LIGHT
        * (
            relative_error(current.n_night, target.n_night)
            + relative_error(current.n_day, target.n_day)
        )
        / 2.0
        + WEIGHT_SPECIES_PROPORTION * species_error
    )

    for species in all_species:
        count = current.species_boxes[species]
        if split == "train":
            if count == 0:
                cost += PENALTY_TRAIN_SPECIES_ZERO
            continue
        if species not in eligible_species:
            continue
        if count == 0:
            cost += PENALTY_COMMON_SPECIES_ZERO
        elif count < COMMON_SPECIES_MIN_BOXES:
            cost += PENALTY_COMMON_SPECIES_LT_50
        elif count < COMMON_SPECIES_SOFT_MIN_BOXES:
            cost += PENALTY_COMMON_SPECIES_LT_100
    return cost


def enumerate_exact_assignments(
    groups: dict[str, LocationGroup],
    progress_every: int,
) -> tuple[dict[str, str], float, int, dict[str, Totals], set[str], Counter]:
    locations = tuple(sorted(groups))
    expected = TRAIN_LOCATIONS + VAL_LOCATIONS + TEST_LOCATIONS
    if len(locations) != expected:
        raise SystemExit(
            f"ERROR: exact {TRAIN_LOCATIONS}/{VAL_LOCATIONS}/{TEST_LOCATIONS} "
            f"enumeration requires {expected} locations, but found {len(locations)}"
        )

    total = sum_location_totals(locations, groups)
    fractions = {
        "train": TRAIN_LOCATIONS / expected,
        "val": VAL_LOCATIONS / expected,
        "test": TEST_LOCATIONS / expected,
    }
    targets = {split: scaled_target(total, fraction) for split, fraction in fractions.items()}
    all_species = sorted(total.species_boxes)
    location_counts = species_location_counts(groups)
    eligible_species = eligible_common_species(total, location_counts)

    best_score = float("inf")
    best_key: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    best_totals: dict[str, Totals] | None = None
    evaluated = 0

    location_set = set(locations)
    for val_locations in itertools.combinations(locations, VAL_LOCATIONS):
        val_set = set(val_locations)
        remaining = tuple(location for location in locations if location not in val_set)
        val_totals = sum_location_totals(val_locations, groups)
        val_cost = split_cost(
            val_totals, targets["val"], "val", all_species, eligible_species
        )

        for test_locations in itertools.combinations(remaining, TEST_LOCATIONS):
            test_set = set(test_locations)
            train_locations = location_set - val_set - test_set
            test_totals = sum_location_totals(test_locations, groups)
            train_totals = total.copy()
            train_totals.subtract(val_totals)
            train_totals.subtract(test_totals)

            score = (
                val_cost
                + split_cost(
                    test_totals,
                    targets["test"],
                    "test",
                    all_species,
                    eligible_species,
                )
                + split_cost(
                    train_totals,
                    targets["train"],
                    "train",
                    all_species,
                    eligible_species,
                )
            )
            evaluated += 1
            key = (tuple(val_locations), tuple(test_locations))
            if score < best_score or (
                math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
                and (best_key is None or key < best_key)
            ):
                best_score = score
                best_key = key
                best_totals = {
                    "train": train_totals,
                    "val": val_totals.copy(),
                    "test": test_totals,
                }

            if progress_every and evaluated % progress_every == 0:
                print(
                    f"[split] evaluated {evaluated:,} assignments; "
                    f"best score={best_score:.6f}"
                )

    assert best_key is not None
    assert best_totals is not None
    val_locations, test_locations = best_key
    val_set = set(val_locations)
    test_set = set(test_locations)
    assignment = {
        location: (
            "val" if location in val_set else "test" if location in test_set else "train"
        )
        for location in locations
    }
    return assignment, best_score, evaluated, best_totals, eligible_species, location_counts


def apply_assignment(
    records: list[ImageRecord], assignment: dict[str, str]
) -> None:
    for record in records:
        record.split = assignment[record.location]


def validate_assignment(records: list[ImageRecord]) -> dict:
    duplicates = [
        image_id
        for image_id, count in Counter(record.image_id for record in records).items()
        if count > 1
    ]
    if duplicates:
        raise RuntimeError(f"duplicate image IDs, e.g. {duplicates[0]}")

    def crossings(attribute: str) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = defaultdict(set)
        for record in records:
            value = getattr(record, attribute)
            if value:
                mapping[value].add(record.split)
        return {key: values for key, values in mapping.items() if len(values) > 1}

    leaked_locations = crossings("location")
    leaked_sequences = crossings("seq_id")
    if leaked_locations:
        raise RuntimeError(f"location leakage, e.g. {next(iter(leaked_locations.items()))}")
    if leaked_sequences:
        raise RuntimeError(f"sequence leakage, e.g. {next(iter(leaked_sequences.items()))}")

    location_splits = {
        split: {record.location for record in records if record.split == split}
        for split in SPLITS
    }
    expected = {
        "train": TRAIN_LOCATIONS,
        "val": VAL_LOCATIONS,
        "test": TEST_LOCATIONS,
    }
    for split in SPLITS:
        if len(location_splits[split]) != expected[split]:
            raise RuntimeError(
                f"{split} has {len(location_splits[split])} locations; "
                f"expected {expected[split]}"
            )

    return {
        "duplicate_images": len(duplicates),
        "leaked_sequences": len(leaked_sequences),
        "leaked_locations": len(leaked_locations),
        "location_counts": {split: len(location_splits[split]) for split in SPLITS},
    }


def species_box_counts(records: Iterable[ImageRecord]) -> Counter:
    output = Counter()
    for record in records:
        output.update(record.species)
    return output


def species_frame_counts(records: Iterable[ImageRecord]) -> Counter:
    output = Counter()
    for record in records:
        if record.is_empty:
            output["empty"] += 1
        else:
            output.update(set(record.species))
    return output


def objective_config() -> dict:
    return {
        "allocator": "exhaustive_exact_location_enumeration",
        "exact_location_counts": {
            "train": TRAIN_LOCATIONS,
            "val": VAL_LOCATIONS,
            "test": TEST_LOCATIONS,
        },
        "weights": {
            "images": WEIGHT_IMAGES,
            "boxes": WEIGHT_BOXES,
            "empty": WEIGHT_EMPTY,
            "light": WEIGHT_LIGHT,
            "species_proportion": WEIGHT_SPECIES_PROPORTION,
        },
        "common_species": {
            "min_total_boxes": COMMON_SPECIES_MIN_TOTAL_BOXES,
            "min_locations": COMMON_SPECIES_MIN_LOCATIONS,
            "min_boxes": COMMON_SPECIES_MIN_BOXES,
            "soft_min_boxes": COMMON_SPECIES_SOFT_MIN_BOXES,
        },
        "penalties": {
            "train_species_zero": PENALTY_TRAIN_SPECIES_ZERO,
            "common_species_zero": PENALTY_COMMON_SPECIES_ZERO,
            "common_species_lt_50": PENALTY_COMMON_SPECIES_LT_50,
            "common_species_lt_100": PENALTY_COMMON_SPECIES_LT_100,
        },
    }


def build_summary(
    records: list[ImageRecord],
    args: argparse.Namespace,
    diagnostics: dict,
    balance_score: float,
    evaluated: int,
    validation: dict,
    eligible_species: set[str],
    location_counts: Counter,
) -> dict:
    overall_boxes = species_box_counts(records)
    overall_frames = species_frame_counts(records)
    summary = {
        "version": 3,
        "balance_score": balance_score,
        "assignments_evaluated": evaluated,
        "objective": objective_config(),
        "eligible_common_species": sorted(eligible_species),
        "species_location_counts": dict(sorted(location_counts.items())),
        "night_definition": {
            "start_hour": args.night_start_hour,
            "end_hour": args.night_end_hour,
        },
        "source_paths": {
            "images": str(args.images),
            "annotations": str(args.annotations),
            "image_labels": str(args.image_labels),
        },
        "diagnostics": diagnostics,
        "validation": validation,
        "overall": {
            "n_images": len(records),
            "n_locations": len({record.location for record in records}),
            "n_sequences": len({record.seq_id for record in records}),
            "n_boxes": sum(record.n_boxes for record in records),
            "n_empty": sum(record.is_empty for record in records),
            "n_night": sum(record.is_night is True for record in records),
            "n_day": sum(record.is_night is False for record in records),
            "n_unknown_light": sum(record.is_night is None for record in records),
            "species_boxes": dict(sorted(overall_boxes.items())),
            "species_frames": dict(sorted(overall_frames.items())),
        },
        "splits": {},
    }
    for split in SPLITS:
        selected = [record for record in records if record.split == split]
        summary["splits"][split] = {
            "locations": sorted({record.location for record in selected}),
            "n_images": len(selected),
            "fraction_images": len(selected) / len(records) if records else 0.0,
            "n_locations": len({record.location for record in selected}),
            "n_sequences": len({record.seq_id for record in selected}),
            "n_boxes": sum(record.n_boxes for record in selected),
            "n_empty": sum(record.is_empty for record in selected),
            "n_night": sum(record.is_night is True for record in selected),
            "n_day": sum(record.is_night is False for record in selected),
            "n_unknown_light": sum(record.is_night is None for record in selected),
            "species_boxes": dict(sorted(species_box_counts(selected).items())),
            "species_frames": dict(sorted(species_frame_counts(selected).items())),
        }
    return summary


def write_manifest(records: list[ImageRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "file_name",
        "image_path",
        "location",
        "seq_id",
        "frame_num",
        "seq_num_frames",
        "date_captured",
        "capture_hour",
        "is_night",
        "group_id",
        "split",
        "is_empty",
        "n_boxes",
        "species",
    ]
    ordered = sorted(
        records,
        key=lambda record: (
            SPLITS.index(record.split),
            record.location,
            record.seq_id,
            record.frame_num if record.frame_num is not None else -1,
            record.image_id,
        ),
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in ordered:
            writer.writerow(
                {
                    "image_id": record.image_id,
                    "file_name": record.file_name,
                    "image_path": record.image_path,
                    "location": record.location,
                    "seq_id": record.seq_id,
                    "frame_num": record.frame_num if record.frame_num is not None else "",
                    "seq_num_frames": (
                        record.seq_num_frames if record.seq_num_frames is not None else ""
                    ),
                    "date_captured": record.date_captured,
                    "capture_hour": (
                        record.capture_hour if record.capture_hour is not None else ""
                    ),
                    "is_night": (
                        "1" if record.is_night is True else "0" if record.is_night is False else ""
                    ),
                    "group_id": f"location:{record.location}",
                    "split": record.split,
                    "is_empty": int(record.is_empty),
                    "n_boxes": record.n_boxes,
                    "species": ";".join(record.species),
                }
            )


def print_diagnostics(data: dict) -> None:
    print("\n[split] discovery")
    for key in (
        "metadata_image_ids",
        "usable",
        "animal",
        "empty",
        "boxes",
        "missing_on_disk",
        "skipped_car",
        "unannotated_or_ambiguous",
        "missing_location",
        "missing_seq_id",
        "missing_or_invalid_date",
    ):
        print(f"  {key:<28} {data.get(key, 0):>8}")


def print_summary(summary: dict) -> None:
    print("\n[split] selected exact assignment")
    print(f"  assignments evaluated: {summary['assignments_evaluated']:,}")
    print(f"  balance score:        {summary['balance_score']:.6f}")
    print("\n[split] partition totals")
    for split in SPLITS:
        data = summary["splits"][split]
        print(
            f"  {split:<5} images={data['n_images']:>6} "
            f"({data['fraction_images'] * 100:5.1f}%) "
            f"locations={data['n_locations']:>3} sequences={data['n_sequences']:>6} "
            f"boxes={data['n_boxes']:>6} empty={data['n_empty']:>6} "
            f"night={data['n_night']:>6} day={data['n_day']:>6}"
        )
        print(f"        location IDs: {', '.join(data['locations'])}")

    eligible = set(summary["eligible_common_species"])
    print("\n[split] species box coverage")
    print(f"  {'species':<20}{'locs':>7}{'train':>10}{'val':>10}{'test':>10}  status")
    for species in sorted(summary["overall"]["species_boxes"]):
        counts = {
            split: summary["splits"][split]["species_boxes"].get(species, 0)
            for split in SPLITS
        }
        n_locations = summary["species_location_counts"].get(species, 0)
        if counts["train"] == 0:
            status = "ERROR: absent from train"
        elif species not in eligible:
            status = "not coverage-constrained"
        elif counts["val"] == 0 or counts["test"] == 0:
            status = "WARNING: eligible species absent"
        elif counts["val"] < COMMON_SPECIES_MIN_BOXES or counts["test"] < COMMON_SPECIES_MIN_BOXES:
            status = "WARNING: eligible species below 50"
        elif counts["val"] < COMMON_SPECIES_SOFT_MIN_BOXES or counts["test"] < COMMON_SPECIES_SOFT_MIN_BOXES:
            status = "note: eligible species below 100"
        else:
            status = "ok"
        print(
            f"  {species:<20}{n_locations:>7}{counts['train']:>10}"
            f"{counts['val']:>10}{counts['test']:>10}  {status}"
        )

    validation = summary["validation"]
    print("\n[split] leakage validation")
    print(f"  duplicate images:  {validation['duplicate_images']}")
    print(f"  leaked sequences:  {validation['leaked_sequences']}")
    print(f"  leaked locations:  {validation['leaked_locations']}")
    print(f"  location counts:   {validation['location_counts']}")


def main() -> int:
    args = build_parser().parse_args()
    summary_path = validate_args(args)

    print(f"[split] reading bounding-box annotations: {args.annotations}")
    bbox_data = read_json_or_zip(args.annotations)
    print(f"[split] reading image-level metadata:      {args.image_labels}")
    image_data = read_json_or_zip(args.image_labels)

    records, diagnostics = discover_records(
        args.images,
        bbox_data,
        image_data,
        args.night_start_hour,
        args.night_end_hour,
    )
    print_diagnostics(diagnostics)
    if not records:
        raise SystemExit("ERROR: no usable records found")

    groups = build_location_groups(records)
    print(
        f"\n[split] enumerating exact {TRAIN_LOCATIONS}/{VAL_LOCATIONS}/"
        f"{TEST_LOCATIONS} assignments across {len(groups)} locations"
    )
    (
        assignment,
        score,
        evaluated,
        _best_totals,
        eligible_species,
        location_counts,
    ) = enumerate_exact_assignments(groups, args.progress_every)

    apply_assignment(records, assignment)
    validation = validate_assignment(records)
    summary = build_summary(
        records,
        args,
        diagnostics,
        score,
        evaluated,
        validation,
        eligible_species,
        location_counts,
    )

    write_manifest(records, args.output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print_summary(summary)
    print(f"\n[split] wrote manifest: {args.output}")
    print(f"[split] wrote summary:  {summary_path}")
    print("[split] Review the exact candidate before committing and freezing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
Dataset loading and stratified frame selection for the evaluation harness.

Owner: Jason Johnson

CCT-20 ships two annotation files and they answer different questions:

  * the bbox file (caltech_bboxes_20200316.json) has species-level boxes for
    frames that contain animals. It has NO entries for animal-free frames.
  * the image-level file (caltech_camera_traps.json) has one label per image
    and is the only source of genuinely empty frames.

Both are needed. Without the image-level file the empty-frame false-positive
metric has no data to compute on.

Stratification
--------------
A plain random sample is not a good sample here. The three ways a frame can
be unrepresentative:

  1. Empty frames absent -> the false-positive metric is undefined.
  2. Day frames mixed in -> a brightness-oriented enhancement may look
     different on a daylight frame, which is a useful control but dilutes
     the night-frame signal if unintentional.
  3. One species dominating -> the result describes squirrels, not wildlife.

So the sample is stratified on: empty vs animal, night vs day, and species.
"""
from __future__ import annotations

import json
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from evaluation.config import StrataConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# Frames containing a car are skipped: a car is a real object but not an
# animal, so scoring it as one would be unfair to every method.
NON_ANIMAL_CATEGORIES = {"car"}


@dataclass
class Frame:
    """One evaluation frame, with its ground truth already scaled to the
    on-disk pixel space."""
    path: str
    image_id: str
    boxes: np.ndarray                 # (n, 4) float xyxy, on-disk pixel space
    species: list[str] = field(default_factory=list)
    is_empty: bool = False
    is_night: bool = True
    datetime: str = ""
    # The annotation's own image dimensions, kept so the boxes can be
    # rescaled if they are not in the on-disk pixel space. CCT-20's images
    # are downsized to a maximum of 1024 px per side, but the bbox JSON
    # carries coordinates in the ORIGINAL image space.
    ann_width: int = 0
    ann_height: int = 0

    @property
    def n_animals(self) -> int:
        return len(self.species)

    @property
    def key(self) -> str:
        return self.path


@dataclass
class Dataset:
    """Loaded annotations plus the on-disk image index."""
    frames: list[Frame]
    categories: dict[str, int]
    n_images_total: int = 0
    n_skipped_car: int = 0
    n_empty_from_labels: int = 0


# ----------------------------------------------------------------------
# Annotation loading
# ----------------------------------------------------------------------
def read_json_or_zip(path: Path) -> dict:
    """Read a JSON annotation file, transparently handling a .zip archive."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist()
                        if n.lower().endswith(".json")
                        and not n.startswith("__MACOSX"))
            return json.loads(z.read(name))
    return json.loads(path.read_text(encoding="utf-8"))


def index_images_on_disk(images_dir: Path) -> dict[str, Path]:
    """Map image basename -> path for every image under images_dir.

    CCT file names are unique, so a flat basename index is enough and avoids
    a 63k-entry directory walk per lookup.
    """
    index: dict[str, Path] = {}
    for p in Path(images_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.name, p)
    return index


def _parse_datetime(value: str) -> datetime | None:
    """Parse the timestamp formats that appear in camera-trap metadata."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(value[:len(fmt) + 6].strip(), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_night_datetime(value: str, strata: StrataConfig) -> bool:
    """Night = hour >= night_start_hour OR hour < night_end_hour.

    Falls back to True (night) when the timestamp is missing or unparseable,
    because the project is about nocturnal imagery. Daylight frames are the
    minority in this corpus anyway, so a fallback toward night keeps the
    sample on-topic.
    """
    dt = _parse_datetime(value)
    if dt is None:
        return True
    h = dt.hour
    if strata.night_start_hour > strata.night_end_hour:
        return h >= strata.night_start_hour or h < strata.night_end_hour
    return strata.night_start_hour <= h < strata.night_end_hour


def empty_image_ids_from_labels(path: Path) -> dict[str, str]:
    """image_id -> file name for frames whose ONLY label is 'empty'."""
    data = read_json_or_zip(path)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    labels: dict[str, set] = defaultdict(set)
    for a in data["annotations"]:
        labels[str(a["image_id"])].add(cats.get(a["category_id"]))
    names = {str(im["id"]): im.get("file_name", f"{im['id']}.jpg")
             for im in data["images"]}
    return {i: names.get(i, f"{i}.jpg")
            for i, l in labels.items() if l == {"empty"}}


# ----------------------------------------------------------------------
# Frame construction
# ----------------------------------------------------------------------
def build_dataset(
    images_dir: Path,
    annotations: Path,
    image_labels: Path | None = None,
    strata: StrataConfig | None = None,
) -> Dataset:
    """Load annotations and build a Frame for every usable image on disk."""
    strata = strata or StrataConfig()
    data = read_json_or_zip(annotations)
    cats = {c["id"]: c["name"] for c in data["categories"]}

    by_img: dict[str, list] = defaultdict(list)
    for a in data["annotations"]:
        by_img[str(a["image_id"])].append(a)

    on_disk = index_images_on_disk(images_dir)
    frames: list[Frame] = []
    n_car = 0
    seen_paths: set[str] = set()

    for im in data["images"]:
        iid = str(im["id"])
        anns = by_img.get(iid, [])
        names = {cats.get(a["category_id"]) for a in anns}
        if names & NON_ANIMAL_CATEGORIES:
            n_car += 1
            continue

        fname = im.get("file_name", f"{iid}.jpg")
        p = on_disk.get(Path(fname).name) or on_disk.get(f"{iid}.jpg")
        if p is None:
            continue

        # Box annotations carry a 'bbox'; the 'empty' label usually does not.
        boxes_raw = [a for a in anns
                     if "bbox" in a and cats.get(a["category_id"]) != "empty"]
        species = [cats.get(a["category_id"], "?") for a in boxes_raw]

        if boxes_raw:
            gt = np.array(
                [[a["bbox"][0], a["bbox"][1],
                  a["bbox"][0] + a["bbox"][2],
                  a["bbox"][1] + a["bbox"][3]] for a in boxes_raw],
                dtype=float).reshape(-1, 4)
            frames.append(Frame(
                path=str(p), image_id=iid, boxes=gt, species=species,
                is_empty=False,
                is_night=is_night_datetime(im.get("datetime", ""), strata),
                datetime=im.get("datetime", ""),
                ann_width=int(im.get("width") or 0),
                ann_height=int(im.get("height") or 0),
            ))
            seen_paths.add(str(p))
        elif "empty" in names:
            frames.append(Frame(
                path=str(p), image_id=iid, boxes=np.zeros((0, 4), float),
                species=[], is_empty=True,
                is_night=is_night_datetime(im.get("datetime", ""), strata),
                datetime=im.get("datetime", ""),
                ann_width=int(im.get("width") or 0),
                ann_height=int(im.get("height") or 0),
            ))
            seen_paths.add(str(p))

    # Empty frames that only the image-level file knows about.
    n_extra = 0
    if image_labels:
        for iid, fname in empty_image_ids_from_labels(image_labels).items():
            p = on_disk.get(Path(fname).name) or on_disk.get(f"{iid}.jpg")
            if p is None or str(p) in seen_paths:
                continue
            frames.append(Frame(
                path=str(p), image_id=iid, boxes=np.zeros((0, 4), float),
                species=[], is_empty=True, is_night=True, datetime="",
            ))
            seen_paths.add(str(p))
            n_extra += 1

    return Dataset(frames=frames, categories=cats,
                   n_images_total=len(data["images"]),
                   n_skipped_car=n_car, n_empty_from_labels=n_extra)


# ----------------------------------------------------------------------
# Stratified selection
# ----------------------------------------------------------------------
def select_frames(dataset: Dataset, max_images: int, strata: StrataConfig,
                  seed: int = 0) -> list[Frame]:
    """Pick a stratified sample of at most `max_images` frames.

    Allocation:
      * `empty_fraction` of the budget goes to animal-free frames.
      * The rest goes to animal frames, split between night and day using
        `day_fraction` (of the animal budget).
      * Animal frames are then spread across species using `species_balance`.

    The result is sorted by path so the selection is deterministic for a
    given seed, independent of dict ordering.
    """
    rng = random.Random(seed)

    animals = [f for f in dataset.frames if not f.is_empty]
    empties = [f for f in dataset.frames if f.is_empty]

    n_empty = min(int(round(max_images * strata.empty_fraction)), len(empties))
    n_animal = min(max_images - n_empty, len(animals))

    # Day/night split inside the animal budget.
    n_day = min(int(round(n_animal * strata.day_fraction)),
                sum(1 for f in animals if not f.is_night))
    n_night = min(n_animal - n_day, sum(1 for f in animals if f.is_night))

    by_species_day: dict[str, list[Frame]] = defaultdict(list)
    by_species_night: dict[str, list[Frame]] = defaultdict(list)
    for f in animals:
        key = f.species[0] if f.species else "?"
        (by_species_night if f.is_night else by_species_day)[key].append(f)

    def take_spread(buckets: dict[str, list[Frame]], quota: int) -> list[Frame]:
        """Spread `quota` frames across species.

        "natural": proportional to how many frames each species has, which
        keeps the sample representative of the dataset.
        "uniform": as even as the available frames allow.
        """
        if quota <= 0 or not buckets:
            return []
        for v in buckets.values():
            rng.shuffle(v)
        out: list[Frame] = []
        if strata.species_balance == "uniform":
            names = sorted(buckets)
            cursor = {n: 0 for n in names}
            while len(out) < quota:
                progressed = False
                for n in names:
                    if len(out) >= quota:
                        break
                    i = cursor[n]
                    if i < len(buckets[n]):
                        out.append(buckets[n][i])
                        cursor[n] = i + 1
                        progressed = True
                if not progressed:
                    break
        else:
            names = sorted(buckets, key=lambda n: -len(buckets[n]))
            total = sum(len(buckets[n]) for n in names) or 1
            # Proportional quota with a floor so rare species are represented.
            quotas = {}
            for n in names:
                quotas[n] = max(strata.min_per_species,
                                int(round(quota * len(buckets[n]) / total)))
            for n in names:
                take = min(quotas[n], len(buckets[n]))
                out.extend(buckets[n][:take])
            # Top up or trim to the exact quota.
            if len(out) > quota:
                rng.shuffle(out)
                out = out[:quota]
            else:
                pool = [f for n in names for f in buckets[n][quotas[n]:]]
                rng.shuffle(pool)
                out.extend(pool[:quota - len(out)])
        return out

    picked = take_spread(by_species_night, n_night)
    picked += take_spread(by_species_day, n_day)
    picked += empties[:n_empty]
    picked.sort(key=lambda f: f.path)
    return picked


def scale_boxes_to_disk(frame: Frame, on_disk_shape: tuple[int, int]) -> Frame:
    """Rescale a frame's ground-truth boxes into the on-disk pixel space.

    CCT-20's benchmark images are downsized to a maximum of 1024 px per
    side, but caltech_bboxes_20200316.json carries bbox coordinates in the
    ORIGINAL image space. The annotation's own width/height are recorded on
    the Frame, so the scale factor is
    (on-disk width / annotation width, on-disk height / annotation height).

    If the annotation dimensions are missing, or already match the on-disk
    dimensions, the boxes are left alone. If the image cannot be read, they
    are left alone too, and the caller warns.

    Skipping this makes every ground-truth box land in the wrong place, so
    recall collapses to exactly zero while the detector still fires. That
    failure looks like "the detector is useless" rather than "the boxes are
    misaligned", so it is worth being explicit about.
    """
    h_disk, w_disk = on_disk_shape[:2]
    if not frame.ann_width or not frame.ann_height:
        return frame
    sx = w_disk / frame.ann_width
    sy = h_disk / frame.ann_height
    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        return frame
    if frame.boxes.size:
        scaled = frame.boxes.copy()
        scaled[:, 0] *= sx
        scaled[:, 2] *= sx
        scaled[:, 1] *= sy
        scaled[:, 3] *= sy
        frame.boxes = scaled
    return frame


def summarise_sample(frames: list[Frame]) -> dict:
    """Counts for the run manifest: how the sample is composed."""
    from collections import Counter
    species = Counter()
    for f in frames:
        for s in (f.species or ["empty"]):
            species[s] += 1
    return {
        "n_frames": len(frames),
        "n_empty": sum(1 for f in frames if f.is_empty),
        "n_night": sum(1 for f in frames if f.is_night),
        "n_day": sum(1 for f in frames if not f.is_night),
        "n_boxes": sum(f.n_animals for f in frames),
        "species": dict(species.most_common()),
        # True only if every frame carries a real timestamp. The CCT-20
        # box annotations have no datetime field, so the day/night split
        # is a DEFAULT in that case and must not be reported as a
        # measurement.
        "day_night_measured": all(bool(f.datetime) for f in frames),
    }

"""Split filtering and pixel-source plumbing.

Everything here uses synthetic fixtures: four 0-byte jpegs, a four-image
annotation file and a four-row split manifest. No CCT data, no checkpoints,
no GPU. The point is to pin the behaviour that decides WHICH frames are
scored, because a wrong frame set silently invalidates every comparison
downstream and nothing else in the suite would catch it.

Owner: Jason Johnson
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.config import RunConfig                        # noqa: E402
from evaluation.dataset import (build_dataset, filter_by_split,  # noqa: E402
                                load_split_map, resolve_frame_path)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
def _write_fixtures(root: Path) -> tuple[Path, Path, Path]:
    """Four frames: a=train animal, b=val animal, c=test empty, d=test animal."""
    import cv2
    import numpy as np

    imgs = root / "raw"
    imgs.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        # Real (tiny) JPEGs, so cv2.imread succeeds in the predictor tests.
        ok = cv2.imwrite(str(imgs / name),
                         np.full((8, 8, 3), 100, np.uint8))
        assert ok, f"could not write fixture image {name}"

    ann = {
        "categories": [{"id": 1, "name": "coyote"},
                       {"id": 2, "name": "empty"}],
        "images": [
            {"id": "a", "file_name": "a.jpg", "width": 64, "height": 64,
             "datetime": "2018-01-01 20:00:00"},
            {"id": "b", "file_name": "b.jpg", "width": 64, "height": 64,
             "datetime": "2018-01-01 21:00:00"},
            {"id": "c", "file_name": "c.jpg", "width": 64, "height": 64,
             "datetime": "2018-01-01 22:00:00"},
            {"id": "d", "file_name": "d.jpg", "width": 64, "height": 64,
             "datetime": "2018-01-01 23:00:00"},
        ],
        "annotations": [
            {"image_id": "a", "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"image_id": "b", "category_id": 1, "bbox": [5, 5, 15, 15]},
            {"image_id": "c", "category_id": 2},
            {"image_id": "d", "category_id": 1, "bbox": [1, 2, 3, 4]},
        ],
    }
    ann_path = root / "annotations.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")

    manifest = root / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "file_name", "split", "is_empty",
                    "image_path", "location"])
        for iid, name, split in (("a", "a.jpg", "train"),
                                 ("b", "b.jpg", "val"),
                                 ("c", "c.jpg", "test"),
                                 ("d", "d.jpg", "test")):
            w.writerow([iid, name, split, 0,
                        f"raw/{name}", f"location:{split}"])
    return imgs, ann_path, manifest


# ----------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------
def test_load_split_map_reads_all_rows() -> None:
    with tempfile.TemporaryDirectory() as td:
        _, _, manifest = _write_fixtures(Path(td))
        m = load_split_map(manifest)
    assert m == {"a": "train", "b": "val", "c": "test", "d": "test"}, m


def test_load_split_map_rejects_missing_split_column() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.csv"
        p.write_text("image_id,file_name\na,a.jpg\n", encoding="utf-8")
        try:
            load_split_map(p)
        except SystemExit as exc:
            assert "split" in str(exc), exc
            return
        raise AssertionError("expected SystemExit for a manifest with no split column")


def test_empty_split_returns_every_frame() -> None:
    with tempfile.TemporaryDirectory() as td:
        imgs, ann, manifest = _write_fixtures(Path(td))
        ds = build_dataset(imgs, ann)
        frames = filter_by_split(ds.frames, load_split_map(manifest), "")
    assert len(frames) == 4, f"expected all 4 frames, got {len(frames)}"


def test_split_filter_keeps_only_that_split() -> None:
    with tempfile.TemporaryDirectory() as td:
        imgs, ann, manifest = _write_fixtures(Path(td))
        ds = build_dataset(imgs, ann)
        split_map = load_split_map(manifest)
        test_ids = sorted(f.image_id for f in filter_by_split(ds.frames, split_map, "test"))
        train_ids = sorted(f.image_id for f in filter_by_split(ds.frames, split_map, "train"))
    assert test_ids == ["c", "d"], test_ids
    assert train_ids == ["a"], train_ids


def test_split_filter_preserves_ground_truth() -> None:
    """Filtering must not mutate or drop boxes on the frames it keeps."""
    with tempfile.TemporaryDirectory() as td:
        imgs, ann, manifest = _write_fixtures(Path(td))
        ds = build_dataset(imgs, ann)
        test_frames = filter_by_split(ds.frames, load_split_map(manifest), "test")
    by_id = {f.image_id: f for f in test_frames}
    assert by_id["c"].is_empty, "c should be an empty frame"
    assert len(by_id["c"].boxes) == 0
    assert by_id["d"].boxes.shape == (1, 4), by_id["d"].boxes
    assert by_id["d"].species == ["coyote"], by_id["d"].species


def test_fingerprint_unchanged_for_default_config() -> None:
    """Backwards compatibility: a pre-existing cache key must stay valid."""
    cfg = RunConfig()
    inputs = cfg.fingerprint_inputs(["a", "b"], "")
    assert "split" not in inputs, "default config must not add a split key"
    assert "pixel_source" not in inputs, "default config must not add a pixel_source key"


def test_fingerprint_tracks_split_and_pixel_source() -> None:
    ids = ["a", "b"]
    base = RunConfig()
    base_fp = base.analysis_fingerprint(ids, "")

    split_cfg = RunConfig(split="test")
    assert "split" in split_cfg.fingerprint_inputs(ids, "")
    assert split_cfg.analysis_fingerprint(ids, "") != base_fp, \
        "changing the split must change the fingerprint"

    pre_cfg = RunConfig(pixel_source="precomputed")
    assert "pixel_source" in pre_cfg.fingerprint_inputs(ids, "")
    assert pre_cfg.analysis_fingerprint(ids, "") != base_fp, \
        "changing the pixel source must change the fingerprint"


def test_scoring_params_still_do_not_touch_the_fingerprint() -> None:
    """conf/bootstrap/seed are scoring-only: they must never invalidate a cache."""
    ids = ["a", "b", "c"]
    a = RunConfig(split="test", pixel_source="precomputed", conf=0.25)
    b = RunConfig(split="test", pixel_source="precomputed", conf=0.70)
    c = RunConfig(split="test", pixel_source="precomputed", bootstrap=5, seed=99)
    fa = a.analysis_fingerprint(ids, "")
    assert fa == b.analysis_fingerprint(ids, ""), "conf must not invalidate the cache"
    assert fa == c.analysis_fingerprint(ids, ""), "bootstrap/seed must not invalidate the cache"


def test_yaml_round_trip_preserves_new_fields() -> None:
    with tempfile.TemporaryDirectory() as td:
        cfg = RunConfig(images_dir="data/raw", annotations="data/ann.json",
                        split_manifest="data/cct20_splits/split_manifest.csv",
                        split="test", pixel_source="precomputed", methods=["raw"])
        p = Path(td) / "config.yaml"
        cfg.to_yaml(p)
        back = RunConfig.from_yaml(p)
    assert back.split == "test", back.split
    assert back.pixel_source == "precomputed", back.pixel_source
    assert back.split_manifest == "data/cct20_splits/split_manifest.csv", back.split_manifest


def test_resolved_makes_split_manifest_absolute() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = RunConfig(split_manifest="data/split_manifest.csv")
        out = cfg.resolved(root)
    assert Path(out.split_manifest).is_absolute(), out.split_manifest
    assert Path(out.split_manifest) == (root / "data/split_manifest.csv")


# ----------------------------------------------------------------------
# Pixel source: the branch that decides whether the enhancement runs
# ----------------------------------------------------------------------
class _Tensorish:
    """Minimal stand-in for a torch tensor: only cpu() and numpy()."""

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Tensorish(xyxy)
        self.conf = _Tensorish(conf)
        self.cls = _Tensorish(cls)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


def _fake_model(calls: dict):
    """One detection per frame, at fixed coordinates."""
    def model(_img, conf=0.001, imgsz=640, verbose=False):
        calls["detector"] = calls.get("detector", 0) + 1
        return [_Result(_Boxes([[1, 1, 5, 5]], [0.9], [0]))]
    return model


def _run_frames(td: Path, pixel_source: str) -> tuple[object, dict]:
    """Run the predictor over two fixture frames with a counting enhancer."""
    import numpy as np

    from evaluation import predict as predict_mod
    from evaluation.dataset import Frame

    imgs, _ann, _manifest = _write_fixtures(td)
    frames = [Frame(path=str(imgs / n), image_id=n[:-4],
                    boxes=np.zeros((0, 4), float))
              for n in ("c.jpg", "d.jpg")]

    calls = {"enhance": 0}

    class _StubEnh:
        def enhance(self, img):
            calls["enhance"] += 1
            return img

    real = predict_mod._get_enhancement
    predict_mod._get_enhancement = lambda *a, **k: _StubEnh()
    try:
        preds = predict_mod.run_method(
            "raw", frames, _fake_model(calls), 64, 0.001, td,
            pixel_source=pixel_source,
        )
    finally:
        predict_mod._get_enhancement = real
    return preds, calls


def test_manifest_path_relative_resolves_under_repo_root() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixtures(root)
        got = resolve_frame_path("raw/a.jpg", None, root)
    assert got is not None and got.name == "a.jpg", got


def test_manifest_path_absolute_that_exists_is_used() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixtures(root)
        target = root / "raw" / "b.jpg"
        got = resolve_frame_path(str(target), None, root)
    assert got == target, got


def test_absolute_path_from_another_machine_falls_back_to_images_dir() -> None:
    """A manifest written on someone else's /work must still resolve here."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        imgs, _ann, _m = _write_fixtures(root)
        foreign = "/work/someone-else/cs5243-lowlight-yolo/data/cct20_snr/images/c.jpg"
        got = resolve_frame_path(foreign, imgs, root)
    assert got is not None, "did not fall back to the images dir"
    assert got.name == "c.jpg", got
    assert str(got).startswith(str(imgs)), f"resolution escaped images_dir: {got}"


def test_unresolvable_path_returns_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        got = resolve_frame_path("/work/nobody/images/nope.jpg", None, root)
    assert got is None, got


def test_precomputed_pixels_are_never_enhanced() -> None:
    """The whole point: precomputed frames must reach the detector unchanged."""
    with tempfile.TemporaryDirectory() as td:
        preds, calls = _run_frames(Path(td), "precomputed")
    assert calls["enhance"] == 0, \
        f"enhancement ran {calls['enhance']} times on already-enhanced pixels"
    assert calls["detector"] == 2, calls["detector"]
    assert preds.n_frames == 2
    assert preds.enh_ms == 0.0, f"expected 0 enh latency, got {preds.enh_ms}"


def test_on_the_fly_enhances_every_frame() -> None:
    with tempfile.TemporaryDirectory() as td:
        preds, calls = _run_frames(Path(td), "on_the_fly")
    assert calls["enhance"] == 2, calls["enhance"]
    assert calls["detector"] == 2, calls["detector"]
    assert preds.n_frames == 2


# ----------------------------------------------------------------------
def main() -> int:
    checks = [
        ("load_split_map reads every row", test_load_split_map_reads_all_rows),
        ("load_split_map rejects a manifest with no split column",
         test_load_split_map_rejects_missing_split_column),
        ("empty split returns every frame", test_empty_split_returns_every_frame),
        ("split filter keeps only that split", test_split_filter_keeps_only_that_split),
        ("split filter preserves ground truth", test_split_filter_preserves_ground_truth),
        ("default config adds no fingerprint keys", test_fingerprint_unchanged_for_default_config),
        ("fingerprint tracks split and pixel_source", test_fingerprint_tracks_split_and_pixel_source),
        ("scoring params do not touch the fingerprint",
         test_scoring_params_still_do_not_touch_the_fingerprint),
        ("yaml round trip preserves new fields", test_yaml_round_trip_preserves_new_fields),
        ("resolved() makes split_manifest absolute",
         test_resolved_makes_split_manifest_absolute),
        ("precomputed pixels are never enhanced",
         test_precomputed_pixels_are_never_enhanced),
        ("on_the_fly enhances every frame",
         test_on_the_fly_enhances_every_frame),
        ("manifest relative path resolves under repo root",
         test_manifest_path_relative_resolves_under_repo_root),
        ("manifest absolute path that exists is used",
         test_manifest_path_absolute_that_exists_is_used),
        ("foreign absolute path falls back to images dir",
         test_absolute_path_from_another_machine_falls_back_to_images_dir),
        ("unresolvable path returns None", test_unresolvable_path_returns_none),
    ]
    passed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  ok   {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

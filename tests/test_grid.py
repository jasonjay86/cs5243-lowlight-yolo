"""End-to-end grid run on synthetic fixtures.

No real weights, no CCT images, no GPU. Two conditions, two checkpoints,
three cells. What is being pinned is the plumbing that decides which pixels
each cell sees and that the table comes out with the right shape.

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

from evaluation.config import RunConfig          # noqa: E402
from evaluation.grid import run_grid             # noqa: E402


# ----------------------------------------------------------------------
# Fixtures: two conditions with their own image directory and manifest
# ----------------------------------------------------------------------
def _write_fixtures(root: Path) -> dict:
    import cv2

    raw, enh = root / "raw", root / "enh"
    raw.mkdir(parents=True)
    enh.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        assert cv2.imwrite(str(raw / name), np.full((8, 8, 3), 80, np.uint8))
        assert cv2.imwrite(str(enh / name), np.full((8, 8, 3), 180, np.uint8))

    ann = {
        "categories": [{"id": 1, "name": "coyote"}, {"id": 2, "name": "empty"}],
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": 64, "height": 64,
                    "datetime": "2018-01-01 22:00:00"} for i in "abcd"],
        "annotations": [
            {"image_id": "a", "category_id": 1, "bbox": [1, 1, 3, 3]},
            {"image_id": "b", "category_id": 1, "bbox": [2, 2, 4, 4]},
            {"image_id": "c", "category_id": 2},
            {"image_id": "d", "category_id": 1, "bbox": [1, 2, 3, 4]},
        ],
    }
    ann_path = root / "annotations.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")

    splits = {"a": "train", "b": "val", "c": "test", "d": "test"}
    manifests = {}
    for cond, sub in (("raw", "raw"), ("enh", "enh")):
        p = root / f"{cond}_manifest.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image_id", "image_path", "split"])
            for iid, split in splits.items():
                w.writerow([iid, f"{sub}/{iid}.jpg", split])
        manifests[cond] = p

    return {"raw_dir": raw, "ann": ann_path, "manifests": manifests}


# ----------------------------------------------------------------------
# Fake detector
# ----------------------------------------------------------------------
class _T:
    def __init__(self, a):
        self._a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)


class _Res:
    def __init__(self):
        self.boxes = _Boxes([[1, 1, 5, 5]], [0.93], [0])


class _Model:
    def __call__(self, _img, conf=0.001, imgsz=640, verbose=False):
        return [_Res()]


def _install_fake_model():
    import evaluation.predict as predict_mod
    real = predict_mod._load_model
    predict_mod._load_model = lambda *a, **k: _Model()
    return lambda: setattr(predict_mod, "_load_model", real)


def _config(fx: dict, root: Path, **kw) -> RunConfig:
    cfg = RunConfig(
        images_dir=str(fx["raw_dir"]),
        annotations=str(fx["ann"]),
        split_manifest=str(fx["manifests"]["raw"]),
        split="test",
        pixel_source="precomputed",
        methods=["raw"],
        weights={"raw": str(root / "raw.pt"), "enh": str(root / "enh.pt")},
        conditions={k: str(v) for k, v in fx["manifests"].items()},
        cell_mode="minimal",
        bootstrap=50,
        conf=0.25,
        run_id="grid-test",
        repo_root=str(root),
    )
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


def _run(root: Path, cfg: RunConfig) -> dict:
    return run_grid(cfg, root, experiments_root=root / "experiments")


def test_minimal_grid_produces_three_cells() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _write_fixtures(root)
        restore = _install_fake_model()
        try:
            out = _run(root, _config(fx, root))
        finally:
            restore()
    kinds = [(r["model"], r["condition"], r["kind"]) for r in out["rows"]]
    assert kinds == [
        ("raw", "raw", "Q1+Q2"),
        ("raw", "enh", "Q1"),
        ("enh", "enh", "Q2"),
    ], kinds


def test_every_cell_scores_the_test_frames_only() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _write_fixtures(root)
        restore = _install_fake_model()
        try:
            out = _run(root, _config(fx, root))
        finally:
            restore()
    # c and d are the test rows
    assert {r["n_frames"] for r in out["rows"]} == {2}, \
        [r["n_frames"] for r in out["rows"]]


def test_csv_is_written_with_every_cell() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _write_fixtures(root)
        restore = _install_fake_model()
        try:
            out = _run(root, _config(fx, root))
        finally:
            restore()
        # Read inside the temporary directory: it is deleted on exit.
        path = Path(out["csv"])
        assert path.is_file(), path
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # Provenance: the table must be reproducible from what was written.
        cfg_path = Path(out["run_dir"]) / "config.yaml"
        assert cfg_path.is_file(), f"no config.yaml written to {out['run_dir']}"
        import yaml
        stored = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert stored["split"] == "test", stored.get("split")
        assert set(stored["weights"]) == {"raw", "enh"}, stored.get("weights")
    assert len(rows) == 3, rows
    for column in ("map50", "d_map50", "empty_fp", "kind", "precision",
                   "enh_ms", "yolo_ms", "gflops_mp", "n_detections",
                   "map50_lo", "map50_hi", "fp_lo", "fp_hi"):
        assert column in rows[0], f"missing column {column}"


def test_delta_is_zero_for_the_baseline_cell() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _write_fixtures(root)
        restore = _install_fake_model()
        try:
            out = _run(root, _config(fx, root))
        finally:
            restore()
    base = out["rows"][0]
    assert base["kind"] == "Q1+Q2"
    assert abs(base["d_map50"]) < 1e-9, base["d_map50"]


def test_second_run_reuses_cached_predictions() -> None:
    """The whole point of the .npz cache: no re-inference on a rescore."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _write_fixtures(root)
        cfg = _config(fx, root)
        calls = {"n": 0}

        import evaluation.predict as predict_mod
        real_load = predict_mod._load_model

        class _Counting(_Model):
            def __call__(self, img, conf=0.001, imgsz=640, verbose=False):
                calls["n"] += 1
                return super().__call__(img, conf, imgsz, verbose)

        predict_mod._load_model = lambda *a, **k: _Counting()
        try:
            _run(root, cfg)
            first = calls["n"]
            _run(root, cfg)
            second = calls["n"] - first
        finally:
            predict_mod._load_model = real_load
    assert first == 6, f"expected 6 detector calls on the first run (3 cells x 2 frames), got {first}"
    assert second == 0, f"second run re-ran inference {second} time(s)"


def test_breakdown_sidecar_is_written() -> None:
    """Night/day, presence and species views, per cell, next to the csv."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _write_fixtures(root)
        restore = _install_fake_model()
        try:
            out = _run(root, _config(fx, root))
            path = Path(out["run_dir"]) / "grid_breakdown.json"
            assert path.is_file(), f"no breakdown written to {out['run_dir']}"
            data = json.loads(path.read_text(encoding="utf-8"))
        finally:
            restore()
    assert set(data) == {"raw__raw", "raw__enh", "enh__enh"}, set(data)
    for cell, entry in data.items():
        for key in ("by_light", "by_presence", "by_species"):
            assert key in entry, f"{cell} missing {key}"


def test_mismatched_condition_frame_set_is_rejected() -> None:
    """If a condition is missing a frame, the delta is meaningless."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        fx = _write_fixtures(root)
        # Drop 'd' from the enh manifest, so that condition resolves 1 frame.
        p = fx["manifests"]["enh"]
        rows = p.read_text(encoding="utf-8").splitlines()
        p.write_text("\n".join(r for r in rows if not r.startswith("d,")) + "\n",
                     encoding="utf-8")
        restore = _install_fake_model()
        try:
            _run(root, _config(fx, root))
        except SystemExit as exc:
            assert "same frames" in str(exc), exc
            return
        finally:
            restore()
    raise AssertionError("a short condition frame set should have been rejected")


# ----------------------------------------------------------------------
def main() -> int:
    checks = [
        ("minimal grid produces three cells",
         test_minimal_grid_produces_three_cells),
        ("every cell scores the test frames only",
         test_every_cell_scores_the_test_frames_only),
        ("csv written with every cell", test_csv_is_written_with_every_cell),
        ("baseline delta is zero", test_delta_is_zero_for_the_baseline_cell),
        ("second run reuses cached predictions",
         test_second_run_reuses_cached_predictions),
        ("mismatched frame sets are rejected",
         test_mismatched_condition_frame_set_is_rejected),
    ]
    passed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  ok   {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

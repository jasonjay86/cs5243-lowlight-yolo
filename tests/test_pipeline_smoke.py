"""
test_pipeline_smoke.py — end-to-end test of the harness with synthetic data.

Owner: Jason Johnson

The unit tests cover the metric arithmetic. This one exercises the whole
orchestration path with a fake dataset and a fake detector, so that the
plumbing (config -> select -> predict -> cache -> score -> report) is known
to work before anyone spends GPU hours on it.

Run from the repo root:

    python tests/test_pipeline_smoke.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.config import RunConfig, StrataConfig  # noqa: E402
from evaluation.harness import run_evaluation             # noqa: E402


def _write_synthetic_dataset(root: Path) -> tuple[Path, Path, Path]:
    """Build a tiny CCT-shaped dataset: images, bbox JSON, image-label JSON."""
    import cv2

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": 1, "name": "opossum"},
        {"id": 2, "name": "raccoon"},
        {"id": 3, "name": "empty"},
    ]
    images, annotations = [], []
    ann_id = 1

    # 6 animal frames, 4 empty frames.
    for i in range(10):
        fname = f"frame_{i:03d}.jpg"
        path = images_dir / fname
        # A dark image with a bright-ish blob where the animal is.
        img = np.full((240, 320, 3), 25, dtype=np.uint8)
        noise = np.random.default_rng(i).normal(0, 3, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if i < 6:
            x, y = 40 + 20 * i, 60
            img[y:y + 60, x:x + 80] = 150
        cv2.imwrite(str(path), img)

        images.append({
            "id": str(i), "file_name": fname, "width": 320, "height": 240,
            "datetime": f"2021-03-{(i % 28) + 1:02d} 23:{(i * 7) % 60:02d}:00",
        })
        if i < 6:
            annotations.append({
                "id": str(ann_id), "image_id": str(i), "category_id": 1,
                "bbox": [40 + 20 * i, 60, 80, 60],
            })
            ann_id += 1
        else:
            annotations.append({
                "id": str(ann_id), "image_id": str(i), "category_id": 3,
            })
            ann_id += 1

    ann_path = root / "bbox.json"
    ann_path.write_text(json.dumps({
        "images": images, "annotations": annotations, "categories": categories,
    }), encoding="utf-8")

    # Image-level labels: the empties are labelled 'empty' only.
    labels_path = root / "labels.json"
    labels_path.write_text(json.dumps({
        "images": [{"id": str(i), "file_name": f"frame_{i:03d}.jpg"} for i in range(10)],
        "annotations": [{"id": str(100 + i), "image_id": str(i), "category_id": 3}
                        for i in range(6, 10)],
        "categories": categories,
    }), encoding="utf-8")

    return images_dir, ann_path, labels_path


def _install_fake_enhancements():
    """Provide a minimal `enhancements` package if the real one is absent.

    The real package lives at the repo root and is what the harness uses in
    production. In a bare checkout (or a sandbox without the repo) this stub
    stands in so the pipeline plumbing can still be exercised.

    The stub implements the same contract as enhancements/base.py: an
    Enhancement ABC, a @register decorator, and get_enhancement(name).
    """
    try:
        import enhancements.base  # noqa: F401
        return "real"
    except Exception:
        pass

    import types
    from abc import ABC, abstractmethod

    pkg = types.ModuleType("enhancements")
    pkg.__path__ = []
    sys.modules["enhancements"] = pkg

    base = types.ModuleType("enhancements.base")
    _registry: dict = {}

    class Enhancement(ABC):
        @abstractmethod
        def name(self) -> str: ...
        @abstractmethod
        def enhance(self, image): ...

    def register(cls):
        _registry[cls().name()] = cls
        return cls

    def get_enhancement(name, **kwargs):
        if name not in _registry:
            raise KeyError(name)
        return _registry[name](**kwargs)

    base.Enhancement = Enhancement
    base.register = register
    base.get_enhancement = get_enhancement
    base.available_enhancements = lambda: dict(_registry)
    sys.modules["enhancements.base"] = base

    import numpy as np

    raw_mod = types.ModuleType("enhancements.raw")

    @register
    class RawEnhancement(Enhancement):
        def name(self) -> str:
            return "raw"

        def enhance(self, image):
            return image

        def flops_per_megapixel(self):
            return 0.0

    raw_mod.RawEnhancement = RawEnhancement
    sys.modules["enhancements.raw"] = raw_mod

    clahe_mod = types.ModuleType("enhancements.clahe")

    @register
    class CLAHEEnhancement(Enhancement):
        def name(self) -> str:
            return "clahe"

        def enhance(self, image):
            # A cheap stand-in: a gamma lift, like the real CLAHE's effect.
            return np.clip(image.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)

        def flops_per_megapixel(self):
            return None

    clahe_mod.CLAHEEnhancement = CLAHEEnhancement
    sys.modules["enhancements.clahe"] = clahe_mod
    return "stub"


def _install_fake_detector(monkeypatch_target: str = "evaluation.predict._load_model"):
    """Replace the YOLO loader with a deterministic fake detector.

    The fake 'detects' a box on the bright blob in half the frames, at a
    confidence that depends on the input brightness. That gives the metrics
    something real to score without needing ultralytics or a GPU.
    """
    import evaluation.predict as predict_mod

    try:
        import torch
    except Exception:
        torch = None

    class FakeResult:
        """Mimics the shape of an Ultralytics Results object.

        The harness calls .boxes.xyxy.cpu().numpy(), so the tensors have to
        be real torch tensors, not numpy arrays.
        """
        def __init__(self, boxes, confs, classes):
            class _B:
                pass
            b = _B()
            if torch is not None:
                b.xyxy = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                b.conf = torch.tensor(confs, dtype=torch.float32).reshape(-1)
                b.cls = torch.tensor(classes, dtype=torch.float32).reshape(-1)
            else:
                b.xyxy = _NpShim(boxes, (-1, 4))
                b.conf = _NpShim(confs, (-1,))
                b.cls = _NpShim(classes, (-1,))
            self.boxes = b

    class _NpShim:
        """Fallback when torch is unavailable: numpy with a .cpu() method."""
        def __init__(self, data, shape):
            self._a = np.asarray(data, dtype=np.float32).reshape(shape)

        def cpu(self):
            return self

        def numpy(self):
            return self._a

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, image, conf=0.001, imgsz=640, verbose=False):
            self.calls += 1
            # Detect the bright blob if the frame is bright enough to have
            # one. The synthetic animal frames have a 150-valued patch.
            bright = float(image.mean())
            boxes, confs, classes = [], [], []
            if bright > 40.0:
                boxes.append([40, 60, 120, 120])
                confs.append(min(0.95, 0.3 + bright / 300.0))
                classes.append(0)
            return [FakeResult(boxes, confs, classes)]

    model = FakeModel()
    setattr(predict_mod, "_load_model", lambda *a, **k: model)
    return model


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="eval-smoke-"))
    print(f"[smoke] workspace: {tmp}")

    images_dir, ann_path, labels_path = _write_synthetic_dataset(tmp)

    which = _install_fake_enhancements()
    print(f"[smoke] enhancements package: {which}")
    fake = _install_fake_detector()

    cfg = RunConfig(
        images_dir=str(images_dir),
        annotations=str(ann_path),
        image_labels=str(labels_path),
        methods=["raw", "clahe"],
        max_images=8,
        bootstrap=50,
        seed=0,
        strata=StrataConfig(empty_fraction=0.25, day_fraction=0.0),
        run_id="smoke-run",
    )

    out = run_evaluation(
        cfg,
        repo_root=REPO_ROOT,
        experiments_root=tmp / "experiments",
        results_root=tmp / "results",
        use_cache=True,
        write_results=True,
    )

    # ---- assertions ----------------------------------------------------
    problems = []

    rdir = Path(out["run_dir"])
    if not (rdir / "config.yaml").is_file():
        problems.append("config.yaml was not written")
    if not (rdir / "frames.json").is_file():
        problems.append("frames.json was not written")
    if not (rdir / "breakdown.json").is_file():
        problems.append("breakdown.json was not written")
    if not (rdir / "predictions" / "raw.npz").is_file():
        problems.append("raw.npz was not cached")
    if not (rdir / "predictions" / "clahe.npz").is_file():
        problems.append("clahe.npz was not cached")

    results_root = tmp / "results"
    for name in ("method_comparison.csv", "false_positives.csv",
                 "latency.csv", "sample_manifest.csv"):
        if not (results_root / name).is_file():
            problems.append(f"results/{name} was not written")
    if not (results_root / "figures" / "comparison.png").is_file():
        # matplotlib may be absent in a bare environment. The figure is
        # optional for the plumbing test; the CSVs are not.
        try:
            import matplotlib  # noqa: F401
            problems.append("results/figures/comparison.png was not written")
        except Exception:
            print("[smoke] matplotlib absent; skipping the figure check")

    # The cache must be reused on a second run: the fake detector should not
    # be called again.
    calls_before = fake.calls
    cfg2 = RunConfig.from_yaml(rdir / "config.yaml")
    run_evaluation(
        cfg2,
        repo_root=REPO_ROOT,
        experiments_root=tmp / "experiments",
        results_root=tmp / "results",
        use_cache=True,
        write_results=False,
    )
    if fake.calls != calls_before:
        problems.append(f"cache was not reused: detector called "
                        f"{fake.calls - calls_before} more times")

    if not out["results"]:
        problems.append("no results were produced")
    else:
        base = out["results"][0]
        if base.method != "raw":
            problems.append("first method is not the baseline")
        if not (0.0 <= base.ap50 <= 1.0):
            problems.append(f"ap50 out of range: {base.ap50}")
        if np.isnan(base.ap50):
            problems.append("ap50 is NaN")

    if problems:
        print("\n[smoke] FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\n[smoke] PASSED: full pipeline exercised end to end.")
    print(f"[smoke] detector calls (first run): {calls_before}, "
          f"second run reused the cache: {fake.calls == calls_before}")
    print(f"[smoke] artefacts in {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

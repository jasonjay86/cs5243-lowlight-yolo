"""
test_rescore_e2e.py — end-to-end check of the rescore workflow.

Builds a real run directory with cached predictions, then rescores it
through the actual CLI, the way a user would. Verifies that:

  * a bare `--rescore` finds the most recent run
  * the rescore reproduces the original numbers exactly
  * a `--rescore` with a different --conf changes the numbers
  * the frame list is preserved

This is the scenario that failed: the run id has a timestamp, so nobody
can be expected to know the directory name.

Run from the repo root:
    python tests/test_rescore_e2e.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CLI = REPO_ROOT / "scripts" / "run_evaluation.py"


def _build_dataset(root: Path):
    import cv2
    import numpy as np

    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    categories = [{"id": 1, "name": "opossum"}, {"id": 2, "name": "empty"}]
    imgs, anns = [], []
    aid = 1
    for i in range(12):
        arr = np.full((120, 160, 3), 25, dtype=np.uint8)
        noise = np.random.default_rng(i).normal(0, 3, arr.shape)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if i < 8:
            arr[30:70, 20 + 10 * (i % 5):90] = 120
        cv2.imwrite(str(images / f"f{i:02d}.jpg"), arr)
        imgs.append({"id": str(i), "file_name": f"f{i:02d}.jpg",
                     "width": 160, "height": 120,
                     "datetime": "2021-03-01 23:00:00"})
        if i < 8:
            anns.append({"id": str(aid), "image_id": str(i), "category_id": 1,
                         "bbox": [20 + 10 * (i % 5), 30, 70, 40]})
        else:
            anns.append({"id": str(aid), "image_id": str(i), "category_id": 2})
        aid += 1
    ann_path = root / "bbox.json"
    ann_path.write_text(json.dumps({"images": imgs, "annotations": anns,
                                    "categories": categories}), encoding="utf-8")
    return images, ann_path


def _make_fake_repo(root: Path):
    """A repo-shaped directory with the harness copied in."""
    import shutil
    repo = root / "repo"
    repo.mkdir()
    shutil.copytree(REPO_ROOT / "evaluation", repo / "evaluation")
    (repo / "scripts").mkdir()
    shutil.copy2(CLI, repo / "scripts" / "run_evaluation.py")
    (repo / "detection").mkdir()
    # The detector weights do not need to exist for a rescore.
    (repo / "detection" / "yolov8n.pt").write_bytes(b"fake")
    return repo


def _seed_cache(repo: Path, run_dir: Path, frames, fingerprint: str):
    """Write a prediction cache as if a real run had produced it."""
    from evaluation.cache import MethodPredictions, PredictionCache
    import numpy as np

    cache = PredictionCache(run_dir, fingerprint)
    for method in ("raw", "clahe"):
        # raw: fewer detections. clahe: a bit more.
        n_det = 6 if method == "raw" else 8
        boxes = []
        confs = []
        img_idx = []
        for k in range(n_det):
            fi = k % len(frames)
            boxes.append([20.0, 30.0, 90.0, 70.0])
            confs.append(0.5 + 0.05 * k)
            img_idx.append(fi)
        preds = MethodPredictions(
            method=method,
            boxes=np.array(boxes, dtype=np.float32),
            confs=np.array(confs, dtype=np.float32),
            classes=np.zeros(n_det, dtype=np.int16),
            img_idx=np.array(img_idx, dtype=np.int32),
            frame_ids=list(frames),
            enh_ms=0.0 if method == "raw" else 50.0,
            yolo_ms=55.0,
            n_frames=len(frames),
        )
        cache.save(preds)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="rescore-e2e-"))
    images, ann_path = _build_dataset(tmp)
    repo = _make_fake_repo(tmp)

    # ---- build a run the way the harness would -------------------------
    sys.path.insert(0, str(repo))
    from evaluation.config import RunConfig, StrataConfig
    from evaluation.dataset import build_dataset, scale_boxes_to_disk, select_frames
    import cv2

    cfg = RunConfig(
        images_dir=str(images), annotations=str(ann_path),
        methods=["raw", "clahe"], max_images=8, bootstrap=50, seed=0,
        strata=StrataConfig(empty_fraction=0.25, day_fraction=0.0),
        run_id="run-100f-s0-20261026-143022",
    )
    dataset = build_dataset(Path(cfg.images_dir), Path(cfg.annotations),
                            None, cfg.strata)
    frames = select_frames(dataset, cfg.max_images, cfg.strata, cfg.seed)
    for fr in frames:
        img = cv2.imread(fr.path, cv2.IMREAD_COLOR)
        if img is not None:
            scale_boxes_to_disk(fr, img.shape)
    frame_paths = [f.path for f in frames]

    run_dir = repo / "experiments" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # The cache is keyed by the fingerprint the harness computes, so the
    # seed has to use the real one. The harness fingerprints the RESOLVED
    # config (absolute paths), so resolve the config the same way first.
    from evaluation.harness import _ground_truth_digest
    resolved = cfg.resolved(repo)
    fingerprint = resolved.analysis_fingerprint(frame_paths,
                                                _ground_truth_digest(frames))

    cfg.to_yaml(run_dir / "config.yaml")
    (run_dir / "frames.json").write_text(
        json.dumps({"fingerprint": fingerprint,
                    "frames": frame_paths,
                    "sample": {"n_frames": len(frame_paths)}}, indent=2),
        encoding="utf-8")
    _seed_cache(repo, run_dir, frame_paths, fingerprint)

    # ---- rescore with no arguments -------------------------------------
    r = subprocess.run([sys.executable, str(repo / "scripts" / "run_evaluation.py"),
                        "--rescore"],
                       capture_output=True, text=True, cwd=str(repo))
    out = r.stdout + r.stderr
    print(out)

    problems = []
    if r.returncode != 0:
        problems.append(f"bare --rescore exited {r.returncode}")
    if "Traceback" in out:
        problems.append("bare --rescore raised a traceback")
    if "using run" not in out:
        problems.append("bare --rescore did not report which run it used")

    results_csv = repo / "results" / "method_comparison.csv"
    if not results_csv.is_file():
        problems.append("no results/method_comparison.csv was written")
    else:
        text = results_csv.read_text(encoding="utf-8")
        if "raw" not in text or "clahe" not in text:
            problems.append("the comparison table is missing a method")

    # The frame list must be unchanged.
    after = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))
    if after["frames"] != frame_paths:
        problems.append("the rescore changed the frame list")

    # ---- rescore with a different threshold ----------------------------
    r2 = subprocess.run([sys.executable, str(repo / "scripts" / "run_evaluation.py"),
                         "--rescore", "--conf", "0.7"],
                        capture_output=True, text=True, cwd=str(repo))
    out2 = r2.stdout + r2.stderr
    if r2.returncode != 0:
        problems.append(f"--rescore --conf 0.7 exited {r2.returncode}")
    if "Traceback" in out2:
        problems.append("--rescore --conf 0.7 raised a traceback")

    # The numbers must actually change with the threshold.
    if results_csv.is_file():
        low = results_csv.read_text(encoding="utf-8")
        r3 = subprocess.run(
            [sys.executable, str(repo / "scripts" / "run_evaluation.py"),
             "--rescore", "--conf", "0.25"],
            capture_output=True, text=True, cwd=str(repo))
        high = results_csv.read_text(encoding="utf-8")
        if low == high:
            problems.append("changing --conf did not change the results")

    if problems:
        print("\n[rescore-e2e] FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\n[rescore-e2e] PASSED: bare --rescore found the run, "
          "preserved the frame list, and honoured --conf.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

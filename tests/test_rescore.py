"""
test_rescore.py — verify the --rescore path works and stays correct.

Three things are checked:

1. `--rescore` with no arguments finds the most recent run, so the caller
   does not have to know the timestamped directory name.
2. A rescore uses the frame list stored in the run, NOT a freshly selected
   one. If it re-selected, the cached boxes could be scored against
   different frames and the numbers would be silently wrong.
3. A missing config gives a clear message, not a traceback.

Run from the repo root:
    python tests/test_rescore.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CLI = REPO_ROOT / "scripts" / "run_evaluation.py"


def _run(args, cwd, repo=None):
    """Invoke the CLI.

    The CLI derives its repo root from the SCRIPT'S OWN LOCATION, so
    running it from this checkout would use this checkout's data and
    write into this checkout's experiments/ directory. To keep the test
    hermetic, the script is copied into a throwaway repo and run from
    there.
    """
    script = CLI
    if repo is not None:
        scripts_dir = repo / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script = scripts_dir / "run_evaluation.py"
        shutil.copy2(CLI, script)
    return subprocess.run([sys.executable, str(script), *args],
                          capture_output=True, text=True, cwd=str(cwd))


def _make_temp_repo(tmp: Path) -> Path:
    """A minimal repo-shaped directory with the harness in it."""
    import shutil
    repo = tmp / "repo"
    repo.mkdir()
    shutil.copytree(REPO_ROOT / "evaluation", repo / "evaluation")
    (repo / "detection").mkdir()
    (repo / "detection" / "yolov8n.pt").write_bytes(b"fake")
    return repo


def test_missing_config_gives_a_clear_error_not_a_traceback():
    """A bad --config path must say so plainly."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_temp_repo(Path(tmp))
        (repo / "experiments").mkdir()
        r = _run(["--config", "experiments/nope/config.yaml", "--rescore"],
                 repo, repo=repo)
        combined = r.stdout + r.stderr
        assert "Traceback" not in combined, combined
        assert "config not found" in combined.lower(), combined
        assert r.returncode != 0, "should exit non-zero"
        print("    missing config reported cleanly")


def test_rescore_with_no_runs_says_so():
    """With no runs at all, the message must explain what to do."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_temp_repo(Path(tmp))
        (repo / "experiments").mkdir()
        r = _run(["--rescore"], repo, repo=repo)
        combined = r.stdout + r.stderr
        assert "Traceback" not in combined, combined
        assert "no runs found" in combined.lower(), combined
        assert r.returncode != 0
        print("    empty experiments dir reported cleanly")


def test_rescore_does_not_touch_the_real_repo():
    """The CLI must not write into the checkout it lives in.

    The CLI derives its repo root from the script's own location, so a
    test that runs the script in place would read the real dataset and
    write into the real experiments/ directory. Running it from a copied
    location keeps the test hermetic.

    This is a regression test for that exact accident.
    """
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_temp_repo(Path(tmp))
        (repo / "experiments").mkdir()
        r = _run(["--rescore"], repo, repo=repo)
        combined = r.stdout + r.stderr
        # The message must name the TEMP repo, not the real one.
        assert str(REPO_ROOT) not in combined, (
            f"the CLI resolved its repo root to the real checkout "
            f"({REPO_ROOT}):\n{combined}")
        print("    the CLI stayed inside the temporary repo")


def test_find_latest_run_picks_the_newest():
    """The latest run must be found by name order, newest last."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_eval_mod", CLI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_eval_mod"] = mod
    spec.loader.exec_module(mod)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Explicit mtimes: writing three files in quick succession can give
        # them the same timestamp, and then "most recent" is ambiguous.
        runs = [
            ("run-100f-s0-20261026-090000", 1_762_000_000),
            ("run-100f-s0-20261026-143022", 1_762_500_000),
            ("run-50f-s1-20261025-120000", 1_761_000_000),
        ]
        for name, mtime in runs:
            d = root / name
            d.mkdir()
            cfg = d / "config.yaml"
            cfg.write_text("x: 1", encoding="utf-8")
            os.utime(cfg, (mtime, mtime))
        # A directory without a config.yaml is not a run.
        (root / "not-a-run").mkdir()

        latest = mod._find_latest_run(root)
        assert latest is not None, "no run found"
        assert latest.name == "run-100f-s0-20261026-143022", (
            f"picked {latest.name}, expected the most recently modified run")
        print(f"    picked the newest run: {latest.name}")


def test_rescore_uses_the_stored_frame_list():
    """The rescore must score against the run's own frame list.

    This is the correctness bug that matters: if a rescore re-selected the
    sample, the cached predictions could be scored against a different set
    of frames and every number would be quietly wrong.
    """
    from evaluation.config import RunConfig, StrataConfig
    from evaluation.harness import run_evaluation

    # Build a tiny dataset.
    import cv2
    import numpy as np

    tmp = Path(tempfile.mkdtemp(prefix="rescore-test-"))
    images = tmp / "images"
    images.mkdir()
    categories = [{"id": 1, "name": "opossum"}, {"id": 2, "name": "empty"}]
    imgs, anns = [], []
    aid = 1
    for i in range(6):
        arr = np.full((80, 100, 3), 25, dtype=np.uint8)
        if i < 4:
            arr[20:50, 20:60] = 120
        cv2.imwrite(str(images / f"f{i}.jpg"), arr)
        imgs.append({"id": str(i), "file_name": f"f{i}.jpg",
                     "width": 100, "height": 80,
                     "datetime": "2021-03-01 23:00:00"})
        if i < 4:
            anns.append({"id": str(aid), "image_id": str(i), "category_id": 1,
                         "bbox": [20, 20, 40, 30]})
        else:
            anns.append({"id": str(aid), "image_id": str(i), "category_id": 2})
        aid += 1
    ann_path = tmp / "bbox.json"
    ann_path.write_text(json.dumps({"images": imgs, "annotations": anns,
                                    "categories": categories}), encoding="utf-8")

    experiments = tmp / "experiments"
    results_root = tmp / "results"
    run_dir = experiments / "run-test"
    run_dir.mkdir(parents=True)

    cfg = RunConfig(
        images_dir=str(images), annotations=str(ann_path),
        methods=["raw"], max_images=4, bootstrap=20, seed=0,
        strata=StrataConfig(empty_fraction=0.25, day_fraction=0.0),
        run_id="run-test",
    )
    cfg.to_yaml(run_dir / "config.yaml")

    # The frame list a previous run would have stored. Use a specific
    # SUBSET so it cannot coincide with a fresh selection.
    stored_paths = [str(images / "f0.jpg"), str(images / "f1.jpg")]
    (run_dir / "frames.json").write_text(
        json.dumps({"fingerprint": "abc123", "frames": stored_paths,
                    "sample": {"n_frames": 2}}, indent=2), encoding="utf-8")

    cfg2 = RunConfig.from_yaml(run_dir / "config.yaml")
    cfg2.run_id = "run-test"

    # A rescore with no cached predictions fails at the scoring step, but
    # by then the frame list has already been restored from the manifest.
    try:
        run_evaluation(cfg2, repo_root=tmp,
                       experiments_root=experiments, results_root=results_root,
                       use_cache=True, write_results=False, skip_predict=True)
    except SystemExit:
        pass

    # The harness rewrites frames.json each run. It must contain the
    # STORED list, not a freshly selected one.
    rewritten = json.loads((run_dir / "frames.json").read_text(encoding="utf-8"))
    assert rewritten["frames"] == stored_paths, (
        f"the rescore re-selected the sample instead of using the stored "
        f"frame list:\n  stored:    {stored_paths}\n"
        f"  rewritten: {rewritten['frames']}"
    )
    print(f"    rescore kept the stored frame list ({len(stored_paths)} frames)")


def test_a_cache_from_the_old_fingerprint_scheme_still_loads():
    """A cache written before the fingerprint scheme changed must still load.

    The scheme was split so that scoring parameters (conf, bootstrap) do
    not invalidate the prediction cache. That change altered the hash
    without altering a single detection. Without accepting the old
    scheme, every existing cache is thrown away and the inference work
    has to be repeated for no reason.
    """
    import numpy as np
    from evaluation.cache import MethodPredictions, PredictionCache
    from evaluation.config import RunConfig

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        cfg = RunConfig(images_dir="i", annotations="a",
                        methods=["raw", "clahe"], seed=0)
        frames = ["/a.jpg", "/b.jpg"]

        old_fp = cfg.legacy_analysis_fingerprint(frames, "gt1")
        new_fp = cfg.analysis_fingerprint(frames, "gt1")
        assert old_fp != new_fp, (
            "the two schemes should differ, otherwise this test proves "
            "nothing")

        # A cache written under the old scheme.
        cache = PredictionCache(run_dir, old_fp)
        cache.save(MethodPredictions(
            method="raw",
            boxes=np.array([[1.0, 2.0, 3.0, 4.0]]),
            confs=np.array([0.9]),
            classes=np.array([0], dtype=np.int16),
            img_idx=np.array([0], dtype=np.int32),
            frame_ids=frames, enh_ms=0.0, yolo_ms=55.0, n_frames=2,
        ))

        # Without the legacy list it is correctly rejected.
        strict = PredictionCache(run_dir, new_fp)
        assert not strict.has("raw")
        assert "fingerprint changed" in strict.why_stale("raw")

        # With the legacy list it is accepted and loads.
        lenient = PredictionCache(run_dir, new_fp,
                                   legacy_fingerprints=[old_fp])
        assert lenient.has("raw"), (
            "a cache written under the old scheme must not be thrown away")
        loaded = lenient.load("raw")
        assert loaded is not None and loaded.n_detections == 1
        print("    a cache from the old scheme still loads")


def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())

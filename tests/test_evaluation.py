"""
test_evaluation.py — correctness tests for the evaluation harness.

Owner: Jason Johnson

Run from the repo root:

    python tests/test_evaluation.py

or, if pytest is installed:

    python -m pytest tests/test_evaluation.py -q

This file has a standalone runner so it works without pytest, which is not
in environment.yml.

These are the checks that matter. A wrong AP implementation produces numbers
that look perfectly reasonable and mean nothing, so the metric code is
tested against hand-computable cases rather than just smoke-tested.

Covered:
  * AP on a hand-computed case (one perfect detection -> AP 1.0)
  * AP on a case with no detections -> AP 0.0
  * IoU matching at multiple thresholds
  * Precision/recall arithmetic
  * Empty-frame false-positive rate
  * Paired bootstrap determinism and CI sanity
  * Stratified selection: quotas respected, determinism, species spread
  * Prediction cache round-trip
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.cache import MethodPredictions, PredictionCache  # noqa: E402
from evaluation.config import RunConfig, StrataConfig            # noqa: E402
from evaluation.dataset import Frame, select_frames, summarise_sample  # noqa: E402
from evaluation.metrics import (                                    # noqa: E402
    average_precision, build_stats, empty_fp_rate, iou_matrix,
    match_image, prec_rec_at, summarize,
)


# ----------------------------------------------------------------------
# Minimal frame stub so the metric tests do not need the real dataset.
# ----------------------------------------------------------------------
class StubFrame:
    def __init__(self, boxes, is_empty=False, is_night=True, species=()):
        self.boxes = np.array(boxes, dtype=float).reshape(-1, 4)
        self.is_empty = is_empty
        self.is_night = is_night
        self.species = list(species)
        self.path = "stub"

    @property
    def n_animals(self):
        return len(self.species)


def make_stats(frames, per_image):
    return build_stats(frames, per_image)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    ious = iou_matrix(a, b)
    assert abs(ious[0, 0] - 1.0) < 1e-9, ious
    # overlap = 25, union = 100 + 100 - 25 = 175
    assert abs(ious[0, 1] - 25.0 / 175.0) < 1e-9, ious


def test_iou_matrix_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((0, 4))).shape == (0, 0)
    assert iou_matrix(np.zeros((0, 4)), np.ones((3, 4))).shape == (0, 3)


def test_perfect_detection_gives_ap_one():
    """One ground truth, one perfectly overlapping detection -> AP 1.0."""
    frames = [StubFrame([[0, 0, 10, 10]])]
    dets = [(np.array([[0, 0, 10,  10]], dtype=float), np.array([0.9]))]
    s = make_stats(frames, dets)
    ap = average_precision(s, 0, np.ones(1))
    assert abs(ap - 1.0) < 1e-9, f"expected AP 1.0, got {ap}"
    p, r = prec_rec_at(s, 0.25, np.ones(1))
    assert abs(p - 1.0) < 1e-9, p
    assert abs(r - 1.0) < 1e-9, r


def test_no_detection_gives_ap_zero():
    frames = [StubFrame([[0, 0, 10, 10]])]
    dets = [(np.zeros((0, 4)), np.zeros(0))]
    s = make_stats(frames, dets)
    ap = average_precision(s, 0, np.ones(1))
    assert abs(ap - 0.0) < 1e-9, f"expected AP 0.0, got {ap}"
    p, r = prec_rec_at(s, 0.25, np.ones(1))
    assert np.isnan(p), p
    assert abs(r - 0.0) < 1e-9, r


def test_misplaced_detection_gives_zero_recall():
    """A detection far from the ground truth is a false positive."""
    frames = [StubFrame([[0, 0, 10, 10]])]
    dets = [(np.array([[100, 100, 110, 110]], dtype=float), np.array([0.9]))]
    s = make_stats(frames, dets)
    p, r = prec_rec_at(s, 0.25, np.ones(1))
    assert abs(p - 0.0) < 1e-9, p
    assert abs(r - 0.0) < 1e-9, r


def test_partial_overlap_counts_only_where_iou_passes():
    """IoU 0.333 overlap: a false positive at every threshold in IOU_THRS.

    Boxes [0,0,10,10] and [5,0,15,10] intersect in 50 of 150 -> IoU 0.333,
    which is below the lowest threshold (0.5), so it never counts as a TP.
    """
    frames = [StubFrame([[0, 0, 10, 10]])]
    dets = [(np.array([[5, 0, 15, 10]], dtype=float), np.array([0.9]))]
    s = make_stats(frames, dets)
    assert not s.tp[:, 0].any(), "IoU 0.33 must not count as a TP at IoU 0.5"
    assert not s.tp.any(), "IoU 0.33 must not count as a TP at any threshold"

    # match_image returns (confs, tp).
    confs, tp = match_image(np.array([[5, 0, 15, 10]], dtype=float),
                            np.array([0.9]),
                            np.array([[0, 0, 10, 10]], dtype=float))
    assert tp.shape == (1, 10)
    assert not tp[0].any()
    assert abs(float(confs[0]) - 0.9) < 1e-9


def test_high_overlap_counts_as_tp_at_low_threshold_only():
    """IoU 0.6 counts at IoU 0.5 and 0.55, but not at 0.6+.

    Boxes [0,0,10,10] and [2,0,12,10]: intersection 8x10 = 80, union 120,
    IoU = 0.667 -> passes 0.5, 0.55, 0.6 and 0.65, fails 0.7+.
    """
    frames = [StubFrame([[0, 0, 10, 10]])]
    dets = [(np.array([[2, 0, 12, 10]], dtype=float), np.array([0.9]))]
    s = make_stats(frames, dets)
    assert s.tp[0, 0], "IoU 0.67 must count at IoU 0.5"
    assert s.tp[0, 1], "IoU 0.67 must count at IoU 0.55"
    assert not s.tp[0, 4], "IoU 0.67 must not count at IoU 0.7"


def test_precision_accounts_for_false_positives():
    """One TP plus one FP at the same threshold -> precision 0.5."""
    frames = [StubFrame([[0, 0, 10, 10]])]
    dets = [(np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float),
             np.array([0.9, 0.8]))]
    s = make_stats(frames, dets)
    p, r = prec_rec_at(s, 0.25, np.ones(1))
    assert abs(p - 0.5) < 1e-9, p
    assert abs(r - 1.0) < 1e-9, r


def test_empty_frame_fp_rate():
    """A detection on an animal-free frame counts as a false positive."""
    frames = [
        StubFrame([[0, 0, 10, 10]], is_empty=False),   # animal frame
        StubFrame([], is_empty=True),                   # empty frame
    ]
    dets = [
        (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9])),
        (np.array([[20, 20, 30, 30]], dtype=float), np.array([0.6])),
    ]
    s = make_stats(frames, dets)
    fp = empty_fp_rate(s, 0.25, np.ones(2))
    assert abs(fp - 1.0) < 1e-9, f"expected FP rate 1.0, got {fp}"


def test_empty_frame_fp_rate_zero_when_clean():
    frames = [
        StubFrame([[0, 0, 10, 10]], is_empty=False),
        StubFrame([], is_empty=True),
    ]
    dets = [
        (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9])),
        (np.zeros((0, 4)), np.zeros(0)),
    ]
    s = make_stats(frames, dets)
    fp = empty_fp_rate(s, 0.25, np.ones(2))
    assert abs(fp - 0.0) < 1e-9, fp


def test_bootstrap_is_deterministic():
    frames = [StubFrame([[0, 0, 10, 10]]) for _ in range(6)]
    dets = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.5 + 0.05 * i]))
            for i in range(6)]
    s = make_stats(frames, dets)
    a = summarize({"m": s}, 0.25, 50)
    b = summarize({"m": s}, 0.25, 50)
    assert a[0].ap50 == b[0].ap50
    assert a[0].ap50_ci == b[0].ap50_ci


def test_bootstrap_ci_brackets_the_point_estimate():
    frames = [StubFrame([[0, 0, 10, 10]]) for _ in range(20)]
    dets = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.7]))
            for _ in range(20)]
    s = make_stats(frames, dets)
    r = summarize({"m": s}, 0.25, 200)[0]
    assert r.ap50_ci[0] <= r.ap50 <= r.ap50_ci[1], r


def test_delta_is_zero_for_the_baseline():
    frames = [StubFrame([[0, 0, 10, 10]])]
    dets = [(np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9]))]
    s = make_stats(frames, dets)
    res = summarize({"raw": s, "clahe": s}, 0.25, 20)
    assert abs(res[0].d_ap50) < 1e-12
    assert abs(res[1].d_ap50) < 1e-12


def test_subset_stats_remaps_frame_indices():
    """Restricting to a subset must renumber the frame indices.

    Regression test: `s.img` held the ORIGINAL frame numbers while `n_gt`
    had been compressed, so `w[s.img[...]]` indexed out of bounds whenever
    the subset skipped a frame. A detection on the last frame with a
    one-frame subset is the minimal reproducer.
    """
    from evaluation.metrics import subset_stats

    frames = [StubFrame([[0, 0, 10, 10]]) for _ in range(4)]
    dets = [
        (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9])),
        (np.zeros((0, 4)), np.zeros(0)),
        (np.zeros((0, 4)), np.zeros(0)),
        (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.8])),
    ]
    s = make_stats(frames, dets)

    # Keep only the last frame.
    mask = np.array([False, False, False, True])
    sub = subset_stats(s, mask)

    assert len(sub.n_gt) == 1, sub.n_gt
    assert sub.img.max() == 0, f"frame index not remapped: {sub.img}"
    assert len(sub.confs) == 1, sub.confs
    # The weight lookup that used to crash must now work.
    ones = np.ones(len(sub.n_gt))
    p, r = prec_rec_at(sub, 0.25, ones)
    assert abs(p - 1.0) < 1e-9, p
    assert abs(r - 1.0) < 1e-9, r


def test_breakdown_does_not_crash_on_a_subset():
    """breakdown() slices Stats by stratum, which is the code path that
    crashed. It must work when a stratum has fewer frames than the whole."""
    from evaluation.metrics import breakdown

    frames = [
        StubFrame([[0, 0, 10, 10]], is_empty=False, is_night=True),
        StubFrame([], is_empty=True, is_night=True),
        StubFrame([[0, 0, 10, 10]], is_empty=False, is_night=False),
        StubFrame([], is_empty=True, is_night=False),
    ]
    dets = [
        (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.9])),
        (np.zeros((0, 4)), np.zeros(0)),
        (np.array([[0, 0, 10, 10]], dtype=float), np.array([0.7])),
        (np.array([[50, 50, 60, 60]], dtype=float), np.array([0.6])),
    ]
    s = make_stats(frames, dets)

    by_light = breakdown(s, 0.25, by="night")
    assert by_light["night"]["n_frames"] == 2, by_light
    assert by_light["day"]["n_frames"] == 2, by_light

    by_presence = breakdown(s, 0.25, by="empty")
    assert by_presence["animal"]["n_frames"] == 2, by_presence
    assert by_presence["empty"]["n_frames"] == 2, by_presence
    # The empty stratum has a detection on the day frame -> FP rate 1.0.
    assert by_presence["empty"]["n_gt"] == 0


def test_scale_boxes_to_disk():
    """Ground-truth boxes must be rescaled from annotation space to disk space.

    Regression test: CCT-20's benchmark images are downsized to a maximum of
    1024 px per side while the bbox JSON carries original-resolution
    coordinates. Skipping the scale makes every box land in the wrong place,
    so recall collapses to exactly zero while the detector still fires.
    """
    from evaluation.dataset import Frame, scale_boxes_to_disk

    fr = Frame(
        path="/x.jpg", image_id="1",
        boxes=np.array([[0, 0, 200, 100]], dtype=float),
        species=["opossum"], ann_width=2048, ann_height=1024,
    )
    # On-disk shape is (height, width): 512 tall, 1024 wide.
    scale_boxes_to_disk(fr, (512, 1024))
    assert abs(fr.boxes[0, 2] - 100.0) < 1e-6, fr.boxes
    assert abs(fr.boxes[0, 3] - 50.0) < 1e-6, fr.boxes

    # Matching dimensions: left alone.
    fr2 = Frame(path="/x.jpg", image_id="1",
                boxes=np.array([[0, 0, 200, 100]], dtype=float),
                ann_width=1024, ann_height=512)
    scale_boxes_to_disk(fr2, (512, 1024))
    assert abs(fr2.boxes[0, 2] - 200.0) < 1e-6, fr2.boxes

    # Missing annotation dimensions: left alone.
    fr3 = Frame(path="/x.jpg", image_id="1",
                boxes=np.array([[0, 0, 200, 100]], dtype=float))
    scale_boxes_to_disk(fr3, (512, 1024))
    assert abs(fr3.boxes[0, 2] - 200.0) < 1e-6, fr3.boxes


def test_scaled_boxes_produce_nonzero_recall():
    """End to end: unscaled boxes give 0 recall, scaled boxes give 1.0.

    This is the exact symptom of the bug that was found on the first real
    run: recall exactly 0.000 on every method, while the detector was
    firing normally (non-zero empty-frame FP rate).

    The annotation is 2048x1024 and the on-disk image is 1024x512, i.e. the
    benchmark images are downsized to a maximum of 1024 px per side. The
    on-disk shape is passed as (height, width), matching cv2.
    """
    from evaluation.dataset import Frame, scale_boxes_to_disk

    # A detection sits at the animal's TRUE location on the downsized image.
    det_box = np.array([[100, 50, 200, 150]], dtype=float)
    dets = [(det_box, np.array([0.9]))]

    # The annotation says the animal is at 200..400, 100..300 in ORIGINAL
    # space; on disk that is 100..200, 50..150.
    ann_box = np.array([[200, 100, 400, 300]], dtype=float)

    # Unscaled: the ground truth is at twice the detection's position.
    frames_unscaled = [StubFrame([[200, 100, 400, 300]])]
    s1 = make_stats(frames_unscaled, dets)
    _, r_unscaled = prec_rec_at(s1, 0.25, np.ones(1))
    assert abs(r_unscaled) < 1e-9, f"expected 0 recall unscaled, got {r_unscaled}"

    # Scaled: the ground truth lands on the detection.
    fr = Frame(path="/x.jpg", image_id="1", boxes=ann_box.copy(),
               species=["opossum"], ann_width=2048, ann_height=1024)
    scale_boxes_to_disk(fr, (512, 1024))     # (height, width) on disk
    assert abs(fr.boxes[0, 0] - 100.0) < 1e-6, fr.boxes
    assert abs(fr.boxes[0, 2] - 200.0) < 1e-6, fr.boxes
    assert abs(fr.boxes[0, 1] - 50.0) < 1e-6, fr.boxes
    assert abs(fr.boxes[0, 3] - 150.0) < 1e-6, fr.boxes

    frames_scaled = [StubFrame(fr.boxes)]
    s2 = make_stats(frames_scaled, dets)
    _, r_scaled = prec_rec_at(s2, 0.25, np.ones(1))
    assert abs(r_scaled - 1.0) < 1e-9, f"expected 1.0 recall scaled, got {r_scaled}"


def test_gt_digest_changes_when_boxes_change():
    """The cache fingerprint must react to a change in the resolved boxes."""
    from evaluation.harness import _ground_truth_digest

    a = [StubFrame([[0, 0, 10, 10]])]
    b = [StubFrame([[0, 0, 20, 20]])]
    assert _ground_truth_digest(a) != _ground_truth_digest(b)


def test_fingerprint_includes_gt_digest():
    """A different ground truth must invalidate the cache."""
    cfg = RunConfig(images_dir="x", annotations="y", methods=["raw"], seed=0)
    f1 = cfg.analysis_fingerprint(["/a.jpg"], "digestA")
    f2 = cfg.analysis_fingerprint(["/a.jpg"], "digestB")
    assert f1 != f2, "the fingerprint must include the ground-truth digest"


# ----------------------------------------------------------------------
# Stratified selection
# ----------------------------------------------------------------------
def _make_dataset(n_species=3, per_species=10, n_empty=10, n_day=0):
    frames = []
    for sp_i in range(n_species):
        sp = f"sp{sp_i}"
        for i in range(per_species):
            frames.append(Frame(
                path=f"/img/{sp}_{i}.jpg", image_id=f"{sp}_{i}",
                boxes=np.array([[0, 0, 10, 10]], dtype=float),
                species=[sp], is_empty=False, is_night=True,
            ))
    for i in range(n_empty):
        frames.append(Frame(
            path=f"/img/empty_{i}.jpg", image_id=f"empty_{i}",
            boxes=np.zeros((0, 4)), species=[], is_empty=True, is_night=True,
        ))
    from evaluation.dataset import Dataset
    return Dataset(frames=frames, categories={})


def test_selection_respects_the_empty_quota():
    ds = _make_dataset()
    strata = StrataConfig(empty_fraction=0.25, day_fraction=0.0)
    picked = select_frames(ds, 20, strata, seed=0)
    n_empty = sum(1 for f in picked if f.is_empty)
    assert n_empty == 5, f"expected 5 empty frames, got {n_empty}"
    assert len(picked) == 20


def test_selection_is_deterministic_for_a_seed():
    ds = _make_dataset()
    strata = StrataConfig(empty_fraction=0.25, day_fraction=0.0)
    a = select_frames(ds, 20, strata, seed=7)
    b = select_frames(ds, 20, strata, seed=7)
    assert [f.path for f in a] == [f.path for f in b]


def test_different_seeds_can_differ():
    ds = _make_dataset()
    strata = StrataConfig(empty_fraction=0.25, day_fraction=0.0)
    a = [f.path for f in select_frames(ds, 20, strata, seed=1)]
    b = [f.path for f in select_frames(ds, 20, strata, seed=2)]
    # Not guaranteed to differ, but with 30 candidates and 20 slots it should.
    assert a != b, "different seeds produced an identical sample"


def test_uniform_species_balance_spreads_across_species():
    ds = _make_dataset(n_species=4, per_species=20, n_empty=0)
    strata = StrataConfig(empty_fraction=0.0, day_fraction=0.0,
                          species_balance="uniform")
    picked = select_frames(ds, 20, strata, seed=0)
    species = {f.species[0] for f in picked}
    assert len(species) >= 3, f"expected spread across species, got {species}"


def test_sample_summary_counts():
    ds = _make_dataset()
    strata = StrataConfig(empty_fraction=0.25, day_fraction=0.0)
    picked = select_frames(ds, 20, strata, seed=0)
    s = summarise_sample(picked)
    assert s["n_frames"] == 20
    assert s["n_empty"] == 5
    assert s["n_boxes"] == 15


# ----------------------------------------------------------------------
# Cache round-trip
# ----------------------------------------------------------------------
def test_prediction_cache_round_trip(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    cache = PredictionCache(tmp, "fp123")
    preds = MethodPredictions(
        method="clahe",
        boxes=np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        confs=np.array([0.9, 0.4], dtype=np.float32),
        classes=np.array([0, 16], dtype=np.int16),
        img_idx=np.array([0, 1], dtype=np.int32),
        frame_ids=["/a.jpg", "/b.jpg"],
        enh_ms=12.5, yolo_ms=30.0, n_frames=2,
    )
    cache.save(preds)
    assert cache.has("clahe")
    loaded = cache.load("clahe")
    assert loaded is not None
    assert loaded.n_detections == 2
    assert np.allclose(loaded.boxes, preds.boxes)
    assert np.allclose(loaded.confs, preds.confs)
    assert loaded.frame_ids == preds.frame_ids
    assert abs(loaded.enh_ms - 12.5) < 1e-6
    # A different fingerprint must invalidate the cache.
    other = PredictionCache(tmp, "different")
    assert not other.has("clahe")
    assert other.load("clahe") is None


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def test_config_fingerprint_changes_with_inputs():
    """The fingerprint must change when the DETECTIONS would change.

    The fingerprint guards the cached predictions, so it must change for
    anything that affects inference: the frame list, the ground truth,
    the methods, the detector input size, the confidence floor.

    It must NOT change for anything that only affects scoring: `conf` and
    `bootstrap`. Those are applied when the cached detections are scored,
    not when they are made, so a threshold sweep must not throw away the
    inference work.
    """
    cfg = RunConfig(images_dir="x", annotations="y", methods=["raw"], seed=0)
    f1 = cfg.analysis_fingerprint(["/a.jpg", "/b.jpg"])
    f2 = cfg.analysis_fingerprint(["/a.jpg", "/b.jpg"])
    assert f1 == f2, "same inputs must give the same fingerprint"

    # A different frame list changes the detections.
    assert f1 != cfg.analysis_fingerprint(["/a.jpg"])

    # A different ground truth changes the scores, and the cache guard
    # tracks it so a stale score can never be silently reused.
    assert f1 != cfg.analysis_fingerprint(["/a.jpg", "/b.jpg"], "digest2")

    # A different method set changes the detections.
    cfg_more = RunConfig(images_dir="x", annotations="y",
                         methods=["raw", "clahe"], seed=0)
    assert f1 != cfg_more.analysis_fingerprint(["/a.jpg", "/b.jpg"])

    # A different detector input size changes the detections.
    cfg_small = RunConfig(images_dir="x", annotations="y", methods=["raw"],
                          seed=0, imgsz=320)
    assert f1 != cfg_small.analysis_fingerprint(["/a.jpg", "/b.jpg"])

    # Scoring parameters must NOT change it.
    cfg_conf = RunConfig(images_dir="x", annotations="y", methods=["raw"],
                         seed=0, conf=0.7)
    assert f1 == cfg_conf.analysis_fingerprint(["/a.jpg", "/b.jpg"]), (
        "changing --conf must not invalidate the prediction cache")

    cfg_boot = RunConfig(images_dir="x", annotations="y", methods=["raw"],
                         seed=0, bootstrap=50)
    assert f1 == cfg_boot.analysis_fingerprint(["/a.jpg", "/b.jpg"]), (
        "changing --bootstrap must not invalidate the prediction cache")


def test_config_yaml_round_trip(tmp_path=None):
    import tempfile
    tmp = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    cfg = RunConfig(images_dir="a", annotations="b", methods=["raw", "clahe"],
                    max_images=123, seed=5,
                    strata=StrataConfig(empty_fraction=0.3, day_fraction=0.1))
    p = tmp / "config.yaml"
    cfg.to_yaml(p)
    back = RunConfig.from_yaml(p)
    assert back.images_dir == cfg.images_dir
    assert back.methods == cfg.methods
    assert back.max_images == cfg.max_images
    assert back.seed == cfg.seed
    assert abs(back.strata.empty_fraction - 0.3) < 1e-9
    assert abs(back.strata.day_fraction - 0.1) < 1e-9


def test_openmp_flag_is_set_before_torch_import():
    """KMP_DUPLICATE_LIB_OK must be set before anything imports torch.

    The OMP Error #15 crash happens when the first enhancement module
    imports torch mid-run, not at import time, so the flag has to be in the
    environment before that happens.
    """
    import os
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE", (
        "KMP_DUPLICATE_LIB_OK is not set; the harness will hit "
        "OMP Error #15 on Windows when the first torch-importing "
        "enhancement is loaded"
    )


def test_openmp_flag_survives_importing_the_package():
    """Importing evaluation must not clear the flag."""
    import os
    import evaluation  # noqa: F401
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
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

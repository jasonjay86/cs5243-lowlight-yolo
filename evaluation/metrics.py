"""
Detection metrics for the evaluation harness.

Owner: Jason Johnson

The metric definitions and the class-agnostic scoring rule are taken from
scripts/evaluate_enhancements.py (Krutin Patel, PR #8), which was the first
implementation of these metrics in the project. This module factors them out
so the harness can compute them from a cached set of detections.

Class-agnostic scoring
-----------------------
YOLOv8n is a COCO model. It has no bobcat, opossum, raccoon or skunk class,
so a per-class score would be meaningless. Instead every ground-truth box
is treated as one "animal" class, and a prediction counts as a true positive
if it overlaps the right place, whatever its COCO label says. This is
reported explicitly in the results table, because it is the assumption that
makes the number interpretable.

The four CCT species that DO have COCO counterparts (bird, dog, cat, cow)
are the only frames where the predicted label can also be correct; on every
other frame the label is ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)   # 0.50 ... 0.95


@dataclass
class Stats:
    """Detections for one method, sorted by descending confidence."""
    confs: np.ndarray      # (D,)
    tp: np.ndarray         # (D, 10) bool, one column per IoU threshold
    img: np.ndarray        # (D,) frame index of each detection
    n_gt: np.ndarray       # (N,) ground-truth boxes per frame
    empty: np.ndarray      # (N,) bool, frame is animal-free
    night: np.ndarray      # (N,) bool, frame is a night frame
    species: list[str]     # (N,) dominant species label per frame
    enh_ms: float = 0.0
    yolo_ms: float = 0.0
    gflops_mp: float | None = None


@dataclass
class MethodResult:
    """Final scored numbers for one method."""
    method: str
    ap50: float
    ap50_ci: tuple[float, float]
    map5095: float
    precision: float
    recall: float
    d_ap50: float
    d_ci: tuple[float, float]
    fp_rate: float
    fp_ci: tuple[float, float]
    enh_ms: float
    yolo_ms: float
    gflops_mp: float | None
    n_detections: int = 0
    extras: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Geometry and matching
# ----------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_image(boxes: np.ndarray, confs: np.ndarray, gt: np.ndarray):
    """Greedy match, highest confidence first, at every IoU threshold.

    Returns (sorted_confs, tp) where tp is (n_det, 10) bool.
    """
    order = np.argsort(-confs)
    boxes, confs = boxes[order], confs[order]
    tp = np.zeros((len(boxes), len(IOU_THRS)), dtype=bool)
    if len(boxes) and len(gt):
        ious = iou_matrix(boxes, gt)
        for ti, thr in enumerate(IOU_THRS):
            used = np.zeros(len(gt), dtype=bool)
            for di in range(len(boxes)):
                cand = np.where(~used & (ious[di] >= thr))[0]
                if cand.size:
                    j = cand[np.argmax(ious[di, cand])]
                    used[j] = True
                    tp[di, ti] = True
    return confs, tp


def build_stats(frames, per_image_dets, enh_ms=0.0, yolo_ms=0.0,
                gflops_mp=None) -> Stats:
    """Assemble a Stats object from per-frame detections.

    frames: list of Frame-like objects with .boxes, .is_empty, .is_night,
            .species
    per_image_dets: list of (boxes, confs) tuples, one per frame, aligned
            with `frames`.
    """
    confs, tps, imgs = [], [], []
    for i, (fr, (boxes, c)) in enumerate(zip(frames, per_image_dets)):
        c2, tp = match_image(boxes, c, fr.boxes)
        confs.append(c2)
        tps.append(tp)
        imgs.append(np.full(len(c2), i, dtype=int))

    confs = np.concatenate(confs) if confs else np.zeros(0)
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_THRS)), bool)
    img = np.concatenate(imgs) if imgs else np.zeros(0, int)

    order = np.argsort(-confs, kind="stable")
    species = [(f.species[0] if f.species else "empty") for f in frames]
    return Stats(
        confs=confs[order], tp=tp[order], img=img[order],
        n_gt=np.array([len(f.boxes) for f in frames]),
        empty=np.array([f.is_empty for f in frames]),
        night=np.array([f.is_night for f in frames]),
        species=species,
        enh_ms=enh_ms, yolo_ms=yolo_ms, gflops_mp=gflops_mp,
    )


# ----------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------
def pr_curve(s: Stats, ti: int, w: np.ndarray):
    """Weighted precision/recall arrays at IoU column ti. w = per-frame weights."""
    ww = w[s.img]
    n_gt = float((w * s.n_gt).sum())
    if n_gt == 0 or len(s.confs) == 0:
        return np.zeros(0), np.zeros(0), n_gt
    tpc = np.cumsum(s.tp[:, ti] * ww)
    fpc = np.cumsum((~s.tp[:, ti]) * ww)
    return tpc / n_gt, tpc / np.maximum(tpc + fpc, 1e-12), n_gt


def average_precision(s: Stats, ti: int, w: np.ndarray) -> float:
    """COCO-style 101-point interpolated average precision."""
    rec, prec, n_gt = pr_curve(s, ti, w)
    if n_gt == 0:
        return float("nan")
    if len(rec) == 0:
        return 0.0
    prec = np.maximum.accumulate(prec[::-1])[::-1]      # precision envelope
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    q = np.zeros(101)
    ok = idx < len(prec)
    q[ok] = prec[idx[ok]]
    return float(q.mean())


def map_50_95(s: Stats, w: np.ndarray) -> float:
    """Mean AP across all ten IoU thresholds."""
    vals = [average_precision(s, ti, w) for ti in range(len(IOU_THRS))]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def prec_rec_at(s: Stats, conf: float, w: np.ndarray):
    """Precision and recall at one confidence threshold."""
    keep = s.confs >= conf
    ww = w[s.img[keep]]
    tp = float((s.tp[keep, 0] * ww).sum())
    n_det = float(ww.sum())
    n_gt = float((w * s.n_gt).sum())
    return (tp / n_det if n_det else float("nan")), (tp / n_gt if n_gt else float("nan"))


def empty_fp_rate(s: Stats, conf: float, w: np.ndarray) -> float:
    """Share of animal-free frames that produced at least one detection."""
    n = len(s.n_gt)
    has_det = np.zeros(n, dtype=bool)
    has_det[s.img[s.confs >= conf]] = True
    denom = float((w * s.empty).sum())
    return float((w * (s.empty & has_det)).sum() / denom) if denom else float("nan")


def bootstrap(all_stats: dict, conf: float, n_boot: int, seed: int = 0) -> dict:
    """Paired bootstrap over frames: every method sees the same resample.

    Pairing is what makes the delta CI meaningful. Independent resamples per
    method would inflate the interval and hide real differences.
    """
    names = list(all_stats)
    n = len(next(iter(all_stats.values())).n_gt)
    rng = np.random.default_rng(seed)
    ap = {m: [] for m in names}
    fp = {m: [] for m in names}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for m in names:
            ap[m].append(average_precision(all_stats[m], 0, w))
            fp[m].append(empty_fp_rate(all_stats[m], conf, w))
    ap = {m: np.array(v) for m, v in ap.items()}
    fp = {m: np.array(v) for m, v in fp.items()}
    base = names[0]
    out = {}
    for m in names:
        d = ap[m] - ap[base]
        out[m] = {
            "ap_ci": tuple(np.nanpercentile(ap[m], [2.5, 97.5])),
            "d_ci": tuple(np.nanpercentile(d, [2.5, 97.5])),
            "fp_ci": tuple(np.nanpercentile(fp[m], [2.5, 97.5])),
        }
    return out


def summarize(all_stats: dict, conf: float, n_boot: int,
              base_method: str | None = None) -> list[MethodResult]:
    """Score every method against the first one (the baseline)."""
    names = list(all_stats)
    base = base_method or names[0]
    ones = np.ones(len(next(iter(all_stats.values())).n_gt))
    boot = bootstrap(all_stats, conf, n_boot)
    base_ap = average_precision(all_stats[base], 0, ones)

    results = []
    for m in names:
        s = all_stats[m]
        p, r = prec_rec_at(s, conf, ones)
        ap50 = average_precision(s, 0, ones)
        results.append(MethodResult(
            method=m,
            ap50=ap50,
            ap50_ci=boot[m]["ap_ci"],
            map5095=map_50_95(s, ones),
            precision=p,
            recall=r,
            d_ap50=ap50 - base_ap,
            d_ci=boot[m]["d_ci"],
            fp_rate=empty_fp_rate(s, conf, ones),
            fp_ci=boot[m]["fp_ci"],
            enh_ms=s.enh_ms,
            yolo_ms=s.yolo_ms,
            gflops_mp=s.gflops_mp,
            n_detections=int(len(s.confs)),
        ))
    return results


# ----------------------------------------------------------------------
# Stratified breakdown
# ----------------------------------------------------------------------
def subset_stats(s: Stats, mask: np.ndarray) -> Stats:
    """Restrict a Stats object to the frames selected by `mask`.

    Detections on excluded frames are dropped, and the per-frame ground
    truth counts are restricted too, so the sub-population is scored on its
    own terms.

    The frame indices in `s.img` are REMAPPED to their new positions in the
    subset. Without that, `s.img` would still hold the original frame
    numbers while `n_gt` had been compressed, and every downstream
    `w[s.img[...]]` lookup would index out of bounds.
    """
    keep_frame = np.zeros(len(s.n_gt), dtype=bool)
    keep_frame[mask] = True

    # old frame index -> new frame index within the subset
    remap = np.full(len(s.n_gt), -1, dtype=int)
    remap[mask] = np.arange(int(mask.sum()))

    det_keep = keep_frame[s.img]
    sub = Stats(
        confs=s.confs[det_keep],
        tp=s.tp[det_keep],
        img=remap[s.img[det_keep]],
        n_gt=s.n_gt[mask],
        empty=s.empty[mask],
        night=s.night[mask],
        species=[sp for sp, m in zip(s.species, mask) if m],
        enh_ms=s.enh_ms, yolo_ms=s.yolo_ms, gflops_mp=s.gflops_mp,
    )
    return sub


def breakdown(s: Stats, conf: float, by: str = "night") -> dict:
    """Per-stratum precision/recall/FP for one Stats object.

    by="night" -> night frames vs day frames
    by="empty" -> animal frames vs empty frames
    """
    out: dict = {}
    n = len(s.n_gt)
    if by == "night":
        groups = {
            "night": s.night,
            "day": ~s.night,
        }
    elif by == "empty":
        groups = {
            "animal": ~s.empty,
            "empty": s.empty,
        }
    else:
        raise ValueError(f"unknown breakdown axis: {by}")

    for name, mask in groups.items():
        if mask.sum() == 0:
            out[name] = {"n_frames": 0, "precision": float("nan"),
                         "recall": float("nan"), "ap50": float("nan"),
                         "n_gt": 0}
            continue
        sub = subset_stats(s, mask)
        ones = np.ones(len(sub.n_gt))
        p, r = prec_rec_at(sub, conf, ones)
        out[name] = {
            "n_frames": int(mask.sum()),
            "n_gt": int(sub.n_gt.sum()),
            "precision": p,
            "recall": r,
            "ap50": average_precision(sub, 0, ones),
        }
    return out


def per_species(s: Stats, conf: float, min_frames: int = 5) -> dict:
    """Precision/recall per species, for species with enough frames."""
    out: dict = {}
    species_arr = np.array(s.species)
    for sp in sorted(set(s.species)):
        mask = species_arr == sp
        if mask.sum() < min_frames:
            continue
        sub = subset_stats(s, mask)
        ones = np.ones(len(sub.n_gt))
        p, r = prec_rec_at(sub, conf, ones)
        out[sp] = {
            "n_frames": int(mask.sum()),
            "n_gt": int(sub.n_gt.sum()),
            "precision": p,
            "recall": r,
        }
    return out

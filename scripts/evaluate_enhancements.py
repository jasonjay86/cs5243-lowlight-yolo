"""Compare raw vs enhanced images through the shared YOLO and save ONE PNG
(results table on top, two plots below).

Run from the repo root, for example:

    python scripts/evaluate_enhancements.py \
        --images data/cct20/eccv_18_all_images_sm \
        --annotations data/cct20/caltech_bboxes_20200316.json \
        --methods raw clahe snr-aware --max-images 500

Layout check without any data or weights (clearly labelled SYNTHETIC):

    python scripts/evaluate_enhancements.py --demo

What it measures (all class-agnostic, see "Why class-agnostic" below)
  * mAP@0.5 and mAP@0.5:0.95   COCO-style 101-point average precision
  * Precision / Recall          at one fixed confidence threshold (--conf, IoU 0.5)
  * Delta mAP@0.5 vs raw        with a paired bootstrap 95% CI over images
  * Empty-frame FP rate         share of animal-free frames with >= 1 detection
  * Latency                     enhancement ms/image and YOLO ms/image
  * GFLOPs per megapixel        from Enhancement.flops_per_megapixel() (n/a if None)

Why class-agnostic: YOLOv8n is a COCO model with no bobcat/opossum/etc. classes,
so a per-class score would be meaningless. Every ground-truth box is treated as
one "animal" class and a prediction counts if it boxes the right place, whatever
its COCO label. Say this in the report.

Fairness: every condition uses the same images, the same YOLO weights, the same
image size and the same thresholds. YOLO is run at a very low confidence floor
(--conf-floor) so the precision-recall curve is complete; the precision, recall
and false-positive numbers are then taken at --conf.

NOTE: this calls detector.model(...) directly (not YOLODetector.detect()) because
detect() drops everything below ultralytics' default threshold, which would make
AP wrong. It still uses Rich's YOLODetector class and his yolov8n.pt weights.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# OpenMP-on-Windows: PyTorch and Ultralytics both link against an OpenMP
# runtime, and on a stock conda+pip install the second one to initialise
# refuses to load (OMP Error #15: "Initializing libomp.dll, but found
# libiomp5md.dll already initialized"). The crash aborts the run partway
# through, which is especially confusing because it happens on the FIRST
# method that touches torch, not at import time.
#
# KMP_DUPLICATE_LIB_OK=TRUE is Intel's documented workaround for the
# inference path (as opposed to training). It is set BEFORE torch and
# ultralytics are imported so both see the flag. The risk of silently
# wrong numerics is real but low for inference on a fixed model.
#
# If you would rather set it in your shell instead of in this file, this
# setdefault() leaves an externally-set value alone.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

warnings.filterwarnings("ignore", message="All-NaN slice")

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)  # 0.50 ... 0.95

# Colours: validated categorical slots 1-4 (blue, orange, aqua, yellow), fixed
# order per method. Text never wears these; they only mark lines/bars/swatches.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class Item:
    path: str
    gt: np.ndarray      # (n, 4) xyxy in the pixel space of the image on disk
    empty: bool         # annotated as an empty frame (no animal)


@dataclass
class Stats:
    """Everything needed to score one method; detections sorted by confidence."""
    confs: np.ndarray       # (D,)
    tp: np.ndarray          # (D, 10) bool, one column per IoU threshold
    img: np.ndarray         # (D,) image index of each detection
    n_gt: np.ndarray        # (N,) ground-truth boxes per image
    empty: np.ndarray       # (N,) bool
    enh_ms: float
    yolo_ms: float
    gflops_mp: float | None


# --------------------------------------------------------------------------
# Geometry and matching
# --------------------------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_image(boxes: np.ndarray, confs: np.ndarray, gt: np.ndarray):
    """Greedy match (highest confidence first) at every IoU threshold."""
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


def build_stats(items, dets, enh_ms, yolo_ms, gflops_mp) -> Stats:
    confs, tps, imgs = [], [], []
    for i, (it, (boxes, c)) in enumerate(zip(items, dets)):
        c2, tp = match_image(boxes, c, it.gt)
        confs.append(c2)
        tps.append(tp)
        imgs.append(np.full(len(c2), i, dtype=int))
    confs = np.concatenate(confs) if confs else np.zeros(0)
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_THRS)), bool)
    img = np.concatenate(imgs) if imgs else np.zeros(0, int)
    order = np.argsort(-confs, kind="stable")
    return Stats(confs[order], tp[order], img[order],
                 np.array([len(it.gt) for it in items]),
                 np.array([it.empty for it in items]),
                 enh_ms, yolo_ms, gflops_mp)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def pr_curve(s: Stats, ti: int, w: np.ndarray):
    """Weighted precision/recall arrays at IoU column ti. w = per-image weights."""
    ww = w[s.img]
    n_gt = float((w * s.n_gt).sum())
    if n_gt == 0 or len(s.confs) == 0:
        return np.zeros(0), np.zeros(0), n_gt
    tpc = np.cumsum(s.tp[:, ti] * ww)
    fpc = np.cumsum((~s.tp[:, ti]) * ww)
    return tpc / n_gt, tpc / np.maximum(tpc + fpc, 1e-12), n_gt


def average_precision(s: Stats, ti: int, w: np.ndarray) -> float:
    rec, prec, n_gt = pr_curve(s, ti, w)
    if n_gt == 0:
        return float("nan")
    if len(rec) == 0:
        return 0.0
    prec = np.maximum.accumulate(prec[::-1])[::-1]       # precision envelope
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    q = np.zeros(101)
    ok = idx < len(prec)
    q[ok] = prec[idx[ok]]
    return float(q.mean())


def map_50_95(s: Stats, w: np.ndarray) -> float:
    return float(np.mean([average_precision(s, ti, w) for ti in range(len(IOU_THRS))]))


def prec_rec_at(s: Stats, conf: float, w: np.ndarray):
    keep = s.confs >= conf
    ww = w[s.img[keep]]
    tp = float((s.tp[keep, 0] * ww).sum())
    n_det = float(ww.sum())
    n_gt = float((w * s.n_gt).sum())
    return (tp / n_det if n_det else float("nan")), (tp / n_gt if n_gt else float("nan"))


def empty_fp_rate(s: Stats, conf: float, w: np.ndarray) -> float:
    n = len(s.n_gt)
    has_det = np.zeros(n, dtype=bool)
    has_det[s.img[s.confs >= conf]] = True
    denom = float((w * s.empty).sum())
    return float((w * (s.empty & has_det)).sum() / denom) if denom else float("nan")


def bootstrap(all_stats: dict, conf: float, n_boot: int, seed: int = 0):
    """Paired bootstrap over images: same resample for every method."""
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


def summarize(all_stats: dict, conf: float, n_boot: int) -> list[dict]:
    names = list(all_stats)
    ones = np.ones(len(next(iter(all_stats.values())).n_gt))
    boot = bootstrap(all_stats, conf, n_boot)
    base_ap = average_precision(all_stats[names[0]], 0, ones)
    rows = []
    for m in names:
        s = all_stats[m]
        p, r = prec_rec_at(s, conf, ones)
        ap50 = average_precision(s, 0, ones)
        rows.append(dict(
            method=m, ap50=ap50, ap50_ci=boot[m]["ap_ci"],
            map5095=map_50_95(s, ones), precision=p, recall=r,
            d_ap50=ap50 - base_ap, d_ci=boot[m]["d_ci"],
            fp_rate=empty_fp_rate(s, conf, ones), fp_ci=boot[m]["fp_ci"],
            enh_ms=s.enh_ms, yolo_ms=s.yolo_ms, gflops_mp=s.gflops_mp,
        ))
    return rows


# --------------------------------------------------------------------------
# Dataset loading (CCT-style COCO-camera-traps JSON)
# --------------------------------------------------------------------------
def is_night_frame(img: np.ndarray, thresh: float = 3.0) -> bool:
    """Grayscale/IR heuristic: channels nearly equal. Not ground truth."""
    import cv2
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.int16)
    b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
    return float((np.abs(b - g) + np.abs(g - r)).mean() / 2.0) <= thresh


def read_json_or_zip(path: Path) -> dict:
    import zipfile
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".json") and not n.startswith("__MACOSX"))
            return json.loads(z.read(name))
    return json.loads(path.read_text())


def empty_images_from_labels(path: Path) -> dict:
    """image id -> file name for images whose ONLY image-level label is 'empty'."""
    data = read_json_or_zip(path)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    labels: dict = {}
    for a in data["annotations"]:
        labels.setdefault(a["image_id"], set()).add(cats.get(a["category_id"]))
    names = {im["id"]: im.get("file_name", f"{im['id']}.jpg") for im in data["images"]}
    return {i: names.get(i, f"{i}.jpg") for i, l in labels.items() if l == {"empty"}}


def load_items(images_dir: Path, ann_path: Path, max_images: int, night_only: bool,
               seed: int, empty_fraction: float = 0.25,
               unannotated_as_empty: bool = False,
               image_labels: Path | None = None) -> list[Item]:
    """Pick frames that exist on disk. Animal frames have >= 1 animal box. Empty
    frames come from the image-level labels file (--image-labels, only label is
    'empty'). Frames with NO annotation in the bbox file are NOT empty: checking
    them showed they contain animals that simply have no box, so they are skipped
    unless --unannotated-as-empty is given.
    Frames containing a 'car' box are skipped: a car is a real COCO object, not
    an animal, so it would be scored unfairly either way."""
    import cv2
    data = json.loads(Path(ann_path).read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    by_img: dict = {}
    for a in data["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    on_disk = {p.name: p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    animals, empties, n_unann, n_car = [], [], 0, 0
    for im in data["images"]:
        p = on_disk.get(Path(im["file_name"]).name) or on_disk.get(f"{im['id']}.jpg")
        if p is None:
            continue
        anns = by_img.get(im["id"], [])
        names = {cats.get(a["category_id"]) for a in anns}
        if "car" in names:
            n_car += 1
            continue
        boxes = [a for a in anns if "bbox" in a and cats.get(a["category_id"]) != "empty"]
        if boxes:
            animals.append((im, p, boxes, False))
        elif "empty" in names:
            empties.append((im, p, [], True))
        elif not anns and unannotated_as_empty:
            n_unann += 1
            empties.append((im, p, [], True))
    n_labelled_empty = 0
    if image_labels is not None:
        seen = {str(p) for _, p, _, _ in empties}
        have_boxes = {im["id"] for im, _, _, _ in animals}
        for iid, fname in empty_images_from_labels(image_labels).items():
            if iid in have_boxes:
                continue
            p = on_disk.get(Path(fname).name) or on_disk.get(f"{iid}.jpg")
            if p is None or str(p) in seen:
                continue
            seen.add(str(p))
            empties.append(({"id": iid}, p, [], True))
            n_labelled_empty += 1
    print(f"On disk and usable: {len(animals)} animal frames, {len(empties)} empty frames "
          f"({n_labelled_empty} from --image-labels, {n_unann} unannotated and assumed empty); "
          f"skipped {n_car} frames with a car.")
    if not empties:
        print("WARNING: no empty frames available, so the empty-frame false-positive rate will be n/a. "
              "Pass --image-labels (the image-level annotations from LILA) to get real empty frames.")
    if not animals and not empties:
        sys.exit("No annotated images from the JSON were found in the images folder. "
                 "Check --images and --annotations (file names must match).")
    rng = random.Random(seed)
    rng.shuffle(animals)
    rng.shuffle(empties)

    def take(cands, quota):
        out = []
        for im, p, boxes, is_empty in cands:
            if len(out) >= quota:
                break
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None or (night_only and not is_night_frame(img)):
                continue
            h, w = img.shape[:2]
            sx = w / im["width"] if im.get("width") else 1.0   # boxes may be in original-size pixels
            sy = h / im["height"] if im.get("height") else 1.0
            gt = np.array([[a["bbox"][0] * sx, a["bbox"][1] * sy,
                            (a["bbox"][0] + a["bbox"][2]) * sx,
                            (a["bbox"][1] + a["bbox"][3]) * sy] for a in boxes], dtype=float).reshape(-1, 4)
            out.append(Item(str(p), gt, is_empty))
        return out

    got_empty = take(empties, int(round(max_images * empty_fraction)))
    got_animal = take(animals, max_images - len(got_empty))
    if len(got_animal) + len(got_empty) < max_images:   # animals ran short: top up with empties
        extra = take(empties[len(got_empty) * 3:], max_images - len(got_animal) - len(got_empty))
        got_empty += extra
    items = got_animal + got_empty
    print(f"Selected {len(items)} frames: {len(got_animal)} with animals, {len(got_empty)} empty.")
    for it in got_empty[:5]:
        print("  empty-frame example (open it and check it really is empty):", it.path)
    return items


# --------------------------------------------------------------------------
# Running a method through YOLO
# --------------------------------------------------------------------------
def run_method(name, items, model, imgsz, conf_floor):
    import cv2
    import enhancements.raw  # noqa: F401  (registers "raw")
    import enhancements.clahe  # noqa: F401
    try:
        import enhancements.snr_aware  # noqa: F401
    except Exception as e:  # torch/weights missing: only matters if requested
        print(f"(snr-aware not importable: {e})")
    from enhancements.base import get_enhancement

    try:
        enh = get_enhancement(name)
    except KeyError:
        raise SystemExit(
            f"ERROR: enhancement '{name}' is not registered. "
            f"A module may have failed to import (see any warning above). "
            f"Run with --methods raw clahe snr-aware only after the module "
            f"imports cleanly."
        )
    dets, enh_t, yolo_t, mps = [], [], [], []
    for k, it in enumerate(items, 1):
        img = cv2.imread(it.path, cv2.IMREAD_COLOR)
        t0 = time.perf_counter()
        out = enh.enhance(img)
        enh_t.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        res = model(out, conf=conf_floor, imgsz=imgsz, verbose=False)[0]
        yolo_t.append((time.perf_counter() - t0) * 1000)
        dets.append((res.boxes.xyxy.cpu().numpy().reshape(-1, 4), res.boxes.conf.cpu().numpy()))
        mps.append(img.shape[0] * img.shape[1] / 1e6)
        if k % 50 == 0:
            print(f"  {name}: {k}/{len(items)}")
    # skip first image in latency means (warm-up)
    skip = 1 if len(items) > 5 else 0
    return dets, float(np.mean(enh_t[skip:])), float(np.mean(yolo_t[skip:])), enh.flops_per_megapixel()


def make_samples(items, methods, out_path, height=360):
    """Save a grid (rows = frames, columns = methods) of enhanced images so you can
    LOOK at what each method does. Ground-truth boxes are drawn in green on the
    first column. No YOLO involved, so it is fast. Also prints mean brightness."""
    import cv2
    import enhancements.raw  # noqa: F401
    import enhancements.clahe  # noqa: F401
    try:
        import enhancements.snr_aware  # noqa: F401
    except Exception as e:
        print(f"(snr-aware not importable: {e})")
    from enhancements.base import get_enhancement

    enhs = {m: get_enhancement(m) for m in methods}

    def tile(img, text):
        h, w = img.shape[:2]
        img = cv2.resize(img, (max(1, int(w * height / h)), height))
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    rows = []
    print("frame  kind     " + "  ".join(f"{m:>10}" for m in methods) + "   (mean brightness 0-255)")
    for k, it in enumerate(items, 1):
        img = cv2.imread(it.path, cv2.IMREAD_COLOR)
        tiles, bright = [], []
        for j, (m, e) in enumerate(enhs.items()):
            out = e.enhance(img).copy()
            if out.shape != img.shape:
                print(f"  note: {m} changed the image shape {img.shape} -> {out.shape}")
            bright.append(float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()))
            tag = m + ("  (green = ground truth)" if j == 0 else "") + ("  [EMPTY FRAME]" if it.empty else "")
            if j == 0:
                for x1, y1, x2, y2 in it.gt:
                    cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), max(2, img.shape[1] // 300))
            tiles.append(tile(out, tag))
        w = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
        rows.append(np.hstack(tiles))
        print(f"{k:>5}  {'empty' if it.empty else 'animal':<7}  " + "  ".join(f"{b:>10.1f}" for b in bright))
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90] if str(out_path).lower().endswith(".jpg") else [])


# --------------------------------------------------------------------------
# Synthetic data for --demo (layout check only; NOT results)
# --------------------------------------------------------------------------
def synthetic(seed=0, n=400):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        empty = rng.random() < 0.3
        k = 0 if empty else rng.integers(1, 3)
        gt = []
        for _ in range(k):
            w, h = rng.uniform(60, 250), rng.uniform(60, 200)
            x, y = rng.uniform(0, 1024 - w), rng.uniform(0, 768 - h)
            gt.append([x, y, x + w, y + h])
        items.append(Item(f"synthetic_{i}", np.array(gt, float).reshape(-1, 4), bool(empty)))

    def make(hit, fp_rate, jitter):
        dets = []
        for it in items:
            b, c = [], []
            for g in it.gt:
                if rng.random() < hit:
                    d = g + rng.normal(0, jitter, 4)
                    b.append(d); c.append(rng.beta(4, 3))
            while rng.random() < fp_rate:
                w, h = rng.uniform(40, 200), rng.uniform(40, 200)
                x, y = rng.uniform(0, 1024 - w), rng.uniform(0, 768 - h)
                b.append([x, y, x + w, y + h]); c.append(rng.beta(2, 5))
            dets.append((np.array(b, float).reshape(-1, 4), np.array(c, float)))
        return dets
    conds = {"raw": (0.30, 0.20, 10, 0.0, 9.0, None),
             "clahe": (0.45, 0.45, 12, 11.0, 9.0, None),
             "snr-aware": (0.40, 0.25, 9, 60.0, 9.0, 120.0)}
    out = {}
    for m, (hit, fp, jit, e_ms, y_ms, fl) in conds.items():
        out[m] = build_stats(items, make(hit, fp, jit), e_ms, y_ms, fl)
    return items, out


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def fmt(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def fmt_ci(v, ci, d=3, signed=False):
    s = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
    return f"{s}\n[{ci[0]:+.{d}f}, {ci[1]:+.{d}f}]" if signed else f"{s}\n[{ci[0]:.{d}f}, {ci[1]:.{d}f}]"


def draw(rows, all_stats, conf, caption, title, out_path, demo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["method"] for r in rows]
    color = {m: SERIES[i % len(SERIES)] for i, m in enumerate(names)}
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig = plt.figure(figsize=(15, 10), dpi=200, facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.55, 1.25], hspace=0.32, wspace=0.18,
                          left=0.05, right=0.97, top=0.90, bottom=0.10)
    fig.text(0.05, 0.955, title, fontsize=17, fontweight="bold", ha="left")
    if demo:
        fig.text(0.05, 0.925, "SYNTHETIC DEMO DATA: layout check only, not real results",
                 fontsize=13, fontweight="bold", color="#b3261e", ha="left")

    # ---- table ----
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    head = ["", "Condition", "mAP@0.5\n[95% CI]", "mAP@\n0.5:0.95", f"Precision\n(conf>={conf})",
            f"Recall\n(conf>={conf})", "Δ mAP@0.5 vs raw\n[95% CI]", "Empty-frame\nFP rate [95% CI]",
            "Enhance\n(ms/img)", "YOLO\n(ms/img)", "Enhance\nGFLOPs/MP"]
    cells = []
    for r in rows:
        base = r["method"] == names[0]
        cells.append(["", r["method"] + " + YOLO", fmt_ci(r["ap50"], r["ap50_ci"]), fmt(r["map5095"]),
                      fmt(r["precision"]), fmt(r["recall"]),
                      "baseline" if base else fmt_ci(r["d_ap50"], r["d_ci"], signed=True),
                      fmt_ci(r["fp_rate"], r["fp_ci"], 2), fmt(r["enh_ms"], 1), fmt(r["yolo_ms"], 1),
                      fmt(r["gflops_mp"], 1) if r["gflops_mp"] is not None else "n/a"])
    widths = [0.015, 0.14, 0.11, 0.07, 0.08, 0.08, 0.15, 0.13, 0.075, 0.075, 0.085]
    tb = ax.table(cellText=cells, colLabels=head, colWidths=widths, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(11); tb.scale(1, 2.9)
    for (ri, ci), c in tb.get_celld().items():
        c.set_edgecolor(GRID); c.set_linewidth(0.8)
        if ri == 0:
            c.set_facecolor("#efeeea"); c.set_text_props(fontweight="bold", color=INK, fontsize=10)
        else:
            c.set_facecolor(SURFACE); c.set_text_props(color=INK)
            if ci == 0:
                c.set_facecolor(color[names[ri - 1]])
            if ci == 1:
                c.set_text_props(fontweight="bold", ha="left"); c._loc = "left"
    ax.text(0.0, -0.02, "Class-agnostic: every ground-truth box is one 'animal' class; a detection counts if it "
            "boxes the right place, whatever its COCO label. Higher is better except FP rate and latency.",
            transform=ax.transAxes, fontsize=10, color=INK2, va="top")

    # ---- panel A: PR curves ----
    a1 = fig.add_subplot(gs[1, 0]); a1.set_facecolor(SURFACE)
    n_img = len(next(iter(all_stats.values())).n_gt)
    ones = np.ones(n_img)
    for m in names:
        rec, prec, _ = pr_curve(all_stats[m], 0, ones)
        prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
        a1.plot(rec, prec, color=color[m], lw=2, label=m + " + YOLO")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1.02)
    a1.set_xlabel("Recall (IoU 0.5)"); a1.set_ylabel("Precision")
    a1.set_title("A. Precision-recall curve", loc="left", fontsize=13, fontweight="bold")
    a1.grid(color=GRID, lw=0.8); a1.set_axisbelow(True)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.legend(frameon=False, loc="upper right", fontsize=11)

    # ---- panel B: delta mAP bars ----
    a2 = fig.add_subplot(gs[1, 1]); a2.set_facecolor(SURFACE)
    others = [r for r in rows if r["method"] != names[0]]
    xs = np.arange(len(others))
    lo_all = min([0.0] + [r["d_ci"][0] for r in others])
    hi_all = max([0.0] + [r["d_ci"][1] for r in others])
    pad = (hi_all - lo_all) * 0.18 or 0.01
    a2.set_ylim(lo_all - pad, hi_all + pad * 1.6)
    a2.set_xlim(-0.7, len(others) - 0.3)
    for x, r in zip(xs, others):
        lo, hi = r["d_ci"]
        a2.bar(x, r["d_ap50"], width=0.38, color=color[r["method"]])
        a2.errorbar(x, r["d_ap50"], yerr=[[r["d_ap50"] - lo], [hi - r["d_ap50"]]], color=INK2, capsize=5, lw=1.6)
        a2.text(x, hi + pad * 0.25, f"{r['d_ap50']:+.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    a2.axhline(0, color=INK2, lw=1.2)
    a2.set_xticks(xs); a2.set_xticklabels([r["method"] + " + YOLO" for r in others], fontsize=11)
    a2.set_xlabel("below 0 = the enhancement hurt detection; a CI that crosses 0 = no clear difference")
    a2.set_ylabel("Δ mAP@0.5 vs raw (95% bootstrap CI)")
    a2.set_title("B. Change in mAP@0.5 vs raw baseline", loc="left", fontsize=13, fontweight="bold")
    a2.grid(axis="y", color=GRID, lw=0.8); a2.set_axisbelow(True)
    for s_ in ("top", "right", "bottom"):
        a2.spines[s_].set_visible(False)
    a2.tick_params(axis="x", length=0)

    fig.text(0.05, 0.03, caption, fontsize=10, color=INK2, ha="left", va="center", wrap=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="folder with the extracted CCT images (searched recursively)")
    ap.add_argument("--annotations", help="CCT bbox JSON, e.g. caltech_bboxes_20200316.json")
    ap.add_argument("--methods", nargs="+", default=["raw", "clahe", "snr-aware"],
                    help="registered enhancement names, FIRST is the baseline (default: raw clahe snr-aware)")
    ap.add_argument("--max-images", type=int, default=500)
    ap.add_argument("--empty-fraction", type=float, default=0.25,
                    help="share of the sample that should be empty frames (needed for the FP-rate metric; default 0.25)")
    ap.add_argument("--image-labels",
                    help="LILA image-level annotations (caltech_camera_traps.json or .json.zip); source of REAL empty frames")
    ap.add_argument("--unannotated-as-empty", action="store_true",
                    help="NOT recommended: treat bbox-file frames with no annotation as empty (they usually contain animals)")
    ap.add_argument("--all-frames", action="store_true", help="do not filter to grayscale/IR night-like frames")
    ap.add_argument("--conf", type=float, default=0.25, help="threshold for precision/recall/FP rate (default 0.25)")
    ap.add_argument("--conf-floor", type=float, default=0.001, help="YOLO confidence floor for the PR curve")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/figures/yolo_comparison.png")
    ap.add_argument("--samples-only", type=int, metavar="N",
                    help="no YOLO: save a before/after picture grid of N frames (default file results/figures/enhanced_samples.jpg)")
    ap.add_argument("--demo", action="store_true", help="synthetic data, no YOLO or dataset (layout check)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.samples_only:
        if not args.images or not args.annotations:
            sys.exit("--samples-only needs --images and --annotations.")
        items = load_items(Path(args.images), Path(args.annotations), args.samples_only,
                           not args.all_frames, args.seed, args.empty_fraction,
                           args.unannotated_as_empty,
                           Path(args.image_labels) if args.image_labels else None)
        out = Path(args.out) if args.out != "results/figures/yolo_comparison.png" else Path("results/figures/enhanced_samples.jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        make_samples(items, args.methods, out)
        print(f"\nSaved {out}")
        return

    if args.demo:
        items, all_stats = synthetic(args.seed)
        weights = "synthetic"
    else:
        if not args.images or not args.annotations:
            sys.exit("Need --images and --annotations (or use --demo).")
        items = load_items(Path(args.images), Path(args.annotations), args.max_images,
                           not args.all_frames, args.seed, args.empty_fraction,
                           args.unannotated_as_empty,
                           Path(args.image_labels) if args.image_labels else None)
        if not items:
            sys.exit("No images selected. Try --all-frames.")
        from detection.yolo_detector import YOLODetector
        model = YOLODetector().model
        weights = "detection/yolov8n.pt"
        all_stats = {}
        for m in args.methods:
            print(f"Running {m} on {len(items)} images ...")
            dets, e_ms, y_ms, fl = run_method(m, items, model, args.imgsz, args.conf_floor)
            all_stats[m] = build_stats(items, dets, e_ms, y_ms, fl)

    rows = summarize(all_stats, args.conf, args.bootstrap)
    n_empty = sum(it.empty for it in items)
    n_boxes = sum(len(it.gt) for it in items)
    try:
        import torch
        dev = "GPU" if torch.cuda.is_available() else "CPU"
    except Exception:
        dev = "n/a"
    if args.demo:
        dev = "n/a (synthetic)"
    caption = (f"{len(items)} frames ({len(items) - n_empty} with animals, {n_empty} empty), {n_boxes} ground-truth boxes. "
               f"Same YOLO ({weights}), imgsz {args.imgsz}, P/R/FP at conf {args.conf}, IoU 0.5. "
               f"Latency measured on {dev}. CIs: paired bootstrap over images ({args.bootstrap} resamples).")
    draw(rows, all_stats, args.conf, caption, "Raw vs enhanced + YOLO on CCT-20 night frames", out_path, args.demo)

    print("\n" + f"{'condition':<12}{'mAP50':>8}{'mAP50-95':>10}{'P':>7}{'R':>7}{'dAP50':>8}{'emptyFP':>9}{'enh ms':>8}{'yolo ms':>9}")
    for r in rows:
        print(f"{r['method']:<12}{fmt(r['ap50']):>8}{fmt(r['map5095']):>10}{fmt(r['precision']):>7}{fmt(r['recall']):>7}"
              f"{r['d_ap50']:>+8.3f}{fmt(r['fp_rate'], 2):>9}{fmt(r['enh_ms'], 1):>8}{fmt(r['yolo_ms'], 1):>9}")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
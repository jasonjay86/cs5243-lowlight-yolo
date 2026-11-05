#!/usr/bin/env python3
"""Enhance every image in the frozen CCT-20 split manifest and write a new
manifest that points at the enhanced copies.

Why this exists
---------------
The fine-tuning plan trains one YOLO model per input condition (raw, CLAHE,
SNR-Aware, RetinexFormer). Each condition needs its own copy of the dataset,
enhanced once and cached on disk, with the *same* train/val/test assignment.
This script produces that copy without touching the originals and without
changing ``scripts/prepare_yolo_dataset.py``: the new manifest has exactly the
same columns and rows, only ``image_path`` differs.

Usage (from the repo root)
--------------------------
    python scripts/enhance_dataset.py \\
        --manifest data/cct20_splits/split_manifest.csv \\
        --method clahe \\
        --output data/cct20_clahe \\
        --workers 16

    python scripts/prepare_yolo_dataset.py \\
        --manifest data/cct20_clahe/split_manifest.csv \\
        --annotations data/cct20/caltech_bboxes_20200316.json \\
        --output data/cct20_yolo_clahe

Output layout
-------------
    <output>/
    +-- images/<image_id>.jpg       enhanced images (same size as the originals)
    +-- split_manifest.csv          same rows/columns, image_path -> enhanced file
    +-- enhance_summary.json        method, settings, counts, timing

Notes
-----
* Resumable: images that already exist (non-empty) are skipped, so a job that
  hits its time limit can simply be resubmitted. Files are written to a
  temporary name and renamed, so a killed job never leaves a half-written JPEG.
* The manifest is written only if every image succeeded. Failures are listed
  in ``enhance_failures.json`` and the script exits non-zero.
* Enhanced images are saved as JPEG (default quality 95), which adds a second
  round of lossy compression on top of the source JPEG. Use the same setting
  for every method so the comparison stays fair.
* CPU methods (raw, clahe) parallelise with ``--workers``. GPU methods
  (snr-aware, retinexformer) should use ``--workers 1 --device cuda``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
GPU_METHODS = {"snr-aware", "retinexformer"}

# Per-process state, set once in each worker by _init_worker().
_ENH = None
_QUALITY = 95


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, required=True,
                   help="Frozen split_manifest.csv (the raw one).")
    p.add_argument("--method", required=True,
                   help="Registered enhancement name: raw, clahe, snr-aware, retinexformer.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory for the enhanced copy.")
    p.add_argument("--workers", type=int, default=1,
                   help="Worker processes. Use >1 for CPU methods only. Default 1.")
    p.add_argument("--device", default=None,
                   help="cpu / cuda / cuda:0, passed to GPU methods only.")
    p.add_argument("--jpeg-quality", type=int, default=95,
                   help="JPEG quality of the enhanced images. Default 95.")
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS),
                   help="Only enhance these splits. Default: all three.")
    p.add_argument("--limit-per-split", type=int, default=None,
                   help="Smoke test: only the first N rows of each split.")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT,
                   help="Repo root (for imports and relative image paths).")
    return p


def _init_worker(method: str, device: str | None, quality: int, repo_root: str) -> None:
    """Build one enhancement object per worker process."""
    global _ENH, _QUALITY
    import cv2

    cv2.setNumThreads(1)  # avoid workers x OpenCV threads oversubscription
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from enhancements.base import get_enhancement

    for mod in ("raw", "clahe", "snr_aware", "retinexformer"):
        try:
            __import__(f"enhancements.{mod}")
        except Exception as exc:  # a broken module must not block the others
            print(f"[enhance] (enhancements.{mod} not importable: {exc})", flush=True)
    kwargs = {"device": device} if (device and method in GPU_METHODS) else {}
    _ENH = get_enhancement(method, **kwargs)
    _QUALITY = quality


def _process(job: tuple[str, str]) -> tuple[str, str, float, str | None]:
    """Enhance one image. Returns (src, dst, ms, error)."""
    import cv2

    src, dst = job
    try:
        if os.path.isfile(dst) and os.path.getsize(dst) > 0:
            return src, dst, 0.0, None  # already done (resume)
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            return src, dst, 0.0, "could not read source image"
        t0 = time.perf_counter()
        out = _ENH.enhance(img)
        ms = (time.perf_counter() - t0) * 1000.0
        if out.shape != img.shape:
            return src, dst, ms, f"shape changed {img.shape} -> {out.shape}"
        tmp = dst[:-4] + ".tmp.jpg"
        ok = cv2.imwrite(tmp, out, [cv2.IMWRITE_JPEG_QUALITY, _QUALITY])
        if not ok:
            return src, dst, ms, "cv2.imwrite failed"
        os.replace(tmp, dst)
        return src, dst, ms, None
    except Exception as exc:  # report, do not crash the whole pool
        return src, dst, 0.0, f"{type(exc).__name__}: {exc}"


def read_manifest(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SystemExit(f"ERROR: manifest has no header: {path}")
        need = {"image_id", "image_path", "split"}
        missing = need - set(reader.fieldnames)
        if missing:
            raise SystemExit(f"ERROR: manifest missing columns: {sorted(missing)}")
        return list(reader.fieldnames), list(reader)


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    if not args.manifest.is_file():
        raise SystemExit(f"ERROR: manifest not found: {args.manifest}")
    if args.workers < 1:
        raise SystemExit("ERROR: --workers must be at least 1")
    if args.workers > 1 and args.method in GPU_METHODS:
        raise SystemExit(
            f"ERROR: {args.method} is a GPU method; use --workers 1 --device cuda."
        )
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("ERROR: --jpeg-quality must be 1-100")

    fieldnames, rows = read_manifest(args.manifest)

    # Keep only the requested splits, optionally the first N rows of each.
    selected, seen = [], {s: 0 for s in SPLITS}
    for row in rows:
        split = row["split"].strip()
        if split not in args.splits:
            continue
        if args.limit_per_split is not None and seen[split] >= args.limit_per_split:
            continue
        seen[split] += 1
        selected.append(row)
    if not selected:
        raise SystemExit("ERROR: no manifest rows selected")

    out_dir = args.output.resolve()
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    jobs, dst_by_id = [], {}
    for row in selected:
        raw = Path(row["image_path"])
        src = raw if raw.is_absolute() else (repo_root / raw)
        src = src.resolve()
        dst = (img_dir / f"{row['image_id']}.jpg").resolve()
        if dst == src:
            raise SystemExit(f"ERROR: output would overwrite the source image: {src}")
        jobs.append((str(src), str(dst)))
        dst_by_id[row["image_id"]] = str(dst)

    print(f"[enhance] method={args.method} images={len(jobs)} workers={args.workers}")
    print(f"[enhance] output: {out_dir}", flush=True)

    t_start = time.perf_counter()
    failures, enh_ms, done = [], [], 0
    init_args = (args.method, args.device, args.jpeg_quality, str(repo_root))
    if args.workers == 1:
        _init_worker(*init_args)
        results = map(_process, jobs)
        pool = None
    else:
        pool = Pool(args.workers, initializer=_init_worker, initargs=init_args)
        results = pool.imap_unordered(_process, jobs, chunksize=16)

    try:
        for src, dst, ms, err in results:
            done += 1
            if err:
                failures.append({"source": src, "error": err})
            elif ms > 0:
                enh_ms.append(ms)
            if done % 1000 == 0 or done == len(jobs):
                rate = done / max(time.perf_counter() - t_start, 1e-9)
                print(f"[enhance] {done}/{len(jobs)}  {rate:.1f} img/s  "
                      f"failures={len(failures)}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    wall = time.perf_counter() - t_start
    summary = {
        "method": args.method,
        "device": args.device,
        "jpeg_quality": args.jpeg_quality,
        "source_manifest": str(args.manifest),
        "splits": args.splits,
        "limit_per_split": args.limit_per_split,
        "rows_selected": len(selected),
        "rows_per_split": seen,
        "newly_enhanced": len(enh_ms),
        "skipped_existing": len(jobs) - len(enh_ms) - len(failures),
        "failures": len(failures),
        "mean_enhance_ms_per_image": (sum(enh_ms) / len(enh_ms)) if enh_ms else None,
        "wall_seconds": round(wall, 1),
    }

    if failures:
        (out_dir / "enhance_failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8")
        (out_dir / "enhance_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"ERROR: {len(failures)} images failed; see "
              f"{out_dir / 'enhance_failures.json'}. Manifest NOT written.")
        return 1

    new_manifest = out_dir / "split_manifest.csv"
    with new_manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            new_row = dict(row)
            new_row["image_path"] = dst_by_id[row["image_id"]]
            writer.writerow(new_row)
    (out_dir / "enhance_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[enhance] done in {wall:.0f}s; {summary['newly_enhanced']} new, "
          f"{summary['skipped_existing']} skipped")
    print(f"[enhance] manifest: {new_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
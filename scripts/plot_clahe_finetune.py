#!/usr/bin/env python3
"""Score fine-tuned YOLO models on the held-out TEST split and draw one table + plot.

Four evaluations (model trained on X, scored on test images processed as Y):

    raw   model on raw   test images   <- baseline
    raw   model on CLAHE test images   (Q1: does enhancing the input help a raw-trained model?)
    CLAHE model on CLAHE test images   (Q2: does training on enhanced data beat raw training?)
    CLAHE model on raw   test images   (extra cell of the 4x4 grid)

Uses Ultralytics' own validation on the `test:` split of each dataset.yaml, so only
held-out camera locations are scored. Metrics are Ultralytics' (mAP at conf 0.001,
precision/recall at the best-F1 point), NOT the harness numbers in the zero-shot
table, so do not mix the two in one table.

Run on a GPU node (writes a CSV, then the figure):

    python scripts/plot_clahe_finetune.py \\
        --raw-weights   <path to raw best.pt> \\
        --clahe-weights runs/detect/experiments/yolo_finetune/yolov8n_cct_animal_640_clahe/weights/best.pt \\
        --raw-data      data/cct20_yolo/dataset.yaml \\
        --clahe-data    data/cct20_yolo_clahe/dataset.yaml

Re-draw from the CSV without a GPU (e.g. on a laptop):

    python scripts/plot_clahe_finetune.py --from-csv results/finetune_clahe_comparison.csv

Check the drawing only, with invented numbers (clearly watermarked):

    python scripts/plot_clahe_finetune.py --demo --out /tmp/demo.png --csv /tmp/demo.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# (trained on, evaluated on); the first row is the baseline.
COMBOS = [("raw", "raw"), ("raw", "clahe"), ("clahe", "clahe"), ("clahe", "raw")]
FIELDS = ["model", "input", "map50", "map50_95", "precision", "recall"]

# Same hues as the zero-shot figure: raw = blue, CLAHE = orange.
COLORS = {"raw": "#2a76d8", "clahe": "#ee6a35"}
LABEL = {"raw": "raw", "clahe": "CLAHE"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw-weights")
    p.add_argument("--clahe-weights")
    p.add_argument("--raw-data", help="dataset.yaml of the raw-image YOLO dataset")
    p.add_argument("--clahe-data", help="dataset.yaml of the CLAHE-image YOLO dataset")
    p.add_argument("--device", default="0", help="Ultralytics device (default 0)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--csv", type=Path, default=Path("results/finetune_clahe_comparison.csv"))
    p.add_argument("--out", type=Path, default=Path("results/figures/finetune_clahe_comparison.png"))
    p.add_argument("--from-csv", type=Path, help="skip evaluation; draw from this CSV")
    p.add_argument("--demo", action="store_true", help="invented numbers, watermarked")
    return p


def evaluate(args) -> list[dict]:
    needed = {
        "--raw-weights": args.raw_weights, "--clahe-weights": args.clahe_weights,
        "--raw-data": args.raw_data, "--clahe-data": args.clahe_data,
    }
    missing = [k for k, v in needed.items() if not v]
    if missing:
        raise SystemExit(f"ERROR: missing {', '.join(missing)}")
    for k, v in needed.items():
        if not Path(v).is_file():
            raise SystemExit(f"ERROR: {k} not found: {v}")

    from ultralytics import YOLO  # here so --from-csv/--demo and bad args need no torch

    weights = {"raw": args.raw_weights, "clahe": args.clahe_weights}
    data = {"raw": args.raw_data, "clahe": args.clahe_data}
    rows = []
    for model_name, input_name in COMBOS:
        print(f"[eval] {model_name} model on {input_name} test images", flush=True)
        model = YOLO(weights[model_name])
        res = model.val(
            data=data[input_name], split="test", imgsz=args.imgsz, device=args.device,
            plots=False, verbose=False,
            name=f"eval_{model_name}model_{input_name}test",
        )
        rows.append({
            "model": model_name, "input": input_name,
            "map50": float(res.box.map50), "map50_95": float(res.box.map),
            "precision": float(res.box.mp), "recall": float(res.box.mr),
        })
        print("       " + "  ".join(f"{k}={rows[-1][k]:.3f}" for k in FIELDS[2:]), flush=True)
    return rows


def demo_rows() -> list[dict]:
    vals = [(0.80, 0.50, 0.85, 0.72), (0.76, 0.47, 0.83, 0.70),
            (0.82, 0.52, 0.86, 0.74), (0.74, 0.45, 0.82, 0.68)]
    return [dict(model=m, input=i, map50=a, map50_95=b, precision=c, recall=d)
            for (m, i), (a, b, c, d) in zip(COMBOS, vals)]


def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in FIELDS[2:]:
            r[k] = float(r[k])
    return rows


def draw(rows, out: Path, demo: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = rows[0]["map50"]
    fig = plt.figure(figsize=(14, 7.5), facecolor="#fcfcfb")
    fig.text(0.04, 0.95, "Fine-tuned YOLOv8n on CCT-20: raw vs CLAHE (held-out test locations)",
             fontsize=17, fontweight="bold", va="top")
    if demo:
        fig.text(0.04, 0.90, "SYNTHETIC DEMO: invented numbers, layout check only",
                 fontsize=13, color="#b00020", fontweight="bold", va="top")

    # ---- table -------------------------------------------------------------
    ax = fig.add_axes([0.04, 0.50, 0.92, 0.36])
    ax.axis("off")
    cols = ["Trained on", "Test images", "mAP@0.5", "mAP@0.5:0.95",
            "Precision", "Recall", "Δ mAP@0.5\nvs raw/raw"]
    cells = []
    for i, r in enumerate(rows):
        delta = "baseline" if i == 0 else f"{r['map50'] - base:+.3f}"
        cells.append([LABEL[r["model"]], LABEL[r["input"]], f"{r['map50']:.3f}",
                      f"{r['map50_95']:.3f}", f"{r['precision']:.3f}",
                      f"{r['recall']:.3f}", delta])
    tbl = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#efeeeb")
            cell.set_text_props(fontweight="bold")
        elif col == 1:
            cell.set_text_props(fontweight="bold", color=COLORS[rows[row - 1]["input"]])

    # ---- bar chart ---------------------------------------------------------
    bx = fig.add_axes([0.08, 0.09, 0.55, 0.32], facecolor="#fcfcfb")
    labels = [f"{LABEL[r['model']]} model\non {LABEL[r['input']]}" for r in rows]
    bars = bx.bar(range(len(rows)), [r["map50"] for r in rows],
                  color=[COLORS[r["input"]] for r in rows], width=0.6)
    for b, r in zip(bars, rows):
        bx.text(b.get_x() + b.get_width() / 2, r["map50"] + 0.01, f"{r['map50']:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    bx.axhline(base, color="#555555", lw=1, ls="--")
    bx.set_xticks(range(len(rows)))
    bx.set_xticklabels(labels, fontsize=11)
    bx.set_ylabel("mAP@0.5 (test split)")
    bx.set_ylim(0, min(1.0, max(r["map50"] for r in rows) + 0.12))
    bx.set_title("mAP@0.5 by trained model and test input (dashed = raw/raw baseline)",
                 fontsize=12, loc="left", fontweight="bold")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    # ---- legend / notes ----------------------------------------------------
    fig.text(0.68, 0.43, "Bar colour = test input (blue raw, orange CLAHE)",
             fontsize=10.5, fontweight="bold", va="top")
    notes =("Q1 (test time): row 1 vs row 2.\n"
             "Does enhancing the input help a\nraw-trained model?\n\n"
             "Q2 (train time): row 1 vs row 3.\n"
             "Does training on enhanced data\nbeat raw training?\n\n"
             "One 'animal' class, so scoring is\nclass-agnostic. Ultralytics metrics\n"
             "(mAP at conf 0.001, P/R at best-F1);\nno confidence intervals.")
    fig.text(0.68, 0.385, notes, fontsize=9.5, color="#444444", va="top", linespacing=1.35)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


def main() -> int:
    args = build_parser().parse_args()
    if args.demo:
        rows = demo_rows()
    elif args.from_csv:
        rows = read_csv(args.from_csv)
    else:
        rows = evaluate(args)
    if not args.from_csv:
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    draw(rows, args.out, args.demo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

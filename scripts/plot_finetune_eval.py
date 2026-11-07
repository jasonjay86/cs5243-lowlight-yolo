#!/usr/bin/env python3
"""Score fine-tuned YOLO models on the held-out TEST split and draw table + plots.

Works for any number of enhancement conditions (raw, CLAHE, SNR-Aware,
RetinexFormer, ...). Give one fine-tuned model and one YOLO dataset per
condition, with the baseline (raw) FIRST:

    python scripts/plot_finetune_eval.py \\
        --model raw=<raw best.pt>  --data raw=data/cct20_yolo/dataset.yaml \\
        --model clahe=<clahe best.pt>  --data clahe=data/cct20_yolo_clahe/dataset.yaml \\
        --model snr-aware=<best.pt>  --data snr-aware=data/cct20_yolo_snr/dataset.yaml \\
        --model retinexformer=<best.pt>  --data retinexformer=data/cct20_yolo_retinex/dataset.yaml

Cells of the (trained on) x (test images) grid:

    Q1, test time:  the FIRST (raw) model scored on each enhanced test set
    Q2, train time: each model scored on its OWN matched test set (the diagonal)

By default every cell is scored (N x N runs). --cells minimal scores only the
Q1 row and the Q2 diagonal (2N-1 runs).

Only the `test:` split of each dataset.yaml is scored, so only held-out camera
locations are used. Metrics are Ultralytics' (mAP at conf 0.001, precision/recall
at the best-F1 point), not the harness numbers of the zero-shot table: do not mix
the two. Score the test split once, after models and thresholds are final.

Re-draw from the CSV without a GPU:

    python scripts/plot_finetune_eval.py --from-csv results/finetune_comparison.csv

Check the drawing only, with invented numbers (watermarked):

    python scripts/plot_finetune_eval.py --demo --out /tmp/demo.png --csv /tmp/demo.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

FIELDS = ["model", "input", "map50", "map50_95", "precision", "recall"]

PRETTY = {"raw": "raw", "clahe": "CLAHE", "snr-aware": "SNR-Aware",
          "retinexformer": "RetinexFormer"}
# Same hues as the zero-shot figure; extra hues are validated categorical colours.
PALETTE = ["#2a76d8", "#ee6a35", "#1ab27a", "#eda300", "#8a63d2", "#d6457a"]


def pretty(name: str) -> str:
    return PRETTY.get(name, name)


def color_of(name: str, order: list[str]) -> str:
    return PALETTE[order.index(name) % len(PALETTE)]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", action="append", default=[], metavar="NAME=WEIGHTS",
                   help="fine-tuned weights per condition; repeat; baseline FIRST")
    p.add_argument("--data", action="append", default=[], metavar="NAME=YAML",
                   help="YOLO dataset.yaml per condition; repeat; same names as --model")
    p.add_argument("--cells", choices=["full", "minimal"], default="full")
    p.add_argument("--device", default="0", help="Ultralytics device (default 0)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--csv", type=Path, default=Path("results/finetune_comparison.csv"))
    p.add_argument("--out", type=Path, default=Path("results/figures/finetune_comparison.png"))
    p.add_argument("--from-csv", type=Path, help="skip evaluation; draw from this CSV")
    p.add_argument("--demo", action="store_true", help="invented numbers, watermarked")
    return p


def parse_pairs(items: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"ERROR: {flag} expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"ERROR: {flag} expects NAME=PATH, got {item!r}")
        if name in out:
            raise SystemExit(f"ERROR: {flag} given twice for {name!r}")
        out[name] = path
    return out


def cells_to_run(names: list[str], mode: str) -> list[tuple[str, str]]:
    base = names[0]
    if mode == "full":
        return [(m, i) for m in names for i in names]
    cells = [(base, i) for i in names]                    # Q1 row (includes baseline)
    cells += [(m, m) for m in names if m != base]         # Q2 diagonal
    return cells


def evaluate(args) -> list[dict]:
    models = parse_pairs(args.model, "--model")
    data = parse_pairs(args.data, "--data")
    if len(models) < 2:
        raise SystemExit("ERROR: give at least two --model NAME=WEIGHTS (baseline first)")
    if list(models) != list(data):
        raise SystemExit(
            "ERROR: --model and --data must name the same conditions in the same order.\n"
            f"  models: {list(models)}\n  data:   {list(data)}"
        )
    for flag, mapping in (("--model", models), ("--data", data)):
        for name, path in mapping.items():
            if not Path(path).is_file():
                raise SystemExit(f"ERROR: {flag} {name}: file not found: {path}")

    from ultralytics import YOLO  # here so --from-csv/--demo and bad args need no torch

    names = list(models)
    loaded: dict[str, object] = {}
    rows = []
    for model_name, input_name in cells_to_run(names, args.cells):
        print(f"[eval] {pretty(model_name)} model on {pretty(input_name)} test images", flush=True)
        if model_name not in loaded:
            loaded[model_name] = YOLO(models[model_name])
        res = loaded[model_name].val(
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
    names = ["raw", "clahe", "snr-aware", "retinexformer"]
    base = {"raw": 0.80, "clahe": 0.81, "snr-aware": 0.77, "retinexformer": 0.78}
    rows = []
    for m in names:
        for i in names:
            v = base[m] - (0 if m == i else 0.05) + (0.01 if i == "clahe" else 0.0)
            v = v - (0.03 if i in ("snr-aware", "retinexformer") and m != i else 0.0)
            rows.append(dict(model=m, input=i, map50=round(v, 3), map50_95=round(v * 0.62, 3),
                             precision=round(v + 0.05, 3), recall=round(v - 0.07, 3)))
    return rows


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


def order_of(rows) -> list[str]:
    seen: list[str] = []
    for r in rows:
        for n in (r["model"], r["input"]):
            if n not in seen:
                seen.append(n)
    return seen


def draw(rows, out: Path, demo: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = order_of(rows)
    n = len(names)
    cell = {(r["model"], r["input"]): r for r in rows}
    base_name = names[0]
    base = cell[(base_name, base_name)]["map50"]

    fig = plt.figure(figsize=(14, 9.2), facecolor="#fcfcfb")
    fig.text(0.04, 0.965, "Fine-tuned YOLOv8n on CCT-20: raw vs enhanced (held-out test locations)",
             fontsize=17, fontweight="bold", va="top")
    if demo:
        fig.text(0.04, 0.925, "SYNTHETIC DEMO: invented numbers, layout check only",
                 fontsize=13, color="#b00020", fontweight="bold", va="top")

    # ---- A. heatmap of the full grid ---------------------------------------
    hx = fig.add_axes([0.13, 0.43, 0.30, 0.30])
    grid = np.full((n, n), np.nan)
    for (m, i), r in cell.items():
        grid[names.index(m), names.index(i)] = r["map50"]
    vmax = np.nanmax(grid)
    vmin = max(0.0, np.nanmin(grid) - 0.05)
    hx.imshow(grid, cmap="Blues", vmin=vmin, vmax=vmax + 0.02, aspect="auto")
    for a in range(n):
        for b in range(n):
            v = grid[a, b]
            if np.isnan(v):
                hx.text(b, a, "n/a", ha="center", va="center", fontsize=10, color="#999999")
            else:
                dark = (v - vmin) / max(vmax + 0.02 - vmin, 1e-9) > 0.55
                hx.text(b, a, f"{v:.3f}", ha="center", va="center", fontsize=11,
                        fontweight="bold" if a == b else "normal",
                        color="white" if dark else "#222222")
    for k in range(n):  # outline the diagonal (Q2: matched train/test)
        hx.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec="#111111", lw=2))
    hx.set_xticks(range(n))
    hx.set_xticklabels([pretty(x) for x in names], fontsize=10)
    hx.set_yticks(range(n))
    hx.set_yticklabels([pretty(x) for x in names], fontsize=10)
    hx.set_xlabel("test images", fontsize=10)
    hx.set_ylabel("model trained on", fontsize=10)
    hx.xaxis.set_label_position("top"); hx.xaxis.tick_top()
    hx.set_title("A. mAP@0.5 grid\n(outlined = matched, Q2; first row = Q1)",
                 fontsize=11, loc="left", fontweight="bold", pad=34)

    # ---- B. grouped bars: Q1 and Q2 per enhancement ------------------------
    bx = fig.add_axes([0.57, 0.43, 0.39, 0.29], facecolor="#fcfcfb")
    enh = names[1:]
    x = np.arange(len(enh))
    w = 0.36
    for k, m in enumerate(enh):
        q1 = cell.get((base_name, m), {}).get("map50")
        q2 = cell.get((m, m), {}).get("map50")
        c = color_of(m, names)
        for off, v, alpha in ((-w / 2, q1, 0.45), (w / 2, q2, 1.0)):
            if v is None:
                continue
            bx.bar(k + off, v, w, color=c, alpha=alpha)
            bx.text(k + off, v + 0.01, f"{v:.3f}", ha="center", fontsize=9.5, fontweight="bold")
    bx.axhline(base, color="#555555", lw=1, ls="--")
    bx.set_xticks(x)
    bx.set_xticklabels([pretty(m) for m in enh], fontsize=11)
    bx.set_ylabel("mAP@0.5 (test split)")
    top = max([r["map50"] for r in rows] + [base])
    bx.set_ylim(0, min(1.0, top + 0.12))
    bx.set_title(f"B. light = {pretty(base_name)}-trained model on enhanced input (Q1); "
                 f"solid = matched (Q2)\n    dashed = {pretty(base_name)}/{pretty(base_name)} baseline",
                 fontsize=10.5, loc="left", fontweight="bold")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)

    # ---- C. summary table: Q1 and Q2 per condition -------------------------
    tx = fig.add_axes([0.04, 0.06, 0.92, 0.26])
    tx.axis("off")
    cols = ["Condition", "Q1: raw-trained model\non enhanced input", "Δ vs raw/raw",
            "Q2: trained and tested\non enhanced", "Δ vs raw/raw",
            "Q2 mAP@0.5:0.95", "Q2 precision", "Q2 recall"]
    body = []
    for m in names:
        q1 = cell.get((base_name, m))
        q2 = cell.get((m, m))
        if m == base_name:
            body.append([pretty(m), f"{q2['map50']:.3f}", "baseline", f"{q2['map50']:.3f}",
                         "baseline", f"{q2['map50_95']:.3f}", f"{q2['precision']:.3f}",
                         f"{q2['recall']:.3f}"])
            continue
        body.append([
            pretty(m),
            f"{q1['map50']:.3f}" if q1 else "n/a",
            f"{q1['map50'] - base:+.3f}" if q1 else "n/a",
            f"{q2['map50']:.3f}" if q2 else "n/a",
            f"{q2['map50'] - base:+.3f}" if q2 else "n/a",
            f"{q2['map50_95']:.3f}" if q2 else "n/a",
            f"{q2['precision']:.3f}" if q2 else "n/a",
            f"{q2['recall']:.3f}" if q2 else "n/a",
        ])
    tbl = tx.table(cellText=body, colLabels=cols, loc="upper center", cellLoc="center",
                   colWidths=[0.13, 0.15, 0.11, 0.15, 0.11, 0.13, 0.11, 0.11])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.9)
    for (row, col), c in tbl.get_celld().items():
        c.set_edgecolor("#dddddd")
        if row == 0:
            c.set_facecolor("#efeeeb")
            c.set_text_props(fontweight="bold")
            c.set_height(c.get_height() * 1.35)
        elif col == 0:
            c.set_text_props(fontweight="bold", color=color_of(names[row - 1], names))

    fig.text(0.04, 0.025,
             "One 'animal' class, so scoring is class-agnostic. Ultralytics metrics (mAP at conf 0.001; "
             "P/R at best-F1); no confidence intervals. Not comparable with the zero-shot table.",
             fontsize=9.5, color="#444444")

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

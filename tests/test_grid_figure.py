"""Grid figure tests: any number of methods, no clipped text.

Owner: Jason Johnson

The figure has to survive 1 method, 2, 5, and a full N x N grid, because the
number of enhancement methods is not fixed and will grow when CLAHE and
Retina arrive. These tests pin that, plus the two layout failures this
project has already paid for once: text outside the canvas and a figure that
describes a run it did not measure.
"""
from __future__ import annotations

import csv
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from evaluation.grid_figure import (draw_grid_figure, load_rows,  # noqa: E402
                                    draw_summary_figure)
from evaluation.grid_figure import main as figure_cli  # noqa: E402


# ----------------------------------------------------------------------
def _cell(model: str, cond: str, kind: str, map50: float, d: float,
          lo: float, hi: float) -> dict:
    return {
        "model": model, "condition": cond, "kind": kind, "n_frames": 6761,
        "map50": round(map50, 4),
        "map50_lo": round(map50 - 0.008, 4),
        "map50_hi": round(map50 + 0.008, 4),
        "d_map50": round(d, 4), "d_lo": round(lo, 4), "d_hi": round(hi, 4),
        "map50_95": round(map50 * 0.6, 4),
        "precision": 0.87, "recall": 0.71,
        "empty_fp": 0.21, "fp_lo": 0.19, "fp_hi": 0.23,
        "enh_ms": None, "yolo_ms": 1.8, "gflops_mp": 8.6,
        "n_detections": 6431,
    }


def _minimal_rows(methods: list[str], base: float = 0.91) -> list[dict]:
    """Same cell layout the grid produces: baseline row + diagonal."""
    rows = [_cell(methods[0], methods[0], "Q1+Q2", base, 0.0, 0.0, 0.0)]
    for name in methods[1:]:
        rows.append(_cell(methods[0], name, "Q1", base * 0.5, -0.46,
                          -0.48, -0.44))
    for name in methods[1:]:
        rows.append(_cell(name, name, "Q2", base * 0.8, -0.17, -0.18, -0.16))
    return rows


def _full_rows(methods: list[str]) -> list[dict]:
    rows = []
    for i, m in enumerate(methods):
        for j, c in enumerate(methods):
            kind = "Q1+Q2" if (i == 0 and j == 0) else (
                "Q2" if i == j else "Q1")
            rows.append(_cell(m, c, kind, 0.9 - 0.1 * (i + j),
                              0.0 if (i == 0 and j == 0) else -0.1 * (i + j),
                              -0.12 * (i + j + 1), -0.08 * (i + j + 1)))
    return rows


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "condition", "kind", "n_frames", "map50", "map50_lo",
              "map50_hi", "d_map50", "d_lo", "d_hi", "map50_95", "precision",
              "recall", "empty_fp", "fp_lo", "fp_hi"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "map50_lo": r["map50"], "map50_hi": r["map50"],
                        "precision": 0.8, "recall": 0.7,
                        "fp_lo": 0.18, "fp_hi": 0.24})
    return path


def _all_texts(fig) -> list:
    """Every text actually rendered: titles, labels, in-view ticks, notes."""
    fig.canvas.draw()
    items = list(fig.texts)
    sup = getattr(fig, "_suptitle", None)
    # A suptitle is registered in fig.texts on current matplotlib; only add
    # it when it is genuinely missing, or every title pair "overlaps" itself.
    if sup is not None and not any(t is sup for t in items):
        items.append(sup)
    for ax in fig.axes:
        items += [ax.title, ax.xaxis.label, ax.yaxis.label]
        items += list(ax.texts)
        for axis, limits in ((ax.xaxis, ax.get_xlim()),
                             (ax.yaxis, ax.get_ylim())):
            lo, hi = min(limits), max(limits)
            for tick in axis.get_major_ticks():
                loc = tick.get_loc()
                if loc is None or loc < lo - 1e-9 or loc > hi + 1e-9:
                    continue          # outside the view: not rendered
                items += [tick.label1, tick.label2]
    out = []
    for t in items:
        if not t.get_text() or not t.get_visible():
            continue
        b = t.get_window_extent(renderer=fig.canvas.get_renderer())
        if b.x1 < b.x0 - 0.5 or b.y1 < b.y0 - 0.5:
            continue
        out.append((t, b))
    return out


def _overlaps(fig) -> list[tuple]:
    """Pairs of rendered texts whose boxes intersect by more than 2 px."""
    pairs = _all_texts(fig)
    bad = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            (t1, b1), (t2, b2) = pairs[i], pairs[j]
            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ox > 2 and oy > 2:
                bad.append((t1.get_text()[:30], t2.get_text()[:30],
                            round(ox), round(oy)))
    return bad


def _out_of_canvas(fig) -> list[tuple]:
    """Text whose rendered extent leaves the figure canvas."""
    fb = fig.bbox
    tol = 2.0
    bad = []
    for t, b in _all_texts(fig):
        if (b.x0 < fb.x0 - tol or b.x1 > fb.x1 + tol
                or b.y0 < fb.y0 - tol or b.y1 > fb.y1 + tol):
            bad.append((t.get_text()[:36], round(b.x0 - fb.x0, 1),
                        round(b.x1 - fb.x1, 1), round(b.y0 - fb.y0, 1),
                        round(b.y1 - fb.y1, 1)))
    return bad


# ----------------------------------------------------------------------
def test_two_methods() -> None:
    rows = _minimal_rows(["raw", "snr-aware"])
    with tempfile.TemporaryDirectory() as td:
        out = draw_grid_figure(rows, Path(td) / "g.png", split="test")
        assert out.is_file()
        assert out.stat().st_size > 20_000, out.stat().st_size


def test_single_method() -> None:
    """One method scores one cell; the delta panel must say so, not crash."""
    rows = [_cell("raw", "raw", "Q1+Q2", 0.91, 0.0, 0.0, 0.0)]
    with tempfile.TemporaryDirectory() as td:
        out = draw_grid_figure(rows, Path(td) / "g.png")
        assert out.is_file()


def test_five_methods() -> None:
    methods = ["raw", "clahe", "snr-aware", "retinexformer", "agcwd"]
    rows = _minimal_rows(methods)
    with tempfile.TemporaryDirectory() as td:
        out, fig = draw_grid_figure(rows, Path(td) / "g.png", split="test",
                                    return_fig=True)
        try:
            assert out.is_file()
            assert out.stat().st_size > 30_000
        finally:
            plt.close(fig)


def test_full_grid_all_cells() -> None:
    methods = ["raw", "clahe", "snr-aware"]
    rows = _full_rows(methods)          # 9 cells, no gaps
    with tempfile.TemporaryDirectory() as td:
        out = draw_grid_figure(rows, Path(td) / "g.png")
        assert out.is_file()


def test_minimal_grid_marks_unscored_cells() -> None:
    """A 3-method minimal run has 5 rows but a 3x3 grid: 4 holes."""
    rows = _minimal_rows(["raw", "clahe", "snr"])
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", return_fig=True)
        try:
            ax = fig.axes[0]
            dashes = [t for t in ax.texts if t.get_text() == "—"]
            assert len(dashes) == 4, f"expected 4 unscored cells, got {len(dashes)}"
        finally:
            plt.close(fig)


def test_no_text_leaves_the_canvas_five_methods() -> None:
    rows = _minimal_rows(["raw", "clahe", "snr-aware", "retinexformer", "agcwd"])
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", split="test",
                                  return_fig=True)
        try:
            bad = _out_of_canvas(fig)
        finally:
            plt.close(fig)
    assert not bad, f"text outside canvas: {bad}"


def test_no_text_leaves_the_canvas_full_grid() -> None:
    rows = _full_rows(["raw", "clahe", "snr-aware", "retinex"])
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", return_fig=True)
        try:
            bad = _out_of_canvas(fig)
        finally:
            plt.close(fig)
    assert not bad, f"text outside canvas: {bad}"


def test_long_method_names_do_not_overflow() -> None:
    methods = ["raw", "retinexformer", "snr-aware", "clahe+gamma"]
    rows = _minimal_rows(methods)
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", return_fig=True)
        try:
            bad = _out_of_canvas(fig)
        finally:
            plt.close(fig)
    assert not bad, f"long labels overflowed: {bad}"


def test_roundtrip_from_csv() -> None:
    """The figure must be drawable from the file alone, with no config."""
    rows = _minimal_rows(["raw", "clahe", "snr-aware"])
    with tempfile.TemporaryDirectory() as td:
        csv_path = _write_csv(Path(td) / "grid.csv", rows)
        loaded = load_rows(csv_path)
        assert len(loaded) == len(rows)
        out = draw_grid_figure(loaded, Path(td) / "g.png")
        assert out.is_file()


def test_caption_has_no_local_paths() -> None:
    rows = _minimal_rows(["raw", "snr"])
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", split="test",
                                  return_fig=True)
        try:
            texts = [t.get_text() for t in fig.texts]
            texts.append(fig._suptitle.get_text())
        finally:
            plt.close(fig)
    joined = " ".join(texts)
    assert "C:\\" not in joined and "/work/" not in joined, joined
    assert "test" in joined.lower(), joined


def test_no_overlap_two_methods() -> None:
    rows = _minimal_rows(["raw", "snr-aware"])
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", split="test",
                                  return_fig=True)
        try:
            bad = _overlaps(fig)
        finally:
            plt.close(fig)
    assert not bad, f"overlapping text: {bad}"


def test_no_overlap_five_methods() -> None:
    """Long row labels on the delta panel must not run into the heatmap."""
    rows = _minimal_rows(["raw", "clahe", "snr-aware", "retinexformer", "agcwd"])
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", split="test",
                                  return_fig=True)
        try:
            bad = _overlaps(fig)
        finally:
            plt.close(fig)
    assert not bad, f"overlapping text: {bad}"


def test_no_overlap_full_grid() -> None:
    rows = _full_rows(["raw", "clahe", "snr-aware", "retinex"])
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", return_fig=True)
        try:
            bad = _overlaps(fig)
        finally:
            plt.close(fig)
    assert not bad, f"overlapping text: {bad}"


def test_no_overlap_single_method() -> None:
    rows = [_cell("raw", "raw", "Q1+Q2", 0.91, 0.0, 0.0, 0.0)]
    with tempfile.TemporaryDirectory() as td:
        _, fig = draw_grid_figure(rows, Path(td) / "g.png", return_fig=True)
        try:
            bad = _overlaps(fig)
        finally:
            plt.close(fig)
    assert not bad, f"overlapping text: {bad}"


def test_summary_figure_renders() -> None:
    rows = _minimal_rows(["raw", "clahe", "snr-aware"])
    with tempfile.TemporaryDirectory() as td:
        out = draw_summary_figure(rows, Path(td) / "s.png", split="test")
        assert out.is_file()
        assert out.stat().st_size > 40_000


def test_summary_figure_survives_many_rows() -> None:
    rows = _full_rows(["raw", "clahe", "snr-aware", "retinex"])   # 16 rows
    with tempfile.TemporaryDirectory() as td:
        out = draw_summary_figure(rows, Path(td) / "s.png")
        assert out.is_file()


def test_summary_reads_a_csv_without_the_new_columns() -> None:
    """An older grid.csv has no latency, CI or empty-FP columns. Still plots."""
    rows = _minimal_rows(["raw", "snr-aware"])
    old_cols = ["model", "condition", "kind", "n_frames", "map50", "map50_lo",
                "map50_hi", "d_map50", "d_lo", "d_hi", "map50_95", "precision",
                "recall", "empty_fp", "fp_lo", "fp_hi"]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "grid.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=old_cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        loaded = load_rows(path)
        assert loaded[0]["enh_ms"] is None
        out = draw_summary_figure(loaded, Path(td) / "s.png")
        assert out.is_file()


def test_slide_variant_is_sixteen_by_nine() -> None:
    rows = _minimal_rows(["raw", "clahe", "snr-aware"])
    with tempfile.TemporaryDirectory() as td:
        out = draw_summary_figure(rows, Path(td) / "s.png", slide=True)
        with out.open("rb") as fh:
            fh.read(16)
            width, height = struct.unpack(">II", fh.read(8))
    ratio = width / height
    assert abs(ratio - 16 / 9) < 0.01, f"{width}x{height} = {ratio:.3f}"
    assert width >= 3800, f"expected ~300 dpi at 13.3in, got {width}px"


def test_slide_variant_of_the_heatmap_is_sixteen_by_nine() -> None:
    rows = _minimal_rows(["raw", "snr-aware"])
    with tempfile.TemporaryDirectory() as td:
        out = draw_grid_figure(rows, Path(td) / "g.png", slide=True)
        with out.open("rb") as fh:
            fh.read(16)
            width, height = struct.unpack(">II", fh.read(8))
    assert abs(width / height - 16 / 9) < 0.01, f"{width}x{height}"


def _image_size(path: Path) -> tuple[int, int]:
    with Path(path).open("rb") as fh:
        fh.read(16)
        return struct.unpack(">II", fh.read(8))


def test_slide_is_clean_at_two_methods() -> None:
    """The exact geometry of a real 2-method run: 3 cells, wide panels.

    A 5-cell slide test passed while a 3-cell one clipped, because the delta
    panel is wider with fewer rows and its title ran off the edge.
    """
    rows = _minimal_rows(["raw", "snr-aware"])
    for fn in (draw_grid_figure, draw_summary_figure):
        with tempfile.TemporaryDirectory() as td:
            _, fig = fn(rows, Path(td) / "x.png", slide=True, return_fig=True)
            try:
                bad = _out_of_canvas(fig)
            finally:
                plt.close(fig)
        assert not bad, f"{fn.__name__} clipped at 2 methods: {bad}"


def test_slide_titles_are_inside_the_canvas() -> None:
    """The enlarged slide title clipped the top edge; it must not."""
    rows = _minimal_rows(["raw", "clahe", "snr-aware"])
    for fn in (draw_grid_figure, draw_summary_figure):
        with tempfile.TemporaryDirectory() as td:
            _, fig = fn(rows, Path(td) / "x.png", slide=True, return_fig=True)
            try:
                bad = _out_of_canvas(fig)
            finally:
                plt.close(fig)
        assert not bad, f"{fn.__name__} clipped: {bad}"


def test_slide_grows_when_the_grid_needs_more_room() -> None:
    """16 cells cannot fit 16:9 legibly; the canvas grows instead of cramming."""
    rows = _full_rows(["raw", "clahe", "snr-aware", "retinex"])   # 16 cells
    with tempfile.TemporaryDirectory() as td:
        out = draw_summary_figure(rows, Path(td) / "s.png", slide=True)
        width, height = _image_size(out)
    ratio = width / height
    assert ratio < 1.70, f"expected a taller canvas for 16 rows, got {ratio:.3f}"


def test_cli_writes_the_names_the_run_uses() -> None:
    """The CLI must not invent a different suffix order than grid.py does."""
    rows = _minimal_rows(["raw", "snr-aware"])
    with tempfile.TemporaryDirectory() as td:
        csv_path = _write_csv(Path(td) / "experiments" / "r" / "grid.csv", rows)
        rc = figure_cli([str(csv_path), "--figure", "both", "--slide"])
        assert rc == 0
        d = csv_path.parent
        for name in ("grid-slide.png", "grid-summary-slide.png"):
            assert (d / name).is_file(), f"{name} missing; got {sorted(p.name for p in d.iterdir())}"


def test_slide_is_clean_with_the_real_four_condition_names() -> None:
    """Long real names rotated 30 degrees used to collide with each other.

    The synthetic tests use short names ("m1", "m2"), which never reproduced
    it: "retinexformer" at 14 pt across a 4-column heatmap is 370 px wide and
    ran into its neighbour. Display names are shortened, so no rendered text
    may overlap or leave the canvas.
    """
    rows = _minimal_rows(["raw", "clahe", "snr-aware", "retinexformer"])
    for fn in (draw_grid_figure, draw_summary_figure):
        with tempfile.TemporaryDirectory() as td:
            _, fig = fn(rows, Path(td) / "x.png", slide=True, return_fig=True)
            try:
                clipped = _out_of_canvas(fig)
                overlapping = _overlaps(fig)
            finally:
                plt.close(fig)
        assert not clipped, f"{fn.__name__} clipped at 4 conditions: {clipped}"
        assert not overlapping, (
            f"{fn.__name__} has overlapping text at 4 conditions: {overlapping}")


# ----------------------------------------------------------------------
def main() -> int:
    checks = [
        ("two methods", test_two_methods),
        ("single method", test_single_method),
        ("five methods", test_five_methods),
        ("full grid", test_full_grid_all_cells),
        ("minimal grid marks unscored cells",
         test_minimal_grid_marks_unscored_cells),
        ("no overlap, single method", test_no_overlap_single_method),
        ("no overlap, two methods", test_no_overlap_two_methods),
        ("no overlap, five methods", test_no_overlap_five_methods),
        ("no overlap, full grid", test_no_overlap_full_grid),
        ("summary figure renders", test_summary_figure_renders),
        ("summary survives many rows", test_summary_figure_survives_many_rows),
        ("summary reads an old csv",
         test_summary_reads_a_csv_without_the_new_columns),
        ("slide variant is 16:9", test_slide_variant_is_sixteen_by_nine),
        ("slide clean at 2 methods", test_slide_is_clean_at_two_methods),
        ("slide clean with real 4-condition names",
         test_slide_is_clean_with_the_real_four_condition_names),
        ("slide titles stay inside", test_slide_titles_are_inside_the_canvas),
        ("slide grows for a big grid", test_slide_grows_when_the_grid_needs_more_room),
        ("cli names match the run", test_cli_writes_the_names_the_run_uses),
        ("slide heatmap is 16:9",
         test_slide_variant_of_the_heatmap_is_sixteen_by_nine),
        ("no overflow, five methods", test_no_text_leaves_the_canvas_five_methods),
        ("no overflow, full grid", test_no_text_leaves_the_canvas_full_grid),
        ("no overflow, long names", test_long_method_names_do_not_overflow),
        ("roundtrip from csv", test_roundtrip_from_csv),
        ("caption has no local paths", test_caption_has_no_local_paths),
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

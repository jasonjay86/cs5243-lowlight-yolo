"""
test_figure_layout.py — verify the comparison figure has no overlapping text.

Not a guess. The table is scaled to 2.9x its axes height, so it grows
downward past the axes boundary. A footnote placed at a fixed axes-relative
coordinate lands inside the table's last row. This test measures the real
rendered geometry and fails if the footnote's box intersects the table's.

Also checks:
  * the title reflects the sample composition (night / day / mixed)
  * the caption shows a repo-relative weights path, not an absolute one

Run from the repo root:
    python tests/test_figure_layout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.metrics import Stats, summarize  # noqa: E402


def _fake_stats(n=40):
    out = {}
    for name in ("raw", "clahe", "snr-aware", "retinexformer"):
        rng = np.random.default_rng(abs(hash(name)) % 9999)
        nd = int(n * 0.7)
        out[name] = Stats(
            confs=rng.uniform(0.05, 0.95, nd),
            tp=(rng.random((nd, 10)) < 0.25),
            img=rng.integers(0, n, nd).astype(np.int32),
            n_gt=np.ones(n, dtype=np.int64),
            empty=np.zeros(n, dtype=bool),
            night=np.ones(n, dtype=bool),
            species=[["opossum"]] * n,
            enh_ms=100.0, yolo_ms=55.0, gflops_mp=None,
        )
    return out


def _render(tmp_path: Path):
    """Render the figure and return (fig, table_bbox, footnote_bbox).

    draw_figure() builds its own figure via plt.figure(). To measure the
    result, the Agg canvas has to be the one attached to that figure, so
    the figure is re-rendered into a live Figure whose canvas is Agg.
    """
    from evaluation.report import draw_figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    stats = _fake_stats()
    results = summarize(stats, 0.25, 50)

    out = tmp_path / "fig.png"
    draw_figure(results, stats, 0.25, "caption text", "Title text", out)

    # Re-render into a live figure so the geometry can be measured.
    import matplotlib.figure as mfig
    real_fig = mfig.Figure(figsize=(15, 10), dpi=200)
    FigureCanvasAgg(real_fig)

    orig = plt.figure
    plt.figure = lambda *a, **k: real_fig
    try:
        draw_figure(results, stats, 0.25, "caption text", "Title text", out)
    finally:
        plt.figure = orig

    real_fig.canvas.draw()
    renderer = real_fig.canvas.get_renderer()
    return real_fig, renderer


def _text_boxes(fig, renderer):
    """Collect (text, bbox) for every text artist in the figure."""
    boxes = []
    for ax in fig.axes:
        for t in ax.texts:
            boxes.append((t.get_text(), t.get_window_extent(renderer)))
        for t in fig.texts:
            boxes.append((t.get_text(), t.get_window_extent(renderer)))
    return boxes


def _table_bbox(fig, renderer):
    for ax in fig.axes:
        for t in ax.tables:
            return t.get_window_extent(renderer)
    return None


def test_table_and_footnote_do_not_overlap():
    """The class-agnostic footnote must sit BELOW the table, not inside it."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="figtest-"))
    fig, renderer = _render(tmp)

    tb = _table_bbox(fig, renderer)
    assert tb is not None, "no table found in the figure"

    foot = None
    for text, box in _text_boxes(fig, renderer):
        if "Class-agnostic" in text:
            foot = box
            break
    assert foot is not None, "the class-agnostic footnote is missing"

    # In display coordinates, higher y is higher up the figure. The table
    # occupies tb.y0 (its bottom edge) to tb.y1 (its top edge). The
    # footnote occupies foot.y0 (bottom) to foot.y1 (top). They overlap
    # when the footnote's top edge is ABOVE the table's bottom edge,
    # i.e. foot.y1 > tb.y0.
    overlap = foot.y1 - tb.y0
    if overlap > 0:
        raise AssertionError(
            f"the footnote overlaps the table by {overlap:.0f} px "
            f"(table bottom {tb.y0:.0f}, footnote top {foot.y1:.0f})"
        )
    gap = tb.y0 - foot.y1
    print(f"    no overlap: table bottom {tb.y0:.0f}, "
          f"footnote top {foot.y1:.0f}, gap {gap:.0f} px")


def test_footnote_is_inside_the_figure():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="figtest-"))
    fig, renderer = _render(tmp)

    fb = fig.get_window_extent(renderer)
    for text, box in _text_boxes(fig, renderer):
        if "Class-agnostic" in text:
            assert box.y0 >= fb.y0 - 1, (
                f"the footnote fell off the bottom of the figure "
                f"({box.y0:.0f} < {fb.y0:.0f})")
            print(f"    footnote inside the figure bounds")


def _render_measured(tmp_path: Path, title="Title text", caption="caption text"):
    """Render into a live figure so geometry can be measured."""
    from evaluation.report import draw_figure
    import matplotlib.figure as mfig
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    stats = _fake_stats()
    results = summarize(stats, 0.25, 50)

    out = tmp_path / "fig.png"
    real_fig = mfig.Figure(figsize=(15, 10), dpi=200)
    FigureCanvasAgg(real_fig)

    orig = plt.figure
    plt.figure = lambda *a, **k: real_fig
    try:
        draw_figure(results, stats, 0.25, caption, title, out)
    finally:
        plt.figure = orig

    real_fig.canvas.draw()
    return real_fig, real_fig.canvas.get_renderer()


def test_no_table_cell_text_overflows_its_cell():
    """Every table cell's text must fit inside the cell.

    The longest method name used to be "retinexformer + YOLO", which is
    wider than its column, so the text ran into the neighbouring mAP
    column and was cut off. The condition column now holds the
    enhancement name alone.

    This measures the real rendered geometry for every cell, so any
    future column change that reintroduces the overflow fails here.
    """
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="figtest-"))
    fig, renderer = _render_measured(tmp)

    worst = 0.0
    worst_at = None
    checked = 0
    for ax in fig.axes:
        for t in ax.tables:
            for (ri, ci), cell in t.get_celld().items():
                text = cell.get_text()
                if not text.get_text():
                    continue
                cell_box = cell.get_window_extent(renderer)
                text_box = text.get_window_extent(renderer)
                over_right = text_box.x1 - cell_box.x1
                over_left = cell_box.x0 - text_box.x0
                over = max(over_right, over_left)
                checked += 1
                if over > worst:
                    worst = over
                    worst_at = (ri, ci, text.get_text())
    assert checked > 0, "no table cells were checked"
    assert worst <= 1.0, (
        f"table text overflows its cell by {worst:.0f} px at {worst_at}; "
        f"the text is cut off or runs into the neighbouring column")
    print(f"    no cell overflows: worst margin {worst:.0f} px "
          f"over {checked} cells")


def test_the_longest_method_name_fits():
    """The longest registered method name must fit the condition column."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="figtest-"))
    fig, renderer = _render_measured(tmp)

    for ax in fig.axes:
        for t in ax.tables:
            cells = t.get_celld()
            for (ri, ci), cell in cells.items():
                text = cell.get_text().get_text()
                if text == "retinexformer":
                    cell_box = cell.get_window_extent(renderer)
                    text_box = cell.get_text().get_window_extent(renderer)
                    assert text_box.x1 <= cell_box.x1, (
                        "'retinexformer' is cut off in the condition column")
                    assert text_box.x0 >= cell_box.x0
                    print(f"    'retinexformer' fits: text {text_box.width:.0f}px "
                          f"in a {cell_box.width:.0f}px cell")
                    return
    raise AssertionError("the 'retinexformer' row was not found in the table")


def test_title_reflects_the_sample_composition():
    """The title must not claim 'night frames' unless that is measured.

    The CCT-20 box annotation file has no datetime field, so the
    day/night split cannot be measured from it: every frame falls back
    to "night". Asserting "night frames" on that basis claims a
    measurement the data does not support.

    So the title names the corpus only, unless the caller confirms the
    split was actually measured.
    """
    from evaluation.harness import _figure_title

    # A genuinely all-night sample, with the split measured.
    assert _figure_title(n_day=0, n_total=100, n_night=100) == \
        "Raw vs enhanced + YOLO on CCT-20 night frames"
    # A genuinely all-day sample.
    assert _figure_title(n_day=100, n_total=100, n_night=0) == \
        "Raw vs enhanced + YOLO on CCT-20 day frames"
    # A mixed sample.
    assert _figure_title(n_day=25, n_total=100, n_night=75) == \
        "Raw vs enhanced + YOLO on CCT-20"
    # The CCT-20 case: no timestamps, so the split is a default and must
    # not be claimed.
    assert _figure_title(n_day=0, n_total=100, n_night=None) == \
        "Raw vs enhanced + YOLO on CCT-20"
    # The old call signature, with no n_night at all.
    assert _figure_title(n_day=0, n_total=100) == \
        "Raw vs enhanced + YOLO on CCT-20"
    print("    title matches the sample composition")


def test_caption_uses_a_relative_weights_path():
    """The caption must not leak an absolute path with the local username."""
    import tempfile
    from evaluation.harness import _relative_weights_path

    # A stand-in repo root. The real path is not used, so this test is
    # portable and does not embed anyone's directory layout in the repo.
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        (repo / "detection").mkdir(parents=True)
        weights = repo / "detection" / "yolov8n.pt"
        weights.write_bytes(b"not a real checkpoint")

        rel = _relative_weights_path(str(weights), repo)
        # Windows uses backslashes; check the components, not the separator.
        assert Path(rel).parts == ("detection", "yolov8n.pt"), rel
        assert not Path(rel).is_absolute(), rel
        assert str(repo) not in rel, rel
        assert "Users" not in rel, rel

        # A path outside the repo falls back to the file name.
        outside = _relative_weights_path("/opt/weights/yolov8n.pt", repo)
        assert outside == "yolov8n.pt", outside
        assert "/" not in outside and "\\" not in outside, outside
    print("    caption path is relative and leaks nothing")


def test_sample_reports_whether_day_night_was_measured():
    """The sample summary must say whether the day/night split is real.

    The CCT-20 box annotations carry no datetime field, so the split
    falls back to "night" for every frame. A count that is not backed by
    a timestamp must not be presented as a measurement.
    """
    import numpy as np
    from evaluation.dataset import Frame, summarise_sample

    no_ts = [
        Frame(path="/a.jpg", image_id="1", boxes=np.zeros((0, 4)),
              species=["opossum"], is_empty=False, is_night=True,
              datetime="", ann_width=100, ann_height=80),
    ]
    assert summarise_sample(no_ts)["day_night_measured"] is False, (
        "frames with no timestamp must not count as a measured split")

    with_ts = [
        Frame(path="/b.jpg", image_id="2", boxes=np.zeros((0, 4)),
              species=["opossum"], is_empty=False, is_night=True,
              datetime="2021-03-01 23:00:00", ann_width=100, ann_height=80),
    ]
    assert summarise_sample(with_ts)["day_night_measured"] is True
    print("    the sample reports whether day/night was measured")


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

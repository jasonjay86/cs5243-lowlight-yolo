"""Grid cell selection for the fine-tuned comparison.

Pure logic: no images, no checkpoints, no GPU. The cells decide WHICH
(model x test-condition) pairs get scored, so an off-by-one here silently
drops a column from the results table and nothing downstream notices.

Owner: Jason Johnson
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.cells import (cell_kind, cells_to_run,  # noqa: E402
                              validate_cells)
from evaluation.config import RunConfig                 # noqa: E402

THREE = ["raw", "clahe", "snr-aware"]


def test_minimal_scores_baseline_row_and_diagonal() -> None:
    cells = cells_to_run(THREE, "minimal")
    assert cells == [
        ("raw", "raw"),        # baseline: Q1 and Q2 at once
        ("raw", "clahe"),      # Q1
        ("raw", "snr-aware"),  # Q1
        ("clahe", "clahe"),    # Q2
        ("snr-aware", "snr-aware"),  # Q2
    ], cells


def test_minimal_cell_count_is_2n_minus_1() -> None:
    for n in (2, 3, 4, 5):
        names = [f"c{i}" for i in range(n)]
        assert len(cells_to_run(names, "minimal")) == 2 * n - 1, n


def test_full_scores_every_pair() -> None:
    cells = cells_to_run(THREE, "full")
    assert len(cells) == 9, cells
    assert len(set(cells)) == 9, "duplicate cells in the full grid"
    assert cells[0] == ("raw", "raw"), cells[0]
    assert ("clahe", "snr-aware") in cells


def test_baseline_is_the_first_name() -> None:
    """The baseline must be the first entry, since every delta uses it."""
    cells = cells_to_run(["baseline", "other"], "minimal")
    assert cells[0] == ("baseline", "baseline")
    assert ("baseline", "other") in cells


def test_invalid_mode_is_rejected() -> None:
    for bad in ("", "all", "FULL", None):
        try:
            cells_to_run(THREE, bad)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"mode {bad!r} should have been rejected")


def test_empty_names_is_rejected() -> None:
    try:
        cells_to_run([], "minimal")
    except ValueError:
        return
    raise AssertionError("empty name list should have been rejected")


def test_cell_kind_labels() -> None:
    assert cell_kind("raw", "raw", "raw") == "Q1+Q2"
    assert cell_kind("raw", "clahe", "raw") == "Q1"
    assert cell_kind("snr-aware", "snr-aware", "raw") == "Q2"
    assert cell_kind("clahe", "snr-aware", "raw") == "cross"


def test_validate_cells_accepts_a_complete_grid() -> None:
    models = {"raw": "a.pt", "clahe": "b.pt", "snr-aware": "c.pt"}
    conditions = {"raw": "a.csv", "clahe": "b.csv", "snr-aware": "c.csv"}
    validate_cells(cells_to_run(THREE, "minimal"), models, conditions)


def test_validate_cells_rejects_a_missing_weight() -> None:
    models = {"raw": "a.pt", "clahe": "b.pt"}          # snr-aware missing
    conditions = {"raw": "a.csv", "clahe": "b.csv", "snr-aware": "c.csv"}
    try:
        validate_cells(cells_to_run(THREE, "minimal"), models, conditions)
    except ValueError as exc:
        assert "snr-aware" in str(exc), exc
        return
    raise AssertionError("a missing checkpoint should have been rejected")


def test_validate_cells_rejects_a_missing_condition() -> None:
    models = {"raw": "a.pt", "clahe": "b.pt", "snr-aware": "c.pt"}
    conditions = {"raw": "a.csv", "clahe": "b.csv"}    # snr-aware manifest missing
    try:
        validate_cells(cells_to_run(THREE, "minimal"), models, conditions)
    except ValueError as exc:
        assert "snr-aware" in str(exc), exc
        return
    raise AssertionError("a missing manifest should have been rejected")


# ----------------------------------------------------------------------
# Config integration
# ----------------------------------------------------------------------
def test_empty_grid_maps_do_not_add_fingerprint_keys() -> None:
    """Zero-shot runs must keep their existing cache keys."""
    inputs = RunConfig().fingerprint_inputs(["a", "b"], "")
    assert "weights" not in inputs, "empty weights map changed the fingerprint"
    assert "conditions" not in inputs, "empty conditions map changed the fingerprint"


def test_weights_and_conditions_enter_the_fingerprint() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        w = root / "best.pt"
        w.write_bytes(b"checkpoint-bytes")
        m = root / "split_manifest.csv"
        m.write_text("image_id,split\na,test\n", encoding="utf-8")

        base = RunConfig()
        with_w = RunConfig(weights={"raw": str(w)})
        with_c = RunConfig(conditions={"raw": str(m)})

        base_fp = base.analysis_fingerprint(["a"], "")
        assert with_w.analysis_fingerprint(["a"], "") != base_fp
        assert with_c.analysis_fingerprint(["a"], "") != base_fp
        assert with_w.analysis_fingerprint(["a"], "") != \
            with_c.analysis_fingerprint(["a"], "")


def test_changing_a_checkpoint_changes_the_fingerprint() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a, b = root / "a.pt", root / "b.pt"
        a.write_bytes(b"model-a")
        b.write_bytes(b"model-b")
        fa = RunConfig(weights={"raw": str(a)}).analysis_fingerprint(["x"], "")
        fb = RunConfig(weights={"raw": str(b)}).analysis_fingerprint(["x"], "")
    assert fa != fb, "two different checkpoints produced the same cache key"


def test_resolved_makes_grid_paths_absolute() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = RunConfig(weights={"raw": "models/raw.pt"},
                        conditions={"raw": "data/raw.csv"})
        out = cfg.resolved(root)
    assert Path(out.weights["raw"]) == root / "models/raw.pt", out.weights
    assert Path(out.conditions["raw"]) == root / "data/raw.csv", out.conditions


def test_yaml_round_trip_preserves_the_grid() -> None:
    with tempfile.TemporaryDirectory() as td:
        cfg = RunConfig(
            weights={"raw": "a.pt", "snr-aware": "b.pt"},
            conditions={"raw": "a.csv", "snr-aware": "b.csv"},
            cell_mode="full",
        )
        p = Path(td) / "config.yaml"
        cfg.to_yaml(p)
        back = RunConfig.from_yaml(p)
    assert back.weights == cfg.weights, back.weights
    assert back.conditions == cfg.conditions, back.conditions
    assert back.cell_mode == "full", back.cell_mode


# ----------------------------------------------------------------------
def main() -> int:
    checks = [
        ("minimal = baseline row + diagonal",
         test_minimal_scores_baseline_row_and_diagonal),
        ("minimal has 2N-1 cells", test_minimal_cell_count_is_2n_minus_1),
        ("full scores every pair", test_full_scores_every_pair),
        ("baseline is the first name", test_baseline_is_the_first_name),
        ("invalid cell mode rejected", test_invalid_mode_is_rejected),
        ("empty name list rejected", test_empty_names_is_rejected),
        ("cell_kind labels Q1/Q2/both/cross", test_cell_kind_labels),
        ("validate_cells accepts a complete grid",
         test_validate_cells_accepts_a_complete_grid),
        ("validate_cells rejects a missing checkpoint",
         test_validate_cells_rejects_a_missing_weight),
        ("validate_cells rejects a missing manifest",
         test_validate_cells_rejects_a_missing_condition),
        ("empty grid maps add no fingerprint keys",
         test_empty_grid_maps_do_not_add_fingerprint_keys),
        ("weights and conditions enter the fingerprint",
         test_weights_and_conditions_enter_the_fingerprint),
        ("different checkpoints get different keys",
         test_changing_a_checkpoint_changes_the_fingerprint),
        ("resolved() makes grid paths absolute",
         test_resolved_makes_grid_paths_absolute),
        ("yaml round trip preserves the grid",
         test_yaml_round_trip_preserves_the_grid),
    ]
    passed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  ok   {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{passed}/{len(checks)} passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

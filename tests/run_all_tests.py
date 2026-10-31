"""
run_all_tests.py — run every evaluation test file and report a summary.

Owner: Jason Johnson

Run from the repo root:

    python tests/run_all_tests.py

Each test file also runs standalone:

    python tests/test_evaluation.py
    python tests/test_pipeline_smoke.py
    python tests/test_figure_layout.py

They are plain scripts with their own runners, so pytest is not required
(it is not in environment.yml). This script just runs all of them in one
go and gives you a single pass/fail.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# (filename, expected pass marker in the output)
TEST_FILES = [
    ("test_evaluation.py", "passed"),
    ("test_pipeline_smoke.py", "PASSED"),
    ("test_figure_layout.py", "passed"),
    ("test_rescore.py", "passed"),
    ("test_rescore_e2e.py", "PASSED"),
]

OPTIONAL_TESTS = [
    ("test_diagnose.py", "passed"),
]


def run_one(path: Path) -> tuple[bool, str, float]:
    """Run a test file. Returns (ok, output, seconds)."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True, text=True, cwd=str(path.parent.parent),
    )
    elapsed = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out, elapsed


def main() -> int:
    print("=" * 62)
    print("Evaluation harness test suite")
    print("=" * 62)

    failures = []
    total = 0.0

    for filename, marker in TEST_FILES + OPTIONAL_TESTS:
        path = TESTS_DIR / filename
        if not path.is_file():
            print(f"  SKIP  {filename} (not present)")
            continue

        ok, out, elapsed = run_one(path)
        total += elapsed
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {filename}  ({elapsed:.1f}s)")

        if not ok:
            failures.append((filename, out))
            # Show the tail so the failure is visible without scrolling.
            tail = out.strip().splitlines()[-15:]
            for line in tail:
                print(f"        {line}")
        else:
            # Echo the summary line so the counts are visible.
            for line in out.strip().splitlines():
                if marker in line and ("passed" in line or "PASSED" in line):
                    print(f"        {line.strip()}")
                    break

    print("-" * 62)
    print(f"total: {total:.1f}s")
    if failures:
        print(f"FAILED: {len(failures)} file(s)")
        return 1
    print("all test files passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

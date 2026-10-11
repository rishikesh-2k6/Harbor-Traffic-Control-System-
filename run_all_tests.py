"""
Run every test suite and summarise the results.

    python run_all_tests.py          # all suites
    python run_all_tests.py -v       # verbose per-test output

Exit code is 0 only when every suite passes.
"""

from __future__ import annotations

import io
import os
import sys
import time
import unittest

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SUITES = [
    ("Prediction validation", "test_prediction_validation.py"),
    ("Integration & regression", "test_integration.py"),
]


def run_suite(label: str, filename: str, verbose: bool) -> tuple:
    """Run one unittest module in an isolated namespace."""
    loader = unittest.TestLoader()
    suite = loader.discover(HERE, pattern=filename)

    # Silence the simulation's chatty print() output during tests
    devnull = open(os.devnull, "w")
    real_stdout = sys.stdout
    stream = sys.stderr if verbose else devnull
    sys.stdout = devnull
    start = time.time()
    try:
        result = unittest.TextTestRunner(stream=stream, verbosity=2 if verbose else 0).run(suite)
    finally:
        sys.stdout = real_stdout
        devnull.close()
    elapsed = time.time() - start

    status = "PASS" if result.wasSuccessful() else "FAIL"
    print(f"  {status}  {label:32s} "
          f"{result.testsRun:3d} tests, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors  ({elapsed:.1f}s)")
    return result.wasSuccessful(), result.testsRun, result


def main() -> int:
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    print("=" * 78)
    print("HARBOR TRAFFIC CONTROL SYSTEM - FULL TEST RUN")
    print("=" * 78)

    total = 0
    all_ok = True
    failures = []

    for label, filename in SUITES:
        ok, count, result = run_suite(label, filename, verbose)
        total += count
        if not ok:
            all_ok = False
            for case, tb in result.failures + result.errors:
                failures.append((label, str(case), tb))

    print("-" * 78)
    if failures:
        print("\nFailure detail:\n")
        for label, case, tb in failures:
            print(f"[{label}] {case}")
            print("    " + tb.strip().replace("\n", "\n    ")[:1200])
            print()
    print("=" * 78)
    print(f"TOTAL: {total} tests | RESULT: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
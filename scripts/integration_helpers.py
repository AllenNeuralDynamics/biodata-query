#!/usr/bin/env python
"""Integration test for biodata_query.helpers against real backends.

Makes real network calls to zombie-squirrel and AIND DocumentDB.

Usage:
    python scripts/integration_helpers.py

Exit code 0 means all tests passed; non-zero means at least one failed.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable

from biodata_query.helpers import find_assets

# ── helpers ────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def _run(label: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        _results.append((label, True, ""))
        print(f"  PASS  {label}")
    except Exception:
        msg = traceback.format_exc()
        _results.append((label, False, msg))
        print(f"  FAIL  {label}\n{msg}")


# ── test functions ─────────────────────────────────────────────────────────────

SUBJECT_ID = "841303"
MODALITIES = ["ecephys"]


def _test_find_ecephys_derived_assets() -> None:
    """Query raw assets for subject 841303, resolve ecephys derived assets."""
    df = find_assets(
        modalities=MODALITIES,
        query={"subject.subject_id": SUBJECT_ID},
    )
    print(f"    rows: {len(df)}")
    print(f"    columns: {df.columns.tolist()}")
    assert set(df.columns) >= {"raw_asset_name", "derived_asset_name", "modality"}, (
        f"missing expected columns, got {df.columns.tolist()}"
    )
    assert (df["modality"] == "ecephys").all(), "unexpected modality values"
    print(f"    raw assets: {df['raw_asset_name'].nunique()}")
    print(f"    derived assets: {df['derived_asset_name'].nunique()}")
    import pandas as pd
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(df.to_string(index=False))


def _test_find_ecephys_derived_assets_with_qc() -> None:
    """Same query but also fetch QC status for two metrics."""
    metric_names = ["Drift Map", "Unit Yield"]
    df = find_assets(
        modalities=MODALITIES,
        query={"subject.subject_id": SUBJECT_ID},
        metric_names=metric_names,
    )
    print(f"    rows: {len(df)}")
    expected_cols = {
        "raw_asset_name",
        "derived_asset_name",
        "modality",
        "ecephys",
        "Drift Map",
        "Unit Yield",
        "all_metrics_pass",
        "qc_pass",
    }
    assert expected_cols <= set(df.columns), (
        f"missing columns: {expected_cols - set(df.columns)}"
    )
    import pandas as pd
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(df.to_string(index=False))


# ── main ───────────────────────────────────────────────────────────────────────

TESTS = [
    ("find ecephys derived assets via query", _test_find_ecephys_derived_assets),
    ("find ecephys derived assets with QC",   _test_find_ecephys_derived_assets_with_qc),
]

if __name__ == "__main__":
    print(f"Running integration tests for biodata_query.helpers (subject {SUBJECT_ID}) ...\n")
    for label, fn in TESTS:
        _run(label, fn)

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print(f"\n{passed}/{len(_results)} passed", end="")
    if failed:
        print(f", {failed} FAILED")
        sys.exit(1)
    else:
        print()

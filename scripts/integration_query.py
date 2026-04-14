#!/usr/bin/env python
"""Integration tests for biodata_query.query against real backends.

These tests make real network calls to zombie-squirrel and the AIND DocumentDB
API. They are intentionally kept out of the pytest suite.

Usage:
    python scripts/integration_query.py

Exit code 0 means all tests passed; non-zero means at least one failed.
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable

from biodata_query.query import API_GATEWAY_HOST, QueryResult, run_query
from aind_data_access_api.document_db import MetadataDbClient

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


def _test_empty_query_cache() -> None:
    """Empty query → cache path, non-empty result."""
    result = run_query({}, names_only=True)
    assert result.backend == "cache", f"expected 'cache', got '{result.backend}'"
    assert len(result.asset_names) > 0, "expected at least one asset"
    assert result.records is None
    print(f"    total assets in cache: {len(result.asset_names)}, elapsed: {result.elapsed_seconds:.2f}s")


def _test_project_name_filter_cache() -> None:
    """Filter by project_name → should hit the cache."""
    result = run_query(
        {"data_description.project_name": {"$regex": "Brain", "$options": "i"}},
        names_only=True,
    )
    assert result.backend == "cache", f"expected 'cache', got '{result.backend}'"
    print(f"    matching assets: {len(result.asset_names)}, elapsed: {result.elapsed_seconds:.2f}s")


def _test_data_level_filter_cache() -> None:
    """Filter by data_level (exact equality) → cache path."""
    result = run_query({"data_description.data_level": "raw"}, names_only=True)
    assert result.backend == "cache"
    assert len(result.asset_names) > 0
    print(f"    raw assets: {len(result.asset_names)}, elapsed: {result.elapsed_seconds:.2f}s")


def _test_date_range_filter_cache() -> None:
    """Filter by acquisition_start_time range → cache path."""
    import pandas as pd

    result = run_query(
        {
            "acquisition.acquisition_start_time": {
                "$gte": pd.Timestamp("2023-01-01"),
                "$lte": pd.Timestamp("2024-12-31"),
            }
        },
        names_only=True,
    )
    assert result.backend == "cache"
    print(f"    assets in 2023–2024: {len(result.asset_names)}, elapsed: {result.elapsed_seconds:.2f}s")


def _test_non_cache_field_routes_to_docdb() -> None:
    """A field not in FIELD_TO_COLUMN → docdb path."""
    result = run_query(
        {"data_description.institution.abbreviation": "AIND"},
        names_only=True,
        limit=10,
    )
    assert result.backend == "docdb", f"expected 'docdb', got '{result.backend}'"
    assert len(result.asset_names) <= 10
    print(f"    docdb results: {len(result.asset_names)}, elapsed: {result.elapsed_seconds:.2f}s")


def _test_unsupported_operator_routes_to_docdb() -> None:
    """$elemMatch is unsupported for cache → routes to docdb."""
    result = run_query(
        {
            "data_description.modalities": {
                "$elemMatch": {"abbreviation": "ecephys"}
            }
        },
        names_only=True,
        limit=10,
    )
    assert result.backend == "docdb", f"expected 'docdb', got '{result.backend}'"
    assert len(result.asset_names) <= 10
    print(f"    docdb results: {len(result.asset_names)}, elapsed: {result.elapsed_seconds:.2f}s")


def _test_cache_path_filtered_names() -> None:
    """Cache path with names_only=True returns a non-empty list."""
    result = run_query({"name": {"$regex": "^ecephys_", "$options": "i"}}, names_only=True)
    assert result.backend == "cache"
    assert result.records is None
    print(f"    cache-filtered assets: {len(result.asset_names)}, elapsed: {result.elapsed_seconds:.2f}s")


def _test_docdb_fetch_single_record() -> None:
    """Directly verify a DocDB fetch using projection={\"_id\": 1} to stay lightweight."""
    # Get one name from the cache so we know it's valid
    cache_result = run_query({}, names_only=True)
    assert len(cache_result.asset_names) > 0, "cache returned no assets"
    name = cache_result.asset_names[0]

    client = MetadataDbClient(host=API_GATEWAY_HOST)
    records = client.retrieve_docdb_records(
        filter_query={"name": name},
        projection={"_id": 1},
    )
    assert len(records) == 1, f"expected 1 record for {name!r}, got {len(records)}"
    assert "_id" in records[0], "expected '_id' key in projected record"
    print(f"    fetched _id for {name!r}")


def _test_in_operator_cache() -> None:
    """$in on project_name → cache path."""
    result = run_query(
        {"data_description.project_name": {"$in": ["Brain Computer Interface", "Omfish"]}},
        names_only=True,
    )
    assert result.backend == "cache"
    print(f"    $in results: {len(result.asset_names)}, elapsed: {result.elapsed_seconds:.2f}s")


# ── main ───────────────────────────────────────────────────────────────────────

TESTS = [
    ("empty query → cache",                   _test_empty_query_cache),
    ("project_name regex → cache",            _test_project_name_filter_cache),
    ("data_level equality → cache",           _test_data_level_filter_cache),
    ("acquisition_start_time range → cache",  _test_date_range_filter_cache),
    ("$in on project_name → cache",           _test_in_operator_cache),
    ("unknown field → docdb",                 _test_non_cache_field_routes_to_docdb),
    ("$elemMatch → docdb",                    _test_unsupported_operator_routes_to_docdb),
    ("cache filtered names_only",             _test_cache_path_filtered_names),
    ("docdb fetch with projection",           _test_docdb_fetch_single_record),
]


if __name__ == "__main__":
    print("Running integration tests for biodata_query.query ...\n")
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

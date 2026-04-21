#!/usr/bin/env python
"""
Compare record-retrieval strategies for a cache-eligible query.

Approaches timed:
  1. Cache filter only — no DocDB at all
  2. Batched $in queries — current default, batch_size=50
  3. Single $in query  — one request with all matched names
  4. Direct DocDB query — no cache, one request using the original filter

Usage:
    .venv/bin/python scripts/integration_batching.py
"""
from __future__ import annotations

import time

from aind_data_access_api.document_db import MetadataDbClient
from zombie_squirrel import asset_basics

from biodata_query.query import _apply_filter_to_dataframe, _fetch_full_records_batched

QUERY = {"data_description.project_name": "Behavior Platform"}
API_GATEWAY_HOST = "api.allenneuraldynamics.org"


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main() -> None:
    # ── 1. Cache filter ────────────────────────────────────────────
    section("1  Cache filter (asset_basics, no DocDB)")
    t0 = time.perf_counter()
    df = asset_basics()
    filtered = _apply_filter_to_dataframe(df, QUERY)
    names = filtered["name"].tolist()
    cache_time = time.perf_counter() - t0
    print(f"  Time : {cache_time:.3f}s")
    print(f"  Names: {len(names)}")

    client = MetadataDbClient(host=API_GATEWAY_HOST)
    n_batches = (len(names) + 49) // 50

    # ── 2. Batched $in (batch_size=50) ─────────────────────────────
    section(f"2  Batched $in queries  (batch_size=50, {n_batches} requests)")
    t0 = time.perf_counter()
    records_batched = _fetch_full_records_batched(names, batch_size=50)
    batched_time = time.perf_counter() - t0
    print(f"  Time    : {batched_time:.3f}s")
    print(f"  Requests: {n_batches}")
    print(f"  Records : {len(records_batched)}")

    # ── 4. Direct DocDB query ──────────────────────────────────────
    section("4  Direct DocDB query  (1 request, no cache)")
    t0 = time.perf_counter()
    records_direct = client.retrieve_docdb_records(filter_query=QUERY)
    direct_time = time.perf_counter() - t0
    print(f"  Time    : {direct_time:.3f}s")
    print(f"  Requests: 1")
    print(f"  Records : {len(records_direct)}")

    # ── Summary ────────────────────────────────────────────────────
    section("Summary")
    rows = [
        ("Cache filter only",              cache_time,   "—"),
        (f"Batched $in (×{n_batches})",    batched_time, f"×{n_batches}"),
        ("Single $in",                     single_time,  "×1"),
        ("Direct DocDB query",             direct_time,  "×1"),
    ]
    print(f"  {'Approach':<32} {'Time':>8}  {'Requests':>10}")
    print(f"  {'─' * 32} {'─' * 8}  {'─' * 10}")
    for label, t, reqs in rows:
        print(f"  {label:<32} {t:>7.3f}s  {reqs:>10}")


if __name__ == "__main__":
    main()

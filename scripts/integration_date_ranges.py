#!/usr/bin/env python
"""Integration tests comparing cache vs DocDB results for date-range queries.

Tests both $gte/$lte (ISO-8601 strings) and $regex approaches across a variety
of short date windows and modalities, to measure how much the two backends
diverge and understand when range operators are safe to use.

Usage:
    python scripts/integration_date_ranges.py

Results are printed as a table. Discrepancies are flagged but do not cause an
error exit by default — the goal is measurement, not a pass/fail gate.
Pass --strict to exit 1 on any discrepancy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from biodata_query.query import retrieve_records


# ── comparison helper ──────────────────────────────────────────────────────────

@dataclass
class Comparison:
    label: str
    cache_count: int
    docdb_count: int
    only_in_cache: list[str]
    only_in_docdb: list[str]

    @property
    def match(self) -> bool:
        return not self.only_in_cache and not self.only_in_docdb

    def print(self) -> None:
        status = "OK  " if self.match else "DIFF"
        print(f"  [{status}] {self.label}")
        print(f"         cache={self.cache_count}  docdb={self.docdb_count}")
        if self.only_in_cache:
            print(f"         only in cache ({len(self.only_in_cache)}): {self.only_in_cache[:5]}")
        if self.only_in_docdb:
            print(f"         only in docdb ({len(self.only_in_docdb)}): {self.only_in_docdb[:5]}")


def _compare(label: str, query: dict) -> Comparison:
    cache = retrieve_records(query, names_only=True, force_backend="cache")
    docdb = retrieve_records(query, names_only=True, force_backend="docdb")
    cs, ds = set(cache.asset_names), set(docdb.asset_names)
    return Comparison(
        label=label,
        cache_count=len(cs),
        docdb_count=len(ds),
        only_in_cache=sorted(cs - ds),
        only_in_docdb=sorted(ds - cs),
    )


# ── test cases ─────────────────────────────────────────────────────────────────

def _gte_lte(start: str, end: str) -> dict:
    """Range filter using $gte/$lte ISO-8601 strings."""
    return {"acquisition.acquisition_start_time": {"$gte": start, "$lte": end}}


def _regex(pattern: str) -> dict:
    """Range filter using $regex prefix match."""
    return {"acquisition.acquisition_start_time": {"$regex": pattern}}


def _with_modality(base: dict, abbrev: str) -> dict:
    return {**base, "data_description.modalities": {"$elemMatch": {"abbreviation": abbrev}}}


def _with_data_level(base: dict, level: str) -> dict:
    return {**base, "data_description.data_level": level}


# ── scenarios ──────────────────────────────────────────────────────────────────

SCENARIOS: list[tuple[str, dict]] = [
    # --- regex year-level (should always match perfectly) ---
    ("regex ^2024",
     _regex("^2024")),
    ("regex ^2025",
     _regex("^2025")),
    ("regex ^2025-01",
     _regex("^2025-01")),
    ("regex ^2025-01 + ecephys raw",
     _with_data_level(_with_modality(_regex("^2025-01"), "ecephys"), "raw")),

    # --- $gte/$lte full-year ranges ---
    ("gte/lte 2024-01-01 to 2024-12-31",
     _gte_lte("2024-01-01T00:00:00", "2024-12-31T23:59:59")),
    ("gte/lte 2025-01-01 to 2025-12-31",
     _gte_lte("2025-01-01T00:00:00", "2025-12-31T23:59:59")),

    # --- $gte/$lte narrow windows ---
    ("gte/lte 2024-01 (Jan 2024)",
     _gte_lte("2024-01-01T00:00:00", "2024-01-31T23:59:59")),
    ("gte/lte 2024-07 (Jul 2024)",
     _gte_lte("2024-07-01T00:00:00", "2024-07-31T23:59:59")),
    ("gte/lte 2025-01 (Jan 2025)",
     _gte_lte("2025-01-01T00:00:00", "2025-01-31T23:59:59")),
    ("gte/lte 2025-03 (Mar 2025)",
     _gte_lte("2025-03-01T00:00:00", "2025-03-31T23:59:59")),
    ("gte/lte 2026-01 (Jan 2026)",
     _gte_lte("2026-01-01T00:00:00", "2026-01-31T23:59:59")),
    ("gte/lte 2026-04 to 2026-05 (Apr–May 2026)",
     _gte_lte("2026-04-01T00:00:00", "2026-05-31T23:59:59")),

    # --- $gte/$lte narrow + modality ---
    ("gte/lte 2024-01 + ecephys",
     _with_modality(_gte_lte("2024-01-01T00:00:00", "2024-01-31T23:59:59"), "ecephys")),
    ("gte/lte 2024-06 + behavior raw",
     _with_data_level(_with_modality(
         _gte_lte("2024-06-01T00:00:00", "2024-06-30T23:59:59"), "behavior"), "raw")),
    ("gte/lte 2025-01 + SPIM raw",
     _with_data_level(_with_modality(
         _gte_lte("2025-01-01T00:00:00", "2025-01-31T23:59:59"), "SPIM"), "raw")),
    ("gte/lte 2025-03 + ecephys raw",
     _with_data_level(_with_modality(
         _gte_lte("2025-03-01T00:00:00", "2025-03-31T23:59:59"), "ecephys"), "raw")),
    ("gte/lte 2026-01 + ecephys raw",
     _with_data_level(_with_modality(
         _gte_lte("2026-01-01T00:00:00", "2026-01-31T23:59:59"), "ecephys"), "raw")),

    # --- cross-check: regex vs gte/lte for same month ---
    ("regex ^2024-01 + ecephys raw",
     _with_data_level(_with_modality(_regex("^2024-01"), "ecephys"), "raw")),
    ("regex ^2024-06 + behavior raw",
     _with_data_level(_with_modality(_regex("^2024-06"), "behavior"), "raw")),
    ("regex ^2025-01 + SPIM raw",
     _with_data_level(_with_modality(_regex("^2025-01"), "SPIM"), "raw")),
    ("regex ^2025-03 + ecephys raw",
     _with_data_level(_with_modality(_regex("^2025-03"), "ecephys"), "raw")),
    ("regex ^2026-01 + ecephys raw",
     _with_data_level(_with_modality(_regex("^2026-01"), "ecephys"), "raw")),
]


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    strict = "--strict" in sys.argv
    results: list[Comparison] = []

    print("Date-range cache vs DocDB comparison\n")

    for label, query in SCENARIOS:
        try:
            c = _compare(label, query)
            c.print()
            results.append(c)
        except Exception as exc:
            print(f"  [ERR ] {label}: {exc}")
            results.append(Comparison(label, 0, 0, [], [f"ERROR: {exc}"]))
        print()

    matched = sum(1 for r in results if r.match)
    diffed = len(results) - matched
    print(f"Summary: {matched}/{len(results)} scenarios matched between cache and docdb")
    if diffed:
        print(f"         {diffed} scenario(s) had discrepancies")
        if strict:
            sys.exit(1)

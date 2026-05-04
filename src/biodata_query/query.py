"""Query execution engine with cache-aware routing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger(__name__)

import pandas as pd
from aind_data_access_api.document_db import MetadataDbClient
from zombie_squirrel import asset_basics

API_GATEWAY_HOST = "api.allenneuraldynamics.org"

# Mapping from MongoDB document field paths to asset_basics column names
FIELD_TO_COLUMN: dict[str, str] = {
    "name": "name",
    "data_description.project_name": "project_name",
    "data_description.modality": "modalities",
    "data_description.modalities": "modalities",
    "data_description.modalities.abbreviation": "modalities",
    "data_description.data_level": "data_level",
    "subject.subject_id": "subject_id",
    "subject.subject_details.genotype": "genotype",
    "acquisition.acquisition_start_time": "acquisition_start_time",
    "acquisition.acquisition_end_time": "acquisition_end_time",
    "process_date": "process_date",
}

# MongoDB operators that cannot be handled by the pandas cache path
_UNSUPPORTED_OPS: frozenset[str] = frozenset(
    {"$or", "$not", "$elemMatch", "$exists", "$nor", "$expr", "$where", "$text"}
)

# MongoDB operators that ARE supported in the pandas cache path
_SUPPORTED_OPS: frozenset[str] = frozenset(
    {"$in", "$all", "$regex", "$options", "$gte", "$lte", "$gt", "$lt"}
)

# Columns whose values are stored as timezone-aware ISO-8601 strings
_DATETIME_COLUMNS: frozenset[str] = frozenset(
    {"acquisition_start_time", "acquisition_end_time", "process_date"}
)

# Column that stores modalities as a comma-separated string of abbreviations
_MODALITIES_COLUMN = "modalities"


@dataclass
class QueryResult:
    """Result of a query execution."""

    backend: Literal["cache", "docdb"]
    elapsed_seconds: float
    asset_names: list[str]
    records: list[dict] | None  # None if names_only=True or projection is cache-servable
    dataframe: pd.DataFrame | None = None  # set on cache path when projection is cache-servable


def _has_unsupported_operators(value: object) -> bool:
    """Return True if the value dict uses any unsupported MongoDB operators."""
    if not isinstance(value, dict):
        return False
    for key in value:
        if key in _UNSUPPORTED_OPS:
            return True
        if key.startswith("$") and key not in _SUPPORTED_OPS:
            return True
    return False


def is_cache_eligible(query: dict) -> bool:
    """Check if all top-level keys in the query map to asset_basics columns."""
    for field, value in query.items():
        if field not in FIELD_TO_COLUMN:
            logger.debug("Cache ineligible: field %r not in FIELD_TO_COLUMN", field)
            return False
        if _has_unsupported_operators(value):
            logger.debug("Cache ineligible: field %r uses unsupported operators", field)
            return False
    return True


def _to_utc_series(series: pd.Series) -> pd.Series:
    """Parse a string or Timestamp series into UTC-aware datetimes."""
    return pd.to_datetime(series, utc=True, errors="coerce")


def _to_utc_timestamp(operand: object) -> pd.Timestamp:
    """Coerce a string, date, or Timestamp operand to a UTC-aware Timestamp."""
    ts = pd.Timestamp(operand)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _projection_is_cache_servable(projection: dict | None) -> bool:
    """Return True if every requested field is available in the local cache.

    ``None`` means the caller did not specify a projection, so we conservatively
    assume full DocDB records may be needed.
    """
    if projection is None:
        return False
    return all(field in FIELD_TO_COLUMN for field in projection)


def _modality_series_contains(series: pd.Series, value: str) -> pd.Series:
    """Boolean mask: rows where *value* is an exact modality term."""

    def _check(cell: object) -> bool:
        if pd.isna(cell):
            return False
        return value in [m.strip() for m in str(cell).split(",")]

    return series.apply(_check)


def _modality_series_contains_any(series: pd.Series, values: list) -> pd.Series:
    """Boolean mask: rows where any element of *values* is an exact modality term."""
    value_set = set(values)

    def _check(cell: object) -> bool:
        if pd.isna(cell):
            return False
        return bool({m.strip() for m in str(cell).split(",")} & value_set)

    return series.apply(_check)


def _modality_series_contains_all(series: pd.Series, values: list) -> pd.Series:
    """Boolean mask: rows where all elements of *values* are exact modality terms."""
    value_set = set(values)

    def _check(cell: object) -> bool:
        if pd.isna(cell):
            return False
        return value_set <= {m.strip() for m in str(cell).split(",")}

    return series.apply(_check)


def _apply_filter_to_dataframe(df: pd.DataFrame, query: dict) -> pd.DataFrame:
    """Translate a MongoDB-style filter dict into pandas DataFrame operations.

    Supported operators: simple equality, $in, $all, $regex (with $options: "i"),
    $gte, $lte, $gt, $lt. Multiple top-level keys are ANDed together.

    Notes
    -----
    * Datetime columns (``acquisition_start_time``, ``acquisition_end_time``,
      ``process_date``) are stored as ISO-8601 strings with timezone offsets.
      They are parsed to UTC before any comparison.
    * The ``modalities`` column is a comma-separated string of abbreviations
      (e.g. ``"ecephys, behavior-videos"``).  Equality and ``$in`` checks
      split the string and test exact term membership; ``$regex`` operates
      on the raw concatenated string.
    """
    mask = pd.Series(True, index=df.index)

    for field, value in query.items():
        col = FIELD_TO_COLUMN[field]
        series = df[col]

        if col == _MODALITIES_COLUMN:
            if isinstance(value, dict):
                if "$all" in value:
                    mask &= _modality_series_contains_all(series, value["$all"])
                elif "$in" in value:
                    mask &= _modality_series_contains_any(series, value["$in"])
                elif "$regex" in value:
                    case_insensitive = "i" in value.get("$options", "")
                    mask &= series.str.contains(
                        value["$regex"],
                        case=not case_insensitive,
                        na=False,
                        regex=True,
                    )
            else:
                mask &= _modality_series_contains(series, value)

        elif col in _DATETIME_COLUMNS:
            series_dt = _to_utc_series(series)
            if isinstance(value, dict):
                for op, operand in value.items():
                    operand_ts = _to_utc_timestamp(operand)
                    if op == "$gte":
                        mask &= series_dt >= operand_ts
                    elif op == "$lte":
                        mask &= series_dt <= operand_ts
                    elif op == "$gt":
                        mask &= series_dt > operand_ts
                    elif op == "$lt":
                        mask &= series_dt < operand_ts
            else:
                mask &= series_dt == _to_utc_timestamp(value)

        else:
            if isinstance(value, dict):
                if "$in" in value:
                    mask &= series.isin(value["$in"])
                elif "$regex" in value:
                    case_insensitive = "i" in value.get("$options", "")
                    mask &= series.str.contains(
                        value["$regex"],
                        case=not case_insensitive,
                        na=False,
                        regex=True,
                    )
                else:
                    for op, operand in value.items():
                        if op == "$gte":
                            mask &= series >= operand
                        elif op == "$lte":
                            mask &= series <= operand
                        elif op == "$gt":
                            mask &= series > operand
                        elif op == "$lt":
                            mask &= series < operand
            else:
                mask &= series == value

    return df[mask]


def _fetch_full_records_batched(names: list[str], batch_size: int = 50) -> list[dict]:
    """Fetch full records from DocDB by batching $in queries on the name field."""
    if not names:
        return []
    client = MetadataDbClient(host=API_GATEWAY_HOST)
    records = []
    for i in range(0, len(names), batch_size):
        batch = names[i : i + batch_size]
        batch_records = client.retrieve_docdb_records(filter_query={"name": {"$in": batch}})
        records.extend(batch_records)
    return records


def retrieve_records(
    filter_query: dict,
    projection: dict | None = None,
    limit: int = 0,
    names_only: bool = False,
    force_backend: Optional[Literal["cache", "docdb"]] = None,
) -> QueryResult:
    """Execute a query, routing through the local cache or DocDB as appropriate.

    A query is routed to the cache when every top-level filter key maps to a
    column in the ``asset_basics`` table and no unsupported operators are used.
    Otherwise the query is forwarded directly to DocumentDB.

    Parameters
    ----------
    filter_query:
        MongoDB-style filter dictionary.
    projection:
        Optional MongoDB projection dict to limit returned fields.
    limit:
        Maximum number of results to return. 0 means no limit. Only applied
        on the DocDB path; the cache path applies it as a post-filter slice.
    names_only:
        When True, skip fetching full records and return only asset names.
    force_backend:
        ``"cache"`` to force the local-cache path (raises ``ValueError`` if
        the query is not cache-eligible), ``"docdb"`` to skip the cache and
        always hit DocumentDB, or ``None`` (default) to auto-route.
    """
    logger.debug(
        "retrieve_records called: filter_query=%r names_only=%s limit=%s force_backend=%s",
        filter_query, names_only, limit, force_backend,
    )
    start = time.time()

    if force_backend == "cache" and not is_cache_eligible(filter_query):
        raise ValueError(
            "force_backend='cache' requested but the query is not cache-eligible. "
            "Use force_backend=None or force_backend='docdb'."
        )

    use_cache = (
        force_backend == "cache"
        or (force_backend is None and is_cache_eligible(filter_query))
    )

    if use_cache:
        logger.debug("Routing to cache backend")
        df = asset_basics()
        filtered = _apply_filter_to_dataframe(df, filter_query)
        if limit:
            filtered = filtered.iloc[:limit]
        names = filtered["name"].tolist()
        cache_elapsed = time.time() - start
        logger.debug("Cache filter complete: %.3fs → %d names", cache_elapsed, len(names))
        records = None
        result_df = None
        if not names_only:
            if _projection_is_cache_servable(projection):
                logger.debug("Projection is cache-servable; skipping DocDB batch fetch")
                result_df = filtered.reset_index(drop=True)
            else:
                fetch_start = time.time()
                logger.debug("Fetching %d full records from DocDB (batched)", len(names))
                records = _fetch_full_records_batched(names)
                logger.debug("DocDB batch fetch complete: %.3fs", time.time() - fetch_start)
        backend = "cache"
    else:
        result_df = None
        logger.debug("Routing to docdb backend")
        client = MetadataDbClient(host=API_GATEWAY_HOST)
        if names_only:
            kwargs: dict = {"filter_query": filter_query, "projection": {"name": 1}}
            if limit:
                kwargs["limit"] = limit
            raw = client.retrieve_docdb_records(**kwargs)
            names = [r["name"] for r in raw]
            records = None
        else:
            kwargs = {"filter_query": filter_query}
            if limit:
                kwargs["limit"] = limit
            if projection is not None:
                kwargs["projection"] = projection
            raw = client.retrieve_docdb_records(**kwargs)
            names = [r["name"] for r in raw]
            records = raw
        backend = "docdb"

    elapsed = time.time() - start
    logger.info(
        "Query complete: backend=%s elapsed=%.3fs results=%d names_only=%s",
        backend, elapsed, len(names), names_only,
    )
    return QueryResult(
        backend=backend,
        elapsed_seconds=elapsed,
        asset_names=names,
        records=records,
        dataframe=result_df,
    )


def retrieve_aggregation(pipeline: list) -> QueryResult:
    """Execute an aggregation pipeline directly against DocumentDB.

    The pipeline is never routed through the local cache.

    Parameters
    ----------
    pipeline:
        A MongoDB aggregation pipeline (a list of stage dicts).

    Raises
    ------
    ValueError
        If *pipeline* is not a non-empty list of dicts.
    """
    if not isinstance(pipeline, list) or not pipeline:
        raise ValueError("pipeline must be a non-empty list of stage dicts")
    for i, stage in enumerate(pipeline):
        if not isinstance(stage, dict):
            raise ValueError(f"pipeline stage {i} is not a dict: {stage!r}")

    logger.debug("retrieve_aggregation called: %d stages", len(pipeline))
    start = time.time()

    client = MetadataDbClient(host=API_GATEWAY_HOST)
    records = client.aggregate_docdb_records(pipeline=pipeline)
    names = [r["name"] for r in records if "name" in r]

    elapsed = time.time() - start
    logger.info("Aggregation complete: elapsed=%.3fs results=%d", elapsed, len(records))
    return QueryResult(
        backend="docdb",
        elapsed_seconds=elapsed,
        asset_names=names,
        records=records,
        dataframe=None,
    )

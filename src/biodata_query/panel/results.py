"""QueryResults Panel component for displaying query execution results."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd
import panel as pn
import param

from biodata_query.query import QueryResult, retrieve_records, retrieve_aggregation

# Fields needed for the results table — all are available in the local cache.
_DISPLAY_PROJECTION: dict[str, int] = {
    "name": 1,
    "data_description.project_name": 1,
    "data_description.data_level": 1,
    "data_description.modalities": 1,
    "subject.subject_id": 1,
    "acquisition.acquisition_start_time": 1,
}

# Flat column names in asset_basics that correspond to _DISPLAY_PROJECTION.
_DISPLAY_COLUMNS: list[str] = [
    "name",
    "project_name",
    "data_level",
    "modalities",
    "subject_id",
    "acquisition_start_time",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_records(records: list[dict]) -> pd.DataFrame:
    """Extract key columns from full DocDB records for tabular display."""
    rows = []
    for r in records:
        dd = r.get("data_description") or {}
        subj = r.get("subject") or {}
        acq = r.get("acquisition") or {}

        # Modalities may be a list of dicts with an "abbreviation" key
        raw_modalities = dd.get("modality") or dd.get("modalities") or []
        if isinstance(raw_modalities, list):
            modalities_str = ", ".join(
                m.get("abbreviation", "") if isinstance(m, dict) else str(m)
                for m in raw_modalities
            )
        else:
            modalities_str = str(raw_modalities)

        rows.append(
            {
                "name": r.get("name", ""),
                "project_name": dd.get("project_name", ""),
                "data_level": dd.get("data_level", ""),
                "modalities": modalities_str,
                "subject_id": subj.get("subject_id", ""),
                "acquisition_start_time": acq.get("acquisition_start_time", ""),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# QueryResults
# ---------------------------------------------------------------------------

class QueryResults(pn.custom.PyComponent):
    """Panel component that executes a query and displays results in a table.

    Connect to a :class:`~biodata_query.panel.builder.QueryBuilder` by
    assigning its ``query`` param::

        builder = QueryBuilder()
        results = QueryResults()
        builder.param.watch(lambda e: results.param.update(query=e.new), "query")
        builder.param.watch(lambda e: results.param.update(pipeline=e.new), "pipeline")

    Or trigger manually via :meth:`run`.
    """

    query = param.Dict(default={}, doc="MongoDB filter query to execute")
    pipeline = param.List(default=[], doc="MongoDB aggregation pipeline to execute")
    force_backend = param.Selector(
        default=None,
        objects=[None, "cache", "docdb"],
        doc="Force routing to a specific backend. None = auto-route. Ignored for aggregation pipelines.",
    )

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        self._last_result: QueryResult | None = None

        self._status = pn.pane.Markdown("*No query run yet.*")
        self._tabulator = pn.widgets.Tabulator(
            pd.DataFrame(),
            pagination="local",
            page_size=50,
            show_index=False,
            sizing_mode="stretch_width",
            header_filters=True,
        )

        self.param.watch(self._on_query_change, "query")
        self.param.watch(self._on_pipeline_change, "pipeline")

    # ------------------------------------------------------------------
    # Watcher
    # ------------------------------------------------------------------

    def _on_query_change(self, event: Any) -> None:  # noqa: ARG002
        if event.new:
            self._execute(event.new)

    def _on_pipeline_change(self, event: Any) -> None:  # noqa: ARG002
        if event.new:
            self._execute_aggregation(event.new)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: dict | None = None, names_only: bool = False) -> None:
        """Execute *query* (or the current ``self.query``) and update the display."""
        self._execute(query if query is not None else self.query, names_only=names_only)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(self, query: dict, names_only: bool = False) -> None:
        logger.debug("Executing query: %r (names_only=%s force_backend=%s)", query, names_only, self.force_backend)
        self._tabulator.loading = True
        try:
            result = retrieve_records(
                query,
                projection=_DISPLAY_PROJECTION,
                names_only=names_only,
                force_backend=self.force_backend,
            )
        except Exception as exc:
            self._status.object = f"**Error:** {exc}"
            self._tabulator.loading = False
            return

        self._last_result = result

        self._status.object = (
            f"**Backend:** {result.backend} &nbsp;|&nbsp; "
            f"**Time:** {result.elapsed_seconds:.2f}s &nbsp;|&nbsp; "
            f"**Results:** {len(result.asset_names)}"
        )

        if result.dataframe is not None:
            df = result.dataframe[_DISPLAY_COLUMNS].copy()
        elif result.records:
            df = _flatten_records(result.records)
        else:
            df = pd.DataFrame({"name": result.asset_names})

        self._tabulator.value = df
        self._tabulator.loading = False

    def _execute_aggregation(self, pipeline: list) -> None:
        logger.debug("Executing aggregation: %d stages", len(pipeline))
        self._tabulator.loading = True
        try:
            result = retrieve_aggregation(pipeline)
        except Exception as exc:
            self._status.object = f"**Error:** {exc}"
            self._tabulator.loading = False
            return

        self._last_result = result

        self._status.object = (
            f"**Backend:** {result.backend} (aggregation) &nbsp;|&nbsp; "
            f"**Time:** {result.elapsed_seconds:.2f}s &nbsp;|&nbsp; "
            f"**Results:** {len(result.records)}"
        )

        df = _flatten_records(result.records) if result.records else pd.DataFrame()
        self._tabulator.value = df
        self._tabulator.loading = False

    # ------------------------------------------------------------------
    # Panel interface
    # ------------------------------------------------------------------

    def __panel__(self) -> pn.viewable.Viewable:
        return pn.Column(self._status, self._tabulator, sizing_mode="stretch_width", width_policy="max")

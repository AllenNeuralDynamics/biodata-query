"""QueryBuilder Panel component for interactively constructing MongoDB-style queries."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import panel as pn
import panel_material_ui as pmu
import param
from zombie_squirrel import asset_basics, unique_project_names, unique_subject_ids

from biodata_query.query import FIELD_TO_COLUMN, _apply_filter_to_dataframe

# ---------------------------------------------------------------------------
# Modality abbreviations from aind-data-schema-models
# ---------------------------------------------------------------------------
from aind_data_schema_models.modalities import Modality as _Modality

_MODALITY_ABBREVIATIONS: list[str] = sorted(
    {v.abbreviation for v in vars(_Modality).values() if hasattr(v, "abbreviation")}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_regex_str(value: Any) -> str:
    """Return the plain string from a value that may be a str or a ``$regex`` dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("$regex", "")
    return ""


# ---------------------------------------------------------------------------
# QueryBuilder
# ---------------------------------------------------------------------------

class QueryBuilder(pn.custom.PyComponent):
    """Panel component that builds a MongoDB filter query interactively.

    Exposes a ``query`` param that external components can watch.
    """

    query = param.Dict(default={}, doc="Current MongoDB filter query dictionary")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

        project_options: list[str] = [""] + sorted(p for p in unique_project_names() if p is not None)
        subject_options: list[str] = sorted(str(s) for s in unique_subject_ids() if s is not None)

        # --- widgets -------------------------------------------------------
        self._w_name = pmu.TextInput(name="name", placeholder="e.g. ecephys_*", size="small")
        self._w_project_name = pmu.AutocompleteInput(
            name="data_description.project_name",
            options=project_options,
            min_characters=0,
            sizing_mode="stretch_width",
            size="small",
        )
        self._w_modality = pmu.MultiSelect(
            name="data_description.modalities", options=_MODALITY_ABBREVIATIONS, size=3
        )
        self._w_data_level = pmu.Select(
            name="data_description.data_level", options=["", "raw", "derived"], size="small"
        )
        self._w_subject_id = pmu.AutocompleteInput(
            name="subject.subject_id",
            options=subject_options,
            placeholder="e.g. 000000",
            min_characters=1,
            sizing_mode="stretch_width",
            size="small",
        )
        self._w_genotype = pmu.TextInput(
            name="subject.subject_details.genotype", placeholder="e.g. Ai32", size="small"
        )
        self._w_acq_start_min = pmu.DatetimePicker(
            name="Min: acquisition.acquisition_start_time", disabled=True
        )
        self._w_acq_start_max = pmu.DatetimePicker(
            name="Max: acquisition.acquisition_start_time", disabled=True
        )
        self._w_process_date = pmu.DatePicker(name="process_date")
        self._w_query_dict = pmu.TextAreaInput(
            name="Query JSON", value="{}", height=56
        )
        self._w_run = pmu.Button(name="Submit", button_type="primary", size="small")

        # Prevent re-entrant sync loops
        self._syncing = False

        # Wire up watchers
        for w in (
            self._w_name,
            self._w_project_name,
            self._w_modality,
            self._w_data_level,
            self._w_subject_id,
            self._w_genotype,
            self._w_acq_start_min,
            self._w_acq_start_max,
            self._w_process_date,
        ):
            w.param.watch(self._on_widget_change, "value")

        self._w_query_dict.param.watch(self._on_text_change, "value")
        self._w_run.on_click(self._on_run_click)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query_from_widgets(self) -> dict:
        """Construct the MongoDB filter dict from current widget values."""
        q: dict[str, Any] = {}

        if self._w_name.value:
            q["name"] = {"$regex": self._w_name.value, "$options": "i"}
        if self._w_project_name.value:
            q["data_description.project_name"] = self._w_project_name.value
        if self._w_modality.value:
            q["data_description.modalities"] = {"$in": list(self._w_modality.value)}
        if self._w_data_level.value:
            q["data_description.data_level"] = self._w_data_level.value
        if self._w_subject_id.value:
            q["subject.subject_id"] = self._w_subject_id.value
        if self._w_genotype.value:
            q["subject.subject_details.genotype"] = {
                "$regex": self._w_genotype.value,
                "$options": "i",
            }

        acq_filter: dict[str, Any] = {}
        if self._w_acq_start_min.value:
            acq_filter["$gte"] = self._w_acq_start_min.value.isoformat()
        if self._w_acq_start_max.value:
            acq_filter["$lte"] = self._w_acq_start_max.value.isoformat()
        if acq_filter:
            q["acquisition.acquisition_start_time"] = acq_filter

        if self._w_process_date.value:
            q["process_date"] = str(self._w_process_date.value)

        return q

    def _has_non_time_filter(self) -> bool:
        return any(
            [
                self._w_name.value,
                self._w_project_name.value,
                self._w_modality.value,
                self._w_data_level.value,
                self._w_subject_id.value,
                self._w_genotype.value,
                self._w_process_date.value,
            ]
        )

    def _update_time_picker_bounds(self) -> None:
        """Constrain the datetime pickers to the range found in the filtered cache."""
        try:
            df = asset_basics()
            partial_query: dict[str, Any] = {}
            if self._w_name.value:
                partial_query["name"] = {"$regex": self._w_name.value, "$options": "i"}
            if self._w_project_name.value:
                partial_query["data_description.project_name"] = self._w_project_name.value
            if self._w_modality.value:
                partial_query["data_description.modalities"] = {"$in": list(self._w_modality.value)}
            if self._w_data_level.value:
                partial_query["data_description.data_level"] = self._w_data_level.value
            if self._w_subject_id.value:
                partial_query["subject.subject_id"] = self._w_subject_id.value
            if self._w_genotype.value:
                partial_query["subject.subject_details.genotype"] = {
                    "$regex": self._w_genotype.value,
                    "$options": "i",
                }

            if partial_query:
                df = _apply_filter_to_dataframe(df, partial_query)

            if len(df) > 0 and "acquisition_start_time" in df.columns:
                times = pd.to_datetime(df["acquisition_start_time"], utc=True, errors="coerce").dropna()
                if len(times) > 0:
                    min_dt = times.min().to_pydatetime()
                    max_dt = times.max().to_pydatetime()
                    self._w_acq_start_min.start = min_dt
                    self._w_acq_start_min.end = max_dt
                    self._w_acq_start_max.start = min_dt
                    self._w_acq_start_max.end = max_dt
        except Exception:
            pass  # Best-effort; never crash the UI

    # ------------------------------------------------------------------
    # Watcher callbacks
    # ------------------------------------------------------------------

    def _on_widget_change(self, event: Any) -> None:  # noqa: ARG002
        if self._syncing:
            return
        self._syncing = True
        try:
            has_filter = self._has_non_time_filter()
            self._w_acq_start_min.disabled = not has_filter
            self._w_acq_start_max.disabled = not has_filter
            if has_filter:
                self._update_time_picker_bounds()

            q = self._build_query_from_widgets()
            self.query = q
            self._w_query_dict.value = json.dumps(q, indent=2, default=str)
        finally:
            self._syncing = False

    def _on_run_click(self, event: Any) -> None:  # noqa: ARG002
        self.query = self._build_query_from_widgets()

    def _on_text_change(self, event: Any) -> None:
        if self._syncing:
            return
        try:
            q = json.loads(event.new)
        except (json.JSONDecodeError, ValueError):
            return  # Ignore invalid JSON while the user is mid-edit

        self._syncing = True
        try:
            self.query = q

            # Update widgets best-effort; ignore unknown values
            self._w_name.value = _extract_regex_str(q.get("name", ""))

            proj = q.get("data_description.project_name", "")
            if proj in self._w_project_name.options:
                self._w_project_name.value = proj
            else:
                self._w_project_name.value = ""

            modalities_val = q.get("data_description.modalities", {})
            if isinstance(modalities_val, dict) and "$in" in modalities_val:
                self._w_modality.value = [
                    m for m in modalities_val["$in"] if m in _MODALITY_ABBREVIATIONS
                ]
            else:
                self._w_modality.value = []

            dl = q.get("data_description.data_level", "")
            self._w_data_level.value = dl if dl in ("", "raw", "derived") else ""

            self._w_subject_id.value = q.get("subject.subject_id", "")
            self._w_genotype.value = _extract_regex_str(
                q.get("subject.subject_details.genotype", "")
            )
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Panel interface
    # ------------------------------------------------------------------

    def __panel__(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.Row(
                pn.Column(
                    self._w_project_name,
                    self._w_subject_id,
                    sizing_mode="stretch_width",
                    margin=(2, 4),
                ),
                pn.Column(
                    self._w_run,
                    align="center",
                    width=120,
                    margin=(2, 4),
                ),
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self._w_acq_start_min,
                self._w_acq_start_max,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self._w_name,
                self._w_genotype,
                self._w_data_level,
                self._w_process_date,
                sizing_mode="stretch_width",
            ),
            self._w_modality,
            self._w_query_dict,
            sizing_mode="stretch_width",
            margin=(4, 8),
        )

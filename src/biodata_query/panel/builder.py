"""QueryBuilder Panel component for interactively constructing MongoDB-style queries."""

from __future__ import annotations

import datetime
import json
import os
import threading
import urllib.parse
from typing import Any

import pandas as pd
import panel as pn
import panel_material_ui as pmu
import param
import requests as _requests
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
# Button sizing constants
# ---------------------------------------------------------------------------
_BUTTON_WIDTH: int = 100
_BUTTON_SIZE: str = "small"


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

        project_options: list[str] = sorted(p for p in unique_project_names() if p is not None)
        subject_options: list[str] = sorted(str(s) for s in unique_subject_ids() if s is not None)
        _df = asset_basics()
        genotype_options: list[str] = sorted(
            g for g in _df["genotype"].dropna().unique() if g and str(g).strip()
        )

        # --- widgets -------------------------------------------------------
        self._w_name = pmu.TextInput(name="name", placeholder="e.g. ecephys_*", size="small")
        self._w_project_name = pmu.MultiChoice(
            name="data_description.project_name",
            value=[],
            options=project_options,
            sizing_mode="stretch_width",
        )
        self._w_modality = pmu.MultiChoice(
            name="data_description.modalities", value=[], options=_MODALITY_ABBREVIATIONS
        )
        self._w_data_level = pmu.Select(
            name="data_description.data_level", options=["", "raw", "derived"], size="small"
        )
        self._w_subject_id = pmu.MultiChoice(
            name="subject.subject_id",
            value=[],
            options=subject_options,
            sizing_mode="stretch_width",
        )
        self._w_genotype = pmu.MultiChoice(
            name="subject.subject_details.genotype",
            value=[],
            options=genotype_options,
            sizing_mode="stretch_width",
        )
        self._w_acq_start_min = pmu.DatetimePicker(
            name="Min: acquisition.acquisition_start_time", disabled=True
        )
        self._w_acq_start_max = pmu.DatetimePicker(
            name="Max: acquisition.acquisition_start_time", disabled=True
        )
        self._w_process_date = pmu.DatePicker(name="process_date")
        self._w_query_dict = pmu.TextAreaInput(
            name="Query JSON", value="{}", height=150
        )
        self._w_run = pmu.Button(name="Submit", button_type="primary", size=_BUTTON_SIZE, width=_BUTTON_WIDTH)

        # --- Paste-mode flag (True when the user manually edited the JSON area)
        self._paste_mode: bool = False

        # --- Chat widgets -------------------------------------------------
        self._w_chat_input = pmu.TextInput(
            name="",
            placeholder="Ask AI to modify the query… (press Enter or Submit)",
            size="small",
            sizing_mode="stretch_width",
        )
        self._w_chat_submit = pmu.Button(name="Ask AI", button_type="success", size=_BUTTON_SIZE, width=_BUTTON_WIDTH)
        self._w_chat_error = pn.pane.Markdown("", visible=False, sizing_mode="stretch_width")

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

        # Chat watchers
        self._w_chat_input.param.watch(self._on_chat_enter, "value")
        self._w_chat_submit.on_click(self._on_chat_click)

        # Restore state from URL on first page load
        pn.state.onload(self._init_from_url)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query_from_widgets(self) -> dict:
        """Construct the MongoDB filter dict from current widget values."""
        q: dict[str, Any] = {}

        if self._w_name.value:
            q["name"] = {"$regex": self._w_name.value, "$options": "i"}
        if self._w_project_name.value:
            vals = list(self._w_project_name.value)
            q["data_description.project_name"] = vals[0] if len(vals) == 1 else {"$in": vals}
        if self._w_modality.value:
            vals = list(self._w_modality.value)
            if len(vals) == 1:
                q["data_description.modalities.abbreviation"] = vals[0]
            else:
                q["data_description.modalities.abbreviation"] = {"$all": vals}
        if self._w_data_level.value:
            q["data_description.data_level"] = self._w_data_level.value
        if self._w_subject_id.value:
            vals = list(self._w_subject_id.value)
            q["subject.subject_id"] = vals[0] if len(vals) == 1 else {"$in": vals}
        if self._w_genotype.value:
            vals = list(self._w_genotype.value)
            q["subject.subject_details.genotype"] = vals[0] if len(vals) == 1 else {"$in": vals}

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

    def _set_paste_mode(self, enabled: bool) -> None:
        """Enable or disable paste mode.

        In paste mode all filter widgets are disabled so the only way to
        change the query is by editing the JSON textarea directly.
        """
        self._paste_mode = enabled
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
            w.disabled = enabled

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
                vals = list(self._w_project_name.value)
                partial_query["data_description.project_name"] = vals[0] if len(vals) == 1 else {"$in": vals}
            if self._w_modality.value:
                vals = list(self._w_modality.value)
                if len(vals) == 1:
                    partial_query["data_description.modalities.abbreviation"] = vals[0]
                else:
                    partial_query["data_description.modalities.abbreviation"] = {"$all": vals}
            if self._w_data_level.value:
                partial_query["data_description.data_level"] = self._w_data_level.value
            if self._w_subject_id.value:
                vals = list(self._w_subject_id.value)
                partial_query["subject.subject_id"] = vals[0] if len(vals) == 1 else {"$in": vals}
            if self._w_genotype.value:
                vals = list(self._w_genotype.value)
                partial_query["subject.subject_details.genotype"] = vals[0] if len(vals) == 1 else {"$in": vals}

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
    # URL sync helpers
    # ------------------------------------------------------------------

    def _sync_url(self) -> None:
        """Push current state into browser URL query parameters.

        In paste mode the raw query JSON is stored under ``q``.
        In widget mode each active widget value is stored individually so
        the exact filter configuration can be restored on page reload.
        """
        if pn.state.location is None:
            return

        if self._paste_mode:
            params: dict[str, str] = {
                "mode": "paste",
                "q": json.dumps(self.query, default=str),
            }
        else:
            params = {"mode": "widget"}
            if self._w_name.value:
                params["name"] = self._w_name.value
            if self._w_project_name.value:
                params["project_name"] = json.dumps(list(self._w_project_name.value))
            if self._w_modality.value:
                params["modality"] = json.dumps(list(self._w_modality.value))
            if self._w_data_level.value:
                params["data_level"] = self._w_data_level.value
            if self._w_subject_id.value:
                params["subject_id"] = json.dumps(list(self._w_subject_id.value))
            if self._w_genotype.value:
                params["genotype"] = json.dumps(list(self._w_genotype.value))
            if self._w_acq_start_min.value:
                params["acq_min"] = self._w_acq_start_min.value.isoformat()
            if self._w_acq_start_max.value:
                params["acq_max"] = self._w_acq_start_max.value.isoformat()
            if self._w_process_date.value:
                params["process_date"] = str(self._w_process_date.value)

        pn.state.location.search = "?" + urllib.parse.urlencode(params)

    def _init_from_url(self) -> None:
        """Restore widget / paste state from URL query parameters on page load."""
        if pn.state.location is None:
            return
        search = pn.state.location.search
        if not search or search in ("?", ""):
            return

        raw = urllib.parse.parse_qs(search.lstrip("?"), keep_blank_values=False)

        def _first(key: str, default: str = "") -> str:
            return raw.get(key, [default])[0]

        mode = _first("mode", "widget")

        self._syncing = True
        try:
            if mode == "paste":
                q_str = _first("q", "{}")
                self._w_query_dict.value = q_str
                try:
                    self.query = json.loads(q_str)
                except (json.JSONDecodeError, ValueError):
                    self.query = {}
            else:
                name = _first("name")
                if name:
                    self._w_name.value = name

                proj_str = _first("project_name")
                if proj_str:
                    try:
                        vals = json.loads(proj_str)
                        self._w_project_name.value = [
                            v for v in vals if v in self._w_project_name.options
                        ]
                    except (json.JSONDecodeError, ValueError):
                        pass

                modality_str = _first("modality")
                if modality_str:
                    try:
                        vals = json.loads(modality_str)
                        self._w_modality.value = [
                            v for v in vals if v in _MODALITY_ABBREVIATIONS
                        ]
                    except (json.JSONDecodeError, ValueError):
                        pass

                data_level = _first("data_level")
                if data_level in ("raw", "derived"):
                    self._w_data_level.value = data_level

                subj_str = _first("subject_id")
                if subj_str:
                    try:
                        vals = json.loads(subj_str)
                        self._w_subject_id.value = [
                            v for v in vals if v in self._w_subject_id.options
                        ]
                    except (json.JSONDecodeError, ValueError):
                        pass

                geno_str = _first("genotype")
                if geno_str:
                    try:
                        vals = json.loads(geno_str)
                        self._w_genotype.value = [
                            v for v in vals if v in self._w_genotype.options
                        ]
                    except (json.JSONDecodeError, ValueError):
                        pass

                acq_min_str = _first("acq_min")
                if acq_min_str:
                    try:
                        self._w_acq_start_min.value = datetime.datetime.fromisoformat(acq_min_str)
                    except ValueError:
                        pass

                acq_max_str = _first("acq_max")
                if acq_max_str:
                    try:
                        self._w_acq_start_max.value = datetime.datetime.fromisoformat(acq_max_str)
                    except ValueError:
                        pass

                process_date_str = _first("process_date")
                if process_date_str:
                    try:
                        self._w_process_date.value = datetime.date.fromisoformat(process_date_str)
                    except ValueError:
                        pass

                has_filter = self._has_non_time_filter()
                self._w_acq_start_min.disabled = not has_filter
                self._w_acq_start_max.disabled = not has_filter

                q = self._build_query_from_widgets()
                self._w_query_dict.value = json.dumps(q, indent=2, default=str)
                self.query = q
        finally:
            self._syncing = False

        if mode == "paste":
            self._set_paste_mode(True)

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
            self._w_query_dict.value = json.dumps(q, indent=2, default=str)
        finally:
            self._syncing = False
        self._sync_url()

    def _on_run_click(self, event: Any) -> None:  # noqa: ARG002
        if self._paste_mode:
            try:
                self.query = json.loads(self._w_query_dict.value)
            except (json.JSONDecodeError, ValueError):
                pass
        else:
            self.query = self._build_query_from_widgets()

    def _on_text_change(self, event: Any) -> None:
        if self._syncing:
            return
        try:
            q = json.loads(event.new)
        except (json.JSONDecodeError, ValueError):
            return  # Ignore invalid JSON while the user is mid-edit

        # The user directly edited the textarea — enter paste mode.
        self._set_paste_mode(True)

        self._syncing = True
        try:
            self.query = q

            # Update widgets best-effort; ignore unknown values
            self._w_name.value = _extract_regex_str(q.get("name", ""))

            proj_val = q.get("data_description.project_name", None)
            if isinstance(proj_val, str):
                self._w_project_name.value = [proj_val] if proj_val in self._w_project_name.options else []
            elif isinstance(proj_val, dict) and "$in" in proj_val:
                self._w_project_name.value = [v for v in proj_val["$in"] if v in self._w_project_name.options]
            else:
                self._w_project_name.value = []

            modalities_val = q.get("data_description.modalities.abbreviation", None)
            if isinstance(modalities_val, str):
                self._w_modality.value = [modalities_val] if modalities_val in _MODALITY_ABBREVIATIONS else []
            elif isinstance(modalities_val, dict) and "$all" in modalities_val:
                self._w_modality.value = [
                    m for m in modalities_val["$all"] if m in _MODALITY_ABBREVIATIONS
                ]
            else:
                self._w_modality.value = []

            dl = q.get("data_description.data_level", "")
            self._w_data_level.value = dl if dl in ("", "raw", "derived") else ""

            subj_val = q.get("subject.subject_id", None)
            if isinstance(subj_val, str):
                self._w_subject_id.value = [subj_val] if subj_val in self._w_subject_id.options else []
            elif isinstance(subj_val, dict) and "$in" in subj_val:
                self._w_subject_id.value = [v for v in subj_val["$in"] if v in self._w_subject_id.options]
            else:
                self._w_subject_id.value = []

            geno_val = q.get("subject.subject_details.genotype", None)
            if isinstance(geno_val, str):
                self._w_genotype.value = [geno_val] if geno_val in self._w_genotype.options else []
            elif isinstance(geno_val, dict) and "$in" in geno_val:
                self._w_genotype.value = [v for v in geno_val["$in"] if v in self._w_genotype.options]
            else:
                self._w_genotype.value = []
        finally:
            self._syncing = False
        self._sync_url()

    # ------------------------------------------------------------------
    # Chat / LLM callbacks
    # ------------------------------------------------------------------

    def _on_chat_enter(self, event: Any) -> None:
        """Fires when the user presses Enter in the chat input."""
        if event.new and event.new.strip():
            msg = event.new.strip()
            self._w_chat_input.value = ""  # clear first; fires watcher again with ""
            self._run_chat_async(msg)

    def _on_chat_click(self, event: Any) -> None:  # noqa: ARG002
        msg = self._w_chat_input.value.strip()
        if msg:
            self._w_chat_input.value = ""
            self._run_chat_async(msg)

    def _run_chat_async(self, message: str) -> None:
        """Submit *message* + current query to the LLM endpoint in a background thread.

        The target URL is read from the ``BIODATA_QUERY_LLM_URL`` environment
        variable, defaulting to ``http://127.0.0.1:8765/get-query``.
        """
        self._w_chat_submit.disabled = True
        self._w_chat_input.disabled = True
        self._w_chat_error.visible = False
        self._w_chat_error.object = ""

        query_snapshot = dict(self.query)
        url = os.environ.get("BIODATA_QUERY_LLM_URL", "http://127.0.0.1:8765/get-query")

        def _do_call() -> None:
            try:
                resp = _requests.get(
                    url,
                    params={
                        "query": json.dumps(query_snapshot),
                        "message": message,
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    new_query = resp.json()["query"]
                    pn.state.execute(lambda nq=new_query: self._apply_new_query(nq))
                else:
                    err = resp.text
                    pn.state.execute(lambda e=err: self._show_chat_error(e))
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                pn.state.execute(lambda e=err: self._show_chat_error(e))
            finally:
                pn.state.execute(self._restore_chat_widgets)

        threading.Thread(target=_do_call, daemon=True).start()

    def _apply_new_query(self, new_query: dict) -> None:
        """Update the query textarea; the text-change watcher propagates to self.query."""
        self._w_query_dict.value = json.dumps(new_query, indent=2, default=str)

    def _show_chat_error(self, msg: str) -> None:
        self._w_chat_error.object = f"**Error:** {msg}"
        self._w_chat_error.visible = True

    def _restore_chat_widgets(self) -> None:
        self._w_chat_submit.disabled = False
        self._w_chat_input.disabled = False

    # ------------------------------------------------------------------
    # Panel interface
    # ------------------------------------------------------------------

    def __panel__(self) -> pn.viewable.Viewable:
        return pn.Column(
            # --- Filters -------------------------------------------------
            pn.Row(
                self._w_project_name,
                self._w_subject_id,
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
            pn.Row(
                self._w_acq_start_min,
                self._w_acq_start_max,
                sizing_mode="stretch_width",
            ),
            pn.layout.Divider(),
            # --- Chat / LLM ----------------------------------------------
            pn.Row(
                self._w_chat_input,
                self._w_chat_submit,
                align="center",
                sizing_mode="stretch_width",
            ),
            self._w_chat_error,
            pn.layout.Divider(),
            # --- Query JSON ----------------------------------------------
            pn.Row(
                self._w_query_dict,
                pn.Column(self._w_run, align="center", width=_BUTTON_WIDTH + 16),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            margin=(4, 8),
        )

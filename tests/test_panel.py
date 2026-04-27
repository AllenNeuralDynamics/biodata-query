"""Tests for the Panel UI components in biodata_query.panel."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Skip the entire module if panel is not installed
pn = pytest.importorskip("panel")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_DF = pd.DataFrame(
    {
        "name": ["asset-A", "asset-B", "asset-C"],
        "project_name": ["ProjectX", "ProjectX", "ProjectY"],
        "modalities": ["ecephys", "behavior", "ecephys, behavior"],
        "data_level": ["raw", "raw", "derived"],
        "subject_id": ["111111", "222222", "111111"],
        "genotype": ["Ai32/wt", "wt/wt", "Ai32/wt"],
        "acquisition_start_time": [
            "2024-01-10T08:00:00+00:00",
            "2024-02-15T09:30:00+00:00",
            "2024-03-20T10:00:00+00:00",
        ],
        "acquisition_end_time": [
            "2024-01-10T10:00:00+00:00",
            "2024-02-15T11:30:00+00:00",
            "2024-03-20T12:00:00+00:00",
        ],
        "process_date": ["2024-01-11", "2024-02-16", "2024-03-21"],
    }
)


@pytest.fixture(autouse=True)
def patch_modality_abbreviations():
    """Replace the module-level modality list so tests don't need aind-data-schema-models."""
    with patch("biodata_query.panel.builder._MODALITY_ABBREVIATIONS", ["ecephys", "behavior", "fib"]):
        yield


@pytest.fixture
def builder():
    """Return a QueryBuilder with all zombie_squirrel calls mocked."""
    with (
        patch("biodata_query.panel.builder.unique_project_names", return_value=["ProjectX", "ProjectY"]),
        patch("biodata_query.panel.builder.unique_subject_ids", return_value=["111111", "222222"]),
        patch("biodata_query.panel.builder.asset_basics", return_value=_SAMPLE_DF.copy()),
    ):
        from biodata_query.panel.builder import QueryBuilder

        return QueryBuilder()


@pytest.fixture
def results():
    from biodata_query.panel.results import QueryResults

    return QueryResults()


# ---------------------------------------------------------------------------
# QueryBuilder — instantiation
# ---------------------------------------------------------------------------


def test_query_builder_instantiates(builder):
    from biodata_query.panel.builder import QueryBuilder

    assert isinstance(builder, QueryBuilder)


def test_query_builder_initial_query_is_empty(builder):
    assert builder.query == {}


def test_query_builder_project_options_populated(builder):
    # blank option plus the two project names
    assert "" in builder._w_project_name.options
    assert "ProjectX" in builder._w_project_name.options
    assert "ProjectY" in builder._w_project_name.options


def test_query_builder_modality_options_populated(builder):
    assert "ecephys" in builder._w_modality.options
    assert "behavior" in builder._w_modality.options


# ---------------------------------------------------------------------------
# QueryBuilder — widget → query
# ---------------------------------------------------------------------------


def test_widget_project_name_updates_query(builder):
    builder._w_project_name.value = "ProjectX"
    assert builder.query.get("data_description.project_name") == "ProjectX"


def test_widget_modality_updates_query(builder):
    builder._w_modality.value = ["ecephys"]
    assert builder.query.get("data_description.modalities.abbreviation") == "ecephys"


def test_widget_multiple_modalities_uses_all(builder):
    builder._w_modality.value = ["ecephys", "behavior"]
    assert builder.query.get("data_description.modalities.abbreviation") == {"$all": ["ecephys", "behavior"]}


def test_widget_data_level_updates_query(builder):
    builder._w_data_level.value = "raw"
    assert builder.query.get("data_description.data_level") == "raw"


def test_widget_name_regex_updates_query(builder):
    builder._w_name.value = "asset.*"
    q = builder.query
    assert q.get("name") == {"$regex": "asset.*", "$options": "i"}


def test_widget_subject_id_updates_query(builder):
    builder._w_subject_id.value = "111111"
    assert builder.query.get("subject.subject_id") == "111111"


def test_widget_genotype_updates_query(builder):
    builder._w_genotype.value = "Ai32"
    q = builder.query
    assert q.get("subject.subject_details.genotype") == {"$regex": "Ai32", "$options": "i"}


def test_empty_widget_value_excluded_from_query(builder):
    builder._w_project_name.value = "ProjectX"
    assert "data_description.project_name" in builder.query
    builder._w_project_name.value = ""
    assert "data_description.project_name" not in builder.query


def test_multiple_widgets_anded(builder):
    builder._w_project_name.value = "ProjectX"
    builder._w_data_level.value = "raw"
    q = builder.query
    assert q.get("data_description.project_name") == "ProjectX"
    assert q.get("data_description.data_level") == "raw"


# ---------------------------------------------------------------------------
# QueryBuilder — time picker enabled/disabled
# ---------------------------------------------------------------------------


def test_time_pickers_disabled_when_no_filter(builder):
    assert builder._w_acq_start_min.disabled is True
    assert builder._w_acq_start_max.disabled is True


def test_time_pickers_enabled_when_filter_set(builder):
    with patch("biodata_query.panel.builder.asset_basics", return_value=_SAMPLE_DF.copy()):
        builder._w_project_name.value = "ProjectX"
    assert builder._w_acq_start_min.disabled is False
    assert builder._w_acq_start_max.disabled is False


# ---------------------------------------------------------------------------
# QueryBuilder — JSON text area → widgets (bidirectional)
# ---------------------------------------------------------------------------


def test_json_text_updates_query(builder):
    new_q = {"data_description.project_name": "ProjectY"}
    builder._w_query_dict.value = json.dumps(new_q)
    assert builder.query.get("data_description.project_name") == "ProjectY"


def test_json_text_updates_project_widget(builder):
    new_q = {"data_description.project_name": "ProjectX"}
    builder._w_query_dict.value = json.dumps(new_q)
    assert builder._w_project_name.value == "ProjectX"


def test_json_text_updates_modality_widget(builder):
    new_q = {"data_description.modalities.abbreviation": "ecephys"}
    builder._w_query_dict.value = json.dumps(new_q)
    assert builder._w_modality.value == ["ecephys"]


def test_json_text_updates_multiple_modality_widget(builder):
    new_q = {"data_description.modalities.abbreviation": {"$all": ["ecephys", "behavior"]}}
    builder._w_query_dict.value = json.dumps(new_q)
    assert builder._w_modality.value == ["ecephys", "behavior"]


def test_json_text_updates_data_level_widget(builder):
    new_q = {"data_description.data_level": "derived"}
    builder._w_query_dict.value = json.dumps(new_q)
    assert builder._w_data_level.value == "derived"


def test_json_text_updates_name_widget(builder):
    new_q = {"name": {"$regex": "my_pattern", "$options": "i"}}
    builder._w_query_dict.value = json.dumps(new_q)
    assert builder._w_name.value == "my_pattern"


def test_invalid_json_ignored(builder):
    """Partial / invalid JSON while the user is typing should not crash."""
    original_query = builder.query
    builder._w_query_dict.value = '{"data_description.project_name": '  # incomplete
    # query should be unchanged
    assert builder.query == original_query


def test_unknown_project_not_set_in_widget(builder):
    """A project name not in the dropdown options should leave the widget blank."""
    new_q = {"data_description.project_name": "UnknownProject"}
    builder._w_query_dict.value = json.dumps(new_q)
    assert builder._w_project_name.value == ""


def test_widget_change_updates_json_text(builder):
    builder._w_project_name.value = "ProjectX"
    parsed = json.loads(builder._w_query_dict.value)
    assert parsed.get("data_description.project_name") == "ProjectX"


# ---------------------------------------------------------------------------
# QueryBuilder — _build_query_from_widgets directly
# ---------------------------------------------------------------------------


def test_build_query_from_widgets_all_empty(builder):
    q = builder._build_query_from_widgets()
    assert q == {}


def test_build_query_from_widgets_name_field(builder):
    builder._w_name.value = "test_pattern"
    q = builder._build_query_from_widgets()
    assert q["name"] == {"$regex": "test_pattern", "$options": "i"}


# ---------------------------------------------------------------------------
# QueryResults — instantiation
# ---------------------------------------------------------------------------


def test_query_results_instantiates(results):
    from biodata_query.panel.results import QueryResults

    assert isinstance(results, QueryResults)


def test_query_results_initial_status(results):
    assert "No query run yet" in results._status.object


# ---------------------------------------------------------------------------
# QueryResults — run with mocked retrieve_records
# ---------------------------------------------------------------------------

_MOCK_RESULT_NAMES_ONLY = MagicMock(
    backend="cache",
    elapsed_seconds=0.123,
    asset_names=["asset-A", "asset-B"],
    records=None,
)

_MOCK_RESULT_FULL = MagicMock(
    backend="docdb",
    elapsed_seconds=0.456,
    asset_names=["asset-A"],
    records=[
        {
            "name": "asset-A",
            "data_description": {"project_name": "ProjectX", "data_level": "raw", "modality": []},
            "subject": {"subject_id": "111111"},
            "acquisition": {"acquisition_start_time": "2024-01-10T08:00:00+00:00"},
        }
    ],
)


def test_query_results_run_names_only(results):
    with patch("biodata_query.panel.results.retrieve_records", return_value=_MOCK_RESULT_NAMES_ONLY):
        results.run({"name": "asset.*"}, names_only=True)

    assert "cache" in results._status.object
    assert "0.12" in results._status.object
    assert "2" in results._status.object
    assert list(results._tabulator.value["name"]) == ["asset-A", "asset-B"]


def test_query_results_run_full_records(results):
    with patch("biodata_query.panel.results.retrieve_records", return_value=_MOCK_RESULT_FULL):
        results.run({"name": "asset-A"})

    assert "docdb" in results._status.object
    df = results._tabulator.value
    assert "name" in df.columns
    assert df.iloc[0]["name"] == "asset-A"
    assert df.iloc[0]["project_name"] == "ProjectX"


def test_query_results_error_handling(results):
    with patch("biodata_query.panel.results.retrieve_records", side_effect=RuntimeError("connection failed")):
        results.run({"name": "asset.*"})

    assert "Error" in results._status.object
    assert "connection failed" in results._status.object


def test_query_results_query_param_triggers_run(results):
    with patch("biodata_query.panel.results.retrieve_records", return_value=_MOCK_RESULT_NAMES_ONLY) as mock_rq:
        results.query = {"data_description.project_name": "ProjectX"}

    mock_rq.assert_called_once_with(
        {"data_description.project_name": "ProjectX"}, names_only=False
    )


def test_query_results_empty_query_does_not_run(results):
    with patch("biodata_query.panel.results.retrieve_records") as mock_rq:
        results.query = {}

    mock_rq.assert_not_called()


# ---------------------------------------------------------------------------
# _flatten_records helper
# ---------------------------------------------------------------------------


def test_flatten_records_basic():
    from biodata_query.panel.results import _flatten_records

    records = [
        {
            "name": "asset-X",
            "data_description": {"project_name": "Proj", "data_level": "raw", "modality": [{"abbreviation": "ecephys"}]},
            "subject": {"subject_id": "123"},
            "acquisition": {"acquisition_start_time": "2024-01-01T00:00:00+00:00"},
        }
    ]
    df = _flatten_records(records)
    assert df.iloc[0]["name"] == "asset-X"
    assert df.iloc[0]["project_name"] == "Proj"
    assert df.iloc[0]["modalities"] == "ecephys"
    assert df.iloc[0]["subject_id"] == "123"


def test_flatten_records_missing_fields():
    from biodata_query.panel.results import _flatten_records

    df = _flatten_records([{"name": "bare-asset"}])
    assert df.iloc[0]["name"] == "bare-asset"
    assert df.iloc[0]["project_name"] == ""

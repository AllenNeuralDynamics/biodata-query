"""Tests for biodata_query.query module."""

from __future__ import annotations

from unittest.mock import call, patch

import pandas as pd
import pytest

from biodata_query.query import (
    FIELD_TO_COLUMN,
    QueryResult,
    _apply_filter_to_dataframe,
    _fetch_full_records_batched,
    _has_unsupported_operators,
    _modality_series_contains,
    _modality_series_contains_any,
    _projection_is_cache_servable,
    _to_utc_series,
    _to_utc_timestamp,
    is_cache_eligible,
    retrieve_records,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    """Small DataFrame mimicking real asset_basics() output.

    All columns are plain Python strings, matching the actual parquet schema:
    - datetime columns store ISO-8601 strings with timezone offsets
    - modalities is a comma-separated string of abbreviation terms
    """
    return pd.DataFrame(
        {
            "name": ["asset-A", "asset-B", "asset-C", "asset-D"],
            "project_name": ["ProjectX", "ProjectX", "ProjectY", "ProjectY"],
            # comma-separated modality abbreviations, as returned by zombie_squirrel
            "modalities": ["ecephys", "behavior, behavior-videos", "ecephys, fMRI", "fMRI"],
            "data_level": ["raw", "derived", "raw", "raw"],
            "subject_id": ["100", "200", "300", "400"],
            "genotype": ["wt/wt", "wt/Cre", "wt/wt", "Cre/Cre"],
            # timezone-aware ISO-8601 strings, as returned by zombie_squirrel
            "acquisition_start_time": [
                "2024-01-01 00:00:00+00:00",
                "2024-02-01 00:00:00+00:00",
                "2024-03-01 00:00:00+00:00",
                "2024-04-01 00:00:00+00:00",
            ],
            "acquisition_end_time": [
                "2024-01-02 00:00:00+00:00",
                "2024-02-02 00:00:00+00:00",
                "2024-03-02 00:00:00+00:00",
                "2024-04-02 00:00:00+00:00",
            ],
            "process_date": [
                "2024-01-10 00:00:00+00:00",
                "2024-02-10 00:00:00+00:00",
                "2024-03-10 00:00:00+00:00",
                "2024-04-10 00:00:00+00:00",
            ],
        }
    )


# ── FIELD_TO_COLUMN coverage ──────────────────────────────────────────────────


class TestFieldToColumn:
    """Verify the mapping table contains the expected entries."""

    def test_all_expected_mongo_paths_present(self):
        expected = {
            "name",
            "data_description.project_name",
            "data_description.modality",
            "data_description.modalities",
            "data_description.modalities.abbreviation",
            "data_description.data_level",
            "subject.subject_id",
            "subject.subject_details.genotype",
            "acquisition.acquisition_start_time",
            "acquisition.acquisition_end_time",
            "process_date",
        }
        assert expected == set(FIELD_TO_COLUMN.keys())

    def test_modality_aliases_map_to_same_column(self):
        assert FIELD_TO_COLUMN["data_description.modality"] == FIELD_TO_COLUMN["data_description.modalities"]


# ── _has_unsupported_operators ────────────────────────────────────────────────


class TestHasUnsupportedOperators:
    def test_non_dict_is_safe(self):
        assert _has_unsupported_operators("foo") is False
        assert _has_unsupported_operators(42) is False
        assert _has_unsupported_operators(None) is False
        assert _has_unsupported_operators(["a", "b"]) is False

    def test_empty_dict_is_safe(self):
        assert _has_unsupported_operators({}) is False

    def test_supported_in_is_safe(self):
        assert _has_unsupported_operators({"$in": [1, 2]}) is False

    def test_supported_regex_with_options_is_safe(self):
        assert _has_unsupported_operators({"$regex": "foo", "$options": "i"}) is False

    def test_supported_comparison_ops_are_safe(self):
        assert _has_unsupported_operators({"$gte": 1, "$lte": 10}) is False
        assert _has_unsupported_operators({"$gt": 0, "$lt": 5}) is False

    def test_or_is_unsupported(self):
        assert _has_unsupported_operators({"$or": []}) is True

    def test_not_is_unsupported(self):
        assert _has_unsupported_operators({"$not": {}}) is True

    def test_elemMatch_is_unsupported(self):
        assert _has_unsupported_operators({"$elemMatch": {}}) is True

    def test_exists_is_unsupported(self):
        assert _has_unsupported_operators({"$exists": True}) is True

    def test_nor_is_unsupported(self):
        assert _has_unsupported_operators({"$nor": []}) is True

    def test_unknown_dollar_op_is_unsupported(self):
        assert _has_unsupported_operators({"$unknownOp": 1}) is True


# ── is_cache_eligible ─────────────────────────────────────────────────────────


class TestIsCacheEligible:
    def test_empty_query_is_eligible(self):
        assert is_cache_eligible({}) is True

    def test_single_known_field_eligible(self):
        assert is_cache_eligible({"name": "foo"}) is True

    def test_all_known_fields_eligible(self):
        query = {
            "name": "foo",
            "data_description.project_name": "ProjectX",
            "data_description.modality": "ecephys",
            "data_description.modalities": "ecephys",
            "data_description.data_level": "raw",
            "subject.subject_id": "123",
            "subject.subject_details.genotype": "wt/wt",
            "acquisition.acquisition_start_time": {"$gte": "2024-01-01"},
            "acquisition.acquisition_end_time": {"$lte": "2024-12-31"},
            "process_date": "2024-01-10",
        }
        assert is_cache_eligible(query) is True

    def test_unknown_field_not_eligible(self):
        assert is_cache_eligible({"unknown.field": "foo"}) is False

    def test_unmapped_nested_path_not_eligible(self):
        assert is_cache_eligible({"data_description.institution": "AIND"}) is False

    def test_top_level_dollar_key_not_eligible(self):
        assert is_cache_eligible({"$or": [{"name": "a"}]}) is False

    def test_or_value_not_eligible(self):
        assert is_cache_eligible({"name": {"$or": []}}) is False

    def test_not_value_not_eligible(self):
        assert is_cache_eligible({"name": {"$not": {"$regex": "foo"}}}) is False

    def test_elemMatch_value_not_eligible(self):
        query = {"data_description.modality": {"$elemMatch": {"abbreviation": "ecephys"}}}
        assert is_cache_eligible(query) is False

    def test_exists_value_not_eligible(self):
        assert is_cache_eligible({"name": {"$exists": True}}) is False

    def test_in_value_eligible(self):
        assert is_cache_eligible({"data_description.project_name": {"$in": ["A", "B"]}}) is True

    def test_regex_value_eligible(self):
        assert is_cache_eligible({"name": {"$regex": "pattern", "$options": "i"}}) is True

    def test_comparison_value_eligible(self):
        query = {"acquisition.acquisition_start_time": {"$gte": "2024-01-01", "$lte": "2024-12-31"}}
        assert is_cache_eligible(query) is True

    def test_mixed_eligible_and_ineligible_field_not_eligible(self):
        query = {
            "name": "foo",
            "data_description.institution": "AIND",
        }
        assert is_cache_eligible(query) is False


# ── _apply_filter_to_dataframe ────────────────────────────────────────────────


class TestApplyFilterToDataframe:
    def test_empty_query_returns_all(self, sample_df):
        result = _apply_filter_to_dataframe(sample_df, {})
        assert len(result) == len(sample_df)

    # Simple equality

    def test_equality_match(self, sample_df):
        result = _apply_filter_to_dataframe(sample_df, {"name": "asset-A"})
        assert list(result["name"]) == ["asset-A"]

    def test_equality_no_match(self, sample_df):
        result = _apply_filter_to_dataframe(sample_df, {"name": "no-such-asset"})
        assert len(result) == 0

    def test_equality_via_mapped_path(self, sample_df):
        result = _apply_filter_to_dataframe(sample_df, {"data_description.data_level": "derived"})
        assert list(result["name"]) == ["asset-B"]

    # $in

    def test_in_single_value(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.project_name": {"$in": ["ProjectX"]}}
        )
        assert set(result["name"]) == {"asset-A", "asset-B"}

    def test_in_multiple_values(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.project_name": {"$in": ["ProjectX", "ProjectY"]}}
        )
        assert len(result) == 4

    def test_in_no_match(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.project_name": {"$in": ["NoProject"]}}
        )
        assert len(result) == 0

    # $regex

    def test_regex_case_sensitive(self, sample_df):
        result = _apply_filter_to_dataframe(sample_df, {"name": {"$regex": "asset-[AB]"}})
        assert set(result["name"]) == {"asset-A", "asset-B"}

    def test_regex_case_insensitive_option(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df, {"name": {"$regex": "ASSET-A", "$options": "i"}}
        )
        assert list(result["name"]) == ["asset-A"]

    def test_regex_case_sensitive_no_match_on_uppercase(self, sample_df):
        result = _apply_filter_to_dataframe(sample_df, {"name": {"$regex": "ASSET-A"}})
        assert len(result) == 0

    def test_regex_no_match(self, sample_df):
        result = _apply_filter_to_dataframe(sample_df, {"name": {"$regex": "^zzz"}})
        assert len(result) == 0

    # Comparison operators on datetime

    def test_gte(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {"acquisition.acquisition_start_time": {"$gte": pd.Timestamp("2024-03-01")}},
        )
        assert set(result["name"]) == {"asset-C", "asset-D"}

    def test_lte(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {"acquisition.acquisition_start_time": {"$lte": pd.Timestamp("2024-02-01")}},
        )
        assert set(result["name"]) == {"asset-A", "asset-B"}

    def test_gt(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {"acquisition.acquisition_start_time": {"$gt": pd.Timestamp("2024-02-01")}},
        )
        assert set(result["name"]) == {"asset-C", "asset-D"}

    def test_lt(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {"acquisition.acquisition_start_time": {"$lt": pd.Timestamp("2024-02-01")}},
        )
        assert list(result["name"]) == ["asset-A"]

    def test_range_gte_lte(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {
                "acquisition.acquisition_start_time": {
                    "$gte": pd.Timestamp("2024-02-01"),
                    "$lte": pd.Timestamp("2024-03-01"),
                }
            },
        )
        assert set(result["name"]) == {"asset-B", "asset-C"}

    # Multiple top-level keys (implicit AND)

    def test_multiple_keys_and_both_match(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {
                "data_description.project_name": "ProjectX",
                "data_description.data_level": "raw",
            },
        )
        assert list(result["name"]) == ["asset-A"]

    def test_multiple_keys_and_no_overlap(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {
                "data_description.project_name": "ProjectX",
                "subject.subject_id": "300",  # 300 belongs to ProjectY
            },
        )
        assert len(result) == 0

    def test_multiple_keys_mixed_operators(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {
                "data_description.project_name": {"$in": ["ProjectY"]},
                "data_description.data_level": "raw",
            },
        )
        assert set(result["name"]) == {"asset-C", "asset-D"}

    # Genotype regex

    def test_genotype_regex_case_insensitive(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {"subject.subject_details.genotype": {"$regex": "cre", "$options": "i"}},
        )
        assert set(result["name"]) == {"asset-B", "asset-D"}

    def test_genotype_exact_equality(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df,
            {"subject.subject_details.genotype": "wt/wt"},
        )
        assert set(result["name"]) == {"asset-A", "asset-C"}

    # Modalities — comma-separated string membership checks

    def test_modality_alias_data_description_modality(self, sample_df):
        # asset-A: "ecephys", asset-C: "ecephys, fMRI"
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.modality": "ecephys"}
        )
        assert set(result["name"]) == {"asset-A", "asset-C"}

    def test_modality_alias_data_description_modalities(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.modalities": "ecephys"}
        )
        assert set(result["name"]) == {"asset-A", "asset-C"}

    def test_modality_equality_does_not_partial_match(self, sample_df):
        # "behavior" should NOT match "behavior-videos"
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.modality": "behavior"}
        )
        assert set(result["name"]) == {"asset-B"}

    def test_modality_equality_multi_modality_row(self, sample_df):
        # asset-C has "ecephys, fMRI" — querying for "fMRI" should match it
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.modality": "fMRI"}
        )
        assert set(result["name"]) == {"asset-C", "asset-D"}

    def test_modality_in_single_value(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.modality": {"$in": ["ecephys"]}}
        )
        assert set(result["name"]) == {"asset-A", "asset-C"}

    def test_modality_in_multiple_values(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.modality": {"$in": ["ecephys", "fMRI"]}}
        )
        assert set(result["name"]) == {"asset-A", "asset-C", "asset-D"}

    def test_modality_in_no_match(self, sample_df):
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.modality": {"$in": ["SPIM"]}}
        )
        assert len(result) == 0

    def test_modality_regex_searches_raw_string(self, sample_df):
        # $regex on modalities operates on the full comma-separated string
        result = _apply_filter_to_dataframe(
            sample_df, {"data_description.modality": {"$regex": "behavior"}}
        )
        # "behavior, behavior-videos" contains "behavior" — asset-B
        assert set(result["name"]) == {"asset-B"}


# ── datetime helpers ─────────────────────────────────────────────────────────


class TestToUtcTimestamp:
    def test_naive_timestamp_gets_localized_to_utc(self):
        ts = _to_utc_timestamp(pd.Timestamp("2024-01-01"))
        assert ts.tzinfo is not None
        assert ts == pd.Timestamp("2024-01-01", tz="UTC")

    def test_utc_aware_timestamp_unchanged(self):
        ts = _to_utc_timestamp(pd.Timestamp("2024-01-01", tz="UTC"))
        assert ts == pd.Timestamp("2024-01-01", tz="UTC")

    def test_non_utc_aware_timestamp_converted(self):
        ts = _to_utc_timestamp(pd.Timestamp("2024-01-01 08:00:00", tz="US/Pacific"))
        assert ts.tzinfo is not None
        assert ts.tz_convert("UTC").hour == 16

    def test_string_input_parsed(self):
        ts = _to_utc_timestamp("2024-06-01")
        assert ts == pd.Timestamp("2024-06-01", tz="UTC")


class TestToUtcSeries:
    def test_tz_aware_strings_parsed(self):
        s = pd.Series(["2024-01-01 00:00:00+00:00", "2024-06-01 12:00:00+00:00"])
        result = _to_utc_series(s)
        assert result.dt.tz is not None
        assert result.iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")

    def test_tz_offset_strings_converted_to_utc(self):
        s = pd.Series(["2024-01-01 08:00:00-08:00"])
        result = _to_utc_series(s)
        assert result.iloc[0] == pd.Timestamp("2024-01-01 16:00:00", tz="UTC")

    def test_invalid_entries_become_nat(self):
        s = pd.Series(["not-a-date", "2024-01-01 00:00:00+00:00"])
        result = _to_utc_series(s)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pd.Timestamp("2024-01-01", tz="UTC")

    def test_already_timestamp_series_handled(self):
        s = pd.Series(pd.to_datetime(["2024-01-01", "2024-06-01"]))
        result = _to_utc_series(s)
        assert result.dt.tz is not None


# ── modality helpers ──────────────────────────────────────────────────────────


class TestModalitySeriesContains:
    def _make(self, values):
        return pd.Series(values)

    def test_single_modality_exact_match(self):
        s = self._make(["ecephys", "behavior", "fMRI"])
        result = _modality_series_contains(s, "ecephys")
        assert list(result) == [True, False, False]

    def test_multi_modality_exact_term_match(self):
        s = self._make(["ecephys, fMRI", "behavior, behavior-videos"])
        result = _modality_series_contains(s, "fMRI")
        assert list(result) == [True, False]

    def test_no_partial_match(self):
        # "behavior" must not match "behavior-videos"
        s = self._make(["behavior-videos", "behavior"])
        result = _modality_series_contains(s, "behavior")
        assert list(result) == [False, True]

    def test_nan_values_return_false(self):
        s = pd.Series([None, "ecephys"])
        result = _modality_series_contains(s, "ecephys")
        assert list(result) == [False, True]

    def test_no_match_returns_all_false(self):
        s = self._make(["ecephys", "behavior"])
        result = _modality_series_contains(s, "SPIM")
        assert list(result) == [False, False]


class TestModalitySeriesContainsAny:
    def _make(self, values):
        return pd.Series(values)

    def test_single_value_set(self):
        s = self._make(["ecephys", "behavior", "fMRI"])
        result = _modality_series_contains_any(s, ["ecephys"])
        assert list(result) == [True, False, False]

    def test_multiple_values_or_logic(self):
        s = self._make(["ecephys", "behavior", "fMRI", "ecephys, fMRI"])
        result = _modality_series_contains_any(s, ["ecephys", "behavior"])
        assert list(result) == [True, True, False, True]

    def test_multi_modality_row_any_match(self):
        s = self._make(["behavior, behavior-videos, ecephys"])
        result = _modality_series_contains_any(s, ["SPIM", "ecephys"])
        assert list(result) == [True]

    def test_empty_values_list_returns_all_false(self):
        s = self._make(["ecephys", "behavior"])
        result = _modality_series_contains_any(s, [])
        assert list(result) == [False, False]

    def test_nan_values_return_false(self):
        s = pd.Series([None, "ecephys"])
        result = _modality_series_contains_any(s, ["ecephys"])
        assert list(result) == [False, True]


# ── _fetch_full_records_batched ───────────────────────────────────────────────


class TestFetchFullRecordsBatched:
    def test_empty_names_returns_empty_without_network_call(self):
        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            result = _fetch_full_records_batched([])
        assert result == []
        mock_cls.assert_not_called()

    def test_single_batch_calls_docdb_once(self):
        names = ["a", "b", "c"]
        fake_records = [{"name": n} for n in names]

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = fake_records
            result = _fetch_full_records_batched(names, batch_size=50)

        assert result == fake_records
        mock_cls.return_value.retrieve_docdb_records.assert_called_once_with(
            filter_query={"name": {"$in": names}}
        )

    def test_multiple_batches_calls_docdb_per_batch(self):
        names = [f"asset-{i}" for i in range(5)]

        def fake_retrieve(filter_query):
            return [{"name": n} for n in filter_query["name"]["$in"]]

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.side_effect = fake_retrieve
            result = _fetch_full_records_batched(names, batch_size=3)

        assert len(result) == 5
        assert mock_cls.return_value.retrieve_docdb_records.call_count == 2
        calls = mock_cls.return_value.retrieve_docdb_records.call_args_list
        assert calls[0] == call(filter_query={"name": {"$in": names[:3]}})
        assert calls[1] == call(filter_query={"name": {"$in": names[3:]}})

    def test_exact_batch_boundary(self):
        names = [f"asset-{i}" for i in range(4)]

        def fake_retrieve(filter_query):
            return [{"name": n} for n in filter_query["name"]["$in"]]

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.side_effect = fake_retrieve
            result = _fetch_full_records_batched(names, batch_size=2)

        assert len(result) == 4
        assert mock_cls.return_value.retrieve_docdb_records.call_count == 2

    def test_one_name_single_batch(self):
        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = [{"name": "solo"}]
            result = _fetch_full_records_batched(["solo"], batch_size=50)

        assert result == [{"name": "solo"}]
        mock_cls.return_value.retrieve_docdb_records.assert_called_once()

    def test_results_from_all_batches_are_combined(self):
        names = ["x", "y", "z"]

        def fake_retrieve(filter_query):
            return [{"name": n, "extra": 1} for n in filter_query["name"]["$in"]]

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.side_effect = fake_retrieve
            result = _fetch_full_records_batched(names, batch_size=2)

        assert [r["name"] for r in result] == ["x", "y", "z"]


# ── retrieve_records ─────────────────────────────────────────────────────────────────


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "name": ["asset-A", "asset-B"],
            "project_name": ["ProjectX", "ProjectY"],
            "modalities": ["ecephys", "behavior"],
            "data_level": ["raw", "derived"],
            "subject_id": ["100", "200"],
            "genotype": ["wt/wt", "wt/Cre"],
            "acquisition_start_time": ["2024-01-01 00:00:00+00:00", "2024-02-01 00:00:00+00:00"],
            "acquisition_end_time": ["2024-01-02 00:00:00+00:00", "2024-02-02 00:00:00+00:00"],
            "process_date": ["2024-01-10 00:00:00+00:00", "2024-02-10 00:00:00+00:00"],
        }
    )


class TestRunQuery:
    # Cache path

    def test_cache_path_names_only(self, small_df):
        with patch("biodata_query.query.asset_basics", return_value=small_df):
            result = retrieve_records({"data_description.project_name": "ProjectX"}, names_only=True)

        assert result.backend == "cache"
        assert result.asset_names == ["asset-A"]
        assert result.records is None

    def test_cache_path_full_records(self, small_df):
        fake_records = [{"name": "asset-A", "data_description": {}}]
        with (
            patch("biodata_query.query.asset_basics", return_value=small_df),
            patch(
                "biodata_query.query._fetch_full_records_batched", return_value=fake_records
            ) as mock_fetch,
        ):
            result = retrieve_records({"data_description.project_name": "ProjectX"}, names_only=False)

        assert result.backend == "cache"
        assert result.asset_names == ["asset-A"]
        assert result.records == fake_records
        mock_fetch.assert_called_once_with(["asset-A"])

    def test_cache_path_empty_query_returns_all(self, small_df):
        with patch("biodata_query.query.asset_basics", return_value=small_df):
            result = retrieve_records({}, names_only=True)

        assert result.backend == "cache"
        assert set(result.asset_names) == {"asset-A", "asset-B"}

    def test_cache_path_no_results_still_calls_fetch(self, small_df):
        with (
            patch("biodata_query.query.asset_basics", return_value=small_df),
            patch(
                "biodata_query.query._fetch_full_records_batched", return_value=[]
            ) as mock_fetch,
        ):
            result = retrieve_records({"data_description.project_name": "NoProject"}, names_only=False)

        assert result.asset_names == []
        assert result.records == []
        mock_fetch.assert_called_once_with([])

    # DocDB path

    def test_docdb_path_names_only(self):
        query = {"data_description.institution": "AIND"}  # not in FIELD_TO_COLUMN
        fake_raw = [{"name": "asset-X"}, {"name": "asset-Y"}]

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = fake_raw
            result = retrieve_records(query, names_only=True)

        assert result.backend == "docdb"
        assert result.asset_names == ["asset-X", "asset-Y"]
        assert result.records is None
        mock_cls.return_value.retrieve_docdb_records.assert_called_once_with(
            filter_query=query, projection={"name": 1}
        )

    def test_docdb_path_full_records(self):
        query = {"data_description.institution": "AIND"}
        fake_raw = [{"name": "asset-X", "field": 1}, {"name": "asset-Y", "field": 2}]

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = fake_raw
            result = retrieve_records(query, names_only=False)

        assert result.backend == "docdb"
        assert result.asset_names == ["asset-X", "asset-Y"]
        assert result.records == fake_raw
        mock_cls.return_value.retrieve_docdb_records.assert_called_once_with(filter_query=query)

    def test_docdb_path_unsupported_operator_routes_to_docdb(self):
        query = {"name": {"$elemMatch": {"x": 1}}}  # unsupported op → docdb

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = []
            result = retrieve_records(query, names_only=True)

        assert result.backend == "docdb"

    # QueryResult structure

    def test_result_is_query_result_instance(self, small_df):
        with patch("biodata_query.query.asset_basics", return_value=small_df):
            result = retrieve_records({}, names_only=True)

        assert isinstance(result, QueryResult)

    def test_elapsed_seconds_is_non_negative(self, small_df):
        with patch("biodata_query.query.asset_basics", return_value=small_df):
            result = retrieve_records({}, names_only=True)

        assert result.elapsed_seconds >= 0

    def test_names_only_false_records_not_none_for_cache(self, small_df):
        with (
            patch("biodata_query.query.asset_basics", return_value=small_df),
            patch("biodata_query.query._fetch_full_records_batched", return_value=[{"name": "asset-A"}]),
        ):
            result = retrieve_records({"name": "asset-A"}, names_only=False)

        assert result.records is not None

    # limit parameter

    def test_limit_applied_on_cache_path(self, small_df):
        with patch("biodata_query.query.asset_basics", return_value=small_df):
            result = retrieve_records({}, names_only=True, limit=1)

        assert result.backend == "cache"
        assert len(result.asset_names) == 1

    def test_limit_zero_means_no_limit_on_cache_path(self, small_df):
        with patch("biodata_query.query.asset_basics", return_value=small_df):
            result = retrieve_records({}, names_only=True, limit=0)

        assert len(result.asset_names) == 2  # both rows in small_df

    def test_limit_passed_to_docdb_names_only(self):
        query = {"data_description.institution": "AIND"}
        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = [{"name": "x"}]
            retrieve_records(query, names_only=True, limit=10)

        mock_cls.return_value.retrieve_docdb_records.assert_called_once_with(
            filter_query=query, projection={"name": 1}, limit=10
        )

    def test_limit_passed_to_docdb_full_records(self):
        query = {"data_description.institution": "AIND"}
        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = [{"name": "x"}]
            retrieve_records(query, names_only=False, limit=5)

        mock_cls.return_value.retrieve_docdb_records.assert_called_once_with(
            filter_query=query, limit=5
        )

    # projection parameter — cache path

    def test_cache_path_cache_servable_projection_skips_fetch(self, small_df):
        projection = {
            "name": 1,
            "data_description.project_name": 1,
        }
        with (
            patch("biodata_query.query.asset_basics", return_value=small_df),
            patch("biodata_query.query._fetch_full_records_batched") as mock_fetch,
        ):
            result = retrieve_records(
                {"data_description.project_name": "ProjectX"},
                names_only=False,
                projection=projection,
            )

        mock_fetch.assert_not_called()
        assert result.backend == "cache"
        assert result.asset_names == ["asset-A"]
        assert result.records is None
        assert result.dataframe is not None
        assert list(result.dataframe["name"]) == ["asset-A"]

    def test_cache_path_non_cache_servable_projection_does_fetch(self, small_df):
        projection = {"name": 1, "data_description.institution": 1}  # institution not in FIELD_TO_COLUMN
        fake_records = [{"name": "asset-A"}]
        with (
            patch("biodata_query.query.asset_basics", return_value=small_df),
            patch(
                "biodata_query.query._fetch_full_records_batched", return_value=fake_records
            ) as mock_fetch,
        ):
            result = retrieve_records(
                {"data_description.project_name": "ProjectX"},
                names_only=False,
                projection=projection,
            )

        mock_fetch.assert_called_once_with(["asset-A"])
        assert result.records == fake_records
        assert result.dataframe is None

    def test_cache_path_none_projection_does_fetch(self, small_df):
        """projection=None (default) is treated as non-cache-servable for safety."""
        fake_records = [{"name": "asset-A"}]
        with (
            patch("biodata_query.query.asset_basics", return_value=small_df),
            patch(
                "biodata_query.query._fetch_full_records_batched", return_value=fake_records
            ) as mock_fetch,
        ):
            result = retrieve_records(
                {"data_description.project_name": "ProjectX"},
                names_only=False,
                projection=None,
            )

        mock_fetch.assert_called_once()
        assert result.dataframe is None

    def test_cache_path_names_only_ignores_projection(self, small_df):
        """names_only=True never fetches or returns a dataframe regardless of projection."""
        projection = {"name": 1, "data_description.project_name": 1}
        with (
            patch("biodata_query.query.asset_basics", return_value=small_df),
            patch("biodata_query.query._fetch_full_records_batched") as mock_fetch,
        ):
            result = retrieve_records({}, names_only=True, projection=projection)

        mock_fetch.assert_not_called()
        assert result.records is None
        assert result.dataframe is None

    # projection parameter — docdb path

    def test_docdb_path_projection_passed_to_api(self):
        query = {"data_description.institution": "AIND"}
        projection = {"name": 1, "data_description.institution": 1}
        fake_raw = [{"name": "asset-X"}]

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = fake_raw
            retrieve_records(query, names_only=False, projection=projection)

        mock_cls.return_value.retrieve_docdb_records.assert_called_once_with(
            filter_query=query, projection=projection
        )

    def test_docdb_path_none_projection_not_passed_to_api(self):
        query = {"data_description.institution": "AIND"}
        fake_raw = [{"name": "asset-X"}]

        with patch("biodata_query.query.MetadataDbClient") as mock_cls:
            mock_cls.return_value.retrieve_docdb_records.return_value = fake_raw
            retrieve_records(query, names_only=False, projection=None)

        mock_cls.return_value.retrieve_docdb_records.assert_called_once_with(
            filter_query=query
        )


# ── _projection_is_cache_servable ─────────────────────────────────────────────


class TestProjectionIsCacheServable:
    def test_none_returns_false(self):
        assert _projection_is_cache_servable(None) is False

    def test_empty_dict_returns_true(self):
        assert _projection_is_cache_servable({}) is True

    def test_all_known_fields_returns_true(self):
        projection = {field: 1 for field in FIELD_TO_COLUMN}
        assert _projection_is_cache_servable(projection) is True

    def test_single_known_field_returns_true(self):
        assert _projection_is_cache_servable({"name": 1}) is True

    def test_unknown_field_returns_false(self):
        assert _projection_is_cache_servable({"data_description.institution": 1}) is False

    def test_mixed_known_and_unknown_returns_false(self):
        assert _projection_is_cache_servable({"name": 1, "data_description.institution": 1}) is False

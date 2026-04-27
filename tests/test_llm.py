"""Tests for the LLM query endpoint (Step 3)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from biodata_query.llm.agent import MAX_RETRIES, _extract_json, build_query
from biodata_query.llm.endpoint import handle_get_query


# ── _extract_json ─────────────────────────────────────────────────────────────


class TestExtractJson:
    def test_bare_json(self):
        assert _extract_json('{"foo": "bar"}') == {"foo": "bar"}

    def test_markdown_json_block(self):
        text = '```json\n{"key": 1}\n```'
        assert _extract_json(text) == {"key": 1}

    def test_markdown_block_no_lang(self):
        text = "```\n{\"x\": true}\n```"
        assert _extract_json(text) == {"x": True}

    def test_json_embedded_in_prose(self):
        text = 'Here is the query: {"name": "test"} — enjoy!'
        assert _extract_json(text) == {"name": "test"}

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("no json here at all")


# ── build_query ───────────────────────────────────────────────────────────────


def _make_bedrock_response(text: str) -> dict:
    """Build a minimal converse() response dict."""
    return {"output": {"message": {"content": [{"text": text}]}}}


class TestBuildQuery:
    def test_success_on_first_attempt(self):
        query_json = '{"data_description.data_level": "raw"}'
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = _make_bedrock_response(query_json)

        with (
            patch("biodata_query.llm.agent.boto3.client", return_value=mock_bedrock),
            patch("biodata_query.llm.agent.retrieve_records") as mock_run,
        ):
            mock_run.return_value = MagicMock(asset_names=[])
            result = build_query({}, "find all raw data")

        assert result == {"data_description.data_level": "raw"}
        mock_run.assert_called_once_with({"data_description.data_level": "raw"}, names_only=True)

    def test_retry_on_invalid_json_then_success(self):
        bad_response = _make_bedrock_response("Sorry, here is the query: oops no json")
        good_json = '{"name": "test-asset"}'
        good_response = _make_bedrock_response(good_json)

        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = [bad_response, good_response]

        with (
            patch("biodata_query.llm.agent.boto3.client", return_value=mock_bedrock),
            patch("biodata_query.llm.agent.retrieve_records") as mock_run,
        ):
            mock_run.return_value = MagicMock(asset_names=[])
            result = build_query({}, "find test-asset")

        assert result == {"name": "test-asset"}
        assert mock_bedrock.converse.call_count == 2

    def test_retry_on_query_execution_error_then_success(self):
        bad_json = '{"$badop": "value"}'
        good_json = '{"name": "fixed"}'

        mock_bedrock = MagicMock()
        mock_bedrock.converse.side_effect = [
            _make_bedrock_response(bad_json),
            _make_bedrock_response(good_json),
        ]

        with (
            patch("biodata_query.llm.agent.boto3.client", return_value=mock_bedrock),
            patch("biodata_query.llm.agent.retrieve_records") as mock_run,
        ):
            mock_run.side_effect = [ValueError("invalid operator"), MagicMock(asset_names=[])]
            result = build_query({}, "find something")

        assert result == {"name": "fixed"}
        assert mock_bedrock.converse.call_count == 2

    def test_raises_after_max_retries(self):
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = _make_bedrock_response("not json at all :(")

        with (
            patch("biodata_query.llm.agent.boto3.client", return_value=mock_bedrock),
        ):
            with pytest.raises(RuntimeError, match="Failed to build a valid query"):
                build_query({}, "give me something")

        assert mock_bedrock.converse.call_count == MAX_RETRIES

    def test_passes_current_query_in_message(self):
        current = {"data_description.data_level": "raw"}
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = _make_bedrock_response(
            '{"data_description.data_level": "raw", "name": "extra"}'
        )

        with (
            patch("biodata_query.llm.agent.boto3.client", return_value=mock_bedrock),
            patch("biodata_query.llm.agent.retrieve_records") as mock_run,
        ):
            mock_run.return_value = MagicMock(asset_names=[])
            build_query(current, "add name filter")

        call_args = mock_bedrock.converse.call_args
        user_text = call_args.kwargs["messages"][0]["content"][0]["text"]
        assert json.dumps(current) in user_text
        assert "add name filter" in user_text

    def test_markdown_json_block_parsed(self):
        query_wrapped = "```json\n{\"subject.subject_id\": \"12345\"}\n```"
        mock_bedrock = MagicMock()
        mock_bedrock.converse.return_value = _make_bedrock_response(query_wrapped)

        with (
            patch("biodata_query.llm.agent.boto3.client", return_value=mock_bedrock),
            patch("biodata_query.llm.agent.retrieve_records") as mock_run,
        ):
            mock_run.return_value = MagicMock(asset_names=[])
            result = build_query({}, "find subject 12345")

        assert result == {"subject.subject_id": "12345"}


# ── handle_get_query ──────────────────────────────────────────────────────────


class TestHandleGetQuery:
    def test_missing_message_returns_400(self):
        event = {"queryStringParameters": {"query": "{}"}}
        response = handle_get_query(event)
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "message" in body["error"]

    def test_no_query_string_parameters(self):
        event = {}
        response = handle_get_query(event)
        assert response["statusCode"] == 400

    def test_invalid_json_in_query_param(self):
        event = {"queryStringParameters": {"query": "not-json", "message": "hello"}}
        response = handle_get_query(event)
        assert response["statusCode"] == 400
        body = json.loads(response["body"])
        assert "Invalid JSON" in body["error"]

    def test_success(self):
        new_q = {"data_description.data_level": "raw"}
        event = {
            "queryStringParameters": {
                "query": "{}",
                "message": "give me raw data",
            }
        }
        with patch("biodata_query.llm.endpoint.build_query", return_value=new_q) as mock_bq:
            response = handle_get_query(event)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["query"] == new_q
        mock_bq.assert_called_once_with({}, "give me raw data")

    def test_build_query_exception_returns_500(self):
        event = {
            "queryStringParameters": {
                "message": "find something",
            }
        }
        with patch(
            "biodata_query.llm.endpoint.build_query",
            side_effect=RuntimeError("max retries exceeded"),
        ):
            response = handle_get_query(event)

        assert response["statusCode"] == 500
        body = json.loads(response["body"])
        assert "max retries exceeded" in body["error"]

    def test_default_empty_query_when_query_param_absent(self):
        new_q = {"name": "foo"}
        event = {"queryStringParameters": {"message": "find foo"}}
        with patch("biodata_query.llm.endpoint.build_query", return_value=new_q) as mock_bq:
            response = handle_get_query(event)

        assert response["statusCode"] == 200
        mock_bq.assert_called_once_with({}, "find foo")

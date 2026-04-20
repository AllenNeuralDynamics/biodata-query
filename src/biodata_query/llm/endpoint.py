"""GET endpoint handler for LLM-powered query building."""

from __future__ import annotations

import json
import logging

from biodata_query.llm.agent import build_query

logger = logging.getLogger(__name__)


def handle_get_query(event: dict) -> dict:
    """Handle a GET /get-query request.

    Query string parameters
    -----------------------
    query : str, optional
        JSON-encoded current query dict.  Defaults to ``"{}"``.
    message : str
        The user's natural-language instruction.

    Returns
    -------
    dict
        An API-Gateway-compatible response dict with ``statusCode`` and ``body``.
    """
    params: dict = (event.get("queryStringParameters") or {})

    raw_query = params.get("query", "{}")
    message = params.get("message", "")

    if not message:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "message parameter is required"}),
        }

    try:
        current_query = json.loads(raw_query)
    except json.JSONDecodeError as exc:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Invalid JSON in query parameter: {exc}"}),
        }

    _ERROR_RESPONSES: dict[str, tuple[int, str]] = {
        "not_possible": (
            422,
            "Cannot construct a filter query to meet the requirements.",
        ),
        "unclear": (400, "Message did not make sense as a data query."),
    }

    try:
        envelope = build_query(current_query, message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("handle_get_query failed")
        return {
            "statusCode": 500,
            "body": str(exc),
        }

    if envelope.get("status") == "error":
        code = envelope.get("code", "")
        status_code, message_text = _ERROR_RESPONSES.get(
            code, (422, "Query could not be fulfilled.")
        )
        return {"statusCode": status_code, "body": message_text}

    return {
        "statusCode": 200,
        "body": json.dumps({"query": envelope.get("query", {})}),
    }

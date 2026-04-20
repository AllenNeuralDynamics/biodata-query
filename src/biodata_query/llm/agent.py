"""LLM-powered query building with Bedrock and a validation retry loop."""

from __future__ import annotations

import json
import logging
import re

import boto3

from biodata_query.llm.prompt import SYSTEM_PROMPT
from biodata_query.query import run_query

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
MAX_RETRIES = 3

# Matches a JSON object inside an optional markdown code block
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BARE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Extract and parse the first JSON object from *text*."""
    # Try markdown code block first
    m = _JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(1))
    # Fall back to the first {...} span
    m = _JSON_BARE_RE.search(text)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("No JSON object found", text, 0)


def build_query(current_query: dict, user_message: str) -> dict:
    """Send the current query + user message to Bedrock and return a
    response envelope.

    Validates the returned query by running it (names_only=True). On
    failure the error is fed back to the model for up to MAX_RETRIES
    attempts.

    Parameters
    ----------
    current_query:
        The query dict currently displayed in the UI (may be empty).
    user_message:
        The natural-language instruction from the user.

    Returns
    -------
    dict
        One of:
        - ``{"status": "ok", "query": <MongoDB filter dict>}``
        - ``{"status": "error", "code": "not_possible"}``
        - ``{"status": "error", "code": "unclear"}``

    Raises
    ------
    RuntimeError
        If a valid query cannot be obtained within MAX_RETRIES attempts.
    """
    bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

    user_content = (
        f"Current query: {json.dumps(current_query)}\n"
        f"User request: {user_message}\n\n"
        "Return ONLY the JSON envelope described in the system prompt."
    )
    messages: list[dict] = [{"role": "user", "content": [{"text": user_content}]}]

    for attempt in range(MAX_RETRIES):
        logger.debug("build_query attempt %d/%d", attempt + 1, MAX_RETRIES)
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
        )
        assistant_text: str = response["output"]["message"]["content"][0]["text"]

        # Append assistant turn so subsequent retries have full context
        messages.append({"role": "assistant", "content": [{"text": assistant_text}]})

        # --- Parse JSON envelope ---
        try:
            envelope = _extract_json(assistant_text)
        except json.JSONDecodeError as exc:
            logger.warning("Attempt %d: JSON parse error: %s", attempt + 1, exc)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "That was not valid JSON. Return ONLY"
                                " the JSON envelope, with no extra text."
                            )
                        }
                    ],
                }
            )
            continue

        # --- Error envelopes are returned immediately (no retry) ---
        if envelope.get("status") == "error":
            logger.debug(
                "build_query returning error envelope: %s",
                envelope.get("code"),
            )
            return envelope

        # --- Validate the query inside an ok envelope ---
        new_query = envelope.get("query", {})
        try:
            run_query(new_query, names_only=True)
            logger.debug("build_query succeeded on attempt %d", attempt + 1)
            return envelope
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Attempt %d: query execution error: %s", attempt + 1, exc
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "The query inside your envelope produced an"
                                f" error when executed: {exc}. "
                                "Fix the query and return ONLY the JSON"
                                " envelope."
                            )
                        }
                    ],
                }
            )
            continue

    raise RuntimeError(
        f"Failed to build a valid query after {MAX_RETRIES} attempts"
    )

# biodata-query Implementation Plan

## Overview

This plan covers migrating the repo to uv/ruff, building a query execution library, Panel-based UI components, and a Bedrock-powered LLM query endpoint.

---

## Step 1: Core Query Library — `src/biodata_query/query.py`

This is the central module. It takes a MongoDB-style query dictionary and routes it through the correct backend.

### 1a. Determine cache-eligible fields

The `asset_basics` table from zombie-squirrel has these columns:
- `_id`, `_last_modified`, `modalities`, `project_name`, `data_level`, `subject_id`, `acquisition_start_time`, `acquisition_end_time`, `code_ocean`, `process_date`, `genotype`, `location`, `name`

A query is **cache-eligible** if every top-level key in the filter dictionary maps to one of these columns. The mapping from MongoDB document paths to `asset_basics` columns is:

| MongoDB field path | asset_basics column |
|---|---|
| `name` | `name` |
| `data_description.project_name` | `project_name` |
| `data_description.modality` / `data_description.modalities` | `modalities` |
| `data_description.data_level` | `data_level` |
| `subject.subject_id` | `subject_id` |
| `subject.subject_details.genotype` | `genotype` |
| `acquisition.acquisition_start_time` | `acquisition_start_time` |
| `acquisition.acquisition_end_time` | `acquisition_end_time` |
| `data_description.process_date` or top-level `process_date` | `process_date` |

Note: Some of these field mappings (e.g. `data_description.modality` vs `data_description.modalities`) should be documented and may need to handle both v1 and v2 schema paths. The implementation should try to be flexible.

### 1b. Module: `src/biodata_query/query.py`

```python
"""Query execution engine with cache-aware routing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

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
    "data_description.data_level": "data_level",
    "subject.subject_id": "subject_id",
    "subject.subject_details.genotype": "genotype",
    "acquisition.acquisition_start_time": "acquisition_start_time",
    "acquisition.acquisition_end_time": "acquisition_end_time",
    "process_date": "process_date",
}


@dataclass
class QueryResult:
    """Result of a query execution."""
    backend: Literal["cache", "docdb"]
    elapsed_seconds: float
    asset_names: list[str]
    records: list[dict] | None  # None if names_only=True


def is_cache_eligible(query: dict) -> bool:
    """Check if all top-level keys in the query map to asset_basics columns."""
    ...


def _apply_filter_to_dataframe(df: pd.DataFrame, query: dict) -> pd.DataFrame:
    """Translate a MongoDB-style filter dict into pandas DataFrame operations."""
    # Handle simple equality, $regex, $in, $gte/$lte/$gt/$lt for dates, etc.
    # Only needs to support the subset of MongoDB operators that make sense
    # for the cached columns.
    ...


def _fetch_full_records_batched(names: list[str], batch_size: int = 50) -> list[dict]:
    """Fetch full records from DocDB by batching $in queries on the name field."""
    client = MetadataDbClient(host=API_GATEWAY_HOST)
    records = []
    for i in range(0, len(names), batch_size):
        batch = names[i : i + batch_size]
        batch_records = client.retrieve_docdb_records(
            filter_query={"name": {"$in": batch}}
        )
        records.extend(batch_records)
    return records


def run_query(query: dict, names_only: bool = False) -> QueryResult:
    """Execute a query, routing through cache or DocDB as appropriate."""
    start = time.time()

    if is_cache_eligible(query):
        df = asset_basics()
        filtered = _apply_filter_to_dataframe(df, query)
        names = filtered["name"].tolist()
        records = None
        if not names_only:
            records = _fetch_full_records_batched(names)
        backend = "cache"
    else:
        client = MetadataDbClient(host=API_GATEWAY_HOST)
        if names_only:
            raw = client.retrieve_docdb_records(
                filter_query=query, projection={"name": 1}
            )
            names = [r["name"] for r in raw]
            records = None
        else:
            raw = client.retrieve_docdb_records(filter_query=query)
            names = [r["name"] for r in raw]
            records = raw
        backend = "docdb"

    elapsed = time.time() - start
    return QueryResult(
        backend=backend,
        elapsed_seconds=elapsed,
        asset_names=names,
        records=records,
    )
```

### 1c. Key implementation details for `_apply_filter_to_dataframe`

This function must handle a reasonable subset of MongoDB query operators when applied to pandas:

- **Simple equality**: `{"name": "foo"}` → `df[df["name"] == "foo"]`
- **`$in`**: `{"project_name": {"$in": ["A", "B"]}}` → `df[df["project_name"].isin(["A", "B"])]`
- **`$regex`**: `{"name": {"$regex": "pattern"}}` → `df[df["name"].str.contains("pattern", case=..., na=False)]` (respect `$options: "i"` for case-insensitive)
- **`$gte`, `$lte`, `$gt`, `$lt`**: For datetime columns like `acquisition_start_time` → standard comparison operators
- **`$and` / implicit AND**: Multiple top-level keys are ANDed together (standard MongoDB behavior)

Operators we do NOT need to support in the cache path: `$or`, `$not`, `$elemMatch`, `$exists`, aggregation pipelines. If a query contains unsupported operators, `is_cache_eligible` should return `False`.

### 1d. Tests: `tests/test_query.py`

- Mock `asset_basics()` to return a small DataFrame
- Mock `MetadataDbClient.retrieve_docdb_records`
- Test `is_cache_eligible` with various query dicts (positive and negative cases)
- Test `_apply_filter_to_dataframe` with each supported operator
- Test `run_query` end-to-end with both cache and docdb paths
- Test `_fetch_full_records_batched` batching logic
- Test `names_only=True` vs `names_only=False` for both backends

---

## Step 2: Panel UI Components — `src/biodata_query/panel/`

These live in a subpackage that is only importable when the `panel` dependency group is installed.

### 2a. Module: `src/biodata_query/panel/__init__.py`

```python
from biodata_query.panel.builder import QueryBuilder
from biodata_query.panel.results import QueryResults
```

### 2b. QueryBuilder component — `src/biodata_query/panel/builder.py`

A `PyComponent` (subclass of `pn.custom.PyComponent`) that provides:

**UI Elements:**
- `project_name`: `Select` widget, populated from `unique_project_names()` (from zombie_squirrel), with a blank/"All" option
- `modality`: `MultiSelect` widget, populated from known modality abbreviations
- `data_level`: `Select` widget, options: `["raw", "derived"]` plus blank
- `subject_id`: `AutocompleteInput` widget, populated from `unique_subject_ids()` (or a text input)
- `genotype`: `TextInput` widget (free text, will use `$regex`)
- `name`: `TextInput` widget (free text, will use `$regex`)
- `acquisition_start_time_min` / `acquisition_start_time_max`: `DatetimePicker` widgets
  - These become enabled/interactive once any other filter is set
  - When enabled, their default min/max values are computed from the filtered `asset_basics()` cache
- `process_date`: `DatetimePicker` or `DatePicker`
- `query_dict`: `TextAreaInput` at the bottom, showing the current query as JSON
  - This is **bidirectional**: when the user edits the text, the query updates; when widgets change, the text updates
  - Include a "Copy" button (or rely on browser text selection)
- A `Run Query` button that triggers execution via `run_query()`

**Behavior:**
- Widget changes → rebuild the query dict → update the text field
- Text field edits → parse JSON → update widget states (best-effort)
- The time pickers should react to other widget changes: when the user selects a project/modality/etc., filter `asset_basics()` by those criteria, then set the min/max range on the time pickers to the range found in the filtered data
- Expose `query` as a `param.Dict` that other components can watch

### 2c. QueryResults component — `src/biodata_query/panel/results.py`

A `PyComponent` that provides:

**UI Elements:**
- A header row showing:
  - Backend used: "cache" or "docdb"
  - Query execution time
  - Number of results
- A `Tabulator` widget displaying the query results as a table
  - When `names_only=True`, just shows asset names
  - When full records are returned, flatten and display key columns

**Behavior:**
- Watches the `QueryBuilder.query` param (or accepts a query dict)
- On query change (or button press), calls `run_query()` and displays results

### 2d. Tests: `tests/test_panel.py`

- Test that `QueryBuilder` can be instantiated and produces valid query dicts
- Test that widget changes produce expected query dictionaries
- Test that `QueryResults` renders with mock data
- Mock all external calls (zombie_squirrel, aind-data-access-api)

---

## Step 3: LLM Query Endpoint — `src/biodata_query/llm/`

### 3a. System prompt — `src/biodata_query/llm/prompt.py`

Create a condensed system prompt (much shorter than the one in `aind-chat-query-builder`). It should include:

1. **Role**: "You are a MongoDB query builder for the AIND metadata database."
2. **Schema overview**: A compact description of the document structure (just field paths and types, NOT full example documents). Something like:
   ```
   Top-level fields: name, location, _id, created, last_modified, schema_version
   data_description: .project_name, .modality[].abbreviation, .data_level ("raw"/"derived"), .subject_id, .creation_time, .institution, .funding_source[], .investigators[], .group
   subject: .subject_id, .subject_details.sex, .subject_details.date_of_birth, .subject_details.genotype, .subject_details.species.name, .subject_details.strain.name
   acquisition: .acquisition_start_time, .acquisition_end_time, .instrument_id, .acquisition_type, .experimenters[], .data_streams[].modalities[]
   procedures: .subject_id, .subject_procedures[].procedures[].procedure_type, .subject_procedures[].procedures[].targeted_structure
   processing: .data_processes[].process_type, .data_processes[].name, .data_processes[].code.url
   quality_control: .metrics[].name, .metrics[].modality, .metrics[].stage, .metrics[].status_history[].status
   ```
3. **Modality list**: The 14 modalities with name and abbreviation (compact table)
4. **Rules**:
   - Output ONLY a valid JSON dictionary representing a MongoDB filter query
   - Use `$regex` with `$options: "i"` when unsure about exact field values
   - Do NOT invent field names; only use documented paths
   - Do NOT produce aggregation pipelines (only filter queries)
   - For date ranges use `$gte`/`$lte`
5. **Few-shot examples** (3-4 compact examples):
   ```
   User: "Find all behavior data for subject 730945"
   Assistant: {"subject.subject_id": "730945", "data_description.modalities": {"$elemMatch": {"abbreviation": "behavior"}}}
   ```

### 3b. Bedrock agent — `src/biodata_query/llm/agent.py`

Use `boto3` directly with `client.converse()` (Bedrock Runtime). No langchain.

```python
"""LLM-powered query building with validation loop."""

import json
import boto3

from biodata_query.llm.prompt import SYSTEM_PROMPT
from biodata_query.query import run_query

BEDROCK_MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"
MAX_RETRIES = 3


def build_query(current_query: dict, user_message: str) -> dict:
    """
    Send the current query + user message to Bedrock, get back a new query dict.
    Validates by running the query (names_only=True). On error, feeds the error
    back to the model for retry.
    """
    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    
    messages = []
    
    # Build the user message including current state
    user_content = f"""Current query: {json.dumps(current_query)}
User request: {user_message}

Return ONLY a valid JSON dictionary representing the updated MongoDB filter query."""

    messages.append({"role": "user", "content": [{"text": user_content}]})

    for attempt in range(MAX_RETRIES):
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
        )
        
        assistant_text = response["output"]["message"]["content"][0]["text"]
        
        # Parse the response as JSON
        try:
            new_query = json.loads(assistant_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            # ... extraction logic ...
            # If still fails, ask model to fix
            messages.append({"role": "assistant", "content": [{"text": assistant_text}]})
            messages.append({"role": "user", "content": [{"text": "That was not valid JSON. Return ONLY a JSON dictionary."}]})
            continue
        
        # Validate by running the query
        try:
            result = run_query(new_query, names_only=True)
            return new_query  # Success
        except Exception as e:
            # Feed error back to model
            messages.append({"role": "assistant", "content": [{"text": assistant_text}]})
            messages.append({"role": "user", "content": [{"text": f"That query produced an error: {e}. Fix the query and return ONLY valid JSON."}]})
            continue
    
    raise RuntimeError(f"Failed to build valid query after {MAX_RETRIES} attempts")
```

### 3c. REST endpoint — `src/biodata_query/llm/endpoint.py`

A minimal function intended to be deployed behind API Gateway / Lambda or similar:

```python
"""GET endpoint handler for LLM query building."""

import json
from biodata_query.llm.agent import build_query


def handle_get_query(event: dict) -> dict:
    """
    Handle a GET /get-query request.
    
    Query parameters:
      - query: JSON string of current query dict (default: "{}")
      - message: user's natural language message
    
    Returns:
      JSON response with the new query dict.
    """
    params = event.get("queryStringParameters", {}) or {}
    current_query = json.loads(params.get("query", "{}"))
    message = params.get("message", "")
    
    if not message:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "message parameter is required"}),
        }
    
    try:
        new_query = build_query(current_query, message)
        return {
            "statusCode": 200,
            "body": json.dumps({"query": new_query}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
```

### 3d. Add dependencies

Add to `pyproject.toml`:
```toml
[dependency-groups]
llm = [
    "boto3>=1.35,<2",
]
```

### 3e. Tests: `tests/test_llm.py`

- Mock `boto3.client("bedrock-runtime")` and its `converse()` method
- Test `build_query` with a mocked successful response
- Test the retry loop: first response is invalid JSON, second succeeds
- Test the retry loop: first response parses but query execution fails, second succeeds
- Test max retries exceeded raises `RuntimeError`
- Test `handle_get_query` with valid/invalid parameters

---

## Step 4: Update pyproject.toml dependencies (final)

```toml
[project]
name = "biodata-query"
version = "0.0.0"
description = "Query builder for AIND metadata with cache-aware routing and LLM support"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [{name = "Allen Institute for Neural Dynamics"}]
classifiers = ["Programming Language :: Python :: 3"]
readme = "README.md"

dependencies = [
    "aind-data-schema>=2.6.0,<3",
    "zombie-squirrel>=0.16.4,<1",
    "aind-data-access-api[docdb]>=1.9.4,<2",
]

[dependency-groups]
dev = [
    "ruff",
    "pytest",
    "pytest-cov",
    "Sphinx",
    "furo",
]
panel = [
    "panel>=1.8.10,<2",
]
llm = [
    "boto3>=1.35,<2",
]
```

---

## Step 5: File structure (final)

```
src/
  biodata_query/
    __init__.py          # version
    query.py             # Step 1: core query engine
    panel/
      __init__.py        # re-exports QueryBuilder, QueryResults
      builder.py         # Step 2b: QueryBuilder PyComponent
      results.py         # Step 2c: QueryResults PyComponent
    llm/
      __init__.py
      prompt.py          # Step 3a: system prompt
      agent.py           # Step 3b: Bedrock agent with retry
      endpoint.py        # Step 3c: REST handler
tests/
  __init__.py
  test_query.py          # Step 1d
  test_panel.py          # Step 2d
  test_llm.py            # Step 3e
```

---

## Execution Order

| Step | Description | Depends on |
|------|-------------|------------|
| 0a-d | Migrate to uv + ruff | — |
| 1    | Core query library | Step 0 |
| 1d   | Query library tests | Step 1 |
| 2    | Panel UI components | Step 1 |
| 2d   | Panel tests | Step 2 |
| 3a   | LLM system prompt | Step 1 |
| 3b-c | LLM agent + endpoint | Step 3a |
| 3e   | LLM tests | Step 3b-c |
| 4    | Final pyproject.toml cleanup | All |

Steps 2 and 3 are independent of each other and can be done in parallel after Step 1.

---

## Notes

- The old `aind-chat-query-builder` repo used langchain's `ChatBedrockConverse` with tool-binding. We replace this with direct `boto3 client.converse()` calls — simpler and fewer dependencies.
- The old repo had separate v1/v2 schema prompts with massive JSON examples. Our new prompt will be compact (field paths only, not full documents) to reduce token usage and improve response speed.
- The validation loop (run query → check for errors → retry) makes this semi-agentic without needing a framework. The `converse()` API's built-in message history makes multi-turn straightforward.
- For the Panel components, we use `PyComponent` (Panel >= 1.0) rather than `Viewer` or older patterns.
- `asset_basics` returns a pandas DataFrame, so cache-path filtering is done with pandas operations, not MongoDB.

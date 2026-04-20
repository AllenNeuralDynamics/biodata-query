# biodata-query

This package supports the creation and deployment of fast queries into metadata records stored at AIND using [aind-data-schema](https://github.com/AllenNeuralDynamics/aind-data-schema).

### Run a query

## Panel apps

`uv add biodata-query --group dev`

### Builder app

The builder app exposes a menu that allows you to select cached fields to construct a query that is guaranteed to run through the cache. The chat menu in the builder app hits the 

### Result app

## Python library

The python library is the backend that accepts query dictionaries and runs them through the cache or sends them to the document database.

`uv add biodata-query`

```
from biodata_query.query import run_query

result = run_query(dict)
```


## /get-query endpoint

The endpoint handler (`biodata_query.llm.endpoint.handle_get_query`) follows the AWS Lambda / API Gateway proxy-integration contract — it accepts an `event` dict with a `queryStringParameters` key and returns a `{"statusCode": ..., "body": ...}` dict.

**Lambda + API Gateway (recommended):**

1. Package the project with its `llm` dependency group and upload to Lambda.
2. Set the Lambda handler to `biodata_query.llm.endpoint.handle_get_query`.
3. Attach an API Gateway HTTP GET route (`/get-query`) with Lambda proxy integration.
4. Grant the Lambda execution role `bedrock:InvokeModel` on `us.anthropic.claude-sonnet-4-20250514-v1:0` in `us-west-2`.

Query string parameters accepted by the deployed endpoint:

| Parameter | Required | Description |
|---|---|---|
| `message` | yes | Natural-language instruction |
| `query` | no | JSON-encoded current query dict (default `{}`) |

## Development

### Running tests

```bash
uv run pytest
```

To include the Panel UI tests, install the panel dependency group first:

```bash
uv sync --group panel --group dev
uv run pytest
```

### Launching the Panel app locally

Install the `panel` dependency group, then serve the demo script:

```bash
uv sync --group panel
uv run panel serve scripts/panel_demo.py --show --autoreload
```

`--show` opens a browser tab automatically; `--autoreload` restarts the server whenever source files change.

The app places the `QueryBuilder` in the sidebar and `QueryResults` in the main area. Select filters and click **Run Query** to execute.

### Running integration tests

Integration tests make real network calls (zombie-squirrel + AIND DocDB) and are kept outside the pytest suite:

```bash
uv run python scripts/integration_query.py
```

### Debugging the LLM /get-query endpoint locally

The LLM endpoint requires AWS credentials that can reach Bedrock in `us-west-2`.  Set them in your environment (e.g. via `aws sso login` or by exporting `AWS_PROFILE`) before starting the server.

**1. Install the `llm` dependency group:**

```bash
uv sync --group llm
```

**2. Start the local HTTP server** (keep this terminal open):

```bash
uv run python scripts/llm_server.py          # default port 8765
uv run python scripts/llm_server.py --port 9000   # custom port
```

The server listens on `http://127.0.0.1:<port>/get-query` and translates incoming GET requests into calls to `handle_get_query`, exactly as a Lambda + API Gateway deployment would.

**3. Run the interactive REPL client** (in a second terminal):

```bash
uv run python scripts/integration_llm.py          # matches default port
uv run python scripts/integration_llm.py --port 9000
```

The REPL maintains the current query dict across turns so you can refine it incrementally:

```
>>> find all raw ecephys sessions
  Updated query:
  {
    "data_description.data_level": "raw",
    "data_description.modality": {"$elemMatch": {"abbreviation": "ecephys"}}
  }
>>> narrow to subject 730945
  Updated query:
  { ...previous filters... "subject.subject_id": "730945" }
>>> show          # print current query without calling the server
>>> reset         # clear the query back to {}
>>> quit
```

You can also hit the endpoint directly:

```bash
curl "http://127.0.0.1:8765/get-query?message=find+all+raw+ecephys+data"
curl "http://127.0.0.1:8765/get-query?message=narrow+to+subject+730945&query=%7B%22data_description.data_level%22%3A%22raw%22%7D"
```

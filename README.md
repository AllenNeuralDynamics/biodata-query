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

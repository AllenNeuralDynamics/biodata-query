#!/usr/bin/env python
"""Local Panel demo — launches QueryBuilder + QueryResults in the browser.

Usage:
    panel serve scripts/panel_demo.py --show --autoreload
"""

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy third-party debug output; show only biodata_query at DEBUG
logging.getLogger("biodata_query").setLevel(logging.DEBUG)
for _noisy in ("bokeh", "tornado", "urllib3", "asyncio", "panel", "param"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import panel as pn

from biodata_query.panel import QueryBuilder, QueryResults

pn.extension(
    'tabulator',
    sizing_mode="stretch_width",
    raw_css=["""
        body, .bk-root { font-size: 12px !important; }
        .tabulator { font-size: 12px !important; }
        .tabulator .tabulator-col-title,
        .tabulator .tabulator-cell { font-size: 12px !important; padding: 2px 6px !important; }
        .tabulator .tabulator-row { min-height: 22px !important; }
    """],
)

builder = QueryBuilder()
results = QueryResults()

# Wire builder → results: when the query/pipeline param changes, update results
builder.param.watch(lambda e: results.param.update(query=e.new), "query")
builder.param.watch(lambda e: results.param.update(pipeline=e.new), "pipeline")

pn.Column(builder, pn.layout.Divider(), results, sizing_mode="stretch_width").servable()

#!/usr/bin/env python
"""Minimal HTTP server that exposes the LLM /get-query endpoint locally.

Usage:
    python scripts/llm_server.py [--port 8765]

The server listens on http://localhost:<port>/get-query and accepts:
    GET /get-query?message=<text>[&query=<json>]

It wraps biodata_query.llm.endpoint.handle_get_query exactly as an API
Gateway Lambda would, so the integration client hits the same logic.

Stop the server with Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import lazily so startup error messages are readable
from biodata_query.llm.endpoint import handle_get_query  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:  # suppress default access log
        logger.info(fmt, *args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path != "/get-query":
            self._respond(404, {"error": f"Unknown path: {parsed.path}"})
            return

        params = dict(urllib.parse.parse_qsl(parsed.query))
        event = {"queryStringParameters": params}

        response = handle_get_query(event)
        status: int = response["statusCode"]
        body: str = response["body"]

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def _respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())


def main() -> None:
    parser = argparse.ArgumentParser(description="Local LLM query server")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), _Handler)
    logger.info("Serving on http://127.0.0.1:%d/get-query", args.port)
    logger.info("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

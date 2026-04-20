#!/usr/bin/env python
"""Interactive integration client for the local LLM /get-query server.

Start the server first:
    python scripts/llm_server.py

Then run this script:
    python scripts/integration_llm.py [--port 8765]

You'll get a REPL where you type natural-language instructions.  The current
query dict is maintained across turns, so you can refine it incrementally.

Commands:
    <message>   Send a message to update the query
    reset       Clear the current query back to {}
    show        Print the current query
    quit / q    Exit
"""

from __future__ import annotations

import argparse
import json
import readline  # noqa: F401 — enables arrow-key history in input()
import sys
import urllib.parse
import urllib.request


def _call_server(base_url: str, query: dict, message: str) -> dict:
    """POST the message + query to the server and return the response dict."""
    params = urllib.parse.urlencode({"query": json.dumps(query), "message": message})
    url = f"{base_url}/get-query?{params}"
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310
        return json.loads(resp.read())


def _pretty(query: dict) -> str:
    return json.dumps(query, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive LLM query client")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    current_query: dict = {}

    print(f"Connected to {base_url}")
    print("Type a natural-language instruction to build a MongoDB query.")
    print("Commands: reset, show, quit\n")

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line.lower() in ("quit", "q", "exit"):
            break

        if line.lower() == "reset":
            current_query = {}
            print("Query reset to {}")
            continue

        if line.lower() == "show":
            print(_pretty(current_query))
            continue

        print("  Calling server…")
        try:
            resp = _call_server(base_url, current_query, line)
        except OSError as exc:
            print(f"  ERROR connecting to server: {exc}")
            print(f"  Is it running?  python scripts/llm_server.py --port {args.port}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            continue

        if "query" in resp:
            current_query = resp["query"]
            print(f"  Updated query:\n{_pretty(current_query)}")
        else:
            print(f"  Server error: {resp.get('error', resp)}")


if __name__ == "__main__":
    main()

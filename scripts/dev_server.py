#!/usr/bin/env python3
"""Local dev server that disables browser caching.

Same as `python3 -m http.server` but sends `Cache-Control: no-cache` so the
browser always revalidates — no more stale pages after moving files around.

Usage: python3 scripts/dev_server.py [port] [directory]
"""
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    handler = partial(NoCacheHandler, directory=directory)
    print(f"Serving {directory} at http://localhost:{port} (no-cache)")
    HTTPServer(("", port), handler).serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local dev server: no browser caching + live reload.

Same as `python3 -m http.server` but (a) sends `Cache-Control: no-cache` so
pages always revalidate, and (b) injects a tiny poller into every HTML page
that reloads the browser when any file under the served directory changes —
so a phone/laptop on the LAN refreshes itself as files are edited.

Usage: python3 scripts/dev_server.py [port] [directory]
"""
import os
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

LIVE_JS = b"""(() => {
  // restore scroll after a live-reload: content renders async, so wait
  // until the page is tall enough (or 3s) before jumping back
  try {
    const y = +sessionStorage.getItem('__live_y');
    if (y) {
      sessionStorage.removeItem('__live_y');
      const t0 = Date.now();
      const tick = () => {
        if (document.documentElement.scrollHeight >= y + innerHeight) {
          scrollTo({top: y, behavior: 'instant'});
        }
        // keep correcting until we're within 2px or 5s elapses: content
        // (markdown fetch, KaTeX, charts) keeps changing page height
        if (Math.abs(scrollY - y) > 2 && Date.now() - t0 < 5000) {
          setTimeout(tick, 50);
        }
      };
      tick();
    }
  } catch (e) {}
  let v = null;
  setInterval(async () => {
    try {
      const t = await (await fetch('/__version', {cache: 'no-store'})).text();
      if (v !== null && t !== v) {
        try { sessionStorage.setItem('__live_y', scrollY); } catch (e) {}
        location.reload();
      }
      v = t;
    } catch (e) {}
  }, 1000);
})();
"""

SKIP_DIRS = {".git", ".claude", "__pycache__", "node_modules"}


def tree_fingerprint(root):
    latest, count = 0.0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            try:
                m = os.stat(os.path.join(dirpath, f)).st_mtime
            except OSError:
                continue
            count += 1
            if m > latest:
                latest = m
    return f"{latest:.0f}-{count}"


class DevHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        if self.path == "/__version":
            body = tree_fingerprint(self.directory).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/__live.js":
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(LIVE_JS)))
            self.end_headers()
            self.wfile.write(LIVE_JS)
            return
        # inject the live-reload poller into HTML responses
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        if path.endswith(".html") and os.path.isfile(path):
            with open(path, "rb") as f:
                body = f.read()
            tag = b'<script src="/__live.js"></script>'
            body = body.replace(b"</body>", tag + b"</body>", 1) if b"</body>" in body else body + tag
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        if "/__version" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    handler = partial(DevHandler, directory=directory)
    print(f"Serving {directory} at http://localhost:{port} (no-cache, live-reload)")
    HTTPServer(("", port), handler).serve_forever()


if __name__ == "__main__":
    main()

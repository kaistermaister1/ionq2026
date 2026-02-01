"""
Local dev server that:
- Serves web_client.html + static files
- Proxies API calls to the live game server to bypass browser CORS

Run:
  cd 2026-IonQ
  python proxy_server.py
Then open:
  http://localhost:5173/web_client.html
"""

from __future__ import annotations

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests


DEFAULT_UPSTREAM = "https://demo-entanglement-distillation-qfhvrahfcq-uc.a.run.app"


def _join_url(base: str, path: str) -> str:
    base = (base or "").rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


class Handler(SimpleHTTPRequestHandler):
    # Serve files from current working directory (2026-IonQ).

    def end_headers(self) -> None:
        # Basic security headers for local dev
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        # CORS preflight for /api/proxy
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _proxy(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        path = (qs.get("path") or [None])[0]
        base = (qs.get("base") or [None])[0]

        if not path or not isinstance(path, str) or not path.startswith("/"):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": {"code": "BAD_REQUEST", "message": "Missing ?path=/v1/..."}}).encode("utf-8"))
            return

        upstream = base or os.environ.get("UPSTREAM_BASE_URL") or DEFAULT_UPSTREAM
        url = _join_url(upstream, path)

        # Forward Authorization if present
        headers = {}
        auth = self.headers.get("Authorization")
        if auth:
            headers["Authorization"] = auth
        headers["Content-Type"] = "application/json"

        method = self.command.upper()
        body: Optional[bytes] = None
        if method in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length else b"{}"

        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                data=body,
                timeout=120,
            )
        except requests.RequestException as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": {"code": "UPSTREAM_ERROR", "message": str(e)}}).encode("utf-8"))
            return

        self.send_response(resp.status_code)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        ct = resp.headers.get("content-type") or "application/json"
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(resp.content)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/proxy"):
            return self._proxy()
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/web_client.html")
            self.end_headers()
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/api/proxy"):
            return self._proxy()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": False, "error": {"code": "NOT_FOUND", "message": "POST not supported here"}}).encode("utf-8"))


def main() -> None:
    host = "127.0.0.1"
    port = 5173
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port} (proxy + static)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()


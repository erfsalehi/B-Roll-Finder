"""Minimal always-on HTTP health endpoint for orchestrators (Coolify/Docker).

The Telegram bot is a long-poller with no HTTP surface, so a PaaS can't tell
whether it's alive — which is why Coolify's zero-downtime deploy leaves the old
container running (two pollers → Telegram 409). A 200-returning /health lets the
orchestrator health-check the bot and stop the old container before the new one
takes over.

Enabled by default; set BOT_HEALTH_PORT (default 8000), or BOT_HEALTH=0 to off.
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8000


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):   # don't spam the bot's stdout/log
        pass

    def _respond(self, body: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_GET(self):
        self._respond(b"ok")
        try:
            self.wfile.write(b"ok")
        except Exception:
            pass

    def do_HEAD(self):
        self._respond(b"")


def start_health_server(port: int = None) -> int | None:
    """Start the health endpoint in a daemon thread. Returns the bound port, or
    None if disabled / it couldn't bind. Pass port=0 to grab a free port."""
    if os.getenv("BOT_HEALTH", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    if port is None:
        try:
            port = int(os.getenv("BOT_HEALTH_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
        except ValueError:
            port = DEFAULT_PORT
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    except Exception as e:
        print(f"[bot.health] could not bind port {port}: {e}")
        return None
    bound = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True, name="BrollHealth").start()
    print(f"[bot.health] health endpoint on :{bound}/health")
    return bound

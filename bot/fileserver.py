"""Tiny tokenized HTTP file server for handing out project-zip download links.

Telegram bots can only upload 50 MB, and clip bundles are usually bigger, so the
bot also serves them over HTTP with short-lived, HMAC-signed URLs:

    http://HOST:PORT/d/<project>.zip?e=<expiry>&t=<token>

The token is ``HMAC(secret, "<relpath>:<expiry>")`` — stateless, so no link
table to maintain, and tamper-proof (you can't fetch a different path or extend
the expiry without the secret). The secret defaults to a hash of the bot token
so links survive restarts. Only files under ``downloads/`` are reachable.

Enable with ``BOT_FILE_SERVER=1``; configure ``BOT_FILE_SERVER_PORT`` (default
8770) and ``BOT_PUBLIC_HOST`` (the host/IP that goes into the URL).
"""

import hashlib
import hmac
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DOWNLOADS_ROOT = os.path.abspath("downloads")
DEFAULT_PORT = 8770
DEFAULT_TTL = 24 * 3600  # link lifetime in seconds


def _secret() -> bytes:
    explicit = os.getenv("BOT_FILE_TOKEN_SECRET", "").strip()
    if explicit:
        return explicit.encode("utf-8")
    # Derive a stable secret from the bot token so links survive restarts
    # without the operator having to set yet another env var.
    seed = os.getenv("TELEGRAM_BOT_TOKEN", "broll-fallback-secret")
    return hashlib.sha256(("brollfs:" + seed).encode("utf-8")).digest()


def sign_token(relpath: str, expiry: int) -> str:
    """HMAC token binding a relative path to an expiry timestamp."""
    msg = f"{relpath}:{expiry}".encode("utf-8")
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:32]


def verify_token(relpath: str, expiry: int, token: str) -> bool:
    if expiry < int(time.time()):
        return False
    return hmac.compare_digest(sign_token(relpath, expiry), token or "")


def build_link(abs_path: str, host: str = None, port: int = DEFAULT_PORT,
               ttl: int = DEFAULT_TTL, scheme: str = "http",
               base: str = None) -> str | None:
    """Build a signed download URL for a file under downloads/. None if the file
    is outside the served root.

    When ``base`` is given (e.g. ``https://broll.tovo.club`` from
    ``public_base_url()``) the link is built against that external base with no
    explicit port — for when the file server sits behind a TLS reverse proxy
    (Traefik/Coolify). Otherwise it falls back to ``scheme://host:port``."""
    rel = os.path.relpath(os.path.abspath(abs_path), DOWNLOADS_ROOT)
    if rel.startswith("..") or os.path.isabs(rel):
        return None
    rel = rel.replace(os.sep, "/")
    expiry = int(time.time()) + ttl
    token = sign_token(rel, expiry)
    q = urllib.parse.urlencode({"e": expiry, "t": token})
    enc = urllib.parse.quote(rel)
    base = (base or "").rstrip("/")
    if base:
        return f"{base}/d/{enc}?{q}"
    return f"{scheme}://{host}:{port}/d/{enc}?{q}"


def public_base_url() -> str:
    """External base URL (``scheme://host[:port]``) for download links when the
    file server is fronted by a reverse proxy / TLS domain. Set
    ``BOT_PUBLIC_URL`` to e.g. ``https://broll.tovo.club``. Empty when unset."""
    return os.getenv("BOT_PUBLIC_URL", "").strip().rstrip("/")


def public_host() -> str:
    """Best-effort public host for links: explicit env, else the primary
    outbound IP, else localhost."""
    h = os.getenv("BOT_PUBLIC_HOST", "").strip()
    if h:
        return h
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet — don't spam the bot's stdout
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/d/"):
            self.send_error(404)
            return
        rel = urllib.parse.unquote(parsed.path[len("/d/"):])
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            expiry = int(qs.get("e", ["0"])[0])
        except ValueError:
            expiry = 0
        token = qs.get("t", [""])[0]

        if not verify_token(rel, expiry, token):
            self.send_error(403, "Invalid or expired link")
            return

        abs_path = os.path.abspath(os.path.join(DOWNLOADS_ROOT, rel))
        # Defence in depth: never serve outside the downloads root.
        if not abs_path.startswith(DOWNLOADS_ROOT + os.sep) or not os.path.isfile(abs_path):
            self.send_error(404)
            return

        size = os.path.getsize(abs_path)

        # Honour a Range request so big files (multi-GB project zips) are
        # resumable — a dropped connection can continue instead of restarting
        # from zero. Browsers/download managers send "Range: bytes=start-end".
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        is_partial = False
        if rng and rng.strip().lower().startswith("bytes="):
            try:
                spec = rng.split("=", 1)[1].split(",", 1)[0].strip()
                s, _, e = spec.partition("-")
                if s:
                    start = int(s)
                    end = int(e) if e else size - 1
                else:  # suffix range: bytes=-N → last N bytes
                    start = max(0, size - int(e))
                    end = size - 1
                if start > end or start >= size:
                    self.send_response(416)  # Range Not Satisfiable
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                is_partial = True
            except (ValueError, IndexError):
                start, end = 0, size - 1
                is_partial = False

        length = end - start + 1
        self.send_response(206 if is_partial else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if is_partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Disposition",
                         f'attachment; filename="{os.path.basename(abs_path)}"')
        self.end_headers()
        if self.command == "HEAD":
            return
        remaining = length
        with open(abs_path, "rb") as f:
            f.seek(start)
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def do_HEAD(self):
        # Lets download managers probe size / Accept-Ranges before fetching.
        self.do_GET()


def start_server(port: int = None) -> int | None:
    """Start the file server in a daemon thread. Returns the bound port, or None
    if disabled / failed to bind."""
    port = port or int(os.getenv("BOT_FILE_SERVER_PORT", str(DEFAULT_PORT)) or DEFAULT_PORT)
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    except Exception as e:
        print(f"[bot.fileserver] could not bind port {port}: {e}")
        return None
    threading.Thread(target=httpd.serve_forever, daemon=True,
                     name="BrollFileServer").start()
    print(f"[bot.fileserver] serving downloads/ on :{port}")
    return port

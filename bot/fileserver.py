"""Tiny tokenized HTTP file server for handing out project-zip download links
(and the reviewer rating page, see :mod:`bot.rating_page`).

Telegram bots can only upload 50 MB, and clip bundles are usually bigger, so the
bot also serves them over HTTP with HMAC-signed URLs:

    http://HOST:PORT/d/<project>.zip?e=<expiry>&t=<token>

The token is ``HMAC(secret, "<relpath>:<expiry>")`` — stateless, so no link
table to maintain, and tamper-proof (you can't fetch a different path or extend
the expiry without the secret). The secret defaults to a hash of the bot token
so links survive restarts. Only files under ``downloads/`` are reachable.

Links do NOT expire by default (``e=0``): this runs on our own server, and an
expired link on a zip that's still on disk is pure friction — you had to
re-issue it to download a file that never went away. The signature is still
required, so a link stays unguessable and can't be pointed at another path. Set
``BOT_LINK_TTL`` to a number of seconds to go back to time-limited links (e.g.
``86400`` for a day); ``/files`` re-issues links for whatever is on disk either
way.

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
# 0 = links never expire (the default — see the module docstring). Any positive
# value is a lifetime in seconds.
DEFAULT_TTL = 0
NEVER = 0   # the expiry value that marks a permanent link


def _secret() -> bytes:
    explicit = os.getenv("BOT_FILE_TOKEN_SECRET", "").strip()
    if explicit:
        return explicit.encode("utf-8")
    # Derive a stable secret from the bot token so links survive restarts
    # without the operator having to set yet another env var.
    seed = os.getenv("TELEGRAM_BOT_TOKEN", "broll-fallback-secret")
    return hashlib.sha256(("brollfs:" + seed).encode("utf-8")).digest()


def link_ttl() -> int:
    """Link lifetime in seconds from ``BOT_LINK_TTL``; 0 (the default) = never
    expires. A malformed or negative value falls back to the default."""
    raw = os.getenv("BOT_LINK_TTL", "").strip()
    if not raw:
        return DEFAULT_TTL
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_TTL


def sign_token(relpath: str, expiry: int) -> str:
    """HMAC token binding a relative path to an expiry timestamp (0 = forever)."""
    msg = f"{relpath}:{expiry}".encode("utf-8")
    return hmac.new(_secret(), msg, hashlib.sha256).hexdigest()[:32]


def verify_token(relpath: str, expiry: int, token: str) -> bool:
    """True when ``token`` signs this path+expiry and the link is still live.

    ``expiry == NEVER`` (0) is a permanent link, so only the signature is
    checked — the timestamp gate is skipped. Every other expiry is enforced,
    so links minted while ``BOT_LINK_TTL`` was set still lapse on schedule."""
    if expiry != NEVER and expiry < int(time.time()):
        return False
    return hmac.compare_digest(sign_token(relpath, expiry), token or "")


def build_link(abs_path: str, host: str = None, port: int = DEFAULT_PORT,
               ttl: int = None, scheme: str = "http",
               base: str = None) -> str | None:
    """Build a signed download URL for a file under downloads/. None if the file
    is outside the served root.

    ``ttl`` defaults to :func:`link_ttl` (``BOT_LINK_TTL``, 0 ⇒ a permanent
    link stamped ``e=0``).

    When ``base`` is given (e.g. ``https://broll.tovo.club`` from
    ``public_base_url()``) the link is built against that external base with no
    explicit port — for when the file server sits behind a TLS reverse proxy
    (Traefik/Coolify). Otherwise it falls back to ``scheme://host:port``."""
    rel = os.path.relpath(os.path.abspath(abs_path), DOWNLOADS_ROOT)
    if rel.startswith("..") or os.path.isabs(rel):
        return None
    rel = rel.replace(os.sep, "/")
    if ttl is None:
        ttl = link_ttl()
    expiry = NEVER if ttl <= 0 else int(time.time()) + ttl
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


def expiry_note() -> str:
    """One-line description of how long issued links live, for bot messages."""
    ttl = link_ttl()
    if ttl <= 0:
        return "link doesn't expire"
    if ttl % 3600 == 0:
        return f"link expires in {ttl // 3600}h"
    return f"link expires in {ttl // 60}min"


def list_zips(root: str = None) -> list:
    """Every ``.zip`` currently under ``downloads/``, newest first.

    Returns dicts of ``{name, path, rel, size, mtime}`` — what ``/files`` needs
    to list what's downloadable on the server right now and mint a fresh link
    for each. Walks recursively so a zip written inside a project folder is
    found too; unreadable entries are skipped rather than raising."""
    base = os.path.abspath(root or DOWNLOADS_ROOT)
    out = []
    if not os.path.isdir(base):
        return out
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            if not fn.lower().endswith(".zip"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                st = os.stat(path)
            except OSError:
                continue
            rel = os.path.relpath(path, base).replace(os.sep, "/")
            out.append({"name": fn, "path": path, "rel": rel,
                        "size": st.st_size, "mtime": st.st_mtime})
    out.sort(key=lambda f: f["mtime"], reverse=True)
    return out


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet — don't spam the bot's stdout
        pass

    def do_GET(self):
        # Reviewer rating page (/rate…) lives on the same server.
        from bot import rating_page
        if rating_page.handle(self):
            return
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/d/"):
            self.send_error(404)
            return
        rel = urllib.parse.unquote(parsed.path[len("/d/"):])
        qs = urllib.parse.parse_qs(parsed.query)
        # A missing or unparseable "e" reads as NEVER (0) — a permanent link.
        # It still has to carry the matching signature for that expiry, so this
        # is a shorthand, not a bypass.
        try:
            expiry = int(qs.get("e", [str(NEVER)])[0])
        except ValueError:
            expiry = NEVER
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

    def do_POST(self):
        from bot import rating_page
        if not rating_page.handle(self):
            self.send_error(404)


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

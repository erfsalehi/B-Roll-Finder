import concurrent.futures
import yt_dlp
import traceback
import os
import re
import threading
import time
import random
import glob
import shutil
import tempfile


# ── In-process metadata cache ────────────────────────────────────────────────
# Avoids re-fetching full info for the same video URL across multiple keyword
# searches in the same session (very common for popular B-roll topics).
_meta_cache: dict = {}
_meta_cache_lock = threading.Lock()
_META_CACHE_MAXSIZE = 2000

def _fetch_full_info_cached(url: str) -> dict:
    """Like _fetch_full_info but backed by an in-process LRU-style cache."""
    with _meta_cache_lock:
        if url in _meta_cache:
            return _meta_cache[url]
    result = _fetch_full_info(url)
    if result:
        with _meta_cache_lock:
            if len(_meta_cache) >= _META_CACHE_MAXSIZE:
                # Evict the oldest entry (dict insertion order, Python 3.7+)
                oldest = next(iter(_meta_cache))
                del _meta_cache[oldest]
            _meta_cache[url] = result
    return result


# ── Cookie helper ────────────────────────────────────────────────────────────
# If Chrome's DPAPI decryption fails (Chrome v127+ App-Bound Encryption),
# we disable cookies for the rest of the session rather than crashing every call.
_cookies_broken = False

def _is_dpapi_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "dpapi" in msg or "failed to decrypt" in msg or "app-bound" in msg

def _is_cookie_error(err: Exception) -> bool:
    """Any failure that means the configured cookie SOURCE is unusable — so we
    should disable cookies for the session and retry without them rather than
    failing every call. Covers Chrome DPAPI *and* a missing/unreadable browser
    cookie database (e.g. YT_COOKIE_BROWSER=firefox on a headless server with no
    Firefox profile → 'could not find firefox cookies database')."""
    if _is_dpapi_error(err):
        return True
    msg = str(err).lower()
    return (
        ("could not find" in msg and "cookies database" in msg)
        or ("could not copy" in msg and "cookie" in msg)
        or "unsupported browser" in msg
    )

def _mark_cookies_broken() -> None:
    global _cookies_broken
    if not _cookies_broken:
        _cookies_broken = True
        print(
            "\n[yt-dlp] WARNING: Configured cookie source is unusable "
            "(decryption failed, or the browser cookie DB doesn't exist here).\n"
            "  Falling back to no-cookie mode for this session.\n"
            "  On a server: unset YT_COOKIE_BROWSER and instead mount a cookies.txt\n"
            "  and point YT_COOKIE_FILE at it (see deploy/README.md).\n"
        )

def _get_cookie_opts() -> dict:
    """Return yt-dlp cookie options from the environment.

    Priority:
      1. YT_COOKIE_FILE env var      → use that cookies.txt (best for servers)
      2. YT_COOKIE_BROWSER env var   → use cookiesfrombrowser (e.g. 'firefox')
      3. cookies.txt in project root → use cookiefile
      4. Neither / source broken     → empty dict (no cookies)

    On a headless server use a cookies FILE (YT_COOKIE_FILE) — a browser cookie
    DB (YT_COOKIE_BROWSER) doesn't exist there and also helps dodge the
    datacenter-IP "confirm you're not a bot" check. Chrome v127+ App-Bound
    Encryption can break the DPAPI reader on Windows; Firefox or a file avoids it.
    """
    if _cookies_broken:
        return {}
    # A cookies FILE always wins over a browser source — so a leftover
    # YT_COOKIE_BROWSER in .env can't override a mounted cookies.txt.
    f = _discover_cookie_file()
    if f:
        sanitized = _sanitized_cookie_file(f)
        # yt-dlp saves the cookie jar back to its cookiefile on close. Fetches run
        # in parallel, so handing them one shared file makes the writebacks race
        # and corrupt its header. Give each call a private throwaway copy instead.
        try:
            return {"cookiefile": _private_cookie_copy(sanitized)}
        except Exception:
            return {"cookiefile": sanitized}
    browser = os.getenv("YT_COOKIE_BROWSER", "").strip().lower()
    if browser and browser != "none":
        return {"cookiesfrombrowser": (browser,)}
    return {}


def _static_youtube_proxies() -> list:
    """Proxies listed inline in ``YT_DLP_PROXY`` / ``YOUTUBE_PROXY``."""
    raw = os.getenv("YT_DLP_PROXY") or os.getenv("YOUTUBE_PROXY") or ""
    return [p.strip() for p in re.split(r"[\s,;]+", raw.strip()) if p.strip()]


def _parse_proxy_lines(text: str) -> list:
    """Parse a fetched proxy list (one per line). Accepts ``protocol://ip:port``
    or a bare ``ip:port`` (assumed http). Skips blanks/comments."""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "://" not in s:
            s = "http://" + s
        out.append(s)
    return out


# Cache for a fetched proxy list (free lists churn, so refetch on a TTL).
_proxy_url_cache = {"ts": 0.0, "list": []}
_proxy_url_lock = threading.Lock()


def _dynamic_youtube_proxies() -> list:
    """Fetch the proxy list from ``YT_DLP_PROXY_URL`` (e.g. a ProxyScrape free
    list), cached for ``YT_DLP_PROXY_URL_TTL`` seconds (default 600). On a fetch
    error the last good list is reused. Returns ``[]`` when the URL is unset."""
    url = (os.getenv("YT_DLP_PROXY_URL") or "").strip()
    if not url:
        return []
    try:
        ttl = float(os.getenv("YT_DLP_PROXY_URL_TTL", "600") or 600)
    except ValueError:
        ttl = 600.0
    now = time.time()
    with _proxy_url_lock:
        if _proxy_url_cache["list"] and now - _proxy_url_cache["ts"] < ttl:
            return _proxy_url_cache["list"]
    try:
        import requests
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        proxies = _parse_proxy_lines(r.text)
    except Exception as e:
        print(f"[yt-dlp] proxy-list fetch failed ({url}): {e}")
        with _proxy_url_lock:
            return list(_proxy_url_cache["list"])   # last good (maybe empty)
    with _proxy_url_lock:
        _proxy_url_cache["ts"] = now
        _proxy_url_cache["list"] = proxies
    print(f"[yt-dlp] fetched {len(proxies)} proxies from {url}")
    return proxies


def youtube_proxies() -> list:
    """All YouTube proxies to use, in order — inline (``YT_DLP_PROXY`` /
    ``YOUTUBE_PROXY``) first, then any fetched from ``YT_DLP_PROXY_URL`` —
    de-duplicated. Routing ONLY yt-dlp through a residential/mobile proxy is the
    fix when the host's datacenter IP is blocked by YouTube ("This content isn't
    available" on every client), without forcing Telegram/stock/LLM traffic
    through the same (often slow/metered) proxy.

    ``YT_DLP_PROXY`` takes one proxy or a LIST (comma / whitespace / newline
    separated). ``YT_DLP_PROXY_URL`` points at a text list (one ``protocol://ip:port``
    per line). The combined pool is round-robined per request (spreads load), and
    a download fails over to the next proxy on a connection error — or, when the
    pool is large, on a YouTube block too (so it hops past blocked IPs). With an
    untrusted free list, set ``YT_DOWNLOAD_NO_COOKIES=1`` so your YouTube session
    cookies aren't sent through random proxies."""
    seen, out = set(), []
    for p in _static_youtube_proxies() + _dynamic_youtube_proxies():
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def youtube_proxy() -> str:
    """First configured proxy (for display / back-compat); '' if none."""
    ps = youtube_proxies()
    return ps[0] if ps else ""


_proxy_rr_lock = threading.Lock()
_proxy_rr_index = 0


def _next_youtube_proxy() -> str:
    """Pick the proxy for the next request. With a dynamic list (pool mode) this
    serves a *validated* working proxy; otherwise it round-robins the inline list
    so a single static pool still spreads load. '' if none configured."""
    from core import proxy_pool
    if proxy_pool.pool_active():
        return proxy_pool.get_proxy()
    ps = youtube_proxies()
    if not ps:
        return ""
    global _proxy_rr_index
    with _proxy_rr_lock:
        p = ps[_proxy_rr_index % len(ps)]
        _proxy_rr_index += 1
    return p


def _youtube_proxy_opts(proxy=None) -> dict:
    """yt-dlp ``proxy`` option. ``proxy=None`` round-robins the pool; an explicit
    value (including '') is used as-is. Returns ``{}`` when there's no proxy."""
    if proxy is None:
        proxy = _next_youtube_proxy()
    return {"proxy": proxy} if proxy else {}


def _search_proxy_opts() -> dict:
    """Proxy option for SEARCH/metadata listing — empty by default.

    Only YouTube *playback/download* is blocked on a datacenter IP; the flat
    ytsearch listing works fine direct, and routing it through slow free proxies
    just makes it time out (the very failures seen in the wild). So search goes
    DIRECT unless ``YT_SEARCH_USE_PROXY`` is set, for the rare host whose IP is
    search-blocked too."""
    if os.getenv("YT_SEARCH_USE_PROXY", "").strip().lower() in ("1", "true", "yes", "on"):
        return _youtube_proxy_opts()
    return {}


# Error substrings (lower-cased) that mean the PROXY itself failed (down, capped,
# unreachable) rather than the video — so we should fail over to a backup proxy.
# Deliberately excludes the "content isn't available" block class: that's handled
# by rotation + the reselect repair loop, and cycling a dead video through every
# slow proxy would just waste time.
_YT_PROXY_FAILOVER_MARKERS = (
    'proxy', 'timed out', 'timeout', 'connection reset', 'connection aborted',
    'unable to connect', 'cannot connect', 'failed to connect',
    'tunnel connection failed', 'remote end closed', 'max retries',
    'connection refused', 'econnreset', 'getaddrinfo', 'name resolution',
    'temporary failure', 'eof occurred',
)


def _is_proxy_failover_error(low: str) -> bool:
    return any(m in low for m in _YT_PROXY_FAILOVER_MARKERS)


# A YouTube playback block ("This content isn't available" / "Video unavailable").
# When there's a POOL of proxies we also fail over on these — a blocked proxy IP
# should be skipped for the next one. With a single proxy we don't (rotation + the
# reselect repair loop handle it, and cycling a dead video would just be slow).
_YT_BLOCK_MARKERS = ("content isn", "video unavailable", "not available on this app",
                     "sign in to confirm", "confirm you")


def _is_block_error(low: str) -> bool:
    return any(m in low for m in _YT_BLOCK_MARKERS)


_sanitized_cookie_cache: dict = {}


def _read_cookie_text(path: str) -> str:
    """Decode a cookies file regardless of how it was saved. Browser exports
    edited/re-saved on Windows are often UTF-16 (PowerShell Out-File / Notepad
    default) or carry a BOM — both make yt-dlp reject the file or read zero
    cookies. Detect the encoding from the BOM (or a NUL-byte heuristic for
    BOM-less UTF-16) and return clean text."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:3] == b"\xef\xbb\xbf":
        return raw[3:].decode("utf-8", "replace")
    if raw[:2] == b"\xff\xfe":
        return raw[2:].decode("utf-16-le", "replace")
    if raw[:2] == b"\xfe\xff":
        return raw[2:].decode("utf-16-be", "replace")
    # BOM-less UTF-16: ASCII text has a NUL beside every character.
    if raw.count(b"\x00") > len(raw) // 4:
        le_nuls = raw[1::2].count(0)   # NUL in high byte → little-endian
        return raw.decode("utf-16-le" if le_nuls >= len(raw) // 4 else "utf-16-be", "replace")
    return raw.decode("utf-8", "replace")


def _sanitized_cookie_file(path: str) -> str:
    """Return a path to a yt-dlp-clean copy of ``path``.

    A UTF-8 BOM at the start makes yt-dlp reject the whole file ("does not look
    like a Netscape format cookies file") — and Windows editors / PowerShell
    ``Out-File`` add one silently. We strip the BOM, normalise newlines to LF,
    and guarantee the Netscape header line, writing the result under .cache.
    Cached by source mtime so we only rewrite when the cookies change. On any
    error we fall back to the original path."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return path
    cached = _sanitized_cookie_cache.get(path)
    if cached and cached[0] == mtime and _looks_like_netscape(cached[1]):
        # Self-heal: only trust the cache if the file still has a valid header.
        # Something external (e.g. a yt-dlp cookie-jar writeback) could have
        # corrupted it; if so, fall through and regenerate from the source.
        return cached[1]
    try:
        lines = _read_cookie_text(path).splitlines()
        # yt-dlp reads the FIRST physical line and rejects the file unless it's
        # the magic header — so drop any leading blank/whitespace-only lines (a
        # leading newline in the upload would otherwise survive here) and strip a
        # stray BOM char before deciding whether to prepend the header.
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines:
            lines[0] = lines[0].lstrip("﻿")
        first = lines[0] if lines else ""
        if not (first.startswith("# Netscape HTTP Cookie File")
                or first.startswith("# HTTP Cookie File")):
            lines = ["# Netscape HTTP Cookie File"] + lines
        out_dir = os.path.join(_cookies_search_root(), ".cache")
        os.makedirs(out_dir, exist_ok=True)
        dest = os.path.join(out_dir, "_yt_cookies.txt")
        with open(dest, "w", encoding="utf-8", newline="\n") as out:
            out.write("\n".join(lines) + "\n")
        _sanitized_cookie_cache[path] = (mtime, dest)
        return dest
    except Exception:
        return path


def _looks_like_netscape(path: str) -> bool:
    """True if ``path``'s first physical line is the Netscape cookie magic header
    yt-dlp requires. Used to detect a cookie file corrupted after we wrote it."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return False
    return first.startswith("# Netscape HTTP Cookie File") or first.startswith("# HTTP Cookie File")


def _private_cookie_copy(sanitized: str) -> str:
    """Return a private throwaway copy of the sanitized cookie file.

    yt-dlp writes the cookie jar back to its ``cookiefile`` when each YoutubeDL
    closes. Deep fetches and searches run in parallel, so pointing them all at
    one shared file makes those writebacks race — truncating/rewriting it at the
    same time and corrupting the header, after which every later fetch fails with
    "does not look like a Netscape format cookies file". Handing each call its
    own copy isolates the writeback so the canonical file stays clean."""
    jar_dir = os.path.join(_cookies_search_root(), ".cache", "_cookiejars")
    os.makedirs(jar_dir, exist_ok=True)
    # Opportunistically prune copies older than an hour so the dir can't grow
    # unbounded across long-running sessions.
    cutoff = time.time() - 3600
    for old in glob.glob(os.path.join(jar_dir, "jar_*.txt")):
        try:
            if os.path.getmtime(old) < cutoff:
                os.remove(old)
        except OSError:
            pass
    fd, dest = tempfile.mkstemp(prefix="jar_", suffix=".txt", dir=jar_dir)
    os.close(fd)
    shutil.copyfile(sanitized, dest)
    return dest


def _cookies_search_root() -> str:
    """Repo root, where auto-detected cookie files live. Separate function so
    tests can point it at a temp dir."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def uploaded_cookie_path() -> str:
    """Where a cookies.txt sent to the Telegram bot is stored — under .cache so
    it persists across restarts (that dir is a mounted volume) without a rebuild."""
    return os.path.join(_cookies_search_root(), ".cache", "cookies.txt")


def reset_cookies_state() -> None:
    """Re-enable cookies after a fresh upload (clears the broken flag + cache)."""
    global _cookies_broken
    _cookies_broken = False
    _sanitized_cookie_cache.clear()


def _discover_cookie_file() -> str | None:
    """Locate a cookies.txt without needing an exact env path. Order:
      1. a file uploaded to the bot (``.cache/cookies.txt``) — most recent intent
      2. ``YT_COOKIE_FILE`` (explicit path)
      3. any ``*.txt`` inside a ``cookies/`` folder at the repo root
         (prefer a name containing 'cookie'/'youtube')
      4. ``cookies.txt`` in the repo root
    Returns the path, or None.
    """
    uploaded = uploaded_cookie_path()
    if os.path.isfile(uploaded):
        return uploaded

    explicit = os.getenv("YT_COOKIE_FILE", "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit

    root = _cookies_search_root()
    cdir = os.path.join(root, "cookies")
    if os.path.isdir(cdir):
        txts = [os.path.join(cdir, n) for n in sorted(os.listdir(cdir))
                if n.lower().endswith(".txt")]
        if txts:
            preferred = [p for p in txts
                         if any(k in os.path.basename(p).lower() for k in ("cookie", "youtube"))]
            return (preferred or txts)[0]

    root_file = os.path.join(root, "cookies.txt")
    if os.path.isfile(root_file):
        return root_file
    return None


def cookie_mode() -> tuple:
    """Describe the active YouTube cookie source for /status, as ``(ok, detail)``.

    ``ok`` is False for states that will hurt YouTube on a server: a configured
    cookie file that's missing, a browser source on a headless host (which can't
    work), or cookies disabled after a failure. ``ok`` is True for a valid file
    and (neutrally) for no-cookies, which still works for search."""
    if _cookies_broken:
        return False, "disabled this session (source failed — see /logs)"

    f = _discover_cookie_file()
    if f:
        return True, f"file {f}"

    # An explicit path was set but the file isn't there.
    cfile = os.getenv("YT_COOKIE_FILE", "").strip()
    if cfile:
        return False, f"file {cfile} NOT FOUND — YouTube will bot-block"

    browser = os.getenv("YT_COOKIE_BROWSER", "").strip().lower()
    if browser and browser != "none":
        # A browser cookie DB cannot exist in a headless container/server.
        return False, (f"browser '{browser}' — won't work on a server; put a "
                       "cookies.txt in a cookies/ folder or set YT_COOKIE_FILE")

    return True, "none (search OK; downloads may bot-block on a datacenter IP)"


class _QuietLogger:
    """Suppress all yt-dlp output (including ERROR lines) for metadata-only calls."""
    def debug(self, msg):   pass
    def info(self, msg):    pass
    def warning(self, msg): pass
    def error(self, msg):   pass


def _extract_info_with_backoff(ydl_opts: dict, target: str,
                                max_attempts: int = 6,
                                maximum_backoff: float = 32.0,
                                **extract_kwargs):
    """Run ``yt-dlp``'s extract_info with truncated exponential backoff.

    Retries only on HTTP 429 ("Too Many Requests") and HTTP 503 ("Service
    Unavailable") — the two responses YouTube uses when it's actively
    rate-limiting or shedding load. Any other error (deleted video, bad
    cookies, geo-block, etc.) is re-raised immediately so the caller's
    existing handlers (e.g. DPAPI fallback) still get to run.

    The wait formula matches Google's recommended pattern:
        min((2 ** attempt) + jitter, maximum_backoff)
    where ``jitter`` is uniform in [0, 1) seconds.

    Per attempt a fresh ``yt_dlp.YoutubeDL`` is opened so cookie state and
    extractor caches stay isolated between retries.

    Extra kwargs (e.g. ``process=False``) flow through to ``extract_info``.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(target, download=False, **extract_kwargs)
        except Exception as e:
            last_exc = e
            error_msg = str(e).lower()
            is_rate_limited = (
                "http error 429" in error_msg
                or "http error 503" in error_msg
                or "too many requests" in error_msg
            )
            if not is_rate_limited or attempt >= max_attempts - 1:
                raise
            wait = min((2 ** attempt) + random.uniform(0, 1), maximum_backoff)
            print(
                f"[yt-dlp] Rate limit hit for '{target}'. "
                f"Backing off for {wait:.2f}s "
                f"(attempt {attempt + 1}/{max_attempts})."
            )
            time.sleep(wait)
    # Should be unreachable, but kept as a defensive fallback so static
    # analyzers don't think the function can return None implicitly.
    if last_exc is not None:
        raise last_exc
    return None


_YT_EXTRACTOR_ARGS = {
    'youtube': {
        # Client choice is critical on a cookie'd server. yt-dlp SKIPS the
        # cookie-incompatible clients (android_vr, tv_simply) when a cookiefile
        # is set, and the 'web'/'mweb' clients only expose 360p unless you
        # supply a GVS PO token. The 'tv', 'web_safari', and 'web_embedded'
        # clients DO accept cookies and return the full HD/4K format list with
        # no PO token — so they're the ones that actually work here. (Deno is
        # still required to solve the nsig challenge so the URLs are downloadable.)
        'player_client': ['tv', 'web_safari', 'web_embedded'],
    }
}


def _yt_video_id(url: str) -> str:
    """Extract an 11-char YouTube video ID from a URL or bare ID string."""
    if not url:
        return ""
    if re.match(r'^[A-Za-z0-9_-]{11}$', url):
        return url
    m = re.search(r'(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else ""


def _yt_thumbnail(url: str, entry: dict) -> str:
    """Return the best thumbnail URL for a YouTube entry.

    yt-dlp's flat search often omits the thumbnail field. Every public
    YouTube video has a guaranteed hqdefault.jpg at a predictable URL, so
    we fall back to constructing it from the video ID rather than leaving
    the thumbnail blank.
    """
    thumb = (entry.get('thumbnail') or '').strip()
    if thumb:
        return thumb
    for t in (entry.get('thumbnails') or []):
        if t.get('url'):
            return t['url']
    vid = _yt_video_id(url)
    return f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg" if vid else ""


def _fetch_full_info(url: str) -> dict:
    """Helper to fetch full metadata for a single video URL.

    Key trick: we call ``extract_info`` with ``process=False`` so yt-dlp
    skips its internal format-selection step entirely. That step is
    what raises:

        ERROR: [youtube] xyz: Requested format is not available.

    when YouTube returns only SABR (Server-Adaptive BitRate) streams
    that yt-dlp's selector can't pick from. Since we only need
    resolution metadata (not playback), there's no reason to run
    selection at all — we just want the raw ``formats`` list. With
    ``process=False`` the call essentially behaves like a list-formats
    probe: every format yt-dlp could extract is returned, no merge or
    playability check is attempted, and the error becomes impossible.

    After extraction we backfill top-level ``height``/``width``/
    ``resolution`` from the formats list, because skipping ``process``
    also means yt-dlp doesn't set those convenience fields itself.
    """
    ydl_opts = {
        'logger': _QuietLogger(),
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'extract_flat': False,
        'socket_timeout': 15,
        # Force a multi-client query. YouTube serves different format
        # lists depending on which player_client yt-dlp identifies as:
        # 'web' often returns only low-res progressive streams (which
        # is why probing a 1080p video could come back as 320x180), while
        # 'tv' and 'mweb' return higher-resolution DASH entries. yt-dlp
        # queries each client in turn and merges their format lists
        # during _real_extract — *before* the process step we've
        # disabled — so this is safe to use alongside process=False.
        # 'ios' is intentionally EXCLUDED: YouTube routinely returns
        # playabilityStatus=DRM responses to the ios client even for
        # videos that aren't actually DRM-protected, which causes the
        # whole extraction to raise "This video is DRM protected" before
        # we can read formats from the other clients' responses.
        'extractor_args': {
            'youtube': {
                # Cookie-compatible, HD-capable, PO-token-free clients. The old
                # 'tv_simply'/'android_vr' set is skipped by yt-dlp when cookies
                # are present, and 'web'/'mweb' cap at 360p without a GVS PO
                # token — both leave only storyboard/low-res, so probes reported
                # 320x180 even for 1080p videos. 'tv' returns the full HD/4K
                # format list with cookies; 'web_safari' is a fallback. (Deno is
                # still required so the nsig-protected URLs are downloadable.)
                'player_client': ['tv', 'web_safari'],
            },
        },
        # DASH is where the high-res entries live — explicit True even
        # though it's the yt-dlp default, in case a future release
        # flips the default.
        'youtube_include_dash_manifest': True,
        # Safety nets for the remaining edge cases:
        #   * ignore_no_formats_error: if a client's response triggers
        #     a "no formats" raise (DRM, age-gate, etc.) yt-dlp downgrades
        #     to a warning instead of raising, so we still return the
        #     metadata we *did* collect from the other clients.
        #   * allow_unplayable_formats: keeps DRM-flagged/region-locked
        #     entries in the formats list rather than dropping them, so
        #     resolution metadata survives even when playability doesn't.
        'ignore_no_formats_error': True,
        'allow_unplayable_formats': True,
        **_get_cookie_opts(),
        **_youtube_proxy_opts(),
    }
    try:
        info = _extract_info_with_backoff(ydl_opts, url, process=False) or {}
    except Exception as e:
        # _QuietLogger silenced yt-dlp's own stderr — surface the actual
        # exception here so callers (Step 5 inspect button, slow-path
        # resolution filter) can be debugged.
        print(f"\n[Deep Fetch Failed] URL: {url} | Error: {e}\n")
        return {}

    # process=False returns raw extracted info. Backfill the convenience
    # fields callers expect (top-level height/width/resolution) from the
    # formats list so behaviour matches the old process=True shape.
    formats = info.get('formats') or []
    if formats:
        heights = [f.get('height') for f in formats if f.get('height')]
        widths = [f.get('width') for f in formats if f.get('width')]
        if heights and not info.get('height'):
            info['height'] = max(heights)
        if widths and not info.get('width'):
            info['width'] = max(widths)
        if not info.get('resolution') and info.get('width') and info.get('height'):
            info['resolution'] = f"{info['width']}x{info['height']}"

    return info


# ── Storyboard-aware resolution inference ────────────────────────────────────
# When yt-dlp's JS solver fails (n-challenge missing, runtime not on PATH,
# etc.) the formats list comes back with only storyboard entries — small
# preview-thumbnail sheets used by the scrubber. Their reported heights
# (90, 180, 360) correlate roughly with the *source video* resolution but
# aren't the same thing. This map captures the typical correspondence
# observed in the wild on YouTube:
#
#     storyboard sheet height → likely source video resolution
#
# Pairs not in the table fall through unchanged. Storyboard heights that
# could plausibly be real video resolutions (360, 720) are mapped only
# when we're confident no real video formats came back from yt-dlp.
_STORYBOARD_TO_RESOLUTION = {
    45:  480,
    90:  720,
    180: 1080,
    270: 1440,
    360: 1440,
    540: 2160,
    720: 2160,
}


def _is_real_video_format(fmt: dict) -> bool:
    """True if this format entry represents an actual video stream (not a
    storyboard sheet or an audio-only track)."""
    if (fmt.get("format_id") or "").startswith("sb"):
        return False
    if fmt.get("protocol") == "mhtml":
        return False
    vcodec = fmt.get("vcodec")
    if vcodec in (None, "none"):
        return False
    return bool(fmt.get("height"))


def infer_video_resolution(info: dict) -> int:
    """Return the best estimate of the *source video* resolution from an
    info_dict returned by ``_fetch_full_info``.

    Logic:
      1. If the formats list contains real video formats (vcodec set,
         non-mhtml protocol, non-sb format_id) → take ``max(height)`` of
         those. yt-dlp's JS solver worked; the number is reliable.
      2. Otherwise the formats list is storyboard-only → look up the
         largest storyboard sheet height in ``_STORYBOARD_TO_RESOLUTION``
         to recover the likely source video resolution.
      3. If neither applies → return whatever ``info['height']`` already
         has (0 if missing).
    """
    if not info:
        return 0
    formats = info.get("formats") or []
    real = [f for f in formats if _is_real_video_format(f)]
    if real:
        heights = [f["height"] for f in real if f.get("height")]
        return max(heights) if heights else (info.get("height") or 0)
    # Storyboard-only path: find the largest storyboard height we have
    sb_heights = [f.get("height") for f in formats if f.get("height")]
    raw = max(sb_heights) if sb_heights else (info.get("height") or 0)
    return _STORYBOARD_TO_RESOLUTION.get(raw, raw)


def search_youtube_single(keyword: str, num_shorts: int = 0, num_longs: int = 3, errors: list = None, min_height: int = 0) -> list:
    """
    Uses yt-dlp to search for a single keyword and returns a mix of shorts and long videos.

    Fast path  (min_height == 0): uses only the flat ytsearch metadata — no
    secondary per-video HTTP calls, typically 5-10× faster.

    Slow path  (min_height  > 0): fetches full per-video metadata in parallel
    so we can filter by actual resolution. Uses a stop_event so workers stop
    as soon as we have enough results.
    """
    if not keyword or keyword.startswith("Error:") or keyword.startswith("No keywords generated"):
        return []

    need = num_shorts + num_longs
    # Resolution-aware pool multiplier.
    # 720p+ qualifies for ~90% of videos → 2× buffer is plenty.
    # 1080p+ qualifies for ~70%           → 3× is safe.
    # 1440p / 4K are rarer               → 4-5× needed.
    if min_height == 0:
        pool_mult = 2
    elif min_height <= 720:
        pool_mult = 2
    elif min_height <= 1080:
        pool_mult = 3
    elif min_height <= 1440:
        pool_mult = 4
    else:
        pool_mult = 5
    search_pool_size = max(need * pool_mult, need + 3)

    ydl_search_opts = {
        'logger': _QuietLogger(),
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'simulate': True,
        'socket_timeout': 20,
        'extractor_args': _YT_EXTRACTOR_ARGS,
        **_get_cookie_opts(),
        **_search_proxy_opts(),   # search goes direct unless YT_SEARCH_USE_PROXY
    }

    try:
        info = _extract_info_with_backoff(
            ydl_search_opts, f"ytsearch{search_pool_size}:{keyword}"
        )

        initial_candidates = []
        for entry in (info.get('entries') or []):
            url = entry.get('url')
            if not url:
                continue
            dur = entry.get('duration')
            is_short = (
                (dur is not None and dur <= 60)
                or 'shorts' in url.lower()
                or 'short' in (entry.get('title') or '').lower()
            )
            initial_candidates.append({
                'title':     entry.get('title', 'Unknown Title'),
                'url':       url,
                'duration':  dur,
                'is_short':  is_short,
                'thumbnail': _yt_thumbnail(url, entry),
                'width':     entry.get('width'),
                'height':    entry.get('height') or 0,
            })

        if not initial_candidates:
            return []

        # ── Fast path: no resolution filter — flat metadata is enough ────────
        if min_height == 0:
            shorts_final, longs_final = [], []
            for item in initial_candidates:
                if item['is_short']:
                    if len(shorts_final) < num_shorts:
                        shorts_final.append(item)
                else:
                    if len(longs_final) < num_longs:
                        longs_final.append(item)
                if len(shorts_final) >= num_shorts and len(longs_final) >= num_longs:
                    break
            return shorts_final + longs_final

        # ── Slow path: full per-video fetch to check actual resolution ────────
        # Rolling-window approach: keep MAX_CONCURRENT futures in-flight at once.
        # As each completes, submit the next candidate only if we still need more
        # results. This means we never run more fetches than (need + MAX_CONCURRENT - 1)
        # in the happy path, vs. submitting all N candidates upfront.
        #
        # Pre-filter: if flat metadata already has a height hint that meets the
        # requirement, accept the candidate without a full-info round-trip.

        MAX_CONCURRENT = min(need + 2, 6)  # e.g. need=3 → 5 concurrent max

        shorts_final: list = []
        longs_final:  list = []

        def _have_enough() -> bool:
            return len(shorts_final) >= num_shorts and len(longs_final) >= num_longs

        def _process_item(item: dict, full_info: dict) -> None:
            h = full_info.get('height') or 0
            if h < min_height:
                return
            dur = full_info.get('duration')
            is_s = (
                (dur is not None and dur <= 60)
                or 'shorts' in item['url'].lower()
                or 'short'  in item['title'].lower()
            )
            thumbs = full_info.get('thumbnails') or []
            item.update({
                'duration':              dur,
                'is_short':              is_s,
                'width':                 full_info.get('width'),
                'height':                h,
                'resolution':            full_info.get('resolution'),
                'available_resolutions': sorted(
                    {f.get('height') for f in full_info.get('formats', []) if f.get('height')},
                    reverse=True,
                ),
                'thumbnail': (thumbs[-1].get('url') if thumbs else full_info.get('thumbnail')) or _yt_thumbnail(item['url'], full_info),
            })
            if is_s:
                if len(shorts_final) < num_shorts:
                    shorts_final.append(item)
            else:
                if len(longs_final) < num_longs:
                    longs_final.append(item)

        remaining = list(initial_candidates)

        # ── Pre-filter pass: accept candidates whose flat height already qualifies ──
        still_needed = []
        for item in remaining:
            if _have_enough():
                break
            flat_h = item.get('height') or 0
            if flat_h >= min_height > 0:
                # Flat metadata already confirms resolution — no full fetch needed
                dur = item.get('duration')
                is_s = (
                    (dur is not None and dur <= 60)
                    or 'shorts' in item['url'].lower()
                    or 'short'  in item['title'].lower()
                )
                item['is_short'] = is_s
                if is_s:
                    if len(shorts_final) < num_shorts:
                        shorts_final.append(item)
                else:
                    if len(longs_final) < num_longs:
                        longs_final.append(item)
            else:
                still_needed.append(item)

        # ── Rolling-window fetch for the rest ────────────────────────────────
        if not _have_enough() and still_needed:
            future_to_item: dict = {}
            pending:        set  = set()

            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_CONCURRENT,
                thread_name_prefix="ytmeta",
            )

            def _submit_next() -> None:
                while still_needed and len(pending) < MAX_CONCURRENT and not _have_enough():
                    item = still_needed.pop(0)
                    f = executor.submit(_fetch_full_info_cached, item['url'])
                    pending.add(f)
                    future_to_item[f] = item

            try:
                _submit_next()

                while pending and not _have_enough():
                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        pending.discard(future)
                        item = future_to_item.pop(future, None)
                        if item is None:
                            continue
                        try:
                            full_info = future.result()
                            if full_info:
                                _process_item(item, full_info)
                        except Exception:
                            pass
                        _submit_next()
            finally:
                # Once we have enough results we don't want to wait for any
                # in-flight yt-dlp metadata fetches — those are slow blocking
                # network calls and the implicit `with`-block shutdown(wait=True)
                # was a major source of stalls. Drop unstarted work and let
                # already-running fetches finish as orphan daemon threads.
                executor.shutdown(wait=False, cancel_futures=True)

        return shorts_final + longs_final

    except Exception as e:
        if _is_cookie_error(e) and not _cookies_broken:
            # Cookie source unusable (DPAPI, or browser cookie DB missing on a
            # headless server) — disable cookies once and retry without them so
            # we don't fail every single search.
            _mark_cookies_broken()
            return search_youtube_single(keyword, num_shorts, num_longs, errors, min_height)
        msg = f"YouTube search failed for '{keyword}': {e}"
        print(msg)
        if errors is not None:
            errors.append(msg)

    return []

def fetch_youtube_results(slots: list, num_shorts: int = 0, num_longs: int = 3, max_workers: int = 5, progress_callback=None, errors: list = None, min_height: int = 0) -> list:
    """
    For each slot, takes the FIRST keyword in 'keywords' and searches YouTube for it.
    Updates each slot with a 'youtube_results' list.
    """
    queries = []
    for idx, slot in enumerate(slots):
        keywords = slot.get('keywords', [])
        if keywords:
            primary_kw = keywords[0]
            queries.append((idx, primary_kw))
            
    total_queries = len(queries)
    if total_queries == 0:
        return slots
        
    completed_queries = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(search_youtube_single, kw, num_shorts, num_longs, errors, min_height): idx
            for idx, kw in queries
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results = future.result()
                slots[idx]['youtube_results'] = results
            except Exception as e:
                slots[idx]['youtube_results'] = []
                msg = f"YouTube search thread failed: {e}"
                print(msg)
                if errors is not None:
                    errors.append(msg)
                
            completed_queries += 1
            if progress_callback:
                progress_callback(completed_queries / total_queries)
                
    for slot in slots:
        if 'youtube_results' not in slot:
            slot['youtube_results'] = []
            
    return slots

from core.ffmpeg_utils import normalize_video

class DownloadInterrupt(Exception):
    pass


# Player clients forced for DOWNLOADS when a cookie file is set. With cookies,
# yt-dlp skips the anonymous android_vr/tv_simply clients, and web/mweb cap at
# 360p without a GVS PO token — so these three are the cookie-compatible,
# HD-capable set. The catch: on a datacenter IP a logged-in (cookie'd) session is
# frequently answered with "This content isn't available" even for public videos,
# and this client set can't recover. The download then falls back to cookie-less
# mode with ``force_clients=None`` (yt-dlp's own anonymous client selection),
# which serves those formats — see the retry in download_video.
_DOWNLOAD_PLAYER_CLIENTS = ('tv', 'web_safari', 'web_embedded')

# Error substrings (lower-cased) that, when cookies are active, are worth one
# automatic cookie-less retry. Covers the PO-token format error AND the
# datacenter-IP "this content isn't available" / "video unavailable" block, both
# of which usually clear without cookies. ("content isn" matches the straight and
# curly apostrophe forms of "isn't".)
_YT_NO_COOKIE_RETRY_MARKERS = (
    'requested format is not available',
    'content isn',
    'video unavailable',
    'sign in to confirm',
    'not available on this app',
    'failed to extract any player response',
)


def download_video(url: str, output_path: str, quality: str, task_state: dict, max_size_mb: float = None, strict_quality: bool = False, normalize: bool = False, no_audio: bool = True, disable_cookies: bool = False, force_clients=_DOWNLOAD_PLAYER_CLIENTS, proxy=None):
    """
    Downloads a video using yt-dlp with progress tracking and interruption support.
    Supports Premiere Pro compatibility, strict quality, and size limits.

    ``disable_cookies`` skips the cookie source for THIS call only — used by the
    no-cookie retry below. (It must not flip the module-global ``_cookies_broken``:
    downloads run in parallel, and a temporary global flip would silently strip
    cookies from every concurrent download.) ``force_clients`` is the YouTube
    player-client list to force, or ``None`` to let yt-dlp pick (used by the
    cookie-less retry so it can reach the anonymous android_vr/tv_simply clients).
    """
    # Escape hatch for servers where logged-in YouTube downloads are IP-blocked
    # but anonymous ones work: YT_DOWNLOAD_NO_COOKIES=1 skips cookies (and the
    # cookie-only client set) for downloads from the very first attempt, so we
    # don't waste a failed cookie'd try on every clip.
    if (not disable_cookies
            and os.getenv("YT_DOWNLOAD_NO_COOKIES", "").strip().lower()
            in ("1", "true", "yes", "on")):
        disable_cookies = True
        if force_clients == _DOWNLOAD_PLAYER_CLIENTS:
            force_clients = None

    # Pick the proxy for this attempt: an explicit one (a failover retry) or the
    # next round-robin from the pool. '' means "no proxy configured".
    chosen_proxy = proxy if proxy is not None else _next_youtube_proxy()

    # Map quality → target height. Accepts "1080", "1080p", or int; "Best"/"Worst"
    # are handled below as special selectors. Parsing the digits (rather than a
    # fixed lookup) means a bare "1080" from the headless/bot path actually caps
    # height instead of silently falling through to no limit.
    def _parse_height(q):
        digits = "".join(ch for ch in str(q) if ch.isdigit())
        return int(digits) if digits else 0

    if strict_quality:
        min_h = _parse_height(quality)
        q_filter = f"[height>={min_h}]" if min_h > 0 else ""
    else:
        max_h = _parse_height(quality)
        q_filter = f"[height<={max_h}]" if max_h > 0 else ""

    # Format selector: video-only by default (no_audio=True) — skips downloading
    # a separate audio stream and the subsequent FFmpeg merge, roughly halving
    # download time. Falls back progressively to avoid "format not available".
    #
    # The last two tiers are unfiltered safety nets: bv*+ba*/b are yt-dlp's
    # wildcard forms that match any video-or-audio stream and any combined
    # stream respectively. Without them, YouTube videos that only ship
    # DASH manifests (no progressive file) and don't match the codec filter
    # at the requested height surface as "Requested format is not available".
    if no_audio:
        format_selector = (
            f"bestvideo{q_filter}[vcodec^=avc1]"   # H.264 video only (Premiere Pro ideal)
            f"/bestvideo{q_filter}"                  # any codec at target res
            f"/best{q_filter}"                       # combined stream at target res
            f"/bestvideo"                            # any video at any res
            f"/bv*"                                  # wildcard: any stream with video
            f"/best"                                 # absolute fallback
        )
        if quality == 'Worst':
            format_selector = 'worstvideo/bv*/worst/b'
        elif quality == 'Best':
            format_selector = 'bestvideo[vcodec^=avc1]/bestvideo/bv*/best'
    else:
        format_selector = (
            f"bestvideo{q_filter}[vcodec^=avc1]+bestaudio[acodec^=mp4a]"
            f"/bestvideo{q_filter}[vcodec^=avc1]+bestaudio"
            f"/bestvideo{q_filter}+bestaudio"
            f"/best{q_filter}"
            f"/bestvideo+bestaudio"                  # any res, separate streams
            f"/bv*+ba*"                              # wildcard merge
            f"/best"                                 # absolute fallback
        )
        if quality == 'Worst':
            format_selector = 'worstvideo+worstaudio/wv*+wa*/worst/b'
        elif quality == 'Best':
            format_selector = (
                'bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]'
                '/bestvideo[vcodec^=avc1]+bestaudio'
                '/bestvideo+bestaudio'
                '/bv*+ba*/best'
            )

    def my_hook(d):
        if task_state.get('status') == 'cancelled':
            raise DownloadInterrupt("Download cancelled by user")

        while task_state.get('status') == 'paused':
            time.sleep(0.5)
        # Re-check cancel after unpausing (user may have cancelled while paused)
        if task_state.get('status') == 'cancelled':
            raise DownloadInterrupt("Download cancelled by user")

        if d['status'] == 'downloading':
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
            if total_bytes:
                downloaded_bytes = d.get('downloaded_bytes', 0)
                task_state['progress'] = downloaded_bytes / total_bytes
                task_state['speed'] = d.get('speed') or 0
                task_state['eta'] = d.get('eta')

                if max_size_mb:
                    size_mb = total_bytes / (1024 * 1024)
                    if size_mb > max_size_mb:
                        raise ValueError(f"Skipped: Video size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")
                
    ydl_opts = {
        'format': format_selector,
        'outtmpl': output_path,
        'progress_hooks': [my_hook],
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        # ── Resume + retry configuration ─────────────────────────────────
        # continuedl=True is yt-dlp's API default, but set explicitly so
        # nobody accidentally turns it off. With this on, the .part file
        # from a previous failed attempt is reused — yt-dlp issues a Range
        # request and downloads only the remaining bytes.
        'continuedl': True,
        # Bumped from 10 → 30: transient HTTP errors during long downloads
        # (large 4K clips, slow proxy, weak Wi-Fi) often need many internal
        # retries before yt-dlp gives up. Each retry costs nothing if the
        # next chunk arrives, so being generous is safe.
        'retries': 30,
        'fragment_retries': 30,
        'extractor_retries': 5,
        'file_access_retries': 5,
        'http_chunk_size': 10485760,  # 10 MB
        'socket_timeout': 60,
        'nocheckcertificate': True,
        'geo_bypass': True,
        'http_headers': {
            # Keep the UA in sync with player_client='web' below
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        },
        # Force the cookie-compatible client set by default; on the cookie-less
        # retry force_clients is None so yt-dlp uses its own anonymous selection
        # (android_vr/tv_simply/…), which serves formats the cookie'd clients are
        # blocked from on a datacenter IP. (Deno solves the nsig challenge either
        # way so the URLs download.)
        'extractor_args': ({'youtube': {'player_client': list(force_clients)}}
                           if force_clients else {}),
        # Premiere Pro compatibility: ensure standard MP4 container
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        # Inject cookies (browser session or cookies.txt) — eliminates bot checks
        **({} if disable_cookies else _get_cookie_opts()),
        # Route through a residential proxy when YT_DLP_PROXY is set — the only
        # real fix when the host's datacenter IP is blocked by YouTube.
        **_youtube_proxy_opts(chosen_proxy),
    }

    try:
        task_state['status'] = 'downloading'
        task_state['progress'] = 0.0
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Check size before download if possible
            if max_size_mb:
                info = ydl.extract_info(url, download=False)
                # Some formats don't have filesize but have filesize_approx
                size = info.get('filesize') or info.get('filesize_approx')
                if size:
                    size_mb = size / (1024 * 1024)
                    if size_mb > max_size_mb:
                        raise ValueError(f"Skipped: Video size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")

            ydl.download([url])
            
        if normalize:
            task_state['status'] = 'processing'
            task_state['speed'] = None
            normalize_video(output_path, task_state=task_state)

        task_state['status'] = 'completed'
        task_state['progress'] = 1.0
        task_state['speed'] = None
    except DownloadInterrupt:
        task_state['status'] = 'cancelled'
        base = os.path.splitext(output_path)[0]
        for path in [output_path, output_path + '.part', output_path + '.ytdl']:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        # Clean up yt-dlp fragment files (e.g. video.f140.mp4, video.f248.webm)
        for pattern in [f"{base}.f[0-9]*.mp4", f"{base}.f[0-9]*.webm", f"{base}.f[0-9]*.m4a"]:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except Exception:
                    pass
    except Exception as e:
        if _is_cookie_error(e) and not _cookies_broken:
            # Cookie source unusable (DPAPI, or missing browser cookie DB) —
            # disable cookies and retry once without them.
            _mark_cookies_broken()
            download_video(url, output_path, quality, task_state,
                           max_size_mb=max_size_mb, strict_quality=strict_quality,
                           normalize=normalize, no_audio=no_audio)
            return

        err_str = str(e)
        low = err_str.lower()

        # Proxy failover: if this attempt's proxy is down/capped/unreachable —
        # or (with a multi-proxy pool) its IP got YouTube-blocked — retry the SAME
        # video through the next untried proxy. Capped by YT_PROXY_MAX_FAILOVER
        # (default 2) so a flaky/free pool can't stall one clip forever. A single
        # proxy never block-fails-over (rotation + the reselect repair loop cover
        # that, and cycling a dead video would just be slow).
        from core import proxy_pool
        pool_mode = proxy_pool.pool_active()
        proxies = youtube_proxies()
        if (chosen_proxy and (pool_mode or len(proxies) > 1)
                and (_is_proxy_failover_error(low) or _is_block_error(low))):
            tried = task_state.setdefault('_proxies_tried', [])
            if chosen_proxy not in tried:
                tried.append(chosen_proxy)
            try:
                max_fail = int(os.getenv("YT_PROXY_MAX_FAILOVER", "2") or 2)
            except ValueError:
                max_fail = 2
            # In pool mode, evict the failed proxy from the validated pool (which
            # triggers a background top-up) and pull the next known-good one.
            # Otherwise just step to the next untried inline proxy.
            nxt = None
            if pool_mode:
                proxy_pool.mark_dead(chosen_proxy)
                for _ in range(5):
                    cand = proxy_pool.get_proxy()
                    if cand and cand not in tried:
                        nxt = cand
                        break
            else:
                nxt = next((p for p in proxies if p not in tried), None)
            if nxt and nxt not in tried and len(tried) <= max_fail:
                print(f"[yt-dlp] {url}: proxy failed ({err_str.strip()[:70]}) — "
                      "failing over to another proxy")
                download_video(url, output_path, quality, task_state,
                               max_size_mb=max_size_mb, strict_quality=strict_quality,
                               normalize=normalize, no_audio=no_audio,
                               disable_cookies=disable_cookies,
                               force_clients=force_clients, proxy=nxt)
                return

        # With cookies configured, two failure classes almost always clear when
        # retried WITHOUT cookies:
        #   * "Requested format is not available" — YouTube kicked the logged-in
        #     request onto a PO-token-gated client (web/mweb) that returned zero
        #     formats.
        #   * "This content isn't available" / "Video unavailable" — on a
        #     datacenter IP a logged-in session is frequently blocked for videos
        #     that are perfectly public anonymously.
        # In both cases the cookie-less request falls back to yt-dlp's anonymous
        # clients (android_vr / tv_simply), which still serve the formats. Retry
        # once without cookies AND without forcing the cookie-only client set
        # (force_clients=None) so that fallback can actually happen.
        if (any(m in low for m in _YT_NO_COOKIE_RETRY_MARKERS)
                and not disable_cookies
                and _get_cookie_opts()
                and not task_state.get('_retried_no_cookies')):
            print(f"[yt-dlp] {url}: \"{err_str.strip()[:90]}\" with cookies — "
                  "retrying cookie-less with anonymous clients")
            task_state['_retried_no_cookies'] = True
            # Strip cookies for this call only (never via the module-global —
            # that would race with concurrent downloads).
            download_video(url, output_path, quality, task_state,
                           max_size_mb=max_size_mb, strict_quality=strict_quality,
                           normalize=normalize, no_audio=no_audio,
                           disable_cookies=True, force_clients=None)
            return

        task_state['status'] = 'error'
        if 'Requested format is not available' in err_str and not _get_cookie_opts():
            err_str = (
                "YouTube returned no formats for this video (blocked by anti-bot / rate limit). "
                "Set YT_COOKIE_BROWSER=firefox (or chrome/edge) in your .env so yt-dlp can use "
                "your browser's logged-in YouTube session. "
                "Original: " + err_str
            )
        task_state['error_msg'] = err_str


# Player-client configs the probe tries, in order. Each is (label, use_cookies,
# player_client-list-or-None). ``None`` lets yt-dlp pick its own default set.
_PROBE_CLIENT_CONFIGS = (
    ("cookies + tv/web_safari/web_embedded", True, ['tv', 'web_safari', 'web_embedded']),
    ("no-cookies + yt-dlp default",          False, None),
    ("no-cookies + android_vr",              False, ['android_vr']),
    ("no-cookies + tv_simply",               False, ['tv_simply']),
    ("no-cookies + ios",                     False, ['ios']),
    ("no-cookies + mweb",                    False, ['mweb']),
    ("no-cookies + web_safari",              False, ['web_safari']),
    ("cookies + default",                    True, None),
)


def _config_has_downloadable_format(info: dict) -> bool:
    """True if an extracted info dict exposes at least one real, fetchable video
    format (a URL on a non-storyboard video stream) — i.e. a download would have
    something to pull."""
    if not info:
        return False
    if info.get("url") and info.get("vcodec") not in (None, "none"):
        return True
    for f in (info.get("formats") or []):
        if f.get("url") and _is_real_video_format(f):
            return True
    return False


def probe_proxy(test_url: str, proxy: str, timeout: int = 10,
                use_cookies: bool = False) -> tuple:
    """Quick liveness + YouTube-playability check of ONE proxy. Returns
    ``(ok, detail)``; ``ok`` means the proxy reached YouTube and a downloadable
    format came back.

    ``use_cookies`` defaults False so a casual probe never sends the YouTube
    session through an untrusted proxy — but the validated-pool builder passes
    True when downloads themselves use cookies, so a proxy that only works *with*
    cookies (dodging the "Sign in to confirm you're not a bot" check) isn't
    wrongly marked dead."""
    opts = {
        'logger': _QuietLogger(),
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
        'socket_timeout': timeout,
        'extractor_args': {},
        'nocheckcertificate': True,
        'proxy': proxy,
    }
    if use_cookies:
        opts.update(_get_cookie_opts())
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(test_url, download=False)
        return (_config_has_downloadable_format(info or {}), "ok")
    except Exception as e:
        return (False, (str(e).strip().splitlines() or ["?"])[-1][:120])


def probe_download_clients(url: str, per_timeout: int = 25) -> list:
    """Diagnostic: try resolving a YouTube video's formats under several
    cookie/player-client combinations and report which actually yield a
    downloadable format. Used by ``/test`` when downloads fail so the operator can
    SEE whether it's a per-video restriction (some combo works) or a systemic
    datacenter-IP / PO-token block (every combo fails the same way) — and exactly
    which client to force if one works.

    Returns a list of ``(label, ok, detail)``. Uses ``extract_info(download=False)``
    with ``process=True`` so YouTube's playability check (the source of "This
    content isn't available") runs without fetching bytes."""
    results = []
    for label, use_cookies, clients in _PROBE_CLIENT_CONFIGS:
        opts = {
            'logger': _QuietLogger(),
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'socket_timeout': per_timeout,
            'extractor_args': ({'youtube': {'player_client': clients}}
                               if clients else {}),
            # Honor YT_DLP_PROXY so re-running /test after setting a residential
            # proxy actually probes through it.
            **_youtube_proxy_opts(),
        }
        if use_cookies:
            ck = _get_cookie_opts()
            if not ck:
                results.append((label, None, "no cookie source configured (skipped)"))
                continue
            opts.update(ck)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            if _config_has_downloadable_format(info or {}):
                h = (info or {}).get("height") or "?"
                results.append((label, True, f"ok — formats available (max {h}p)"))
            else:
                results.append((label, False, "no downloadable format returned"))
        except Exception as e:
            last = (str(e).strip().splitlines() or ["?"])[-1]
            results.append((label, False, last[:140]))
    return results


import re
import time
import random
import threading
import requests
import traceback
from urllib.parse import urlparse


class _TokenBucket:
    """Thread-safe token bucket — limits requests to `rate` per second."""
    def __init__(self, rate: float):
        self._rate   = rate
        self._tokens = rate
        self._last   = time.monotonic()
        self._lock   = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._rate,
                    self._tokens + (now - self._last) * self._rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)

    def __enter__(self):  self.acquire(); return self
    def __exit__(self, *_): pass


# Conservative per-API rate limiters.  Pexels / Pixabay both have per-minute
# quotas; 3 req/s (180/min) stays safely under any paid-tier limit while still
# processing a 200-shot run in ~1 minute for unique queries.
_PEXELS_RL  = _TokenBucket(rate=3.0)
_PIXABAY_RL = _TokenBucket(rate=3.0)


class _RateLimited(Exception):
    """Raised when a host's rate-limit window is in effect.

    Carries ``until`` (epoch seconds) so callers can tell the user when the
    quota resets instead of retrying into a wall.
    """
    def __init__(self, until_epoch: float):
        self.until = until_epoch
        super().__init__(f"rate limited until {until_epoch:.0f}")


def _parse_reset_seconds(headers) -> float | None:
    """Seconds until the rate-limit resets, from response headers.

    Honours ``Retry-After`` (delta seconds) first, then Pexels-style
    ``X-Ratelimit-Reset`` (a UTC epoch timestamp). Returns None if neither is
    present/parseable.
    """
    ra = headers.get("Retry-After")
    if ra:
        try:
            return max(0.0, float(ra))
        except (TypeError, ValueError):
            pass
    reset = headers.get("X-Ratelimit-Reset") or headers.get("X-RateLimit-Reset")
    if reset:
        try:
            return max(0.0, float(reset) - time.time())
        except (TypeError, ValueError):
            pass
    return None


class _RateLimitGate:
    """Process-wide circuit breaker for one API's hourly quota.

    Stock APIs like Pexels cap requests *per hour* (≈200 on the free tier), but
    a single large project fires hundreds of searches in minutes. Once the
    quota is gone, every further call returns 429 with a reset that can be ~an
    hour out — so retrying is useless and just floods the error log.

    This gate tracks the reset moment (learned from a 429 *or* proactively from
    an ``X-Ratelimit-Remaining: 0`` header on a good response) and lets callers
    skip the host until then, turning a flood of per-query failures into one
    actionable "rate limited until HH:MM" line.
    """
    def __init__(self, name: str):
        self.name = name
        self.until = 0.0
        self._lock = threading.Lock()

    def blocked_for(self) -> float:
        with self._lock:
            return max(0.0, self.until - time.time())

    def trip_for(self, seconds: float) -> None:
        with self._lock:
            self.until = max(self.until, time.time() + max(0.0, seconds))

    def note_response(self, headers) -> None:
        """Pre-emptively trip when the server says 0 requests remain."""
        rem = headers.get("X-Ratelimit-Remaining") or headers.get("X-RateLimit-Remaining")
        if rem is None:
            return
        try:
            if int(rem) <= 0:
                secs = _parse_reset_seconds(headers)
                self.trip_for(secs if secs is not None else 3600.0)
        except (TypeError, ValueError):
            return

    def reset_clock_str(self) -> str:
        import datetime as _dt
        with self._lock:
            if self.until <= time.time():
                return ""
            return _dt.datetime.fromtimestamp(self.until).strftime("%H:%M")


_PEXELS_GATE  = _RateLimitGate("Pexels")
_PIXABAY_GATE = _RateLimitGate("Pixabay")


def get_rate_limit_status() -> dict:
    """Current rate-limit state per host, for the UI.

    Returns ``{'pexels': {'blocked': bool, 'seconds': float, 'reset_at': 'HH:MM'},
    'pixabay': {...}}``. ``seconds`` is how long until the quota resets (0 when
    not limited).
    """
    out = {}
    for key, gate in (("pexels", _PEXELS_GATE), ("pixabay", _PIXABAY_GATE)):
        secs = gate.blocked_for()
        out[key] = {
            "blocked": secs > 0,
            "seconds": secs,
            "reset_at": gate.reset_clock_str(),
        }
    return out


def _http_get_with_retry(url, *, headers=None, params=None, timeout=8,
                         max_attempts=4, network_retries=1, rate_gate=None):
    """GET with backoff on 429/503 and a bounded retry on transient timeouts.

    Three failure classes, handled differently:

    * **429 / 503** — server is rate-limiting / briefly unavailable. Back off
      (honouring ``Retry-After``) and retry up to ``max_attempts``.
    * **Timeout / ConnectionError** — a socket stalled or dropped. Often
      transient under the parallel background dispatcher, so retry up to
      ``network_retries`` times (default 1) with a short pause.
    * **SSLError** — the TLS handshake was reset mid-stream (e.g.
      ``UNEXPECTED_EOF_WHILE_READING``). This is almost never transient: it
      means a firewall, DPI, or VPN egress is actively killing connections to
      this host. Retrying just multiplies a ~15s stall, so we raise it
      immediately and let the caller surface a network-path hint.

    ``timeout`` is a single value (applied to connect *and* read). A
    ``(connect, read)`` tuple is unhelpful here: the TLS handshake — exactly
    where a blocked host stalls — is governed by the *connect* value, so the
    read half is misleading.
    """
    # If this host's quota is already known to be exhausted, don't even try —
    # surface the rate-limit state immediately.
    if rate_gate is not None and rate_gate.blocked_for() > 0:
        raise _RateLimited(rate_gate.until)

    net_retries_left = network_retries
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
        except requests.exceptions.SSLError:
            # Host is being intercepted/reset — don't hammer it.
            raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if net_retries_left > 0:
                net_retries_left -= 1
                time.sleep(0.5)
                continue
            raise
        if response.status_code == 429:
            # Quota hit. Learn the reset window and trip the breaker so sibling
            # queries skip this host instead of each burning their own retries.
            reset_in = _parse_reset_seconds(response.headers)
            if rate_gate is not None:
                rate_gate.trip_for(reset_in if reset_in is not None else 3600.0)
            # Only a short, known reset is worth waiting for; an hourly reset is
            # not — fail fast as a rate-limit signal.
            if reset_in is not None and reset_in <= 3 and attempt < max_attempts - 1:
                time.sleep(reset_in + 0.5)
                continue
            raise _RateLimited(
                rate_gate.until if rate_gate is not None
                else time.time() + (reset_in if reset_in is not None else 3600.0)
            )
        if response.status_code == 503 and attempt < max_attempts - 1:
            time.sleep(2 ** attempt + random.uniform(0, 1))
            continue
        # Success: learn remaining-quota so we can stop *before* the next 429.
        if rate_gate is not None and response.ok:
            rate_gate.note_response(response.headers)
        response.raise_for_status()
        return response
    # Exhausted attempts on repeated 503 — raise the last HTTP status.
    response.raise_for_status()
    return response


def _network_hint(exc: Exception) -> str:
    """Return a user-facing suffix when an error looks like a blocked host.

    Distinguishes 'the host is unreachable on your network' from a genuine
    app/API fault, so the Step 3 error list tells the user to check their
    VPN/firewall instead of filing it as a bug.
    """
    s = str(exc).lower()
    if any(k in s for k in ("unexpected_eof", "eof occurred", "sslerror", "ssl:",
                            "wrong_version", "handshake")):
        return ("  [network] TLS connection reset - your network/VPN is likely "
                "blocking this host. Try a different VPN server or exit node.")
    if any(k in s for k in ("timed out", "timeout", "connection refused",
                            "max retries", "failed to establish")):
        return ("  [network] host unreachable/slow - check your internet or VPN "
                "(this host may be blocked on your current connection).")
    return ""


def _rate_limit_message(name: str, gate: "_RateLimitGate") -> str:
    """A single constant-per-window message so the app's error de-dup collapses
    a flood of 429s into one actionable line."""
    when = gate.reset_clock_str()
    return (
        f"{name} rate limit reached - skipping remaining {name} searches"
        + (f" until ~{when}." if when else " for now.")
        + f" ({name}'s free tier allows ~200 requests/hour; reduce results-per-query"
        " or split the project across sessions.)"
    )


def _pexels_slug_to_text(page_url: str) -> str:
    """Extract semantic words from a Pexels page URL.

    Pexels embeds the visual content in the page-URL slug, e.g.
    https://www.pexels.com/video/woman-walking-in-the-park-12345/
    -> 'woman walking in the park'
    """
    if not page_url:
        return ""
    try:
        path = urlparse(page_url).path
        slug = path.strip("/").split("/")[-1]
        slug = re.sub(r"-?\d+$", "", slug)
        return slug.replace("-", " ").strip()
    except Exception:
        return ""


def search_pexels(keyword: str, api_key: str, num_results: int = 3, errors: list = None, page: int = 1) -> list:
    if not api_key or not keyword:
        return []

    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {"query": keyword, "per_page": min(num_results, 80), "page": max(1, page)}

    # Skip immediately if Pexels' quota is known-exhausted — avoids burning a
    # rate-limiter token and a doomed request, and keeps the error log to one
    # shared line instead of one per query.
    if _PEXELS_GATE.blocked_for() > 0:
        if errors is not None:
            errors.append(_rate_limit_message("Pexels", _PEXELS_GATE))
        return []

    results = []
    try:
        with _PEXELS_RL:
            response = _http_get_with_retry(url, headers=headers, params=params,
                                            rate_gate=_PEXELS_GATE)
        data = response.json()

        for video in data.get('videos', []):
            video_files = video.get('video_files', [])
            if not video_files:
                continue

            # Prefer highest-resolution HD/UHD file
            best_file = video_files[0]
            for vf in video_files:
                if vf.get('quality') in ['hd', 'uhd']:
                    if (vf.get('width', 0) * vf.get('height', 0)) > (best_file.get('width', 0) * best_file.get('height', 0)):
                        best_file = vf
                    elif best_file.get('quality') not in ['hd', 'uhd']:
                        best_file = vf

            page_url = video.get('url', '')
            semantic = _pexels_slug_to_text(page_url)
            author = video.get('user', {}).get('name', 'Unknown')
            results.append({
                'title': semantic.title() if semantic else f"Pexels Video {video.get('id')}",
                'url': best_file.get('link'),
                'page_url': page_url,
                'source': 'pexels',
                'thumbnail': video.get('image', ''),
                'description': f"By {author}",
                'duration': video.get('duration'),
                'is_short': False,
                'width': best_file.get('width'),
                'height': best_file.get('height'),
                'quality': best_file.get('quality'),
                'file_size': None,
            })
            if len(results) >= num_results:
                break
    except _RateLimited:
        if errors is not None:
            errors.append(_rate_limit_message("Pexels", _PEXELS_GATE))
    except Exception as e:
        msg = f"Pexels search failed for '{keyword}': {e}{_network_hint(e)}"
        print(msg)
        if errors is not None:
            errors.append(msg)

    return results

def search_pixabay(keyword: str, api_key: str, num_results: int = 3, errors: list = None, page: int = 1) -> list:
    if not api_key or not keyword:
        return []

    url = "https://pixabay.com/api/videos/"
    params = {"key": api_key, "q": keyword, "per_page": min(max(3, num_results), 100), "page": max(1, page)}

    if _PIXABAY_GATE.blocked_for() > 0:
        if errors is not None:
            errors.append(_rate_limit_message("Pixabay", _PIXABAY_GATE))
        return []

    results = []
    try:
        with _PIXABAY_RL:
            response = _http_get_with_retry(url, params=params, rate_gate=_PIXABAY_GATE)
        data = response.json()

        for hit in data.get('hits', []):
            videos = hit.get('videos', {})

            # Find the best resolution available
            best_quality = 'large'
            if 'large' in videos and videos['large'].get('url'):
                best_quality = 'large'
            elif 'medium' in videos and videos['medium'].get('url'):
                best_quality = 'medium'
            elif 'small' in videos and videos['small'].get('url'):
                best_quality = 'small'
            else:
                continue

            v_data = videos[best_quality]

            tags = hit.get('tags', '') or ''
            results.append({
                'title': tags.title() if tags else f"Pixabay Video {hit.get('id', '?')}",
                'url': v_data.get('url'),
                'source': 'pixabay',
                'thumbnail': v_data.get('thumbnail', ''),
                'description': '',
                'duration': hit.get('duration'),
                'is_short': False,
                'width': v_data.get('width'),
                'height': v_data.get('height'),
                'quality': best_quality,
                'file_size': v_data.get('size'),
            })
            if len(results) >= num_results:
                break

        return results
    except _RateLimited:
        if errors is not None:
            errors.append(_rate_limit_message("Pixabay", _PIXABAY_GATE))
    except Exception as e:
        msg = f"Pixabay search failed for '{keyword}': {e}{_network_hint(e)}"
        print(msg)
        if errors is not None:
            errors.append(msg)

    return results


def _truncate(text: str, limit: int) -> str:
    """Trim text to ``limit`` chars, adding an ellipsis when cut."""
    if not text:
        return ""
    text = " ".join(text.split())  # collapse whitespace / newlines
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


_SHORTS_TAGS = ("#shorts", "#short", "#youtubeshorts", "#youtubeshort", "#ytshorts")


def _looks_like_short(title: str, description: str) -> bool:
    """Heuristic: detect YouTube Shorts via hashtag in title/description.

    The Data API v3 doesn't expose pixel dimensions or aspect ratio for
    arbitrary user videos — ``contentDetails.dimension`` only returns
    "2d"/"3d", thumbnails are normalized to 16:9, and there's no
    orientation filter on ``search.list``. The cleapest signal we have
    is the ``#shorts`` hashtag that creators conventionally include in
    Shorts metadata. False positives are possible (a regular video
    referencing "shorts" in its title) but those just get demoted by
    the rank-stage horizontal preference, not excluded — acceptable.
    """
    haystack = (title or "").lower() + " " + (description or "").lower()
    return any(tag in haystack for tag in _SHORTS_TAGS)


def search_youtube_data_api(keyword: str, api_key: str, num_results: int = 3,
                            errors: list = None, min_height: int = 0) -> list:
    """Search YouTube via the Data API v3.
    Note: search.list does NOT return resolutions, so min_height check 
    is only a heuristic for Shorts in this API.
    """
    if not api_key or not keyword:
        return []

    url = "https://www.googleapis.com/youtube/v3/search"
    # Fetch more to allow for heuristic filtering
    fetch_count = num_results * 2 if min_height > 0 else num_results
    params = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "maxResults": max(1, min(fetch_count, 10)),
        "safeSearch": "moderate",
        "key": api_key,
    }

    results = []
    try:
        response = _http_get_with_retry(url, params=params, timeout=10)
        data = response.json()

        for item in data.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            if not video_id:
                continue
            snippet = item.get("snippet", {}) or {}
            title = snippet.get("title") or "Untitled"
            channel = snippet.get("channelTitle") or "Unknown channel"
            raw_desc = snippet.get("description") or ""
            # Pack channel + description into a single 'description' string
            # so the judge sees the channel name as part of the relevance
            # signal. Many channels are topic-specific ("AutoFix Garage")
            # which is itself a strong prior.
            desc = f"by {channel}"
            if raw_desc.strip():
                desc += f" — {_truncate(raw_desc, 200)}"
            thumb = (snippet.get("thumbnails") or {}).get("medium", {}).get("url", "")

            # Detect Shorts via hashtag and synthesize portrait dimensions
            # so the rank-stage horizontal preference can demote them.
            is_short = _looks_like_short(title, raw_desc)
            yt_w = 1080 if is_short else None
            yt_h = 1920 if is_short else None

            results.append({
                "title": _truncate(title, 120),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "page_url": f"https://www.youtube.com/watch?v={video_id}",
                "source": "youtube",
                "thumbnail": thumb,
                "description": desc,
                # search.list does not return real duration or pixel
                # dimensions for normal videos. We synthesize portrait
                # dimensions only for Shorts (detected via hashtag).
                "duration": None,
                "is_short": is_short,
                "width": yt_w,
                "height": yt_h,
                "quality": None,
                "file_size": None,
                "channel": channel,
                "video_id": video_id,
            })
    except Exception as e:
        msg = f"YouTube search failed for '{keyword}': {e}"
        print(msg)
        if errors is not None:
            errors.append(msg)

    return results


def fetch_youtube_definition(video_url_or_id: str, api_key: str,
                              errors: list = None) -> str:
    """Look up a YouTube video's HD/SD definition via the Data API v3.

    Calls ``videos.list?part=contentDetails`` and returns the
    ``contentDetails.definition`` field, which is one of:

      * ``"hd"``      — at least one stream is 720p or higher
      * ``"sd"``      — every stream is below 720p
      * ``"unknown"`` — video missing, API key invalid, network error, or
                        no API key configured

    The Data API doesn't expose exact pixel resolution (only this two-way
    bucket), but the answer is authoritative — it comes directly from
    YouTube's own metadata, not from inferring through yt-dlp's format
    list. That makes it the reliable choice when storyboard-based
    inference is too inaccurate.

    Quota cost: 1 unit per call. With the default 10,000-unit daily
    budget, this comfortably supports thousands of inspect-button
    clicks per day.
    """
    if not api_key:
        return "unknown"

    # Accept either a full URL or a bare video ID. Mirrors the parsing in
    # core/youtube.py:_yt_video_id so callers can pass either form.
    video_id = video_url_or_id
    if "/" in video_id or "=" in video_id:
        import re as _re
        m = _re.search(r'(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})', video_url_or_id)
        if m:
            video_id = m.group(1)
        else:
            return "unknown"
    if not (len(video_id) == 11 and video_id.replace("_", "").replace("-", "").isalnum()):
        return "unknown"

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "contentDetails",
        "id": video_id,
        "key": api_key,
    }
    try:
        response = _http_get_with_retry(url, params=params, timeout=10)
        data = response.json()
        items = data.get("items") or []
        if not items:
            return "unknown"
        return items[0].get("contentDetails", {}).get("definition", "unknown")
    except Exception as e:
        msg = f"YouTube Data API definition lookup failed for '{video_id}': {e}"
        print(msg)
        if errors is not None:
            errors.append(msg)
        return "unknown"


def fetch_youtube_definitions_batch(video_urls_or_ids: list, api_key: str,
                                     errors: list = None) -> dict:
    """Batch-lookup HD/SD definition for many videos in a single API call.

    YouTube Data API's ``videos.list`` accepts up to 50 video IDs in a
    single ``id=`` parameter and costs **1 quota unit per call** regardless
    of how many IDs are inside. That makes this 50× cheaper than calling
    :func:`fetch_youtube_definition` in a loop — perfect for the Step 5
    "check all clips in this shot" button.

    Returns a dict keyed by the **caller's original input** (URL or bare
    ID) mapped to ``"hd"`` / ``"sd"`` / ``"unknown"``. Missing inputs
    (couldn't parse a video ID) map to ``"unknown"``.
    """
    if not api_key or not video_urls_or_ids:
        return {inp: "unknown" for inp in (video_urls_or_ids or [])}

    import re as _re

    # Parse each input to a video ID, remembering the mapping so we can
    # return the result keyed by what the caller passed in.
    input_to_id: dict = {}
    for inp in video_urls_or_ids:
        if not inp:
            continue
        if isinstance(inp, str) and len(inp) == 11 and _re.match(r'^[A-Za-z0-9_-]{11}$', inp):
            input_to_id[inp] = inp
            continue
        m = _re.search(r'(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})', inp or "")
        if m:
            input_to_id[inp] = m.group(1)

    unique_ids = list({vid for vid in input_to_id.values()})
    id_to_def: dict = {}

    BATCH = 50  # YouTube Data API hard limit
    url = "https://www.googleapis.com/youtube/v3/videos"
    for start in range(0, len(unique_ids), BATCH):
        batch = unique_ids[start:start + BATCH]
        params = {
            "part": "contentDetails",
            "id": ",".join(batch),
            "maxResults": BATCH,
            "key": api_key,
        }
        try:
            response = _http_get_with_retry(url, params=params, timeout=15)
            data = response.json()
            for item in data.get("items", []):
                vid_id = item.get("id")
                defn = (item.get("contentDetails") or {}).get("definition", "unknown")
                if vid_id:
                    id_to_def[vid_id] = defn
        except Exception as e:
            msg = f"YouTube Data API batch lookup failed (batch {start}-{start+len(batch)}): {e}"
            print(msg)
            if errors is not None:
                errors.append(msg)

    # Map back to the caller's input keys; anything we couldn't parse
    # or couldn't fetch defaults to 'unknown'.
    return {
        inp: id_to_def.get(input_to_id.get(inp), "unknown")
        for inp in video_urls_or_ids
    }

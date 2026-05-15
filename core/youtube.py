import concurrent.futures
import yt_dlp
import traceback
import os
import re
import threading
import time
import random
import glob


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

def _mark_cookies_broken() -> None:
    global _cookies_broken
    if not _cookies_broken:
        _cookies_broken = True
        print(
            "\n[yt-dlp] WARNING: Cookie decryption failed (Chrome App-Bound Encryption / DPAPI).\n"
            "  Falling back to no-cookie mode for this session.\n"
            "  To fix: set 'YouTube Cookie Browser' to 'firefox' in Setup,\n"
            "  OR run: pip install -U yt-dlp  (requires yt-dlp >= 2024.11)\n"
        )

def _get_cookie_opts() -> dict:
    """Return yt-dlp cookie options from the environment.

    Priority:
      1. YT_COOKIE_BROWSER env var  → use cookiesfrombrowser (e.g. 'firefox')
      2. cookies.txt in project root → use cookiefile
      3. Neither / DPAPI broken      → empty dict (no cookies)

    NOTE: Chrome v127+ uses App-Bound Encryption which breaks yt-dlp's DPAPI
    reader on Windows. Use Firefox, or update yt-dlp to >= 2024.11.
    """
    if _cookies_broken:
        return {}
    browser = os.getenv("YT_COOKIE_BROWSER", "").strip().lower()
    if browser and browser != "none":
        return {"cookiesfrombrowser": (browser,)}
    # Fallback: cookies.txt two directories up (project root)
    cookie_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cookies.txt")
    if os.path.exists(cookie_file):
        return {"cookiefile": cookie_file}
    return {}


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
        # Prefer clients that serve traditional (non-SABR) formats so
        # yt-dlp can inspect the format list without "format not available" errors.
        'player_client': ['web', 'tv_embedded', 'android'],
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
                # 'tv_simply' is listed first because YouTube ships its
                # high-res DASH URLs to that client **without** the
                # n-parameter JS challenge that protects the URLs every
                # other client receives. Without a working JS runtime
                # (Deno / Node / phantomjs), n-challenge solving fails
                # silently and every challenge-protected format is
                # dropped — which is why probes return 320x180 even
                # though 1080p actually exists. tv_simply gives us a
                # working fallback even on machines with no JS runtime
                # installed.
                'player_client': ['tv_simply', 'default', 'tv', 'mweb'],
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

    # If yt-dlp came back with a suspiciously low resolution (the classic
    # n-challenge-failure pattern: 360p or worse on a video that's actually
    # 1080p+), fall back to the Invidious API. Invidious instances handle
    # YouTube's SABR/n-challenge server-side, so we get a clean format list
    # back with one HTTPS GET — no JS runtime needed locally.
    if (info.get('height') or 0) < 720:
        prev = info.get('height') or 0
        inv = _fetch_via_invidious(url)
        if (inv.get('height') or 0) > prev:
            info['height'] = inv['height']
            info['width'] = inv.get('width') or info.get('width')
            info['resolution'] = inv.get('resolution') or info.get('resolution')
            existing_h = {f.get('height') for f in (info.get('formats') or [])}
            for f in inv.get('formats') or []:
                if f.get('height') not in existing_h:
                    info.setdefault('formats', []).append(f)
            print(
                f"[Inspect] yt-dlp returned {prev}p; "
                f"Invidious upgraded to {info['height']}p ({inv.get('_source', '')})."
            )

    return info


# ── Invidious fallback ───────────────────────────────────────────────────────
# Public Invidious instances. yt-dlp sometimes drops the high-res DASH entries
# when its JS solver fails; Invidious handles that server-side and exposes a
# simple JSON API. We try instances in order and use the first that responds.
# The list is short on purpose — most instances don't survive long, so a
# rotating curated handful is more useful than an exhaustive one.
_INVIDIOUS_INSTANCES = [
    "https://yewtu.be",
    "https://invidious.fdn.fr",
    "https://invidious.materialio.us",
    "https://invidious.privacydev.net",
    "https://inv.nadeko.net",
]


def _fetch_via_invidious(url: str) -> dict:
    """Fetch video resolution metadata from Invidious as a yt-dlp fallback.

    Returns a dict shaped roughly like yt-dlp's info_dict (height, width,
    resolution, formats[list of {height, width, ext}]) so the caller can
    merge results without conditional handling. Returns ``{}`` if every
    instance fails or the video ID can't be extracted.
    """
    try:
        import requests as _requests
    except ImportError:
        return {}

    vid_id = _yt_video_id(url)
    if not vid_id:
        return {}

    for instance in _INVIDIOUS_INSTANCES:
        try:
            resp = _requests.get(
                f"{instance}/api/v1/videos/{vid_id}",
                params={"fields": "adaptiveFormats,formatStreams,title"},
                timeout=8,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            formats: list = []
            # adaptiveFormats = DASH (separate video / audio streams).
            # formatStreams = progressive (combined audio+video).
            for fmt in (data.get("adaptiveFormats") or []) + (data.get("formatStreams") or []):
                res = fmt.get("resolution") or ""
                if "x" not in res:
                    continue
                try:
                    w_s, h_s = res.split("x", 1)
                    formats.append({
                        "width":  int(w_s),
                        "height": int(h_s),
                        "ext":    fmt.get("container", ""),
                    })
                except (ValueError, TypeError):
                    continue
            if not formats:
                continue
            heights = [f["height"] for f in formats]
            widths  = [f["width"]  for f in formats]
            return {
                "height":     max(heights),
                "width":      max(widths),
                "resolution": f"{max(widths)}x{max(heights)}",
                "formats":    formats,
                "title":      data.get("title", ""),
                "_source":    f"invidious:{instance.split('//', 1)[-1]}",
            }
        except Exception as e:
            print(f"[Invidious] {instance} failed: {e}")
            continue

    return {}

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
        if _is_dpapi_error(e) and not _cookies_broken:
            # Chrome DPAPI decryption broke — disable cookies and retry once
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

def download_video(url: str, output_path: str, quality: str, task_state: dict, max_size_mb: float = None, strict_quality: bool = False, normalize: bool = False, no_audio: bool = True):
    """
    Downloads a video using yt-dlp with progress tracking and interruption support.
    Supports Premiere Pro compatibility, strict quality, and size limits.
    """
    # Map simple quality strings to yt-dlp format strings
    # We prioritize h264 for Premiere Pro compatibility
    if strict_quality:
        # Require exactly the height or higher (if higher is ok, but usually exact is what they mean by 'only get 1080p')
        # However, yt-dlp's [height=1080] might fail if exactly 1080 isn't available.
        # We'll use [height>=1080] if they want 1080p or better, or [height=1080] if they want ONLY 1080.
        # Let's go with [height>=target] to be safe but strictly above the threshold.
        res_map = {'1080p': 1080, '720p': 720, '480p': 480}
        min_h = res_map.get(quality, 0)
        q_filter = f"[height>={min_h}]" if min_h > 0 else ""
    else:
        res_map = {'1080p': 1080, '720p': 720, '480p': 480}
        max_h = res_map.get(quality, 9999)
        q_filter = f"[height<={max_h}]"

    # Format selector: video-only by default (no_audio=True) — skips downloading
    # a separate audio stream and the subsequent FFmpeg merge, roughly halving
    # download time. Falls back progressively to avoid "format not available".
    if no_audio:
        format_selector = (
            f"bestvideo{q_filter}[vcodec^=avc1]"   # H.264 video only (Premiere Pro ideal)
            f"/bestvideo{q_filter}"                  # any codec at target res
            f"/best{q_filter}"                       # combined stream fallback
            f"/best"                                 # absolute fallback
        )
        if quality == 'Worst':
            format_selector = 'worstvideo/worst'
        elif quality == 'Best':
            format_selector = 'bestvideo[vcodec^=avc1]/bestvideo/best'
    else:
        format_selector = (
            f"bestvideo{q_filter}[vcodec^=avc1]+bestaudio[acodec^=mp4a]"
            f"/bestvideo{q_filter}[vcodec^=avc1]+bestaudio"
            f"/bestvideo{q_filter}+bestaudio"
            f"/best{q_filter}"
            f"/best"
        )
        if quality == 'Worst':
            format_selector = 'worstvideo+worstaudio/worst'
        elif quality == 'Best':
            format_selector = (
                'bestvideo[vcodec^=avc1]+bestaudio[acodec^=mp4a]'
                '/bestvideo[vcodec^=avc1]+bestaudio'
                '/bestvideo+bestaudio/best'
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
        'retries': 10,
        'fragment_retries': 10,
        'extractor_retries': 5,
        'file_access_retries': 5,
        'http_chunk_size': 10485760,  # 10 MB
        'socket_timeout': 30,
        'nocheckcertificate': True,
        'geo_bypass': True,
        'http_headers': {
            # Keep the UA in sync with player_client='web' below
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        },
        'extractor_args': {
            'youtube': {
                # 'web' is the most compatible client and least likely to get
                # 403-blocked. 'tv_embedded' is a solid fallback. 'android'
                # and 'ios' are increasingly rate-limited by YouTube (2024+).
                'player_client': ['web', 'tv_embedded', 'android'],
            }
        },
        # Premiere Pro compatibility: ensure standard MP4 container
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        # Inject cookies (browser session or cookies.txt) — eliminates bot checks
        **_get_cookie_opts(),
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
        if _is_dpapi_error(e) and not _cookies_broken:
            # Chrome DPAPI broke cookie reading — disable and retry once without cookies
            _mark_cookies_broken()
            download_video(url, output_path, quality, task_state,
                           max_size_mb=max_size_mb, strict_quality=strict_quality,
                           normalize=normalize, no_audio=no_audio)
            return
        task_state['status'] = 'error'
        task_state['error_msg'] = str(e)

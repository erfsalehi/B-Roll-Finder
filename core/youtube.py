import concurrent.futures
import yt_dlp
import traceback
import os
import threading
import time
import glob


# ── Cookie helper ────────────────────────────────────────────────────────────
def _get_cookie_opts() -> dict:
    """Return yt-dlp cookie options from the environment.

    Priority:
      1. YT_COOKIE_BROWSER env var  → use cookiesfrombrowser (e.g. 'chrome')
      2. cookies.txt in project root → use cookiefile
      3. Neither                     → empty dict (most restricted)

    Both options dramatically reduce the "Sign in to confirm you're not a bot"
    errors because yt-dlp will use the real YouTube session from the browser.
    """
    browser = os.getenv("YT_COOKIE_BROWSER", "").strip().lower()
    if browser and browser != "none":
        return {"cookiesfrombrowser": (browser,)}
    # Fallback: cookies.txt two directories up (project root)
    cookie_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cookies.txt")
    if os.path.exists(cookie_file):
        return {"cookiefile": cookie_file}
    return {}


def _fetch_full_info(url: str) -> dict:
    """Helper to fetch full metadata for a single video URL."""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'simulate': True,
        'skip_download': True,
        'extract_flat': False,
        'socket_timeout': 15,
        **_get_cookie_opts(),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception:
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
    # Over-fetch more candidates only when we need resolution filtering
    pool_mult = 4 if min_height > 0 else 2
    search_pool_size = max(need * pool_mult, need + 3)

    ydl_search_opts = {
        'extract_flat': True,
        'quiet': True,
        'no_warnings': True,
        'simulate': True,
        'socket_timeout': 20,
        **_get_cookie_opts(),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_search_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{search_pool_size}:{keyword}", download=False)

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
                'thumbnail': entry.get('thumbnail', ''),
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
        shorts_final, longs_final = [], []
        stop_event = threading.Event()

        def _fetch_if_needed(url: str) -> dict:
            """Fetch full info; bail immediately if we already have enough."""
            if stop_event.is_set():
                return {}
            return _fetch_full_info(url)

        max_threads = min(len(initial_candidates), 8)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            future_to_item = {
                executor.submit(_fetch_if_needed, item['url']): item
                for item in initial_candidates
            }
            for future in concurrent.futures.as_completed(future_to_item):
                if stop_event.is_set():
                    break
                item = future_to_item[future]
                try:
                    full_info = future.result()
                    if not full_info:
                        continue
                    h = full_info.get('height') or 0
                    if h < min_height:
                        continue
                    dur  = full_info.get('duration')
                    is_s = (
                        (dur is not None and dur <= 60)
                        or 'shorts' in item['url'].lower()
                        or 'short'  in item['title'].lower()
                    )
                    thumbs = full_info.get('thumbnails') or []
                    item.update({
                        'duration':             dur,
                        'is_short':             is_s,
                        'width':                full_info.get('width'),
                        'height':               h,
                        'resolution':           full_info.get('resolution'),
                        'available_resolutions': sorted(
                            {f.get('height') for f in full_info.get('formats', []) if f.get('height')},
                            reverse=True,
                        ),
                        'thumbnail': thumbs[-1].get('url') if thumbs else full_info.get('thumbnail', ''),
                    })
                    if is_s:
                        if len(shorts_final) < num_shorts:
                            shorts_final.append(item)
                    else:
                        if len(longs_final) < num_longs:
                            longs_final.append(item)
                    if len(shorts_final) >= num_shorts and len(longs_final) >= num_longs:
                        stop_event.set()  # signal other threads to exit early
                        break
                except Exception:
                    continue

        return shorts_final + longs_final

    except Exception as e:
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

def download_video(url: str, output_path: str, quality: str, task_state: dict, max_size_mb: float = None, strict_quality: bool = False, normalize: bool = False):
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

    # Premiere Pro loves H.264 (avc1) and AAC (mp4a)
    format_selector = f"bestvideo{q_filter}[vcodec^=avc1][ext=mp4]+bestaudio[acodec^=mp4a][ext=m4a]/best[ext=mp4]/best"
    
    if quality == 'Worst':
        format_selector = 'worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst'
    elif quality == 'Best':
        format_selector = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

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
        task_state['status'] = 'error'
        task_state['error_msg'] = str(e)

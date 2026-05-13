import os
import threading
import concurrent.futures
from core.stock_apis import search_pexels, search_pixabay, search_youtube_data_api
from core.youtube import search_youtube_single


# ── Cross-shot query result cache ─────────────────────────────────────────────
# Keyed by (source, query, num_results, min_height).  Multiple shots that share
# the same stock query (e.g. "city street") hit the same API only once.
_query_cache: dict = {}
_query_cache_lock = threading.Lock()


def _fetch_query(query: str, source: str, api_key: str, num_results: int,
                 errors: list, min_height: int = 0) -> list:
    # min_height is only meaningful for YouTube; stock APIs (Pexels/Pixabay)
    # are virtually always 720p+ so we skip the resolution check there and
    # let the download-stage filter handle it instead.
    cache_key = (source, query, num_results, min_height)
    with _query_cache_lock:
        if cache_key in _query_cache:
            return _query_cache[cache_key]

    if source == 'pexels':
        results = search_pexels(query, api_key, num_results, errors=errors)
    elif source == 'pixabay':
        results = search_pixabay(query, api_key, num_results, errors=errors)
    elif source == 'youtube':
        results = search_youtube_data_api(query, api_key, num_results, errors=errors, min_height=min_height)
    else:
        return []

    for r in results:
        r['matched_query'] = query

    with _query_cache_lock:
        _query_cache[cache_key] = results
    return results


def clear_query_cache() -> None:
    """Reset the cross-shot query cache (call between Director runs)."""
    with _query_cache_lock:
        _query_cache.clear()


def _normalize_youtube_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if len(url) == 11 and "/" not in url:
        return f"https://www.youtube.com/watch?v={url}"
    return url


def _classic_youtube_candidate(item: dict, query: str) -> dict:
    url = _normalize_youtube_url(item.get("url", ""))
    is_short = bool(item.get("is_short"))
    return {
        "title": item.get("title", "YouTube Video"),
        "url": url,
        "page_url": url,
        "source": "youtube",
        "thumbnail": item.get("thumbnail", ""),
        "description": "",
        "duration": item.get("duration"),
        "is_short": is_short,
        "width": item.get("width") or (1080 if is_short else None),
        "height": item.get("height") or (1920 if is_short else None),
        "available_resolutions": item.get("available_resolutions", []),
        "quality": None,
        "file_size": None,
        "matched_query": query,
    }


def search_youtube_classic(keyword: str, num_results: int = 3,
                            errors: list = None, min_height: int = 0) -> list:
    results = search_youtube_single(keyword, num_shorts=0, num_longs=num_results,
                                    errors=errors, min_height=min_height)
    return [_classic_youtube_candidate(item, keyword) for item in results]


def _process_shot(
    shot: dict,
    use_pexels: bool, use_pixabay: bool,
    use_youtube_api: bool, use_youtube_search: bool,
    pexels_key: str, pixabay_key: str, youtube_key: str,
    pexels_num_results: int, pixabay_num_results: int,
    youtube_api_num_results: int, youtube_search_num_results: int,
    min_height: int,
    errors: list,
    inner_workers: int,
) -> None:
    """Fetch all candidates for one shot in-place. Runs inside the shared executor."""
    queries = shot.get('search_queries', [])
    youtube_queries = shot.get('youtube_keywords') or queries[:1]

    # Build all jobs for Pexels / Pixabay / YouTube Data API.
    # Pexels + Pixabay: every query.  YouTube Data API: first query only (quota).
    jobs = []
    for q in queries:
        if use_pexels and pexels_key and pexels_num_results > 0:
            jobs.append((q, 'pexels', pexels_key, pexels_num_results))
        if use_pixabay and pixabay_key and pixabay_num_results > 0:
            jobs.append((q, 'pixabay', pixabay_key, pixabay_num_results))
    if use_youtube_api and youtube_key and queries and youtube_api_num_results > 0:
        jobs.append((queries[0], 'youtube', youtube_key, youtube_api_num_results))

    # YouTube Classic jobs — added to the same job list so they run concurrently
    # with Pexels/Pixabay rather than sequentially after them.
    yt_classic_jobs = []
    if use_youtube_search and youtube_queries and youtube_search_num_results > 0:
        for q in youtube_queries:
            yt_classic_jobs.append(q)

    results = []
    seen_urls = set()

    def _collect(items: list) -> None:
        for item in items:
            url = item.get('url')
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(item)

    # Run all stock-API jobs + YouTube Classic jobs in one shared pool.
    all_futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as ex:
        for q, src, key, n in jobs:
            f = ex.submit(_fetch_query, q, src, key, n, errors, min_height)
            all_futures[f] = ('stock', q, src)
        for q in yt_classic_jobs:
            f = ex.submit(search_youtube_classic, q, youtube_search_num_results, errors, min_height)
            all_futures[f] = ('yt_classic', q, None)

        for future in concurrent.futures.as_completed(all_futures):
            try:
                _collect(future.result())
            except Exception as e:
                kind, q, src = all_futures[future]
                errors.append(
                    f"Fetch error for shot {shot.get('slot_id')} "
                    f"({kind} '{q}'): {e}"
                )

    shot['video_results'] = results


def fetch_director_footage(
    shots: list,
    use_pexels: bool = True,
    use_pixabay: bool = True,
    use_youtube: bool = False,
    pexels_num_results: int = 3,
    pixabay_num_results: int = 3,
    youtube_api_num_results: int = 3,
    youtube_search_num_results: int = 3,
    max_workers: int = 6,
    progress_callback=None,
    errors: list = None,
    youtube_mode: str = "classic",
    use_youtube_api: bool = False,
    use_youtube_search: bool = None,
    retry_only: bool = False,
    min_height: int = 0,
) -> list:
    """
    Stage 2: Fetch video candidates for every shot in parallel.

    All shots run concurrently inside a single ThreadPoolExecutor.  Within each
    shot, Pexels, Pixabay, and YouTube Classic also run concurrently (one merged
    executor instead of two sequential ones).  Identical (source, query) pairs
    across shots are deduplicated by a module-level cache so the same API call
    is never issued twice in the same session.
    """
    if errors is None:
        errors = []

    pexels_key  = os.getenv("PEXELS_API_KEY", "")
    pixabay_key = os.getenv("PIXABAY_API_KEY", "")
    youtube_key = os.getenv("YOUTUBE_API_KEY", "")

    if use_youtube_search is None:
        use_youtube_search = use_youtube and youtube_mode == "classic"
    use_youtube_api = bool(use_youtube_api or (use_youtube and youtube_mode == "data_api"))

    # Separate shots that need work from those that can be skipped.
    active_shots = []
    for shot in shots:
        if shot.get('priority') == 'none':
            if not retry_only:
                shot['video_results'] = []
            active_shots.append(None)          # placeholder to preserve index
            continue
        if retry_only and len(shot.get('video_results', [])) > 0:
            active_shots.append(None)
            continue
        queries = shot.get('search_queries', [])
        youtube_queries = shot.get('youtube_keywords') or queries[:1]
        has_work = bool(queries) or bool(use_youtube_search and youtube_queries)
        if not has_work:
            if not retry_only:
                shot['video_results'] = []
            active_shots.append(None)
            continue
        active_shots.append(shot)

    work_shots = [s for s in active_shots if s is not None]
    # Use work_shots as the denominator so the bar fills smoothly over
    # real work; skipped shots (priority=none / retry) don't artificially
    # compress progress into a fraction of the range.
    progress_total = max(len(work_shots), 1)
    completed = [0]
    completed_lock = threading.Lock()

    # Per-shot inner concurrency: one thread per (query × source) job.
    inner_workers = max_workers

    def _run_shot(shot: dict) -> None:
        _process_shot(
            shot,
            use_pexels=use_pexels,
            use_pixabay=use_pixabay,
            use_youtube_api=use_youtube_api,
            use_youtube_search=use_youtube_search,
            pexels_key=pexels_key,
            pixabay_key=pixabay_key,
            youtube_key=youtube_key,
            pexels_num_results=pexels_num_results,
            pixabay_num_results=pixabay_num_results,
            youtube_api_num_results=youtube_api_num_results,
            youtube_search_num_results=youtube_search_num_results,
            min_height=min_height,
            errors=errors,
            inner_workers=inner_workers,
        )
        if progress_callback:
            # Hold the lock for the callback so concurrent shot completions
            # are serialized — Streamlit's DeltaGenerator is not thread-safe.
            with completed_lock:
                completed[0] += 1
                progress_callback(completed[0] / progress_total)

    # Outer parallelism: all shots concurrently.
    outer_workers = min(len(work_shots), max(max_workers, 16))
    if work_shots:
        with concurrent.futures.ThreadPoolExecutor(max_workers=outer_workers) as executor:
            futures = {executor.submit(_run_shot, shot): shot for shot in work_shots}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    shot = futures[future]
                    errors.append(f"Shot {shot.get('slot_id')} fetch failed: {e}")

    if progress_callback:
        progress_callback(1.0)

    return shots

import os
import concurrent.futures
from core.stock_apis import search_pexels, search_pixabay, search_youtube_data_api
from core.youtube import search_youtube_single


def _fetch_query(query: str, source: str, api_key: str, num_results: int, errors: list) -> list:
    if source == 'pexels':
        results = search_pexels(query, api_key, num_results, errors=errors)
    elif source == 'pixabay':
        results = search_pixabay(query, api_key, num_results, errors=errors)
    elif source == 'youtube':
        results = search_youtube_data_api(query, api_key, num_results, errors=errors)
    else:
        return []
    for r in results:
        r['matched_query'] = query
    return results


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
        "width": 1080 if is_short else None,
        "height": 1920 if is_short else None,
        "quality": None,
        "file_size": None,
        "matched_query": query,
    }


def search_youtube_classic(keyword: str, num_results: int = 3, errors: list = None) -> list:
    results = search_youtube_single(keyword, num_shorts=0, num_longs=num_results, errors=errors)
    return [_classic_youtube_candidate(item, keyword) for item in results]


def fetch_director_footage(shots: list, use_pexels: bool = True, use_pixabay: bool = True,
                           use_youtube: bool = False,
                           num_results: int = 3, max_workers: int = 6,
                           progress_callback=None, errors: list = None,
                           youtube_num_results: int = 3,
                           youtube_mode: str = "classic",
                           use_youtube_api: bool = False,
                           use_youtube_search: bool = None,
                           retry_only: bool = False) -> list:
    """
    Stage 2: For each shot, run search_queries across the enabled sources
    in parallel and deduplicate by URL.

    Pexels + Pixabay run all 2-3 queries per shot — they have no per-call
    quota. YouTube runs only on the FIRST query per shot to stay within
    the Data API v3 daily quota (10,000 units/day; 100 units per search
    call). For a 60-shot script that costs ~6,000 units of YT quota.
    """
    if errors is None:
        errors = []

    pexels_key = os.getenv("PEXELS_API_KEY", "")
    pixabay_key = os.getenv("PIXABAY_API_KEY", "")
    youtube_key = os.getenv("YOUTUBE_API_KEY", "")
    if use_youtube_search is None:
        use_youtube_search = use_youtube and youtube_mode == "classic"
    use_youtube_api = bool(use_youtube_api or (use_youtube and youtube_mode == "data_api"))

    total = len(shots)

    for idx, shot in enumerate(shots):
        if shot.get('priority') == 'none':
            if not retry_only:
                shot['video_results'] = []
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        if retry_only and len(shot.get('video_results', [])) > 0:
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        queries = shot.get('search_queries', [])
        youtube_queries = shot.get('youtube_keywords') or queries[:1]
        if not queries and not (use_youtube_search and youtube_queries):
            if not retry_only:
                shot['video_results'] = []
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        # Build all (query, source, key, num) jobs for this shot.
        # Pexels/Pixabay: every query. YouTube: first query only (quota).
        jobs = []
        for q in queries:
            if use_pexels and pexels_key:
                jobs.append((q, 'pexels', pexels_key, num_results))
            if use_pixabay and pixabay_key:
                jobs.append((q, 'pixabay', pixabay_key, num_results))
        if use_youtube_api and youtube_key and queries:
            jobs.append((queries[0], 'youtube', youtube_key, youtube_num_results))

        results = []
        seen_urls = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_fetch_query, q, src, key, n, errors): (q, src)
                for q, src, key, n in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    for item in future.result():
                        url = item.get('url')
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            results.append(item)
                except Exception as e:
                    errors.append(f"Fetch error for shot {shot.get('slot_id')}: {e}")

        if use_youtube_search and youtube_queries:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {
                    ex.submit(search_youtube_classic, q, youtube_num_results, errors): q
                    for q in youtube_queries
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        for item in future.result():
                            url = item.get('url')
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                results.append(item)
                    except Exception as e:
                        errors.append(f"YouTube fetch error for shot {shot.get('slot_id')}: {e}")

        shot['video_results'] = results

        if progress_callback:
            progress_callback((idx + 1) / total)

    return shots

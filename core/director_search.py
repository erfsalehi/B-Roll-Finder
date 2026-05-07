import os
import concurrent.futures
from core.stock_apis import search_pexels, search_pixabay


def _fetch_query(query: str, source: str, api_key: str, num_results: int, errors: list) -> list:
    if source == 'pexels':
        results = search_pexels(query, api_key, num_results, errors=errors)
    elif source == 'pixabay':
        results = search_pixabay(query, api_key, num_results, errors=errors)
    else:
        return []
    for r in results:
        r['matched_query'] = query
    return results


def fetch_director_footage(shots: list, use_pexels: bool = True, use_pixabay: bool = True,
                           num_results: int = 3, max_workers: int = 6,
                           progress_callback=None, errors: list = None) -> list:
    """
    Stage 2: For each shot, run ALL search_queries across Pexels + Pixabay in parallel.
    Deduplicates results within each shot by URL.
    Populates shot['video_results'] with up to (num_results * queries * sources) candidates.
    """
    if errors is None:
        errors = []

    pexels_key = os.getenv("PEXELS_API_KEY", "")
    pixabay_key = os.getenv("PIXABAY_API_KEY", "")

    total = len(shots)

    for idx, shot in enumerate(shots):
        if shot.get('priority') == 'none':
            shot['video_results'] = []
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        queries = shot.get('search_queries', [])
        if not queries:
            shot['video_results'] = []
            if progress_callback:
                progress_callback((idx + 1) / total)
            continue

        # Build all (query, source) jobs for this shot
        jobs = []
        for q in queries:
            if use_pexels and pexels_key:
                jobs.append((q, 'pexels', pexels_key))
            if use_pixabay and pixabay_key:
                jobs.append((q, 'pixabay', pixabay_key))

        results = []
        seen_urls = set()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_fetch_query, q, src, key, num_results, errors): (q, src)
                for q, src, key in jobs
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

        shot['video_results'] = results

        if progress_callback:
            progress_callback((idx + 1) / total)

    return shots

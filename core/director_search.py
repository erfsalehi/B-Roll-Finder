import os
import concurrent.futures
from core.stock_apis import search_pexels, search_pixabay
from core.youtube import search_youtube_single

def fetch_director_footage(shots: list, use_youtube: bool, use_pexels: bool, use_pixabay: bool, progress_callback=None) -> list:
    """
    Stage 2: Takes the JSON shot list from the Director and fetches candidates from APIs.
    """
    total_shots = len(shots)
    
    for idx, shot in enumerate(shots):
        if shot.get("priority") == "none":
            # Presenter talking head only, skip search
            if progress_callback:
                progress_callback((idx + 1) / total_shots)
            continue
            
        queries = shot.get("search_queries", [])
        if not queries:
            if progress_callback:
                progress_callback((idx + 1) / total_shots)
            continue
            
        primary_query = queries[0]
        
        # Clear any existing results for a fresh fetch
        shot['video_results'] = []
        
        # Fetch sequentially to respect API rate limits easily, though threads could be used
        if use_pexels and os.getenv("PEXELS_API_KEY"):
            res = search_pexels(primary_query, os.getenv("PEXELS_API_KEY"), num_results=3)
            shot['video_results'].extend(res)
            
        if use_pixabay and os.getenv("PIXABAY_API_KEY"):
            res = search_pixabay(primary_query, os.getenv("PIXABAY_API_KEY"), num_results=3)
            shot['video_results'].extend(res)
            
        if use_youtube:
            # We fetch 1 short and 2 longs as candidates
            res = search_youtube_single(primary_query, num_shorts=1, num_longs=2)
            for r in res:
                r['source'] = 'youtube'
            shot['video_results'].extend(res)
            
        if progress_callback:
            progress_callback((idx + 1) / total_shots)
            
    return shots

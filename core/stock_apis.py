import re
import requests
import traceback
from urllib.parse import urlparse


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


def search_pexels(keyword: str, api_key: str, num_results: int = 3, errors: list = None) -> list:
    if not api_key or not keyword:
        return []

    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": api_key}
    params = {"query": keyword, "per_page": num_results}

    results = []
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
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
    except Exception as e:
        msg = f"Pexels search failed for '{keyword}': {e}"
        print(msg)
        if errors is not None:
            errors.append(msg)

    return results

def search_pixabay(keyword: str, api_key: str, num_results: int = 3, errors: list = None) -> list:
    if not api_key or not keyword:
        return []

    url = "https://pixabay.com/api/videos/"
    # Pixabay per_page minimum is 3. If user wants 1, we fetch 3 and slice later.
    fetch_count = max(3, num_results)
    params = {"key": api_key, "q": keyword, "per_page": fetch_count}

    results = []
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
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

        return results[:num_results]
    except Exception as e:
        msg = f"Pixabay search failed for '{keyword}': {e}"
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


def search_youtube_data_api(keyword: str, api_key: str, num_results: int = 3,
                            errors: list = None) -> list:
    """Search YouTube via the Data API v3.

    Returns dicts in the same shape as ``search_pexels`` / ``search_pixabay``
    so the downstream judge and editor UI need no special-casing. Width and
    height are unavailable from search.list (would require a second
    videos.list call), so they're left as ``None`` — the judge treats that
    as orientation 'unknown', which the rank prompt handles.

    Each call costs 100 quota units (default daily quota: 10,000). Callers
    should rate-limit themselves; ``director_search`` only invokes this
    once per shot, on the first query.
    """
    if not api_key or not keyword:
        return []

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "maxResults": max(1, min(num_results, 5)),
        "safeSearch": "moderate",
        "key": api_key,
    }

    results = []
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
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

            results.append({
                "title": _truncate(title, 120),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "page_url": f"https://www.youtube.com/watch?v={video_id}",
                "source": "youtube",
                "thumbnail": thumb,
                "description": desc,
                # search.list does not return duration or pixel dimensions.
                # Filling these would require a second videos.list call; we
                # skip it to stay on the daily quota.
                "duration": None,
                "is_short": False,
                "width": None,
                "height": None,
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

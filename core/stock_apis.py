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

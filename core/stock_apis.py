import requests
import traceback

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

            title = f"Pexels Video {video.get('id')}"
            results.append({
                'title': title,
                'url': best_file.get('link'),
                'source': 'pexels',
                'is_short': False,
                'width': best_file.get('width'),
                'height': best_file.get('height'),
                'quality': best_file.get('quality'),
                'file_size': None
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

            title = f"Pixabay: {hit.get('tags', 'video')}"
            results.append({
                'title': title,
                'url': v_data.get('url'),
                'source': 'pixabay',
                'is_short': False,
                'width': v_data.get('width'),
                'height': v_data.get('height'),
                'quality': best_quality,
                'file_size': v_data.get('size')
            })

        return results[:num_results]
    except Exception as e:
        msg = f"Pixabay search failed for '{keyword}': {e}"
        print(msg)
        if errors is not None:
            errors.append(msg)

    return results

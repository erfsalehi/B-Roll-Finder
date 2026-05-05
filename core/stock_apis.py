import requests
import traceback

def search_pexels(keyword: str, api_key: str, num_results: int = 3) -> list:
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
                
            # Prefer HD/UHD quality, fallback to the first available
            best_file = video_files[0]
            for vf in video_files:
                if vf.get('quality') in ['hd', 'uhd']:
                    best_file = vf
                    break
                    
            title = f"Pexels Video {video.get('id')}"
            results.append({
                'title': title,
                'url': best_file.get('link'),
                'source': 'pexels',
                'is_short': False # Assume landscape/long for simplicity, though Pexels has portraits
            })
    except Exception as e:
        print(f"Error searching Pexels for '{keyword}': {e}")
        
    return results

def search_pixabay(keyword: str, api_key: str, num_results: int = 3) -> list:
    if not api_key or not keyword:
        return []
        
    url = "https://pixabay.com/api/videos/"
    params = {"key": api_key, "q": keyword, "per_page": num_results}
    
    results = []
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for hit in data.get('hits', []):
            videos = hit.get('videos', {})
            # Prefer large, then medium
            video_url = None
            if 'large' in videos and videos['large'].get('url'):
                video_url = videos['large']['url']
            elif 'medium' in videos and videos['medium'].get('url'):
                video_url = videos['medium']['url']
            elif 'small' in videos and videos['small'].get('url'):
                video_url = videos['small']['url']
                
            if not video_url:
                continue
                
            # Tags are often a good substitute for title
            title = f"Pixabay: {hit.get('tags', 'video')}"
            results.append({
                'title': title,
                'url': video_url,
                'source': 'pixabay',
                'is_short': False
            })
    except Exception as e:
        print(f"Error searching Pixabay for '{keyword}': {e}")
        
    return results

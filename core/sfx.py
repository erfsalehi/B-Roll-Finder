import os
import requests
import json

def search_freesound(query, api_key):
    """
    Searches Freesound.org for a sound effect and returns the best candidate.
    """
    if not api_key:
        return None
        
    url = "https://freesound.org/apiv2/search/text/"
    params = {
        "query": query,
        "token": api_key,
        "fields": "id,name,previews,duration,type",
        "filter": "duration:[0.1 TO 5.0]", # SFX are usually short
        "sort": "rating_desc",
        "page_size": 5
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        results = data.get("results", [])
        if not results:
            return None
            
        # Return the first result (top rated)
        return results[0]
    except Exception as e:
        print(f"Freesound search failed: {e}")
        return None

def download_sfx(sound_data, output_path, api_key):
    """
    Downloads the HQ preview of the sound.
    """
    if not sound_data or not api_key:
        return False
        
    preview_url = sound_data.get("previews", {}).get("preview-hq-mp3")
    if not preview_url:
        return False
        
    try:
        resp = requests.get(preview_url, timeout=20)
        resp.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"Freesound download failed: {e}")
        return False

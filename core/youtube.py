import concurrent.futures
import yt_dlp
import traceback
import os
import threading

def search_youtube_single(keyword: str, num_shorts: int = 0, num_longs: int = 3) -> list:
    """
    Uses yt-dlp to search for a single keyword and returns a mix of shorts and long videos.
    Returns a list of dicts: [{'title': str, 'url': str, 'is_short': bool}, ...]
    """
    ydl_opts = {
        'extract_flat': True,
        'force_generic_extractor': True,
        'quiet': True,
        'no_warnings': True,
        'simulate': True
    }
    
    if not keyword or keyword.startswith("Error:") or keyword.startswith("No keywords generated"):
        return []
        
    results = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Fetch more than we need to allow for filtering
            total_needed = num_shorts + num_longs
            query = f"ytsearch{total_needed * 3}:{keyword}"
            info = ydl.extract_info(query, download=False)
            
            shorts_found = []
            longs_found = []
            
            if 'entries' in info:
                for entry in info['entries']:
                    title = entry.get('title', 'Unknown Title')
                    url = entry.get('url')
                    # duration might be missing, assume long if missing to be safe, or just check
                    duration = entry.get('duration')
                    
                    if not url:
                        continue
                        
                    # YouTube shorts are typically <= 60 seconds
                    is_short = False
                    if duration is not None and duration <= 60:
                        is_short = True
                    elif 'shorts' in url.lower() or 'short' in title.lower():
                        is_short = True
                        
                    item = {'title': title, 'url': url, 'is_short': is_short, 'duration': duration}
                    
                    if is_short and len(shorts_found) < num_shorts:
                        shorts_found.append(item)
                    elif not is_short and len(longs_found) < num_longs:
                        longs_found.append(item)
                        
                    if len(shorts_found) >= num_shorts and len(longs_found) >= num_longs:
                        break
                        
            results = shorts_found + longs_found
    except Exception as e:
        print(f"Error searching YouTube for '{keyword}': {e}")
        
    return results

def fetch_youtube_results(slots: list, num_shorts: int = 0, num_longs: int = 3, max_workers: int = 5, progress_callback=None) -> list:
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
            executor.submit(search_youtube_single, kw, num_shorts, num_longs): idx 
            for idx, kw in queries
        }
        
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results = future.result()
                slots[idx]['youtube_results'] = results
            except Exception as e:
                slots[idx]['youtube_results'] = []
                
            completed_queries += 1
            if progress_callback:
                progress_callback(completed_queries / total_queries)
                
    for slot in slots:
        if 'youtube_results' not in slot:
            slot['youtube_results'] = []
            
    return slots

class DownloadInterrupt(Exception):
    pass

def download_video(url: str, output_path: str, quality: str, task_state: dict):
    """
    Downloads a video using yt-dlp with progress tracking and interruption support.
    `task_state` is a dictionary that must contain:
      - 'status': 'queued', 'downloading', 'paused', 'cancelled', 'completed', 'error'
      - 'progress': float 0.0 to 1.0
    """
    # Map simple quality strings to yt-dlp format strings
    format_map = {
        '1080p': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '720p': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        '480p': 'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'Best': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'Worst': 'worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst'
    }
    
    ydl_format = format_map.get(quality, 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best')
    
    def my_hook(d):
        if task_state.get('status') == 'cancelled':
            raise DownloadInterrupt("Download cancelled by user")
            
        while task_state.get('status') == 'paused':
            if task_state.get('status') == 'cancelled':
                raise DownloadInterrupt("Download cancelled by user")
            threading.Event().wait(1.0) # sleep 1s
            
        if d['status'] == 'downloading':
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
            if total_bytes:
                downloaded_bytes = d.get('downloaded_bytes', 0)
                task_state['progress'] = downloaded_bytes / total_bytes
                
    ydl_opts = {
        'format': ydl_format,
        'outtmpl': output_path,
        'progress_hooks': [my_hook],
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        'retries': 15,
        'fragment_retries': 15,
        'file_access_retries': 5,
        'http_chunk_size': 10485760, # 10MB
    }
    
    try:
        task_state['status'] = 'downloading'
        task_state['progress'] = 0.0
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        task_state['status'] = 'completed'
        task_state['progress'] = 1.0
    except DownloadInterrupt:
        task_state['status'] = 'cancelled'
        # Clean up partial file if needed, yt-dlp usually leaves .part files
        part_file = output_path + '.part'
        if os.path.exists(part_file):
            try: os.remove(part_file)
            except: pass
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except: pass
    except Exception as e:
        task_state['status'] = 'error'
        task_state['error_msg'] = str(e)

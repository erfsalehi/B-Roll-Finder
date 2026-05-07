import concurrent.futures
import yt_dlp
import traceback
import os
import threading
import time
import glob

def search_youtube_single(keyword: str, num_shorts: int = 0, num_longs: int = 3, errors: list = None) -> list:
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
        msg = f"YouTube search failed for '{keyword}': {e}"
        print(msg)
        if errors is not None:
            errors.append(msg)

    return results

def fetch_youtube_results(slots: list, num_shorts: int = 0, num_longs: int = 3, max_workers: int = 5, progress_callback=None, errors: list = None) -> list:
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
            executor.submit(search_youtube_single, kw, num_shorts, num_longs, errors): idx
            for idx, kw in queries
        }

        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results = future.result()
                slots[idx]['youtube_results'] = results
            except Exception as e:
                slots[idx]['youtube_results'] = []
                msg = f"YouTube search thread failed: {e}"
                print(msg)
                if errors is not None:
                    errors.append(msg)
                
            completed_queries += 1
            if progress_callback:
                progress_callback(completed_queries / total_queries)
                
    for slot in slots:
        if 'youtube_results' not in slot:
            slot['youtube_results'] = []
            
    return slots

from core.ffmpeg_utils import normalize_video

class DownloadInterrupt(Exception):
    pass

def download_video(url: str, output_path: str, quality: str, task_state: dict, max_size_mb: float = None, strict_quality: bool = False, normalize: bool = False):
    """
    Downloads a video using yt-dlp with progress tracking and interruption support.
    Supports Premiere Pro compatibility, strict quality, and size limits.
    """
    # Map simple quality strings to yt-dlp format strings
    # We prioritize h264 for Premiere Pro compatibility
    if strict_quality:
        # Require exactly the height or higher (if higher is ok, but usually exact is what they mean by 'only get 1080p')
        # However, yt-dlp's [height=1080] might fail if exactly 1080 isn't available.
        # We'll use [height>=1080] if they want 1080p or better, or [height=1080] if they want ONLY 1080.
        # Let's go with [height>=target] to be safe but strictly above the threshold.
        res_map = {'1080p': 1080, '720p': 720, '480p': 480}
        min_h = res_map.get(quality, 0)
        q_filter = f"[height>={min_h}]" if min_h > 0 else ""
    else:
        res_map = {'1080p': 1080, '720p': 720, '480p': 480}
        max_h = res_map.get(quality, 9999)
        q_filter = f"[height<={max_h}]"

    # Premiere Pro loves H.264 (avc1) and AAC (mp4a)
    format_selector = f"bestvideo{q_filter}[vcodec^=avc1][ext=mp4]+bestaudio[acodec^=mp4a][ext=m4a]/best[ext=mp4]/best"
    
    if quality == 'Worst':
        format_selector = 'worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst[ext=mp4]/worst'
    elif quality == 'Best':
        format_selector = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

    def my_hook(d):
        if task_state.get('status') == 'cancelled':
            raise DownloadInterrupt("Download cancelled by user")

        while task_state.get('status') == 'paused':
            time.sleep(0.5)
        # Re-check cancel after unpausing (user may have cancelled while paused)
        if task_state.get('status') == 'cancelled':
            raise DownloadInterrupt("Download cancelled by user")

        if d['status'] == 'downloading':
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate')
            if total_bytes:
                downloaded_bytes = d.get('downloaded_bytes', 0)
                task_state['progress'] = downloaded_bytes / total_bytes
                task_state['speed'] = d.get('speed') or 0
                task_state['eta'] = d.get('eta')

                if max_size_mb:
                    size_mb = total_bytes / (1024 * 1024)
                    if size_mb > max_size_mb:
                        raise ValueError(f"Skipped: Video size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")
                
    ydl_opts = {
        'format': format_selector,
        'outtmpl': output_path,
        'progress_hooks': [my_hook],
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': 'mp4',
        'retries': 15,
        'fragment_retries': 15,
        'file_access_retries': 5,
        'http_chunk_size': 10485760, # 10MB
        # Premiere Pro compatibility: ensure standard MP4 container
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }
    
    try:
        task_state['status'] = 'downloading'
        task_state['progress'] = 0.0
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Check size before download if possible
            if max_size_mb:
                info = ydl.extract_info(url, download=False)
                # Some formats don't have filesize but have filesize_approx
                size = info.get('filesize') or info.get('filesize_approx')
                if size:
                    size_mb = size / (1024 * 1024)
                    if size_mb > max_size_mb:
                        raise ValueError(f"Skipped: Video size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")

            ydl.download([url])
            
        if normalize:
            task_state['status'] = 'processing'
            task_state['speed'] = None
            normalize_video(output_path)

        task_state['status'] = 'completed'
        task_state['progress'] = 1.0
        task_state['speed'] = None
    except DownloadInterrupt:
        task_state['status'] = 'cancelled'
        base = os.path.splitext(output_path)[0]
        for path in [output_path, output_path + '.part', output_path + '.ytdl']:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        # Clean up yt-dlp fragment files (e.g. video.f140.mp4, video.f248.webm)
        for pattern in [f"{base}.f[0-9]*.mp4", f"{base}.f[0-9]*.webm", f"{base}.f[0-9]*.m4a"]:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                except Exception:
                    pass
    except Exception as e:
        task_state['status'] = 'error'
        task_state['error_msg'] = str(e)

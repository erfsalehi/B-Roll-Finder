import requests
import os
import threading
from core.youtube import DownloadInterrupt

def download_direct_video(url: str, output_path: str, task_state: dict, max_retries: int = 10):
    """
    Downloads an MP4 directly via HTTP using requests.
    Supports pause, resume, cancellation, and auto-resume on network drops.
    """
    try:
        task_state['status'] = 'downloading'
        if 'progress' not in task_state:
            task_state['progress'] = 0.0
            
        downloaded = 0
        total_size = 0
        
        # Check if file exists to resume
        if os.path.exists(output_path):
            downloaded = os.path.getsize(output_path)
            
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                if downloaded > 0:
                    headers['Range'] = f'bytes={downloaded}-'
                    
                response = requests.get(url, headers=headers, stream=True, timeout=15, allow_redirects=True)
                
                # If server doesn't support Range and we asked for it, it might return 200 instead of 206
                if response.status_code == 200 and downloaded > 0:
                    downloaded = 0 # Server didn't respect Range, restart
                    open(output_path, 'wb').close() # Clear file
                    
                response.raise_for_status()
                
                if total_size == 0:
                    content_length = int(response.headers.get('content-length', 0))
                    if response.status_code == 206:
                        total_size = downloaded + content_length
                    else:
                        total_size = content_length
                        
                mode = 'ab' if downloaded > 0 else 'wb'
                
                with open(output_path, mode) as f:
                    for data in response.iter_content(1024 * 1024): # 1MB chunks
                        if task_state.get('status') == 'cancelled':
                            raise DownloadInterrupt("Download cancelled by user")
                            
                        while task_state.get('status') == 'paused':
                            if task_state.get('status') == 'cancelled':
                                raise DownloadInterrupt("Download cancelled by user")
                            threading.Event().wait(1.0)
                            
                        if data:
                            f.write(data)
                            downloaded += len(data)
                            
                            if total_size > 0:
                                task_state['progress'] = downloaded / total_size
                                
                # If we get here, download finished cleanly
                break
                
            except requests.exceptions.RequestException as e:
                print(f"Network drop on {output_path}, attempt {attempt+1}/{max_retries}. Error: {e}")
                if attempt == max_retries - 1:
                    raise e
                threading.Event().wait(2.0) # Wait before retry
                
        task_state['status'] = 'completed'
        task_state['progress'] = 1.0
        
    except DownloadInterrupt:
        task_state['status'] = 'cancelled'
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except: pass
    except Exception as e:
        task_state['status'] = 'error'
        task_state['error_msg'] = str(e)

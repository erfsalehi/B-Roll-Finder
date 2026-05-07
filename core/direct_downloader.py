import requests
import os
import time
from core.youtube import DownloadInterrupt
from core.ffmpeg_utils import normalize_video

def download_direct_video(url: str, output_path: str, task_state: dict, max_retries: int = 10, max_size_mb: float = None, normalize: bool = False):
    """
    Downloads an MP4 directly via HTTP using requests.
    Supports pause, resume, cancellation, auto-resume, and max size limit.
    """
    try:
        task_state['status'] = 'downloading'
        if 'progress' not in task_state:
            task_state['progress'] = 0.0

        downloaded = 0
        total_size = 0

        # Pre-flight size check via HEAD
        if max_size_mb:
            try:
                head_resp = requests.head(url, allow_redirects=True, timeout=10)
                cl = head_resp.headers.get('content-length')
                if cl:
                    size_mb = int(cl) / (1024 * 1024)
                    if size_mb > max_size_mb:
                        raise ValueError(f"Skipped: Video size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")
            except requests.exceptions.RequestException:
                pass  # HEAD failed; secondary check runs during GET

        # Resume from partial file if it exists
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

                # Server ignored Range header — restart from scratch
                if response.status_code == 200 and downloaded > 0:
                    downloaded = 0
                    open(output_path, 'wb').close()

                response.raise_for_status()

                if total_size == 0:
                    content_length = int(response.headers.get('content-length', 0))
                    total_size = (downloaded + content_length) if response.status_code == 206 else content_length

                    # Secondary size check via GET content-length (fallback from HEAD)
                    if max_size_mb and total_size > 0:
                        size_mb = total_size / (1024 * 1024)
                        if size_mb > max_size_mb:
                            raise ValueError(f"Skipped: Video size ({size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)")

                mode = 'ab' if downloaded > 0 else 'wb'
                session_start = time.monotonic()
                session_start_bytes = downloaded

                with open(output_path, mode) as f:
                    for data in response.iter_content(1024 * 1024):  # 1MB chunks
                        if task_state.get('status') == 'cancelled':
                            raise DownloadInterrupt("Download cancelled by user")

                        # Block while paused, then re-check for cancel
                        while task_state.get('status') == 'paused':
                            time.sleep(0.5)
                        if task_state.get('status') == 'cancelled':
                            raise DownloadInterrupt("Download cancelled by user")

                        if data:
                            f.write(data)
                            downloaded += len(data)

                            # Enforce size limit even when no content-length was available
                            if max_size_mb and (downloaded / (1024 * 1024)) > max_size_mb:
                                raise ValueError(f"Skipped: Downloaded size exceeded limit ({max_size_mb:.0f}MB)")

                            elapsed = time.monotonic() - session_start
                            if elapsed > 0:
                                task_state['speed'] = (downloaded - session_start_bytes) / elapsed

                            if total_size > 0:
                                task_state['progress'] = downloaded / total_size

                break  # Download finished cleanly

            except requests.exceptions.RequestException as e:
                print(f"Network drop on {output_path}, attempt {attempt+1}/{max_retries}. Error: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2.0)

        if normalize:
            task_state['status'] = 'processing'
            task_state['speed'] = None
            normalize_video(output_path, task_state=task_state)

        task_state['status'] = 'completed'
        task_state['progress'] = 1.0
        task_state['speed'] = None

    except DownloadInterrupt:
        task_state['status'] = 'cancelled'
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
    except Exception as e:
        task_state['status'] = 'error'
        task_state['error_msg'] = str(e)

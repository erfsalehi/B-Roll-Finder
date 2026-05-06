import concurrent.futures
import uuid
import os
from core.youtube import download_video
from core.direct_downloader import download_direct_video

class DownloadManager:
    def __init__(self, max_workers=3):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.tasks = {} # dict of task_id -> state dict

    def add_download(self, url: str, output_path: str, quality: str, source: str = 'youtube', max_size_mb: float = None, strict_quality: bool = False) -> str:
        task_id = str(uuid.uuid4())
        
        # Ensure output directory exists (handle relative paths safely)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        
        state = {
            'id': task_id,
            'url': url,
            'output_path': output_path,
            'quality': quality,
            'source': source,
            'status': 'queued',
            'progress': 0.0,
            'error_msg': None,
            'max_size_mb': max_size_mb,
            'strict_quality': strict_quality
        }
        self.tasks[task_id] = state
        return task_id

    def start_download(self, task_id: str):
        state = self.tasks.get(task_id)
        if not state:
            return
            
        if state['status'] in ['queued', 'error', 'cancelled']:
            state['status'] = 'queued'
            state['progress'] = 0.0
            state['error_msg'] = None
            if state['source'] == 'youtube':
                self.executor.submit(download_video, state['url'], state['output_path'], state['quality'], state, max_size_mb=state.get('max_size_mb'), strict_quality=state.get('strict_quality'))
            else:
                self.executor.submit(download_direct_video, state['url'], state['output_path'], state, max_size_mb=state.get('max_size_mb'))

    def pause_download(self, task_id: str):
        state = self.tasks.get(task_id)
        if state and state['status'] == 'downloading':
            state['status'] = 'paused'

    def resume_download(self, task_id: str):
        state = self.tasks.get(task_id)
        if state and state['status'] == 'paused':
            state['status'] = 'downloading'

    def cancel_download(self, task_id: str):
        state = self.tasks.get(task_id)
        if state and state['status'] in ['downloading', 'paused', 'queued']:
            state['status'] = 'cancelled'

    def get_all_tasks(self):
        return list(self.tasks.values())
        
    def get_active_tasks(self):
        return [t for t in self.tasks.values() if t['status'] in ['downloading', 'paused']]
        
    def get_failed_tasks(self):
        return [t for t in self.tasks.values() if t['status'] == 'error']
        
    def get_stats(self):
        stats = {'total': len(self.tasks), 'queued': 0, 'downloading': 0, 'paused': 0, 'completed': 0, 'error': 0, 'cancelled': 0}
        for t in self.tasks.values():
            stats[t['status']] = stats.get(t['status'], 0) + 1
        return stats
        
    def retry_all_failed(self):
        for task_id, state in self.tasks.items():
            if state['status'] == 'error':
                self.start_download(task_id)
                
    def cancel_all(self):
        for task_id, state in self.tasks.items():
            if state['status'] in ['downloading', 'paused', 'queued']:
                state['status'] = 'cancelled'

    def clear_and_reset(self):
        """Cancel all in-flight downloads and clear the task list safely."""
        self.cancel_all()
        self.tasks.clear()
        # Recreate the executor so new downloads are not queued behind cancelled ones
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            # Python < 3.9 doesn't support cancel_futures
            self.executor.shutdown(wait=False)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

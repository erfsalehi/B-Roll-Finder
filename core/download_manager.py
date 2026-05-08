import concurrent.futures
import os
import re
import shutil
import uuid
from typing import Optional

from core.youtube import download_video
from core.direct_downloader import download_direct_video


# Per-task auto-retry cap (manager-level; the underlying downloaders also
# have their own internal retries). After this, the user must change
# settings or the URL or wait — preventing accidental infinite loops.
MAX_RETRIES = 3


def _materialize_extras(state: dict) -> None:
    """After a canonical download completes, materialize each extra_path.

    Hardlinks are tried first (free — same inode, same data); if the
    target filesystem doesn't support hardlinks (cross-drive, FAT32,
    ReFS, etc.) we fall back to copy2. Errors are swallowed so a
    materialization failure doesn't poison an otherwise-successful task.
    """
    src   = state.get('output_path', '')
    extras = state.get('extra_paths', []) or []
    if not src or not extras or not os.path.exists(src):
        return
    for dst in extras:
        if not dst or dst == src:
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                continue  # already materialized from a previous run
            try:
                os.link(src, dst)
            except (OSError, NotImplementedError, AttributeError):
                shutil.copy2(src, dst)
        except Exception as e:
            # Best-effort: log but don't fail the parent task.
            print(f"Could not materialize mirror {dst}: {e}")


def _summarize_error(msg: str) -> str:
    """Reduce a typical exception blob to one short, actionable line.

    Network errors out of requests / yt-dlp are long ConnectionPool dumps
    where the actual cause is buried. This walks a small set of known
    patterns and falls back to the first line truncated to 120 chars.
    """
    if not msg:
        return ""
    low = msg.lower()
    if any(s in low for s in ("getaddrinfo failed", "name or service not known",
                              "nameresolutionerror", "name resolution")):
        return "DNS resolution failed (check your VPN / DNS settings)"
    if "winerror 10061" in low or "actively refused" in low:
        return "Connection refused (proxy not running or firewall blocking)"
    if "ssl" in low and ("certificate" in low or "handshake" in low or "tlsv1" in low):
        return "SSL/TLS error (try `yt-dlp -U` or check VPN)"
    if "timed out" in low or "read timed out" in low or "connection timed out" in low:
        return "Connection timed out"
    if "max retries exceeded" in low:
        return "Network unreachable (max retries exceeded)"
    if re.search(r"\b403\b", msg) or "forbidden" in low:
        return "Access denied (HTTP 403)"
    if re.search(r"\b404\b", msg) or "not found" in low:
        return "Video not found (HTTP 404)"
    if "exceeds limit" in low:
        return "Skipped: file size over the configured limit"
    if "video unavailable" in low or "private video" in low or "removed" in low:
        return "Video unavailable / removed by uploader"
    head = msg.split("\n")[0].strip()
    return head[:120] + ("…" if len(head) > 120 else "")


class DownloadManager:
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.tasks: dict = {}                 # task_id -> live state dict
        self.futures: dict = {}               # task_id -> Future (for proper cancellation)
        self.history: list = []               # list of completed/failed/cancelled snapshots

    # ── Adding & launching tasks ────────────────────────────────────────
    def add_download(self, url: str, output_path: str, quality: str,
                     source: str = 'youtube', max_size_mb: float = None,
                     strict_quality: bool = False, normalize: bool = False,
                     extra_paths: Optional[list] = None) -> str:
        """Queue a download.

        ``extra_paths`` is a list of additional output paths the same file
        should appear at. The download runs once (saving to ``output_path``);
        on success the file is hardlinked (or copied) to each extra path.
        Used by the director flow to avoid downloading the same clip
        multiple times when it was selected for several shots.
        """
        task_id = str(uuid.uuid4())

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Dedup against currently-active tasks (same URL + path).
        # Errored / cancelled / completed tasks are NOT considered active —
        # the user can re-add to retry, possibly with new settings.
        for existing in self.tasks.values():
            if (existing['url'] == url and existing['output_path'] == output_path
                    and existing['status'] not in ('cancelled', 'error', 'completed')):
                # Merge any new extra_paths into the existing task so a
                # second add for the same URL doesn't lose mirror requests.
                if extra_paths:
                    have = set(existing.get('extra_paths') or [])
                    have.update(p for p in extra_paths if p)
                    existing['extra_paths'] = sorted(have)
                return existing['id']

        state = {
            'id':              task_id,
            'url':             url,
            'output_path':     output_path,
            'quality':         quality,
            'source':          source,
            'status':          'queued',
            'progress':        0.0,
            'error_msg':       None,
            'error_summary':   None,
            'max_size_mb':     max_size_mb,
            'strict_quality':  strict_quality,
            'normalize':       normalize,
            'speed':           None,
            'eta':             None,
            'attempts':        0,
            'extra_paths':     list(extra_paths) if extra_paths else [],
        }
        self.tasks[task_id] = state
        return task_id

    def start_download(self, task_id: str) -> None:
        state = self.tasks.get(task_id)
        if not state:
            return
        if state['status'] not in ('queued', 'error', 'cancelled'):
            return
        state['status']        = 'queued'
        state['progress']      = 0.0
        state['error_msg']     = None
        state['error_summary'] = None
        state['attempts']      = state.get('attempts', 0) + 1
        future = self.executor.submit(self._run_task, state)
        self.futures[task_id] = future

    def _run_task(self, state: dict) -> None:
        """Wrapper that calls the right downloader and uniformly captures errors.

        The underlying downloaders already set ``state['status']`` to
        'error' / 'completed' / 'cancelled' themselves; this is just a
        defensive net for unexpected exceptions, and the place where we
        compute the user-facing ``error_summary``.
        """
        try:
            if state['source'] == 'youtube':
                download_video(
                    state['url'], state['output_path'], state['quality'], state,
                    max_size_mb=state.get('max_size_mb'),
                    strict_quality=state.get('strict_quality'),
                    normalize=state.get('normalize'),
                )
            else:
                download_direct_video(
                    state['url'], state['output_path'], state,
                    max_size_mb=state.get('max_size_mb'),
                    normalize=state.get('normalize'),
                )
        except Exception as e:
            state['status']    = 'error'
            state['error_msg'] = str(e)

        if state['status'] == 'error' and state.get('error_msg'):
            state['error_summary'] = _summarize_error(state['error_msg'])
        elif state['status'] == 'completed' and state.get('extra_paths'):
            _materialize_extras(state)

    # ── Pause / resume / cancel ─────────────────────────────────────────
    def pause_download(self, task_id: str) -> None:
        state = self.tasks.get(task_id)
        if state and state['status'] == 'downloading':
            state['status'] = 'paused'

    def resume_download(self, task_id: str) -> None:
        state = self.tasks.get(task_id)
        if state and state['status'] == 'paused':
            state['status'] = 'downloading'

    def cancel_download(self, task_id: str) -> None:
        state = self.tasks.get(task_id)
        if state and state['status'] in ('downloading', 'paused', 'queued'):
            state['status'] = 'cancelled'

    def cancel_all(self) -> None:
        for state in self.tasks.values():
            if state['status'] in ('downloading', 'paused', 'queued'):
                state['status'] = 'cancelled'

    # ── Retry ────────────────────────────────────────────────────────────
    def retry_failed(self, task_id: str, overrides: Optional[dict] = None) -> bool:
        """Retry a single failed task, optionally with new settings.

        Returns True if the retry was scheduled, False if blocked
        (task missing, not in 'error' state, or attempts exhausted).
        """
        state = self.tasks.get(task_id)
        if not state or state['status'] != 'error':
            return False
        if state.get('attempts', 0) >= MAX_RETRIES:
            return False
        if overrides:
            for k, v in overrides.items():
                if k in state and v is not None:
                    state[k] = v
        # Critical: drop any partial file so the next attempt doesn't
        # try to "resume" with a wrong byte offset and corrupt the merge.
        if os.path.exists(state['output_path']):
            try:
                os.remove(state['output_path'])
            except Exception:
                pass
        self.start_download(task_id)
        return True

    def retry_all_failed(self, overrides: Optional[dict] = None) -> int:
        """Retry every failed task with optional new settings. Returns count actually scheduled."""
        n = 0
        for task_id in list(self.tasks.keys()):
            if self.retry_failed(task_id, overrides=overrides):
                n += 1
        return n

    # ── Inspection ──────────────────────────────────────────────────────
    def get_all_tasks(self) -> list:
        return list(self.tasks.values())

    def get_active_tasks(self) -> list:
        return [t for t in self.tasks.values()
                if t['status'] in ('downloading', 'paused', 'processing')]

    def get_failed_tasks(self) -> list:
        return [t for t in self.tasks.values() if t['status'] == 'error']

    def get_completed_tasks(self) -> list:
        return [t for t in self.tasks.values() if t['status'] == 'completed']

    def get_history(self) -> list:
        """All completed/failed/cancelled snapshots, including past batches."""
        # Include archived history plus the current batch's terminal tasks.
        terminal_now = [dict(t) for t in self.tasks.values()
                        if t['status'] in ('completed', 'error', 'cancelled')]
        return list(self.history) + terminal_now

    def get_stats(self) -> dict:
        stats = {'total': len(self.tasks), 'queued': 0, 'downloading': 0,
                 'paused': 0, 'processing': 0, 'completed': 0,
                 'error': 0, 'cancelled': 0}
        for t in self.tasks.values():
            stats[t['status']] = stats.get(t['status'], 0) + 1
        return stats

    def can_retry(self, task_id: str) -> bool:
        state = self.tasks.get(task_id)
        if not state or state['status'] != 'error':
            return False
        return state.get('attempts', 0) < MAX_RETRIES

    # ── Reset ───────────────────────────────────────────────────────────
    def clear_and_reset(self) -> None:
        """Cancel in-flight downloads, archive terminal tasks to history,
        wait briefly for workers to honor the cancellation, then reset state."""
        # Archive terminal tasks so the user doesn't lose their record across batches.
        for state in self.tasks.values():
            if state['status'] in ('completed', 'error', 'cancelled'):
                self.history.append(dict(state))

        self.cancel_all()

        # Give the workers up to 2 seconds to honor the cancellation flag —
        # the downloaders poll their state periodically, so most exit promptly.
        # We don't block longer than that to keep the UI responsive.
        for fut in list(self.futures.values()):
            try:
                fut.result(timeout=2.0)
            except concurrent.futures.TimeoutError:
                pass
            except Exception:
                pass

        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            self.executor.shutdown(wait=False)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.tasks.clear()
        self.futures.clear()

    def clear_history(self) -> None:
        self.history.clear()

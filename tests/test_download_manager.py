import threading
from concurrent.futures import CancelledError

from core.download_manager import DownloadManager


def test_cancel_queued_task_never_starts(monkeypatch, tmp_path):
    started = []
    first_started = threading.Event()
    release_first = threading.Event()

    def fake_direct_download(url, output_path, task_state, **kwargs):
        started.append(output_path)
        task_state["status"] = "downloading"
        first_started.set()
        release_first.wait(timeout=2)
        task_state["status"] = "completed"
        task_state["progress"] = 1.0

    monkeypatch.setattr("core.download_manager.download_direct_video", fake_direct_download)

    manager = DownloadManager(max_workers=1)
    try:
        first = manager.add_download(
            "https://example.test/one.mp4",
            str(tmp_path / "one.mp4"),
            "Best",
            source="direct",
        )
        second = manager.add_download(
            "https://example.test/two.mp4",
            str(tmp_path / "two.mp4"),
            "Best",
            source="direct",
        )

        manager.start_download(first)
        assert first_started.wait(timeout=2)
        manager.start_download(second)
        manager.cancel_download(second)
        release_first.set()

        manager.futures[first].result(timeout=2)
        if second in manager.futures:
            try:
                manager.futures[second].result(timeout=2)
            except CancelledError:
                pass

        assert manager.tasks[second]["status"] == "cancelled"
        assert str(tmp_path / "two.mp4") not in started
    finally:
        release_first.set()
        manager.clear_and_reset()

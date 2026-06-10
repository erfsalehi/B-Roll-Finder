import requests

from core.direct_downloader import download_direct_video


class FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield b"x" * 2048


def test_size_limit_error_removes_partial_file(monkeypatch, tmp_path):
    monkeypatch.setattr("core.direct_downloader.requests.get", lambda *a, **k: FakeResponse())
    monkeypatch.setattr("core.direct_downloader.requests.head", lambda *a, **k: FakeResponse())

    output_path = tmp_path / "clip.mp4"
    state = {}

    download_direct_video(
        "https://example.test/clip.mp4",
        str(output_path),
        state,
        max_size_mb=0.0001,
    )

    assert state["status"] == "error"
    assert "exceeded limit" in state["error_msg"].lower()
    assert not output_path.exists()
    assert not (tmp_path / "clip.mp4.part").exists()


def test_success_publishes_final_file_with_no_part_leftover(monkeypatch, tmp_path):
    monkeypatch.setattr("core.direct_downloader.requests.get", lambda *a, **k: FakeResponse())

    output_path = tmp_path / "clip.mp4"
    state = {}

    download_direct_video("https://example.test/clip.mp4", str(output_path), state)

    assert state["status"] == "completed"
    assert output_path.read_bytes() == b"x" * 2048
    assert not (tmp_path / "clip.mp4.part").exists()


def test_midstream_failure_never_leaves_a_final_file(monkeypatch, tmp_path):
    """A download that dies mid-stream must not leave anything at the final
    path — otherwise download_selected_clips' size>0 skip check would treat
    the truncated file as a completed clip on the next run."""

    class DyingResponse(FakeResponse):
        def iter_content(self, chunk_size):
            yield b"x" * 1024
            raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr("core.direct_downloader.requests.get", lambda *a, **k: DyingResponse())
    monkeypatch.setattr("core.direct_downloader.time.sleep", lambda *_: None)

    output_path = tmp_path / "clip.mp4"
    state = {}

    download_direct_video("https://example.test/clip.mp4", str(output_path), state,
                          max_retries=2)

    assert state["status"] == "error"
    assert not output_path.exists()
    # The partial stays as a .part so a retry can resume from it.
    assert (tmp_path / "clip.mp4.part").exists()


def test_resume_appends_to_existing_part_file(monkeypatch, tmp_path):
    """A leftover .part is resumed via Range and the final file holds the
    old + new bytes."""

    seen_headers = {}

    class PartialResponse(FakeResponse):
        status_code = 206

        def iter_content(self, chunk_size):
            yield b"y" * 512

    def fake_get(url, headers=None, **kwargs):
        seen_headers.update(headers or {})
        return PartialResponse()

    monkeypatch.setattr("core.direct_downloader.requests.get", fake_get)

    output_path = tmp_path / "clip.mp4"
    (tmp_path / "clip.mp4.part").write_bytes(b"x" * 1024)
    state = {}

    download_direct_video("https://example.test/clip.mp4", str(output_path), state)

    assert seen_headers.get("Range") == "bytes=1024-"
    assert state["status"] == "completed"
    assert output_path.read_bytes() == b"x" * 1024 + b"y" * 512
    assert not (tmp_path / "clip.mp4.part").exists()

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

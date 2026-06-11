"""YouTube proxy pool: parsing, round-robin rotation, and download failover."""

import os
import core.youtube as yt


def test_youtube_proxies_parsing(monkeypatch):
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY", "http://a:1, http://b:2\n socks5://c:3 ;http://d:4")
    assert yt.youtube_proxies() == ["http://a:1", "http://b:2", "socks5://c:3", "http://d:4"]
    assert yt.youtube_proxy() == "http://a:1"
    monkeypatch.delenv("YT_DLP_PROXY", raising=False)
    assert yt.youtube_proxies() == []
    assert yt.youtube_proxy() == ""


def test_youtube_proxy_alias(monkeypatch):
    monkeypatch.delenv("YT_DLP_PROXY", raising=False)
    monkeypatch.setenv("YOUTUBE_PROXY", "http://only:1")
    assert yt.youtube_proxies() == ["http://only:1"]


def test_youtube_proxy_round_robin(monkeypatch):
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY", "http://a, http://b, http://c")
    monkeypatch.setattr(yt, "_proxy_rr_index", 0)
    picks = [yt._next_youtube_proxy() for _ in range(4)]
    assert picks == ["http://a", "http://b", "http://c", "http://a"]


def test_youtube_proxy_opts(monkeypatch):
    monkeypatch.delenv("YT_DLP_PROXY", raising=False)
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    assert yt._youtube_proxy_opts() == {}
    assert yt._youtube_proxy_opts("http://x:1") == {"proxy": "http://x:1"}
    assert yt._youtube_proxy_opts("") == {}


class _FakeYDL:
    """Minimal yt-dlp stand-in: fails for proxies in ``_dead``, writes a file
    otherwise, and records every proxy it was asked to use."""
    _dead = set()
    _used = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        return {}

    def download(self, urls):
        proxy = self.opts.get("proxy")
        _FakeYDL._used.append(proxy)
        if proxy in _FakeYDL._dead:
            raise Exception("Unable to connect to proxy: timed out")
        with open(self.opts["outtmpl"], "wb") as f:
            f.write(b"ok")


def test_download_video_fails_over_to_backup_proxy(monkeypatch, tmp_path):
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY", "http://A:1, http://B:1")
    monkeypatch.setattr(yt, "_proxy_rr_index", 0)   # first pick = A
    _FakeYDL._dead = {"http://A:1"}                  # A is down
    _FakeYDL._used = []
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", _FakeYDL)

    out = str(tmp_path / "v.mp4")
    ts: dict = {}
    yt.download_video("https://y/1", out, "360", ts, no_audio=True)

    assert ts["status"] == "completed"
    assert _FakeYDL._used == ["http://A:1", "http://B:1"]   # A failed → B
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_download_video_proxy_failover_capped(monkeypatch, tmp_path):
    # All proxies dead + cap=1 → exactly 2 attempts (initial + 1 failover), then error.
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY", "http://A:1, http://B:1, http://C:1")
    monkeypatch.setenv("YT_PROXY_MAX_FAILOVER", "1")
    monkeypatch.setattr(yt, "_proxy_rr_index", 0)
    _FakeYDL._dead = {"http://A:1", "http://B:1", "http://C:1"}
    _FakeYDL._used = []
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", _FakeYDL)

    ts: dict = {}
    yt.download_video("https://y/1", str(tmp_path / "v.mp4"), "360", ts, no_audio=True)

    assert ts["status"] == "error"
    assert len(_FakeYDL._used) == 2          # initial + one failover (capped)

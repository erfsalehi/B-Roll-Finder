"""YouTube proxy pool: parsing, round-robin rotation, and download failover."""

import os

import pytest

import core.youtube as yt


def test_youtube_proxies_parsing(monkeypatch):
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)
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


def test_search_goes_direct_by_default(monkeypatch):
    # Search must NOT use a proxy by default — only playback is IP-blocked, and
    # routing search through free proxies makes it time out.
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY", "http://p:1")
    monkeypatch.delenv("YT_SEARCH_USE_PROXY", raising=False)
    assert yt._search_proxy_opts() == {}                 # direct
    monkeypatch.setenv("YT_SEARCH_USE_PROXY", "1")
    monkeypatch.setattr(yt, "_proxy_rr_index", 0)
    assert yt._search_proxy_opts() == {"proxy": "http://p:1"}   # opt-in


# ── dynamic proxy list (YT_DLP_PROXY_URL) ─────────────────────────────────────

def test_parse_proxy_lines():
    text = "http://1.2.3.4:8080\nsocks5://5.6.7.8:1080\n# a comment\n\n9.10.11.12:3128\n"
    assert yt._parse_proxy_lines(text) == [
        "http://1.2.3.4:8080", "socks5://5.6.7.8:1080", "http://9.10.11.12:3128"]


def test_dynamic_proxies_fetch_merge_and_cache(monkeypatch):
    import requests
    monkeypatch.setenv("YT_DLP_PROXY", "http://static:1")
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setattr(yt, "_proxy_url_cache", {"ts": 0.0, "list": []})

    calls = {"n": 0}

    class _R:
        text = "http://static:1\nhttp://dyn:2\n"   # static dup is de-duped
        def raise_for_status(self):
            pass

    def _get(url, timeout=20):
        calls["n"] += 1
        return _R()
    monkeypatch.setattr(requests, "get", _get)

    assert yt.youtube_proxies() == ["http://static:1", "http://dyn:2"]
    yt.youtube_proxies()                     # within TTL → served from cache
    assert calls["n"] == 1


def test_dynamic_proxies_fetch_error_is_safe(monkeypatch):
    import requests
    monkeypatch.delenv("YT_DLP_PROXY", raising=False)
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setattr(yt, "_proxy_url_cache", {"ts": 0.0, "list": []})

    def _boom(*a, **k):
        raise Exception("list host down")
    monkeypatch.setattr(requests, "get", _boom)

    assert yt.youtube_proxies() == []        # no crash, no last-good → empty


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


def test_next_proxy_uses_pool_when_active(monkeypatch):
    import core.proxy_pool as pp
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setattr(pp, "get_proxy", lambda: "http://pooled:1")
    assert yt._next_youtube_proxy() == "http://pooled:1"


def test_download_failover_pool_mode_marks_dead(monkeypatch, tmp_path):
    import core.proxy_pool as pp
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.delenv("YT_DLP_PROXY", raising=False)
    monkeypatch.setattr(yt, "youtube_proxies", lambda: [])    # avoid real list fetch
    monkeypatch.setattr(pp, "pool_active", lambda: True)

    seq = iter(["http://P1", "http://P2"])
    monkeypatch.setattr(pp, "get_proxy", lambda: next(seq, "http://P2"))
    dead = []
    monkeypatch.setattr(pp, "mark_dead", lambda p: dead.append(p))

    used = []

    class _FakeYDL:
        def __init__(self, opts):
            self.opts = opts
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def extract_info(self, u, download=False):
            return {}
        def download(self, urls):
            p = self.opts.get("proxy")
            used.append(p)
            if p == "http://P1":
                raise Exception("Unable to connect to proxy: timed out")
            with open(self.opts["outtmpl"], "wb") as f:
                f.write(b"ok")
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", _FakeYDL)

    ts: dict = {}
    yt.download_video("https://y/1", str(tmp_path / "v.mp4"), "360", ts, no_audio=True)
    assert ts["status"] == "completed"
    assert dead == ["http://P1"]                 # bad proxy evicted from pool
    assert used == ["http://P1", "http://P2"]    # failed over to a pooled proxy


def test_probe_proxy_ok_and_fail(monkeypatch):
    class _OkYDL:
        def __init__(self, opts):
            self.opts = opts
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def extract_info(self, u, download=False):
            return {"formats": [{"url": "http://f", "vcodec": "avc1", "height": 720}]}
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", _OkYDL)
    ok, _ = yt.probe_proxy("https://y/1", "http://A:1")
    assert ok is True

    class _BadYDL(_OkYDL):
        def extract_info(self, u, download=False):
            raise Exception("Unable to connect to proxy: timed out")
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", _BadYDL)
    ok, detail = yt.probe_proxy("https://y/1", "http://A:1")
    assert ok is False and "timed out" in detail


def test_download_video_fails_over_on_block_with_pool(monkeypatch, tmp_path):
    # A multi-proxy pool hops past a proxy whose IP is YouTube-blocked.
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY", "http://A:1, http://B:1")
    monkeypatch.setattr(yt, "_proxy_rr_index", 0)
    _FakeYDL._dead = set()      # not a connection error — a block error:

    class _BlockYDL(_FakeYDL):
        def download(self, urls):
            proxy = self.opts.get("proxy")
            _FakeYDL._used.append(proxy)
            if proxy == "http://A:1":
                raise Exception("ERROR: [youtube] X: Video unavailable. "
                                "This content isn't available.")
            with open(self.opts["outtmpl"], "wb") as f:
                f.write(b"ok")
    _FakeYDL._used = []
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", _BlockYDL)

    ts: dict = {}
    yt.download_video("https://y/1", str(tmp_path / "v.mp4"), "360", ts, no_audio=True)
    assert ts["status"] == "completed"
    assert _FakeYDL._used == ["http://A:1", "http://B:1"]


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


# ── stall watchdog ────────────────────────────────────────────────────────────

class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _watch(monkeypatch, **env):
    for k in ("YT_STALL_SECONDS", "YT_STALL_MIN_KBPS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    clock = _Clock()
    monkeypatch.setattr(yt, "_stall_clock", clock)
    return yt._StallWatch(), clock


def test_stall_watch_raises_when_a_window_moved_too_few_bytes(monkeypatch):
    w, clock = _watch(monkeypatch)                     # 25 s window, 40 KB/s floor
    w.feed(0)
    clock.t += 24
    w.feed(100_000)                                    # window not over yet
    clock.t += 2
    with pytest.raises(yt.ProxyStall, match="proxy stalled"):
        w.feed(150_000)                                # 150 KB in 26 s = ~5.6 KB/s


def test_stall_watch_lets_a_healthy_download_through_across_windows(monkeypatch):
    w, clock = _watch(monkeypatch)
    got = 0
    for _ in range(10):                                # 250 s at ~200 KB/s
        clock.t += 5
        got += 1_000_000
        w.feed(got)


def test_stall_watch_restarts_its_window_on_a_new_stream(monkeypatch):
    w, clock = _watch(monkeypatch)
    w.feed(0)
    clock.t += 20
    w.feed(5_000_000)                                  # the video stream was fast
    clock.t += 10
    w.feed(100)                                        # audio stream begins: counter went back
    clock.t += 20
    w.feed(2_000_000)                                  # fast again — no false stall


def test_stall_watch_can_be_switched_off(monkeypatch):
    w, clock = _watch(monkeypatch, YT_STALL_SECONDS=0)
    w.feed(0)
    clock.t += 500
    w.feed(1)


def test_a_stalled_proxy_hands_the_clip_to_the_next_proxy(monkeypatch, tmp_path):
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY", "http://A:1, http://B:1")
    monkeypatch.setattr(yt, "_proxy_rr_index", 0)
    _w, clock = _watch(monkeypatch)
    seen = {}

    class _SlowYDL(_FakeYDL):
        def download(self, urls):
            proxy = self.opts.get("proxy")
            _FakeYDL._used.append(proxy)
            seen[proxy] = (self.opts["socket_timeout"], self.opts["retries"])
            hook = self.opts["progress_hooks"][0]
            if proxy == "http://A:1":                  # trickles: 1 KB every 5 s
                for i in range(10):
                    clock.t += 5
                    hook({"status": "downloading", "downloaded_bytes": 1024 * (i + 1),
                          "total_bytes": 10_000_000})
            with open(self.opts["outtmpl"], "wb") as f:
                f.write(b"ok")
    _FakeYDL._dead, _FakeYDL._used = set(), []
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", _SlowYDL)

    ts: dict = {}
    yt.download_video("https://y/1", str(tmp_path / "v.mp4"), "360", ts, no_audio=True)
    assert ts["status"] == "completed"
    assert _FakeYDL._used == ["http://A:1", "http://B:1"]
    assert seen["http://A:1"] == (20, 5)               # fail-fast settings with a failover available


def test_a_lone_proxy_keeps_the_generous_retries_and_is_never_stall_aborted(monkeypatch, tmp_path):
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)
    monkeypatch.setenv("YT_DLP_PROXY", "http://A:1")
    _w, clock = _watch(monkeypatch)
    seen = {}

    class _SlowYDL(_FakeYDL):
        def download(self, urls):
            seen["opts"] = (self.opts["socket_timeout"], self.opts["retries"])
            hook = self.opts["progress_hooks"][0]
            for i in range(10):
                clock.t += 5
                hook({"status": "downloading", "downloaded_bytes": 1024 * (i + 1),
                      "total_bytes": 10_000_000})
            with open(self.opts["outtmpl"], "wb") as f:
                f.write(b"ok")
    monkeypatch.setattr(yt.yt_dlp, "YoutubeDL", _SlowYDL)

    ts: dict = {}
    yt.download_video("https://y/1", str(tmp_path / "v.mp4"), "360", ts, no_audio=True)
    assert ts["status"] == "completed" and seen["opts"] == (60, 30)

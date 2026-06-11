"""Preflight self-test (/test) — aggregation logic + bot command wiring.

The individual checks call the real LLM/yt-dlp/Pexels functions; here we stub
those so the test is hermetic and assert run_self_test's pass/fail/skip rollup
behaves (critical failures fail the run, missing-key checks are skipped, not
failed)."""

import core.selftest as st


def _patch_all(monkeypatch, *, yt_dl_ok=True, ffmpeg=True, pexels_key=True):
    import core.keywords, core.transcription, core.stock_apis, core.direct_downloader
    import core.director_search, core.youtube

    monkeypatch.setattr(st.shutil, "which",
                        lambda n: "/usr/bin/ffmpeg" if ffmpeg else None)
    monkeypatch.setattr(st, "_make_test_audio",
                        lambda: "test.wav" if ffmpeg else None)
    monkeypatch.setattr(core.keywords, "generate_video_topic",
                        lambda t, k: "Coffee mornings")
    monkeypatch.setattr(core.transcription, "transcribe_audio", lambda p, k: [])
    if pexels_key:
        monkeypatch.setenv("PEXELS_API_KEY", "px")
    else:
        monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setattr(core.stock_apis, "search_pexels",
                        lambda *a, **k: [{"url": "http://p/v.mp4"}])

    def _direct(url, out, ts, **k):
        with open(out, "wb") as f:
            f.write(b"x" * 4096)
        ts["status"] = "completed"
    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _direct)

    monkeypatch.setattr(core.director_search, "search_youtube_classic",
                        lambda kw, num_results=4, errors=None: [
                            {"url": "https://y/1", "duration": 80, "title": "A"},
                            {"url": "https://y/2", "duration": 90, "title": "B"}])

    def _ytdl(url, out, quality, ts, no_audio=True, **k):
        if yt_dl_ok:
            with open(out, "wb") as f:
                f.write(b"y" * 4096)
            ts["status"] = "completed"
        else:
            ts["status"] = "error"
            ts["error_msg"] = "Video unavailable. This content isn't available."
    monkeypatch.setattr(core.youtube, "download_video", _ytdl)
    monkeypatch.setattr(core.youtube, "cookie_mode", lambda: (True, "file ok"))


def _by_name(report):
    return {r["name"]: r for r in report["results"]}


def test_self_test_all_pass(monkeypatch):
    _patch_all(monkeypatch)
    report = st.run_self_test(do_downloads=True)
    assert report["ok"] is True
    assert report["errors"] == []
    res = _by_name(report)
    assert res["YouTube download (yt-dlp)"]["ok"] is True
    assert res["Pexels download"]["ok"] is True


def test_self_test_fails_on_youtube_download(monkeypatch):
    _patch_all(monkeypatch, yt_dl_ok=False)
    report = st.run_self_test(do_downloads=True)
    assert report["ok"] is False
    assert any("YouTube download" in e for e in report["errors"])
    # The dead-video reason is surfaced so the user can see it.
    assert any("unavailable" in e.lower() for e in report["errors"])


def test_self_test_skips_pexels_without_key(monkeypatch):
    _patch_all(monkeypatch, pexels_key=False)
    report = st.run_self_test(do_downloads=True)
    res = _by_name(report)
    assert res["Pexels search"]["ok"] is None        # skipped, not failed
    assert res["Pexels download"]["ok"] is None
    assert report["ok"] is True                       # skip never fails the run


def test_self_test_quick_skips_downloads(monkeypatch):
    _patch_all(monkeypatch)
    report = st.run_self_test(do_downloads=False)
    names = {r["name"] for r in report["results"]}
    assert "YouTube download (yt-dlp)" not in names
    assert "YouTube search (yt-dlp)" in names         # cheap checks still run


def test_self_test_fails_without_ffmpeg(monkeypatch):
    _patch_all(monkeypatch, ffmpeg=False)
    report = st.run_self_test(do_downloads=False)
    res = _by_name(report)
    assert res["ffmpeg"]["ok"] is False
    assert report["ok"] is False


# ── bot wiring ────────────────────────────────────────────────────────────────

def test_is_test_command_and_format(monkeypatch):
    from bot import telegram_bot as tb
    assert tb.is_test_command("/test")
    assert tb.is_test_command("/preflight")
    assert tb.is_test_command("/test quick")
    assert not tb.is_test_command("/status")

    passed = tb.format_selftest({"ok": True, "results": [
        {"name": "ffmpeg", "ok": True, "critical": True, "detail": "found", "secs": 0.0},
    ], "errors": []})
    assert "passed" in passed.lower()

    failed = tb.format_selftest({"ok": False, "results": [
        {"name": "YouTube download (yt-dlp)", "ok": False, "critical": True,
         "detail": "all failed", "secs": 5.0},
    ], "errors": ["YouTube download (yt-dlp): all failed"]})
    assert "❌" in failed

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
    # The client probe only runs on download failure; stub it so that path stays
    # hermetic (it otherwise hits real yt-dlp).
    monkeypatch.setattr(core.youtube, "probe_download_clients",
                        lambda url, **k: [("no-cookies + android_vr", True, "ok")])
    # Don't let the real (possibly stale) yt-dlp version / update marker flip the
    # rollup in the generic-path tests.
    monkeypatch.setattr(st, "_ytdlp_version_check", lambda: (True, "fresh"))
    monkeypatch.setattr(st, "_ytdlp_update_check", lambda: (True, "last ran today"))
    # The free-list health check is skipped unless YT_DLP_PROXY_URL is set; keep
    # the generic-path tests deterministic regardless of the dev's shell env.
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)


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


def test_ytdlp_version_flags_stale(monkeypatch):
    import yt_dlp
    monkeypatch.setattr(yt_dlp.version, "__version__", "2020.01.01", raising=False)
    ok, detail = st._ytdlp_version_check()
    assert ok is False and "stale" in detail.lower()


def test_ytdlp_version_ok_when_fresh(monkeypatch):
    import yt_dlp
    from datetime import date
    monkeypatch.setattr(yt_dlp.version, "__version__", date.today().isoformat(),
                        raising=False)
    ok, detail = st._ytdlp_version_check()
    assert ok is True


def test_ytdlp_update_check_flags_failure(monkeypatch):
    import core.app_utils as au
    from datetime import date
    monkeypatch.setattr(au, "_read_update_marker",
                        lambda: {"last_check": date.today().isoformat(),
                                 "error": "pip failed: network unreachable"})
    ok, detail = st._ytdlp_update_check()
    assert ok is False and "FAILED" in detail


def test_ytdlp_update_check_ok_when_recent(monkeypatch):
    import core.app_utils as au
    from datetime import date
    monkeypatch.setattr(au, "_read_update_marker",
                        lambda: {"last_check": date.today().isoformat(),
                                 "version": "2026.06.10", "error": ""})
    ok, detail = st._ytdlp_update_check()
    assert ok is True and "2026.06.10" in detail


def test_ytdlp_update_check_flags_stale_marker(monkeypatch):
    import core.app_utils as au
    monkeypatch.setattr(au, "_read_update_marker",
                        lambda: {"last_check": "2026-01-01", "version": "x", "error": ""})
    ok, detail = st._ytdlp_update_check()
    assert ok is False and "days ago" in detail


def test_ytdlp_update_check_skips_when_absent(monkeypatch):
    import core.app_utils as au
    monkeypatch.setattr(au, "_read_update_marker", lambda: {})
    ok, detail = st._ytdlp_update_check()
    assert ok is None      # absent marker is skipped, not a failure


def test_yt_client_probe_reports_working_config(monkeypatch):
    import core.youtube as yt
    monkeypatch.setattr(yt, "probe_download_clients", lambda url, **k: [
        ("cookies + tv/web_safari/web_embedded", False, "This content isn't available"),
        ("no-cookies + android_vr", True, "ok — formats available (max 1080p)"),
    ])
    ok, detail = st._yt_client_probe_check(
        {"yt_download_failed": True, "yt_results": [{"url": "https://y/1"}]})
    assert ok is True
    assert "WORKING" in detail and "android_vr" in detail


def test_yt_client_probe_reports_systemic_block(monkeypatch):
    import core.youtube as yt
    monkeypatch.setattr(yt, "probe_download_clients", lambda url, **k: [
        ("cookies + tv/web_safari/web_embedded", False, "This content isn't available"),
        ("no-cookies + yt-dlp default", False, "This content isn't available"),
        ("no-cookies + android_vr", False, "This content isn't available"),
    ])
    ok, detail = st._yt_client_probe_check(
        {"yt_download_failed": True, "yt_results": [{"url": "https://y/1"}]})
    assert ok is False
    assert "EVERY client failed" in detail


def test_yt_client_probe_skipped_when_downloads_ok():
    ok, _ = st._yt_client_probe_check({"yt_download_failed": False})
    assert ok is None


def test_youtube_proxy_opts(monkeypatch):
    import core.youtube as yt
    monkeypatch.delenv("YT_DLP_PROXY", raising=False)
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    assert yt.youtube_proxy() == ""
    assert yt._youtube_proxy_opts() == {}
    monkeypatch.setenv("YT_DLP_PROXY", "http://u:p@host:8080")
    assert yt._youtube_proxy_opts() == {"proxy": "http://u:p@host:8080"}


def test_mask_proxy_hides_credentials():
    assert st._mask_proxy("http://user:pass@1.2.3.4:8080") == "http://***@1.2.3.4:8080"
    assert st._mask_proxy("socks5://1.2.3.4:1080") == "socks5://1.2.3.4:1080"


def test_probe_message_suggests_proxy_when_unset(monkeypatch):
    import core.youtube as yt
    monkeypatch.delenv("YT_DLP_PROXY", raising=False)
    monkeypatch.delenv("YOUTUBE_PROXY", raising=False)
    monkeypatch.setattr(yt, "probe_download_clients",
                        lambda url, **k: [("cookies + tv", False, "unavailable")])
    ok, detail = st._yt_client_probe_check(
        {"yt_download_failed": True, "yt_results": [{"url": "u"}]})
    assert ok is False and "YT_DLP_PROXY" in detail


def test_probe_message_blames_proxy_when_set(monkeypatch):
    import core.youtube as yt
    monkeypatch.setenv("YT_DLP_PROXY", "http://user:secret@host:8080")
    monkeypatch.setattr(yt, "probe_download_clients",
                        lambda url, **k: [("cookies + tv", False, "unavailable")])
    ok, detail = st._yt_client_probe_check(
        {"yt_download_failed": True, "yt_results": [{"url": "u"}]})
    assert ok is False
    assert "proxy IP is ALSO blocked" in detail
    assert "secret" not in detail        # credentials masked


def test_self_test_failure_runs_probe(monkeypatch):
    _patch_all(monkeypatch, yt_dl_ok=False)
    report = st.run_self_test(do_downloads=True)
    names = {r["name"] for r in report["results"]}
    assert "YouTube client probe" in names      # probe runs on download failure


def test_dynamic_proxy_health_skipped_when_no_url(monkeypatch):
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)
    ok, detail = st._dynamic_proxy_health_check({})
    assert ok is None and "skipped" in detail


def test_dynamic_proxy_health_reports_working(monkeypatch):
    import core.youtube as yt
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setattr(yt, "_dynamic_youtube_proxies",
                        lambda: ["http://a:1", "http://b:2", "http://c:3"])
    monkeypatch.setattr(yt, "probe_proxy",
                        lambda url, p, timeout=10: (p == "http://b:2", "x"))
    ok, detail = st._dynamic_proxy_health_check({"yt_results": [{"url": "https://y/1"}]})
    assert ok is True and "1/3" in detail


def test_dynamic_proxy_health_reports_all_dead(monkeypatch):
    import core.youtube as yt
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setattr(yt, "_dynamic_youtube_proxies", lambda: ["http://a:1", "http://b:2"])
    monkeypatch.setattr(yt, "probe_proxy", lambda url, p, timeout=10: (False, "dead"))
    ok, detail = st._dynamic_proxy_health_check({})
    assert ok is False and "0/2" in detail


def test_dynamic_proxy_health_empty_list(monkeypatch):
    import core.youtube as yt
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setattr(yt, "_dynamic_youtube_proxies", lambda: [])
    ok, detail = st._dynamic_proxy_health_check({})
    assert ok is False and "0 proxies" in detail


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

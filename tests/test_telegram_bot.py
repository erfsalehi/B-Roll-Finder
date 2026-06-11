"""Telegram bot helpers — auth, message parsing, summary formatting."""

import bot.telegram_bot as tb


# ── authorization (fail-closed) ───────────────────────────────────────────────

def test_allowed_user_ids_parses_env(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111, 222 ;333")
    assert tb.allowed_user_ids() == {111, 222, 333}


def test_is_allowed_requires_membership():
    assert tb.is_allowed(111, {111, 222}) is True
    assert tb.is_allowed(999, {111, 222}) is False


def test_is_allowed_empty_allowlist_denies_everyone():
    assert tb.is_allowed(111, set()) is False   # fail-closed


# ── audio extraction ──────────────────────────────────────────────────────────

def test_extract_voice():
    fid, name = tb.extract_audio({"voice": {"file_id": "V1", "file_unique_id": "u"}})
    assert fid == "V1" and name.endswith(".ogg")


def test_extract_audio_track_keeps_filename():
    fid, name = tb.extract_audio({"audio": {"file_id": "A1", "file_name": "myvoice.mp3"}})
    assert fid == "A1" and name == "myvoice.mp3"


def test_extract_audio_document_by_mime_and_ext():
    fid, _ = tb.extract_audio({"document": {"file_id": "D1", "mime_type": "audio/mpeg", "file_name": "x.bin"}})
    assert fid == "D1"
    fid2, _ = tb.extract_audio({"document": {"file_id": "D2", "file_name": "clip.wav"}})
    assert fid2 == "D2"


# ── oversize-file guard (Telegram's 20 MB getFile cap) ────────────────────────

def test_audio_size_reads_file_size():
    assert tb._audio_size({"voice": {"file_id": "V", "file_size": 1234}}) == 1234
    assert tb._audio_size({"audio": {"file_id": "A", "file_size": 5_000_000}}) == 5_000_000
    assert tb._audio_size({"document": {"file_id": "D", "file_size": 42}}) == 42
    assert tb._audio_size({"voice": {"file_id": "V"}}) == 0      # unknown
    assert tb._audio_size({}) == 0


def test_too_big_message_states_limit_and_fixes():
    msg = tb.too_big_message(30 * 1024 * 1024)
    assert "20 MB" in msg
    assert "ffmpeg" in msg                       # concrete fix
    assert "TELEGRAM_API_BASE" in msg


def test_using_local_bot_api(monkeypatch):
    monkeypatch.delenv("TELEGRAM_API_BASE", raising=False)
    assert tb._using_local_bot_api() is False
    monkeypatch.setenv("TELEGRAM_API_BASE", "https://api.telegram.org")
    assert tb._using_local_bot_api() is False    # the public host is still capped
    monkeypatch.setenv("TELEGRAM_API_BASE", "http://localhost:8081")
    assert tb._using_local_bot_api() is True


def test_call_surfaces_telegram_description(monkeypatch):
    import pytest

    class _Resp:
        ok = False
        status_code = 400
        def json(self):
            return {"ok": False, "error_code": 400,
                    "description": "Bad Request: file is too big"}

    monkeypatch.setattr(tb.requests, "post", lambda *a, **k: _Resp())
    monkeypatch.setattr(tb, "_token", lambda: "T")
    monkeypatch.setattr(tb, "_proxies", lambda: None)
    with pytest.raises(tb.requests.HTTPError) as ei:
        tb._call("getFile", file_id="x")
    assert "file is too big" in str(ei.value)     # not just "400 Bad Request"


def test_extract_audio_ignores_non_audio():
    assert tb.extract_audio({"text": "hi"}) == (None, None)
    assert tb.extract_audio({"document": {"file_id": "z", "file_name": "notes.pdf"}}) == (None, None)


# ── project naming ─────────────────────────────────────────────────────────────

def test_project_name_from_filename():
    assert tb.project_name_from("My Voice.mp3") == "My Voice"


def test_project_name_from_fallback():
    assert tb.project_name_from("", fallback="video_x") == "video_x"


def test_project_name_truncates():
    assert len(tb.project_name_from("a" * 200 + ".mp3")) == 50


# ── summary formatting ─────────────────────────────────────────────────────────

def test_format_summary_includes_counts_and_qa():
    result = {
        "n_shots": 40, "n_selected": 38, "n_clips": 180,
        "download": {"ok": 175, "failed": 5, "skipped": 0, "dir": "/d/proj"},
        "qa": {"overall": "Solid timeline.", "issues": [{"slot_id": 3}]},
    }
    out = tb.format_summary("proj", result)
    assert "40" in out and "180" in out
    assert "175 ok" in out and "5 failed" in out
    assert "Solid timeline." in out and "1 flag" in out
    assert "/d/proj" in out


def test_format_summary_without_download_section():
    out = tb.format_summary("p", {"n_shots": 5, "n_selected": 5, "n_clips": 9,
                                  "download": None, "qa": {"overall": "ok", "issues": []}})
    assert "Downloaded:" not in out and "Shots: 5" in out


# ── health / status command ────────────────────────────────────────────────────

def test_is_status_command_matches_variants():
    assert tb.is_status_command("/status")
    assert tb.is_status_command("/health")
    assert tb.is_status_command("/ping")
    assert tb.is_status_command("/status@MyBot")   # group-mention form
    assert not tb.is_status_command("hello")
    assert not tb.is_status_command("")


def test_is_cancel_command_matches_variants():
    assert tb.is_cancel_command("/cancel")
    assert tb.is_cancel_command("/stop")
    assert tb.is_cancel_command("/abort@MyBot")
    assert not tb.is_cancel_command("/status")
    assert not tb.is_cancel_command("hello")


def test_is_zip_command_matches_variants():
    assert tb.is_zip_command("/zip")
    assert tb.is_zip_command("/zip my-proj")     # with an argument
    assert tb.is_zip_command("/package")
    assert not tb.is_zip_command("/status")


def test_human_size():
    assert tb._human_size(512) == "512.0B"
    assert tb._human_size(2048) == "2.0KB"
    assert tb._human_size(5 * 1024 * 1024) == "5.0MB"
    assert tb._human_size(3 * 1024 ** 3) == "3.0GB"


def test_handle_zip_uses_last_project(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda chat, text: sent.append(text))
    monkeypatch.setitem(tb._LAST, "project", "lastproj")
    import core.output
    monkeypatch.setattr(core.output, "zip_project",
                        lambda name, out_path=None: {"path": f"downloads/{name}.zip",
                                                     "size_bytes": 2048, "files": 3})
    tb.handle_zip(123, "/zip")
    assert any("lastproj" in m for m in sent)
    assert any("2.0KB" in m for m in sent)


def test_handle_zip_no_project(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda chat, text: sent.append(text))
    monkeypatch.setitem(tb._LAST, "project", None)
    tb.handle_zip(123, "/zip")
    assert any("No recent project" in m for m in sent)


def test_check_health_reports_keys_and_pipeline(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("PEXELS_API_KEY", "p")
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(tb, "_probe_internet", lambda timeout=8: (True, "reachable: internet"))
    checks = dict((name, (ok, detail)) for name, ok, detail in tb.check_health())
    assert checks["Groq key"][0] is True
    assert checks["Search sources"][0] is True and "Pexels" in checks["Search sources"][1]
    assert checks["Internet"][0] is True
    assert checks["Pipeline"][0] is True   # core.pipeline imports cleanly


def test_check_health_flags_missing_groq(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(tb, "_probe_internet", lambda timeout=8: (True, "ok"))
    checks = dict((name, ok) for name, ok, _ in tb.check_health())
    assert checks["Groq key"] is False


def test_probe_internet_ok_when_any_host_responds(monkeypatch):
    calls = {"n": 0}
    def _head(url, timeout=8, allow_redirects=True, proxies=None):
        calls["n"] += 1
        if "google" in url:
            return object()          # responded
        raise OSError("no route")    # others unreachable
    monkeypatch.setattr(tb.requests, "head", _head)
    ok, detail = tb._probe_internet()
    assert ok is True and "internet" in detail and "unreachable" in detail


def test_probe_internet_fails_when_all_down(monkeypatch):
    monkeypatch.setattr(tb.requests, "head",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    ok, _ = tb._probe_internet()
    assert ok is False


def test_format_health_shows_busy_state():
    checks = [("Groq key", True, "set")]
    out = tb.format_health(checks, busy={"active": True, "project": "myvid"})
    assert "Busy" in out and "myvid" in out
    idle = tb.format_health(checks, busy={"active": False})
    assert "Idle" in idle

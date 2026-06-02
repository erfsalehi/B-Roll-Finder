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
    def _head(url, timeout=8, allow_redirects=True):
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

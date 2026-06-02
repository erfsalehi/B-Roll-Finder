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

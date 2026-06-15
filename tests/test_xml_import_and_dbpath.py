"""Feeding an exported FCP7 XML to the bot (learn trims) + the env-configurable
Clip Library DB path."""

import bot.telegram_bot as tb
import core.clip_library as lib


# ── XML upload routing ───────────────────────────────────────────────────────

def test_extract_xml_doc_matches_xml_and_captions():
    assert tb.extract_xml_doc({"document": {"file_id": "f", "file_name": "timeline.xml"}}) == ("f", "timeline.xml")
    # caption-driven, any extension
    fid, _ = tb.extract_xml_doc({"document": {"file_id": "g", "file_name": "edit.fcpxml"},
                                 "caption": "/import"})
    assert fid == "g"
    # not an xml / no caption → ignored
    assert tb.extract_xml_doc({"document": {"file_id": "h", "file_name": "notes.txt"}}) == (None, None)
    assert tb.extract_xml_doc({}) == (None, None)


def test_handle_xml_upload_learns_trims(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda chat, text: sent.append(text))
    monkeypatch.setattr(tb, "download_telegram_file", lambda fid, dest: dest)
    import core.xml_reimport as xr
    monkeypatch.setattr(xr, "ingest_reimported_xml", lambda src: {
        "parsed": 5, "video": 4, "matched": 3, "created": 1, "recorded": 4,
        "unmatched": [], "skipped_non_video": 1,
    })
    tb.handle_xml_upload(123, "fileid", "timeline.xml")
    blob = "\n".join(sent)
    assert "Learned trims" in blob
    assert "preferred trims recorded: 4" in blob


def test_handle_xml_upload_reports_parse_error(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda chat, text: sent.append(text))
    monkeypatch.setattr(tb, "download_telegram_file", lambda fid, dest: dest)
    import core.xml_reimport as xr
    monkeypatch.setattr(xr, "ingest_reimported_xml",
                        lambda src: (_ for _ in ()).throw(ValueError("bad xml")))
    tb.handle_xml_upload(123, "fileid", "broken.xml")
    assert any("Couldn't parse" in m for m in sent)


# ── Configurable / persistent DB path ────────────────────────────────────────

def test_db_path_prefers_env(monkeypatch, tmp_path):
    target = str(tmp_path / "server" / "clip_library.db")
    monkeypatch.setenv("CLIP_LIBRARY_DB", target)
    assert lib._db_path() == target


def test_db_path_falls_back_to_module_default(monkeypatch):
    monkeypatch.delenv("CLIP_LIBRARY_DB", raising=False)
    monkeypatch.setattr(lib, "_DB_PATH", "/tmp/fallback.db")
    assert lib._db_path() == "/tmp/fallback.db"


def test_conn_uses_env_db_path(monkeypatch, tmp_path):
    target = tmp_path / "nested" / "lib.db"
    monkeypatch.setenv("CLIP_LIBRARY_DB", str(target))
    lib.init_db()                       # creates the file via _conn()
    assert target.exists()

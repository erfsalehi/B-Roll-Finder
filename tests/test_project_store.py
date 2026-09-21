"""Project store: record what was delivered, then label every clip/image from
the editor's final XML (used on its shot / moved / unused / external)."""

import bot.telegram_bot as tb
from core import project_store as ps


FPS = 30.0


def _item(name, start, end, in_sec=0.0, folder="C:/edit/proj/director"):
    """A parsed clipitem as core.xml_reimport.parse_fcpxml returns it."""
    return {
        "name": name, "local_path": f"{folder}/{name}", "fps": FPS,
        "start_frame": int(start * FPS), "end_frame": int(end * FPS),
        "in_frame": int(in_sec * FPS), "out_frame": int((in_sec + end - start) * FPS),
        "in_seconds": in_sec, "out_seconds": in_sec + end - start,
    }


def _shots():
    return [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 5.0, "text": "intro",
         "selected_results": [
             {"url": "https://yt/a", "local_path": "/srv/d/1-1-a.mp4", "source": "youtube"},
             {"url": "https://px/b", "local_path": "/srv/d/1-2-b.mp4", "source": "pexels"}],
         "images": [{"url": "https://img/1", "local_path": "/srv/i/01-1-x.jpg"}]},
        {"slot_id": 2, "timestamp": 5.0, "end_timestamp": 10.0, "text": "engine",
         "selected_results": [
             {"url": "https://yt/c", "local_path": "/srv/d/2-1-c.mp4"},
             {"url": "https://yt/fail", "local_path": "", "_dl_failed": True}],
         "images": [{"url": "https://img/2", "local_path": "/srv/i/02-1-y.jpg"}]},
        {"slot_id": 3, "timestamp": 10.0, "end_timestamp": 15.0, "text": "outro",
         "selected_results": [{"url": "https://yt/d", "local_path": "/srv/d/3-1-d.mp4"}]},
    ]


def _delivered_project(monkeypatch, exported=None):
    pid = ps.create_project("Camry review", "Camry review", operator_id=42,
                            operator_name="erfan", chat_id=7)
    monkeypatch.setattr(ps, "_exported_placements",
                        lambda path: ({i["name"]: i for i in (exported or [])},
                                      {i["name"] for i in (exported or [])}))
    counts = ps.record_delivery(pid, _shots(), xml_path="x.xml")
    return pid, counts


def test_record_delivery_skips_failed_downloads(monkeypatch):
    pid, counts = _delivered_project(monkeypatch)
    assert counts == {"clips": 4, "images": 2}
    p = ps.get_project(pid)
    assert p["status"] == "delivered" and p["operator_id"] == 42


def test_analyze_labels_each_asset(monkeypatch):
    exported = [_item("1-1-a.mp4", 0, 2.5), _item("1-2-b.mp4", 2.5, 5),
                _item("2-1-c.mp4", 5, 10), _item("3-1-d.mp4", 10, 15),
                _item("overlay-01.mov", 3, 6)]
    pid, _ = _delivered_project(monkeypatch, exported)

    final = [
        _item("1-1-a.mp4", 0, 2.5, in_sec=4.0),     # kept on shot 1, trimmed
        _item("2-1-c.mp4", 11, 14),                 # moved to shot 3
        _item("01-1-x.jpg", 1, 3),                  # image on its shot
        _item("my-own.mp4", 5, 10),                 # editor's own footage
        _item("overlay-01.mov", 3, 6),              # our overlay: ignored
        _item("voice.wav", 0, 15),                  # audio: ignored
    ]
    s = ps.analyze_import(pid, final, imported_by=42, filename="final.xml")

    assert s["clips"] == {"used_here": 1, "used_elsewhere": 1, "unused": 2}
    assert s["images"] == {"used_here": 1, "unused": 1}
    assert s["external_clips"] == 1 and s["external_images"] == 0
    assert s["median_in_move_sec"] == 4.0
    assert s["shots_without_our_footage"] == ["2"]
    assert ps.get_project(pid)["status"] == "learned"

    with ps._conn() as c:
        rows = {r["filename"]: dict(r) for r in c.execute(
            "SELECT filename, verdict, used_slot_id FROM project_assets WHERE project_id=?",
            (pid,))}
    assert rows["2-1-c.mp4"]["verdict"] == "used_elsewhere"
    assert rows["2-1-c.mp4"]["used_slot_id"] == "3"
    assert rows["1-2-b.mp4"]["verdict"] == "unused"


def test_global_shift_is_not_read_as_moves(monkeypatch):
    exported = [_item("1-1-a.mp4", 0, 5), _item("2-1-c.mp4", 5, 10),
                _item("3-1-d.mp4", 10, 15)]
    pid, _ = _delivered_project(monkeypatch, exported)
    # Editor added a 6s cold open: everything slid right by 6s.
    final = [_item("1-1-a.mp4", 6, 11), _item("2-1-c.mp4", 11, 16),
             _item("3-1-d.mp4", 16, 21)]
    s = ps.analyze_import(pid, final)
    assert s["shift_sec"] == 6.0
    assert s["clips"].get("used_elsewhere", 0) == 0
    assert s["clips"]["used_here"] == 3


def test_reimport_replaces_previous_labels(monkeypatch):
    pid, _ = _delivered_project(monkeypatch)
    ps.analyze_import(pid, [_item("1-1-a.mp4", 0, 2)])
    ps.analyze_import(pid, [])
    with ps._conn() as c:
        n = c.execute("SELECT COUNT(*) FROM asset_usage WHERE project_id=?", (pid,)).fetchone()[0]
        verdicts = {r[0] for r in c.execute(
            "SELECT verdict FROM project_assets WHERE project_id=?", (pid,))}
    assert n == 6 and verdicts == {"unused"}


def test_match_counts_finds_the_right_project(monkeypatch):
    pid, _ = _delivered_project(monkeypatch)
    other = ps.create_project("Other")
    counts = ps.match_counts([_item("1-1-a.mp4", 0, 1), _item("02-1-y.jpg", 1, 2),
                              _item("nothing.mp4", 2, 3)])
    assert counts == {pid: 2}
    assert other not in counts


def test_timeline_secs_recovers_transition_edges():
    it = _item("a.mp4", 2, 4)
    it["start_frame"] = -1
    assert ps._timeline_secs(it) == (2.0, 4.0)


# ── bot flow ─────────────────────────────────────────────────────────────────

def test_safe_title():
    assert tb.safe_title("Tesla vs BYD: 2026/27?") == "Tesla vs BYD 2026 27"
    assert tb.safe_title("خودروی برقی") == "خودروی برقی"
    assert tb.safe_title("???") == ""


def test_title_prompt_then_mode_keyboard(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message",
                        lambda chat, text, reply_markup=None: sent.append((text, reply_markup)))
    tb._PENDING_START.clear()
    tb.ask_title(9, "fid", "rec_001.ogg", {"id": 42, "name": "erfan"})
    assert "project title" in sent[-1][0]
    assert sent[-1][1]["inline_keyboard"][0][0]["callback_data"] == "title:default"

    assert tb.apply_title(9, "Camry: first drive", {"id": 1}) is True
    pend = tb._PENDING_START[9]
    assert pend["title"] == "Camry: first drive"
    assert pend["name"] == "Camry first drive.ogg"
    assert pend["operator"] == {"id": 42, "name": "erfan"}   # uploader, not replier
    assert not pend["awaiting_title"]
    assert "start:full:clear" in str(sent[-1][1])
    # No second title once it's set.
    assert tb.apply_title(9, "again") is False
    tb._PENDING_START.clear()


def test_xml_upload_asks_which_project(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message",
                        lambda chat, text, reply_markup=None: sent.append((text, reply_markup)))
    monkeypatch.setattr(tb, "download_telegram_file", lambda fid, dest: dest)
    pid, _ = _delivered_project(monkeypatch)
    import core.xml_reimport as xr
    monkeypatch.setattr(xr, "parse_fcpxml", lambda src: [_item("1-1-a.mp4", 0, 2)])

    tb.handle_xml_upload(5, "fid", "final.xml")
    text, kb = sent[-1]
    assert "Which project" in text
    first = kb["inline_keyboard"][0][0]
    assert first["callback_data"] == f"learn:{pid}" and first["text"].startswith("⭐")
    assert 5 in tb._PENDING_XML
    tb._PENDING_XML.clear()


def test_learn_from_xml_reports_summary(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda chat, text, reply_markup=None: sent.append(text))
    pid, _ = _delivered_project(monkeypatch)
    import core.xml_reimport as xr
    monkeypatch.setattr(xr, "ingest_reimported_xml", lambda src: {"recorded": 1})
    monkeypatch.setattr(xr, "parse_fcpxml", lambda src: [_item("1-1-a.mp4", 0, 2)])
    tb.learn_from_xml(5, "missing.xml", "final.xml", project_id=pid, imported_by=42)
    assert "Learned from the edit of 'Camry review'" in sent[-1]
    assert "1 used on their shot" in sent[-1]

"""Saved projects: snapshots, /download of any project, and reviewing old
projects (backfill + per-project queue)."""

import pytest

import bot.telegram_bot as tb
from core import project_store as ps
from core import ratings


def _cand(key, **kw):
    return dict({"url": key, "page_url": key, "source": "youtube", "title": key}, **kw)


def _shots():
    return [{"slot_id": 1, "timestamp": 0, "end_timestamp": 5, "text": "oil drains out",
             "selected_results": [_cand("yt/pick")],
             "video_results": [_cand("yt/pick"), _cand("yt/r1"), _cand("px/bad", irrelevant=True)]},
            {"slot_id": 2, "timestamp": 5, "end_timestamp": 9, "text": "refill it",
             "selected_results": [_cand("yt/two")], "video_results": [_cand("yt/two")]}]


@pytest.fixture
def sent(monkeypatch):
    out = []
    monkeypatch.setattr(tb, "send_message",
                        lambda chat, text, reply_markup=None: out.append((text, reply_markup)) or {})
    monkeypatch.setattr(tb, "edit_message", lambda *a, **k: None)
    tb._PENDING.clear()
    tb._BUSY.update(active=False)
    yield out
    tb._PENDING.clear()


# ── store ────────────────────────────────────────────────────────────────────

def test_snapshot_roundtrip_and_find():
    pid = ps.create_project("Camry oil change", "Camry oil change")
    ps.create_project("BYD Seal review", "BYD Seal review")
    ps.save_snapshot(pid, {"shots": _shots(), "topic": "cars", "junk": object()}, quality=720)
    snap = ps.load_snapshot(pid)
    assert snap["topic"] == "cars" and snap["quality"] == "720" and "junk" not in snap
    assert [p["id"] for p in ps.find_projects("camry")] == [pid]
    assert ps.find_projects(f"#{pid}")[0]["has_snapshot"] == 1
    assert len(ps.find_projects("")) == 2
    assert "snapshot" not in ps.list_projects()[0]


def test_backfill_queues_old_projects_from_snapshot_or_assets(monkeypatch):
    a = ps.create_project("From snapshot")
    ps.save_snapshot(a, {"shots": _shots()})
    b = ps.create_project("Delivered before reviews")
    monkeypatch.setattr(ps, "_exported_placements", lambda path: ({}, set()))
    ps.record_delivery(b, [{"slot_id": 1, "text": "x", "selected_results": [
        dict(_cand("yt/old"), local_path="/srv/1-1-old.mp4")]}])
    assert ratings.backfill_queue() == 4 + 1        # a: pick, runner-up, rejected, pick · b: pick
    assert ratings.backfill_queue() == 0            # idempotent
    assert ratings.next_task(501, project_id=b)["items"][0]["page_url"] == "yt/old"


def test_project_filter_walks_one_project_in_order():
    a, b = ps.create_project("A"), ps.create_project("B")
    for pid in (a, b):
        ps.record_shots(pid, _shots())
        ratings.record_candidates(pid, _shots())
    t = ratings.next_task(501, project_id=a)
    assert (t["project_id"], t["slot_id"]) == (a, "1")
    for it in t["items"]:
        ratings.save_rating(501, it["id"], "yes")
    assert ratings.next_task(501, project_id=a)["slot_id"] == "2"
    prog = {p["id"]: p for p in ratings.project_progress(501)}
    assert prog[a]["clips_mine"] == 3 and prog[a]["clips"] == 4
    assert prog[b]["clips_mine"] == 0


# ── bot ──────────────────────────────────────────────────────────────────────

def test_review_gate_snapshot_queues_for_reviewers():
    pid = ps.create_project("Paused")
    tb._snapshot_project(pid, {"shots": _shots(), "topic": "t"}, 1080)
    assert ps.load_snapshot(pid)["quality"] == "1080"
    assert ratings.next_task(501, project_id=pid) is not None


def test_download_lookup_single_multiple_none(sent, monkeypatch):
    started = []
    monkeypatch.setattr(tb, "_start_project_download", lambda chat, pid: started.append(pid))
    tb.handle_download_lookup(1, "nothing")
    assert "No saved project matches" in sent[-1][0]

    a = ps.create_project("Camry oil change")
    ps.create_project("Camry brakes")
    tb.handle_download_lookup(1, "oil")
    assert started == [a]
    tb.handle_download_lookup(1, "camry")
    kb = sent[-1][1]["inline_keyboard"]
    assert len(kb) == 3 and kb[0][0]["callback_data"].startswith("dl:")
    assert kb[-1][0]["callback_data"] == "dl:cancel"


def test_download_project_prefers_files_on_disk(sent, monkeypatch):
    pid = ps.create_project("Camry", "Camry")
    delivered, zips = [], []
    monkeypatch.setattr(tb, "deliver_project", lambda chat, proj: delivered.append(proj))
    monkeypatch.setattr(tb, "_send_existing_zip", lambda chat, proj, z: zips.append(z))

    monkeypatch.setattr(tb, "_project_files_on_disk", lambda proj: (True, None))
    tb._run_download_project(1, pid)
    assert delivered == ["Camry"]

    monkeypatch.setattr(tb, "_project_files_on_disk", lambda proj: (False, "/d/Camry.zip"))
    tb._run_download_project(1, pid)
    assert zips == ["/d/Camry.zip"]


def test_download_project_rebuilds_from_snapshot(sent, monkeypatch, tmp_path):
    pid = ps.create_project("Camry", "Camry")
    kept = tmp_path / "ov.mov"
    kept.write_bytes(b"x")
    ps.save_snapshot(pid, {"shots": _shots(), "topic": "cars",
                           "overlays": [{"filepath": str(kept)}, {"filepath": "/gone.mov"}]},
                     quality=720)
    monkeypatch.setattr(tb, "_project_files_on_disk", lambda proj: (False, None))
    calls = {}
    import core.pipeline as pl

    def fake_finalize(shots, proj, **kw):
        calls.update(proj=proj, n=len(shots), **kw)
        return {"download": {"ok": 2}, "xml_path": "/x.xml"}
    monkeypatch.setattr(pl, "finalize_project", fake_finalize)
    done = {}
    monkeypatch.setattr(tb, "_deliver_completed",
                        lambda chat, proj, result: done.update(proj=proj, result=result))
    tb._run_download_project(1, pid)
    assert calls["proj"] == "Camry" and calls["n"] == 2 and calls["quality"] == "720"
    assert calls["overlays"] == [{"filepath": str(kept)}]
    assert done["result"]["project_id"] == pid and done["result"]["xml_path"] == "/x.xml"
    assert any("overlays were cleaned up" in t for t, _ in sent)


def test_download_project_without_snapshot_explains(sent, monkeypatch):
    pid = ps.create_project("Ancient")
    monkeypatch.setattr(tb, "_project_files_on_disk", lambda proj: (False, None))
    tb._run_download_project(1, pid)
    assert "can't be rebuilt" in sent[-1][0]


def test_download_project_paused_elsewhere_uses_that_review_gate(sent, monkeypatch):
    pid = ps.create_project("Paused")
    tb._PENDING[77] = {"project": "Paused", "result": {"project_id": pid}}
    ran = []
    monkeypatch.setattr(tb, "_run_download", lambda chat, pend_chat=None: ran.append((chat, pend_chat)))
    tb._run_download_project(1, pid)
    assert ran == [(1, 77)]

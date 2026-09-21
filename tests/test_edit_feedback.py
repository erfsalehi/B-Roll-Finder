"""Edit feedback → picking: learned labels reach the ranking prompt, demote
clips the editor keeps rejecting, set a default in-point, and order images."""

import pytest

import core.director_rank as dr
import core.edit_feedback as ef
from core import output
from core import project_store as ps


FPS = 30.0


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    # Word-overlap similarity: no embedding model download in tests.
    monkeypatch.setattr(ef, "_embedder", lambda: None)
    monkeypatch.setenv("RANK_JITTER_MAX", "0")


def _item(name, start, end, in_sec=0.0):
    return {"name": name, "local_path": f"C:/edit/{name}", "fps": FPS,
            "start_frame": int(start * FPS), "end_frame": int(end * FPS),
            "in_frame": int(in_sec * FPS), "out_frame": int((in_sec + end - start) * FPS),
            "in_seconds": in_sec, "out_seconds": in_sec + end - start}


def _clip(name, url, source="youtube", title="", channel="", in_rule="default"):
    return {"url": url, "page_url": url, "local_path": f"/srv/{name}", "source": source,
            "title": title or name, "channel": channel, "in_rule": in_rule}


def _learn(monkeypatch, shots, final):
    """Deliver ``shots`` (exported where the final XML has them, else nowhere)
    and learn from ``final``."""
    pid = ps.create_project("p")
    monkeypatch.setattr(ps, "_exported_placements", lambda path: ({}, set()))
    ps.record_delivery(pid, shots)
    ps.analyze_import(pid, final)
    return pid


def _engine_project(monkeypatch):
    shots = [{"slot_id": 1, "timestamp": 0, "end_timestamp": 5,
              "text": "the engine oil drains out of the sump",
              "selected_results": [
                  _clip("1-1-a.mp4", "https://yt/keep", title="Oil change close up",
                        channel="AutoFix Garage"),
                  _clip("1-2-b.mp4", "https://px/drop", source="pexels",
                        title="Woman Drinking Coffee")],
              "images": [{"url": "https://cdn.good.com/1.jpg", "local_path": "/srv/01-1-x.jpg",
                          "page": "https://www.good.com/a"},
                         {"url": "https://bad.com/2.jpg", "local_path": "/srv/01-2-y.jpg",
                          "page": "https://bad.com/b"}]}]
    final = [_item("1-1-a.mp4", 0, 5, in_sec=6.0), _item("01-1-x.jpg", 1, 3)]
    return _learn(monkeypatch, shots, final)


def test_nothing_learned_means_no_history():
    assert ef.load_history() is None
    assert ef.learned_in_offset("youtube") is None


def test_history_note_for_similar_line(monkeypatch):
    _engine_project(monkeypatch)
    h = ef.load_history()
    note = ef.history_note(h, "watch the engine oil drain from the sump")
    assert "EDITOR HISTORY" in note
    assert "kept: [YOUTUBE] Oil change close up" in note
    assert "dropped: [PEXELS] Woman Drinking Coffee" in note
    assert ef.history_note(h, "a cat sleeping on a sofa") == ""


def test_track_record_for_clip_and_channel(monkeypatch):
    _engine_project(monkeypatch)
    _engine_project(monkeypatch)
    h = ef.load_history()
    rec = ef.track_record(h, {"url": "https://yt/keep", "source": "youtube",
                              "description": "by AutoFix Garage — tips"})
    assert "this clip kept 2/2" in rec and "AutoFix Garage kept 2/2" in rec
    assert ef.track_record(h, {"url": "https://new", "source": "youtube"}) == ""


def test_demote_rejected_needs_two_rejections(monkeypatch):
    _engine_project(monkeypatch)
    shot = {"video_results": [{"url": "https://px/drop", "page_url": "https://px/drop"},
                              {"url": "https://other"}]}
    assert ef.demote_rejected([shot]) == 0          # dropped once: benefit of the doubt
    _engine_project(monkeypatch)
    assert ef.demote_rejected([shot]) == 1
    assert shot["video_results"][0]["url"] == "https://other"


def test_ranking_prompt_carries_history(monkeypatch):
    _engine_project(monkeypatch)
    seen = {}

    def fake_llm(client, system_prompt, user_msg, **kw):
        seen["system"], seen["user"] = system_prompt, user_msg
        return {"shots": [{"shot_id": 7, "ranked": [{"index": 0, "reason": "ok"}]}]}
    monkeypatch.setattr(dr, "_call_llm_json", fake_llm)
    shot = {"slot_id": 7, "priority": "medium", "text": "engine oil drains from the sump",
            "shot_intent": "x", "selected_results": [],
            "video_results": [{"url": "https://yt/keep", "source": "youtube",
                               "title": "Oil change", "channel": "AutoFix Garage"}]}
    dr.rank_shot_candidates([shot], api_key="k")
    assert "EDITOR HISTORY" in seen["user"]
    assert "editor record: this clip kept 1/1" in seen["user"]
    assert "EDITOR HISTORY" in seen["system"]          # rule 9 explains it


def test_learned_in_point_habit(monkeypatch):
    clips = [_clip(f"1-{i}-c.mp4", f"https://yt/{i}") for i in range(5)]
    clips.append(_clip("1-9-v.mp4", "https://yt/v", in_rule="verified"))
    shots = [{"slot_id": 1, "timestamp": 0, "end_timestamp": 60, "text": "t",
              "selected_results": clips}]
    final = [_item(f"1-{i}-c.mp4", i * 5, i * 5 + 5, in_sec=6.0 + i * 0.1) for i in range(5)]
    final.append(_item("1-9-v.mp4", 40, 45, in_sec=250.0))   # verified: not a habit
    _learn(monkeypatch, shots, final)
    assert ef.learned_in_offset("youtube") == pytest.approx(6.2)
    assert ef.learned_in_offset("pexels") is None

    # The export uses it for a clip nothing else positioned, and says so.
    monkeypatch.setattr("core.clip_library.find_clip_by_path_or_url", lambda **kw: None)
    cand = {"source": "youtube", "url": "https://yt/new"}
    frame = output._preferred_in_frame("https://yt/new", "x.mp4", 90, 30 * 60, 30.0, cand)
    assert frame == int(round(6.2 * 30)) or abs(frame - 6.2 * 30) <= 1
    assert cand["in_rule"] == "habit"


def test_image_sites_reordered(monkeypatch):
    for _ in range(3):
        _engine_project(monkeypatch)
    cands = [{"url": "https://bad.com/9.jpg", "context": "https://bad.com/z"},
             {"url": "https://x.org/1.jpg", "context": "https://x.org/y"},
             {"url": "https://good.com/5.jpg", "context": "https://good.com/q"}]
    order = [c["url"] for c in ef.order_image_candidates(cands)]
    assert order == ["https://good.com/5.jpg", "https://x.org/1.jpg", "https://bad.com/9.jpg"]


def test_disabled_by_env(monkeypatch):
    _engine_project(monkeypatch)
    monkeypatch.setenv("EDIT_LEARNING", "0")
    assert ef.load_history() is None

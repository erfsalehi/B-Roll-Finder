"""Reviewer ratings: queueing, the rating page API, and how ratings steer
picking (history, demotion, in-points, suggestions, house rules)."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import bot.telegram_bot as tb
import core.director_rank as dr
import core.edit_feedback as ef
from bot import fileserver, rating_page
from core import output
from core import project_store as ps
from core import ratings


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(ef, "_embedder", lambda: None)
    monkeypatch.setattr(ratings, "fetch_title", lambda url: "Suggested video")
    monkeypatch.setenv("RANK_JITTER_MAX", "0")
    monkeypatch.setenv("TELEGRAM_RATERS", "501,502")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "900")


def _cand(key, **kw):
    return dict({"url": key, "page_url": key, "source": "youtube", "title": key}, **kw)


def _project(text="the engine oil drains out of the sump"):
    pid = ps.create_project("Camry")
    shots = [{"slot_id": 1, "timestamp": 0, "end_timestamp": 5, "text": text,
              "selected_results": [_cand("yt/pick")],
              "video_results": [_cand("yt/pick"), _cand("yt/r1"), _cand("yt/r2"),
                                _cand("yt/r3"), _cand("px/bad", source="pexels", irrelevant=True),
                                _cand("px/bad2", irrelevant=True)]},
             {"slot_id": 2, "is_extra": True, "selected_results": [_cand("yt/extra")]}]
    ps.record_delivery(pid, shots)
    return pid, shots


def _items(pid):
    with ps._conn() as c:
        return {r["page_url"]: dict(r) for r in c.execute(
            "SELECT * FROM rating_items WHERE project_id=?", (pid,))}


# ── queueing ─────────────────────────────────────────────────────────────────

def test_record_candidates_queues_picks_runner_ups_and_one_rejected():
    pid, shots = _project()
    assert ratings.record_candidates(pid, shots) == 4
    roles = {k: v["role"] for k, v in _items(pid).items()}
    assert roles == {"yt/pick": "pick", "yt/r1": "runner_up", "yt/r2": "runner_up",
                     "px/bad": "rejected"}
    assert ratings.record_candidates(pid, shots) == 0          # idempotent


def test_next_task_hides_the_bots_choice_and_honours_skip():
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    task = ratings.next_task(501)
    assert task["text"].startswith("the engine oil")
    assert len(task["items"]) == 4
    assert all("role" not in i for i in task["items"])
    assert ratings.next_task(501, skip=[(pid, "1")]) is None


def test_fully_rated_shot_leaves_the_queue_for_that_rater_only():
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    for item in ratings.next_task(501)["items"]:
        ratings.save_rating(501, item["id"], "yes")
    assert ratings.next_task(501) is None
    assert ratings.next_task(502) is not None


def test_save_rating_validates_and_cleans():
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    item = ratings.next_task(501)["items"][0]
    with pytest.raises(ValueError):
        ratings.save_rating(501, item["id"], "maybe")
    with pytest.raises(ValueError):
        ratings.save_rating(501, item["id"], "yes", fit=9)
    ratings.save_rating(501, item["id"], "trim", 4, ["talking_head", "bogus"], "x",
                        [[12, 18], [5, 3], ["a", 1], [1, 2]])
    mine = ratings.next_task(502)  # other rater still sees it
    assert mine is not None
    with ps._conn() as c:
        r = c.execute("SELECT * FROM ratings WHERE item_id=?", (item["id"],)).fetchone()
    assert json.loads(r["reasons"]) == ["talking_head"]
    assert json.loads(r["segments"]) == [[1.0, 2.0], [12.0, 18.0]]


def test_add_suggestion_normalises_youtube():
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    iid = ratings.add_suggestion(501, pid, 1, "https://youtu.be/AbCdEfGhIjK?t=30",
                                 "shows the drain plug", [[30, 36]])
    it = {v["id"]: v for v in _items(pid).values()}[iid]
    assert it["page_url"] == "https://www.youtube.com/watch?v=AbCdEfGhIjK"
    assert it["role"] == "suggested" and it["title"] == "Suggested video"
    with pytest.raises(ValueError):
        ratings.add_suggestion(501, pid, 1, "not a link")


# ── feeding picking ──────────────────────────────────────────────────────────

def _rate_all(pid, rater, usable_by_key, note="", segs=None):
    for key, item in _items(pid).items():
        if key in usable_by_key:
            ratings.save_rating(rater, item["id"], usable_by_key[key], note=note,
                                segments=segs if usable_by_key[key] != "no" else None)


def test_ratings_reach_history_record_and_in_point(monkeypatch):
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    _rate_all(pid, 501, {"yt/r1": "yes", "px/bad": "no"}, note="phone battery, wrong kind",
              segs=[[8, 14]])
    _rate_all(pid, 502, {"yt/r1": "trim", "px/bad": "no"}, segs=[[10, 15]])

    h = ef.load_history()
    note = ef.history_note(h, "engine oil drains from the sump")
    assert "kept: [YOUTUBE] yt/r1" in note
    assert 'reviewer: "phone battery, wrong kind"' in note
    assert "reviewers: usable 2/2" in ef.track_record(h, _cand("yt/r1"))

    shot = {"video_results": [_cand("px/bad"), _cand("fresh")]}
    assert ef.demote_rejected([shot], h) == 1
    assert shot["video_results"][0]["url"] == "fresh"

    assert ef.rated_in_point(_cand("yt/r1"), h) in (8, 10)
    monkeypatch.setattr("core.clip_library.find_clip_by_path_or_url", lambda **kw: None)
    c = _cand("yt/r1")
    frame = output._preferred_in_frame("yt/r1", "x.mp4", 60, 30 * 60, 30.0, c)
    assert c["in_rule"] == "rated" and frame in (240, 300)


def test_other_line_does_not_count_against_the_clip():
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    for r in (501, 502):
        item = _items(pid)["yt/r2"]
        ratings.save_rating(r, item["id"], "no", reasons=["other_line"])
    h = ef.load_history()
    assert not ef._rejected(h, _cand("yt/r2"))


def test_suggestions_offered_on_similar_lines():
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    ratings.add_suggestion(501, pid, 1, "https://www.youtube.com/watch?v=AbCdEfGhIjK",
                           "drain plug close-up", [[30, 36]])
    shot = {"slot_id": 9, "text": "oil drains out of the engine sump",
            "video_results": [_cand("yt/other")]}
    ef.annotate_for_ranking([shot])
    added = [c for c in shot["video_results"] if c.get("reviewer_suggested")]
    assert len(added) == 1
    assert added[0]["verified_in_sec"] == 30
    assert "drain plug close-up" in added[0]["description"]


def test_house_rules_distilled_and_used_in_ranking(monkeypatch):
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    _rate_all(pid, 501, {"px/bad": "no"}, note="presenter talking, not the car")
    import core.keywords as kw
    monkeypatch.setattr(kw, "_call_llm_json",
                        lambda *a, **k: {"rules": ["No presenter footage for car topics"]})
    assert ratings.distill_house_rules() is None               # below threshold
    assert ratings.distill_house_rules(force=True) == ["No presenter footage for car topics"]

    seen = {}
    monkeypatch.setattr(dr, "_call_llm_json", lambda client, system, user, **k: (
        seen.update(system=system) or {"shots": []}))
    dr.rank_shot_candidates([{"slot_id": 1, "text": "x", "video_results": [_cand("a")]}],
                            api_key="k")
    assert "HOUSE RULES" in seen["system"]
    assert "No presenter footage for car topics" in seen["system"]


def test_rater_stats_agreement():
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    _rate_all(pid, 501, {"yt/r1": "yes", "px/bad": "no"})
    _rate_all(pid, 502, {"yt/r1": "yes", "px/bad": "yes"})
    stats = {s["rater_id"]: s for s in ratings.rater_stats()}
    assert stats[501]["peer_shared"] == 2 and stats[501]["peer_agree"] == 1


# ── rating page over HTTP ────────────────────────────────────────────────────

@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), fileserver._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.read().decode("utf-8")


def test_rating_page_end_to_end(server):
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    link = rating_page.rate_link(501, server)
    auth = link.split("?", 1)[1]

    status, html = _get(link)
    assert status == 200 and "B-roll Review" in html

    meta = json.loads(_get(f"{server}/rate/api/meta?{auth}")[1])
    assert any(r["id"] == "talking_head" for r in meta["reasons"])
    task = json.loads(_get(f"{server}/rate/api/next?{auth}")[1])
    body = json.dumps({"project_id": task["project_id"], "slot_id": task["slot_id"],
                       "ratings": [{"item_id": i["id"], "usable": "yes", "fit": 4,
                                    "segments": [[1, 3]]} for i in task["items"]],
                       "suggestions": [{"url": "https://youtu.be/AbCdEfGhIjK",
                                        "segments": [[5, 9]]}]}).encode()
    req = urllib.request.Request(f"{server}/rate/api/submit?{auth}", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.loads(r.read())["saved"] == len(task["items"]) + 1
    assert json.loads(_get(f"{server}/rate/api/next?{auth}")[1]) == {"done": True}


def test_rating_page_rejects_bad_or_revoked_links(server, monkeypatch):
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{server}/rate/api/next?r=501&t=forged")
    assert e.value.code == 403
    link = rating_page.rate_link(501, server)
    monkeypatch.setenv("TELEGRAM_RATERS", "502")               # removed from the list
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(f"{server}/rate/api/meta?{link.split('?', 1)[1]}")
    assert e.value.code == 403


# ── bot ──────────────────────────────────────────────────────────────────────

def test_handle_rate_needs_file_server(monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "send_message", lambda chat, text, reply_markup=None: sent.append(text))
    monkeypatch.setitem(tb._FILESERVER, "port", None)
    tb.handle_rate(1, {"id": 501, "name": "sara"})
    assert "BOT_FILE_SERVER" in sent[-1]

    monkeypatch.setitem(tb._FILESERVER, "port", 8770)
    monkeypatch.setenv("BOT_PUBLIC_URL", "https://broll.example")
    tb.handle_rate(1, {"id": 501, "name": "sara"})
    assert "https://broll.example/rate?r=501&t=" in sent[-1]


# ── checking editor labels + contribution log ────────────────────────────────

FPS = 30.0


def _xml_item(name, start, end, in_sec=0.0):
    return {"name": name, "local_path": f"C:/edit/{name}", "fps": FPS,
            "start_frame": int(start * FPS), "end_frame": int(end * FPS),
            "in_frame": int(in_sec * FPS), "out_frame": int((in_sec + end - start) * FPS),
            "in_seconds": in_sec, "out_seconds": in_sec + end - start}


def _learned_project(monkeypatch):
    """Two lines; the editor used a.mp4 on line 1 and dropped b.mp4."""
    pid = ps.create_project("Edited")
    shots = [{"slot_id": 1, "timestamp": 0, "end_timestamp": 5,
              "text": "the engine oil drains out of the sump",
              "selected_results": [
                  dict(_cand("yt/a"), local_path="/srv/1-1-a.mp4", title="Oil drain"),
                  dict(_cand("px/b", source="pexels"), local_path="/srv/1-2-b.mp4",
                       title="Coffee cup")]},
             {"slot_id": 2, "timestamp": 5, "end_timestamp": 10, "text": "then refill it",
              "selected_results": []}]
    monkeypatch.setattr(ps, "_exported_placements", lambda path: ({}, set()))
    ps.record_delivery(pid, shots)
    ps.analyze_import(pid, [_xml_item("1-1-a.mp4", 0, 5, in_sec=7.0)])
    with ps._conn() as c:
        ids = {r["filename"]: r["id"] for r in c.execute(
            "SELECT id, filename FROM project_assets WHERE project_id=?", (pid,))}
    return pid, ids


def test_label_task_shows_auto_labels_and_lines(monkeypatch):
    pid, ids = _learned_project(monkeypatch)
    task = ratings.next_label_task(501)
    assert task["mode"] == "labels" and task["slot_id"] == "1"
    by = {i["asset_id"]: i for i in task["items"]}
    assert by[ids["1-1-a.mp4"]]["verdict"] == "used_here"
    assert by[ids["1-1-a.mp4"]]["in_sec"] == 7.0
    assert by[ids["1-2-b.mp4"]]["verdict"] == "unused"
    assert [l["slot_id"] for l in task["lines"]] == ["1", "2"]


def test_label_review_confirm_correct_and_validation(monkeypatch):
    pid, ids = _learned_project(monkeypatch)
    with pytest.raises(ValueError):
        ratings.save_label_review(501, ids["1-2-b.mp4"], "correct", "used_elsewhere")  # no line
    with pytest.raises(ValueError):
        ratings.save_label_review(501, ids["1-2-b.mp4"], "correct", "sideways")
    ratings.save_label_review(501, ids["1-1-a.mp4"], "confirm")
    ratings.save_label_review(501, ids["1-2-b.mp4"], "correct", "neutral", note="fine, too many")
    assert ratings.next_label_task(501) is None                 # all checked
    labels = ratings.reviewed_labels()
    assert labels[ids["1-1-a.mp4"]] == ("used_here", "")
    assert labels[ids["1-2-b.mp4"]] == ("neutral", "")
    s = ratings.contribution_summary(501)
    assert s["labels_checked"] == 2
    assert s["impact"]["label_confirmed"] == 1 and s["impact"]["label_corrected"] == 1


def test_corrected_labels_change_what_is_learned(monkeypatch):
    pid, ids = _learned_project(monkeypatch)
    h = ef.load_history()
    assert "dropped: [PEXELS] Coffee cup" in ef.history_note(h, "engine oil drains from the sump")
    # A reviewer says the unused clip was fine, just not needed → no longer a negative.
    ratings.save_label_review(501, ids["1-2-b.mp4"], "correct", "neutral")
    h = ef.load_history()
    assert "Coffee cup" not in ef.history_note(h, "engine oil drains from the sump")
    # …and a clip corrected to "right for line 2" counts as kept there.
    ratings.save_label_review(502, ids["1-2-b.mp4"], "correct", "used_elsewhere", "2")
    ratings.save_label_review(501, ids["1-2-b.mp4"], "correct", "used_elsewhere", "2")
    h = ef.load_history()
    assert "kept: [PEXELS] Coffee cup" in ef.history_note(h, "then refill it")


def test_impact_credits_for_picks_blocks_and_in_points(monkeypatch):
    pid, shots = _project()
    ratings.record_candidates(pid, shots)
    _rate_all(pid, 501, {"yt/r1": "yes", "px/bad": "no"}, segs=[[4, 9]])
    _rate_all(pid, 502, {"px/bad": "no"})
    ratings.add_suggestion(502, pid, 1, "https://www.youtube.com/watch?v=AbCdEfGhIjK")
    h = ef.load_history()

    ef.demote_rejected([{"video_results": [_cand("px/bad"), _cand("x")]}], h)
    sug = "https://www.youtube.com/watch?v=AbCdEfGhIjK"
    ef.credit_delivery(77, [{"selected_results": [
        dict(_cand("yt/r1"), in_rule="rated"),
        dict(_cand(sug), reviewer_suggested=True)]}], h)
    ef.credit_delivery(77, [{"selected_results": [dict(_cand("yt/r1"), in_rule="rated")]}], h)

    a, b = ratings.contribution_summary(501), ratings.contribution_summary(502)
    assert a["impact"]["helped_pick"] == 1                  # once per project+clip
    assert a["impact"]["in_point_used"] == 1
    assert a["impact"]["blocked_bad_clip"] == 1 and b["impact"]["blocked_bad_clip"] == 1
    assert b["impact"]["suggestion_used"] == 1
    assert b["suggested"] == 1
    assert any(e["kind"] == "suggestion_used" for e in b["log"])
    stats = {s["rater_id"]: s for s in ratings.rater_stats()}
    assert stats[501]["changed"] == 3


def test_label_and_impact_endpoints(server, monkeypatch):
    pid, ids = _learned_project(monkeypatch)
    auth = rating_page.rate_link(501, server).split("?", 1)[1]
    task = json.loads(_get(f"{server}/rate/api/labels/next?{auth}")[1])
    body = json.dumps({"reviews": [{"asset_id": i["asset_id"], "action": "confirm"}
                                   for i in task["items"]]}).encode()
    req = urllib.request.Request(f"{server}/rate/api/labels/submit?{auth}", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.loads(r.read())["saved"] == 2
    impact = json.loads(_get(f"{server}/rate/api/impact?{auth}")[1])
    assert impact["labels_checked"] == 2 and impact["impact"]["label_confirmed"] == 2

"""Segment library: editors' cuts become stored, described, reviewed segments,
and new lines pick from them — exact subject when it's identifiable, general
footage when it isn't."""

import json
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import bot.telegram_bot as tb
from bot import fileserver, rating_page
from core import output
from core import project_store as ps
from core import ratings
from core import segment_library as sl

FPS = 30.0
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def fake_embed(text):
    """Bag-of-words vector: similar wording → similar vectors, no model needed."""
    v = np.zeros(256, dtype=np.float32)
    for w in sl._tokens(text):
        v[sum(map(ord, w)) % 256] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


@pytest.fixture(autouse=True)
def _lib(monkeypatch, tmp_path):
    monkeypatch.setenv("SEGMENT_LIBRARY_DIR", str(tmp_path / "segments"))
    monkeypatch.setattr(sl, "_embed", fake_embed)
    monkeypatch.setattr(sl, "tag_lines", lambda shots, topic="": None)
    monkeypatch.setattr(sl, "kick", lambda: None)
    monkeypatch.setenv("TELEGRAM_RATERS", "501,502")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "900")


def _video(path, seconds=6):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=duration={seconds}:size=320x240:rate=30",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def _xml_item(name, start, end, in_sec=0.0):
    return {"name": name, "local_path": f"C:/edit/{name}", "fps": FPS,
            "start_frame": int(start * FPS), "end_frame": int(end * FPS),
            "in_frame": int(in_sec * FPS), "out_frame": int((in_sec + end - start) * FPS),
            "in_seconds": in_sec, "out_seconds": in_sec + end - start}


def _learned_project(monkeypatch, tmp_path, with_file=True):
    clips = tmp_path / "proj" / "director"
    clips.mkdir(parents=True)
    if with_file and HAVE_FFMPEG:
        _video(clips / "1-1-a.mp4")
    monkeypatch.setattr(output, "clip_base_dir", lambda name: str(clips))
    monkeypatch.setattr(ps, "_exported_placements", lambda path: ({}, set()))
    pid = ps.create_project("Camry service", "Camry service")
    shots = [{"slot_id": 1, "timestamp": 0, "end_timestamp": 5,
              "text": "drain the old oil from the Camry",
              "selected_results": [{"url": "https://yt/a", "page_url": "https://yt/a",
                                    "source": "youtube", "title": "Camry oil change",
                                    "channel": "AutoFix", "local_path": "/srv/1-1-a.mp4"}]}]
    ps.record_delivery(pid, shots)
    ps.analyze_import(pid, [_xml_item("1-1-a.mp4", 0, 3, in_sec=1.0)])
    return pid


def _ready(page, subject, identifiable, generic, description, tmp_path, trust="verified",
           seconds=4.0):
    sid, _ = sl.add_segment(page, 0, seconds, source="youtube", trust=trust)
    f = tmp_path / f"seg{sid}.mp4"
    f.write_bytes(b"x")
    seg = {"description": description, "generic_use": generic, "subject": subject}
    with ps._conn() as c:
        c.execute("""UPDATE segments SET file_status='ready', file_path=?, subject=?,
                     identifiable=?, generic_use=?, description=?, embedding=? WHERE id=?""",
                  (str(f), subject, identifiable, generic, description,
                   fake_embed(sl._embed_text(seg)).tobytes(), sid))
    return sid


# ── ingest ───────────────────────────────────────────────────────────────────

def test_editor_cuts_become_segments(monkeypatch, tmp_path):
    pid = _learned_project(monkeypatch, tmp_path)
    assert sl.ingest_from_import(pid) == 1
    assert sl.ingest_from_import(pid) == 0                 # merged, not duplicated
    with ps._conn() as c:
        seg = dict(c.execute("SELECT * FROM segments").fetchone())
        lines = [r[0] for r in c.execute("SELECT line_text FROM segment_lines")]
    assert (seg["src_in"], seg["src_out"]) == (1.0, 4.0)
    assert seg["trust"] == "used" and seg["src_file"].endswith("1-1-a.mp4")
    assert lines == ["drain the old oil from the Camry"]


@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_worker_cuts_describes_and_embeds(monkeypatch, tmp_path):
    pid = _learned_project(monkeypatch, tmp_path)
    sl.ingest_from_import(pid)
    monkeypatch.setattr(sl, "_draft_vision", lambda frames, lines, topic: {
        "description": "Close-up of a hand removing the oil drain plug under a sedan",
        "subject": "Toyota Camry", "identifiable": False,
        "generic_use": "draining engine oil from a car", "shot_type": "close_up",
        "problems": ["bogus"]})
    assert sl.process_pending() == 1
    seg = sl.get_segment(1)
    assert seg["file_status"] == "ready" and len(seg["frames"]) == 3
    assert seg["identifiable"] == 0 and seg["draft_source"] == "vision"
    assert seg["problems"] == [] and seg["embedding"]
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", seg["file_path"]], capture_output=True, text=True)
    assert abs(float(probe.stdout) - 3.0) < 0.3


def test_worker_marks_unfetchable_sources_failed(monkeypatch, tmp_path):
    sl.add_segment("https://gone", 0, 3, source="web")
    import core.direct_downloader as dd
    monkeypatch.setattr(dd, "download_direct_video", lambda url, out, state: None)
    sl.process_pending()
    seg = sl.get_segment(1)
    assert seg["file_status"] == "failed" and "download" in seg["file_error"]


def test_rated_good_parts_become_suggested_segments():
    pid = ps.create_project("P")
    ps.record_shots(pid, [{"slot_id": 1, "text": "tighten the oil filter"}])
    ratings.record_candidates(pid, [{"slot_id": 1, "selected_results": [
        {"url": "https://yt/f", "page_url": "https://yt/f", "source": "youtube"}]}])
    item = ratings.next_task(501)["items"][0]
    ratings.save_rating(501, item["id"], "trim", segments=[[10, 16]])
    assert sl.ingest_from_ratings() == 1
    seg = sl.get_segment(1)
    assert seg["trust"] == "suggested" and (seg["src_in"], seg["src_out"]) == (10, 16)


# ── reviewer step ────────────────────────────────────────────────────────────

def test_review_validates_and_sets_trust(tmp_path):
    sid = _ready("https://yt/a", "", None, "", "", tmp_path, trust="used")
    with pytest.raises(ValueError, match="Too short"):
        sl.save_review(501, sid, usable="yes", description="good clip", identifiable=0,
                       generic_use="oil")
    with pytest.raises(ValueError, match="viewer can tell"):
        sl.save_review(501, sid, usable="yes",
                       description="hand removes the drain plug and oil pours into a pan")
    with pytest.raises(ValueError, match="exact subject"):
        sl.save_review(501, sid, usable="yes", identifiable="1",
                       description="hand removes the drain plug and oil pours into a pan")
    seg = sl.save_review(501, sid, usable="yes", fits_line="yes", identifiable="0",
                         description="hand removes the drain plug and oil pours into a pan",
                         generic_use="draining engine oil", shot_type="close_up")
    assert seg["trust"] == "verified" and seg["draft_source"] == "reviewer"
    sl.save_review(502, sid, usable="no", note="watermark")
    sl.save_review(900, sid, usable="no", note="watermark")
    assert sl.get_segment(sid)["trust"] == "avoid"
    assert ratings.contribution_summary(501)["impact"]["segment_reviewed"] == 1


def test_review_queue_skips_reviewed_and_avoided(tmp_path):
    a = _ready("https://yt/a", "", 0, "oil", "x", tmp_path, trust="used")
    b = _ready("https://yt/b", "", 0, "oil", "x", tmp_path, trust="used")
    assert sl.next_review_task(501)["segment_id"] == a
    sl.save_review(501, a, usable="no")
    assert sl.next_review_task(501)["segment_id"] == b
    assert sl.next_review_task(501, skip=[b]) is None


# ── picking ──────────────────────────────────────────────────────────────────

def test_subject_matching():
    assert sl.subject_matches("Toyota Camry", ["2020 Toyota Camry"])
    assert not sl.subject_matches("Toyota Camry", ["Honda Civic"])
    assert sl.subject_matches("Camry 2019", [], "how to service your camry")
    assert not sl.subject_matches("Camry", [], "how to service your civic")


def test_civic_line_gets_generic_footage_not_the_camry_clip(tmp_path):
    camry = _ready("https://yt/camry", "Toyota Camry", 1, "draining engine oil",
                   "Toyota Camry badge then oil drains from the engine", tmp_path)
    generic = _ready("https://yt/generic", "", 0, "drain the engine oil of a car",
                     "hand drains the engine oil into a pan", tmp_path)
    civic = {"slot_id": 1, "text": "drain the engine oil from your Honda Civic",
             "line_subjects": ["Honda Civic"], "line_topic": "oil change",
             "duration_needed_sec": 4, "video_results": []}
    camry_line = {"slot_id": 2, "text": "drain the engine oil from your Toyota Camry",
                  "line_subjects": ["Toyota Camry"], "line_topic": "oil change",
                  "duration_needed_sec": 4, "video_results": []}
    sl.inject_candidates([civic, camry_line])
    assert [c["library_segment_id"] for c in civic["video_results"]] == [generic]
    assert camry_line["video_results"][0]["library_segment_id"] == camry
    c = camry_line["video_results"][0]
    assert c["segment_path"].endswith(f"seg{camry}.mp4") and c["duration"] == 4.0


def test_segment_replaces_untrimmed_copy_and_recent_ones_rest(tmp_path):
    sid = _ready("https://yt/g", "", 0, "draining engine oil",
                 "oil drains from the engine into a pan", tmp_path)
    shot = {"slot_id": 1, "text": "oil drains from the engine", "line_subjects": [],
            "video_results": [{"url": "https://yt/g", "page_url": "https://yt/g",
                               "source": "youtube"}, {"url": "https://yt/other"}]}
    sl.inject_candidates([shot])
    assert [c.get("library_segment_id") for c in shot["video_results"]] == [sid, None]
    assert shot["video_results"][1]["url"] == "https://yt/other"

    pid = _delivered("Just used it")
    assert sl.mark_used(pid, [{"selected_results": [shot["video_results"][0]]}]) == 1
    fresh = {"slot_id": 1, "text": "oil drains from the engine", "line_subjects": [],
             "video_results": []}
    assert sl.inject_candidates([fresh]) == 0          # used in a recent video
    assert sl.get_segment(sid)["times_used"] == 1


def _delivered(title="P"):
    pid = ps.create_project(title)
    ps.set_status(pid, "delivered")
    return pid


def test_used_segments_rest_longer_when_identifiable(tmp_path):
    generic = _ready("https://yt/g", "", 0, "draining engine oil",
                     "oil drains from the engine into a pan", tmp_path)
    camry = _ready("https://yt/c", "Toyota Camry", 1, "draining engine oil",
                   "Toyota Camry badge then oil drains from the engine", tmp_path)
    sl.mark_used(_delivered(), [{"selected_results": [{"library_segment_id": generic},
                                                      {"library_segment_id": camry}]}])
    ps.create_project("running now")               # runs that didn't deliver don't count
    ps.set_status(ps.create_project("broke"), "failed")

    def usable():
        return {s["id"] for s in sl._usable("clip")}
    assert usable() == set()
    for _ in range(3):
        _delivered()
    assert usable() == {generic}                   # generic rests 3 videos…
    for _ in range(7):
        _delivered()
    assert usable() == {generic, camry}            # …a recognisable subject 10


def test_each_recent_use_costs_score(monkeypatch, tmp_path):
    monkeypatch.setattr(sl, "_MIN_SCORE", -1.0)
    worn = _ready("https://yt/a", "", 0, "draining engine oil",
                  "oil drains from the engine into a pan", tmp_path)
    unused = _ready("https://yt/b", "", 0, "draining engine oil",
                    "oil drains from the engine into a pan", tmp_path)
    for _ in range(2):
        sl.mark_used(_delivered(), [{"selected_results": [{"library_segment_id": worn}]}])
    for _ in range(3):
        _delivered()
    segs = sl._usable("clip")
    assert {s["id"]: s["recent_uses"] for s in segs} == {worn: 2, unused: 0}
    scores = {s["id"]: score for score, s in sl.find_matches(
        {"text": "oil drains from the engine", "line_subjects": []}, segs)}
    assert scores[unused] - scores[worn] == pytest.approx(0.06)


def test_one_library_clip_per_shot_and_once_per_video():
    from core.director_rank import auto_select_top_candidates

    def lib(sid):
        return {"url": f"https://yt/lib{sid}", "source": "youtube", "library_segment_id": sid}

    def fresh(src, n):
        return {"url": f"https://{src}/{n}", "page_url": f"https://{src}/{n}", "source": src}

    shots = [{"slot_id": i, "duration_needed_sec": 8,           # 2 YouTube + 2 Pexels slots
              "video_results": [lib(1), lib(2), fresh("youtube", i * 10 + 1),
                                fresh("youtube", i * 10 + 2), fresh("pexels", i * 10 + 1),
                                fresh("pexels", i * 10 + 2)]} for i in (1, 2)]
    shots.append({"slot_id": 3, "duration_needed_sec": 8,
                  "video_results": [lib(1), lib(2), fresh("pexels", 31), fresh("pexels", 32)]})
    auto_select_top_candidates(shots, lookback=0)
    picks = [[c.get("library_segment_id") or c["url"] for c in s["selected_results"]]
             for s in shots]
    assert picks[0] == [1, "https://youtube/11", "https://pexels/11", "https://pexels/12"]
    assert picks[1] == [2, "https://youtube/21", "https://pexels/21", "https://pexels/22"]
    assert picks[2] == ["https://pexels/31", "https://pexels/32"]   # both already used


def test_library_still_goes_to_one_shot_per_video(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sid = _ready("https://img/a.jpg", "", 0, "oil filter diagram",
                 "diagram of an oil filter cut in half", tmp_path)
    with ps._conn() as c:
        c.execute("UPDATE segments SET kind='image' WHERE id=?", (sid,))
    shots = [{"slot_id": i, "text": "diagram of an oil filter cut in half",
              "line_subjects": [], "images": []} for i in (1, 2)]
    assert sl.add_library_images(shots, "P") == 1
    assert [len(s["images"]) for s in shots] == [1, 0]


def test_short_segments_skipped_for_long_slots(tmp_path):
    _ready("https://yt/s", "", 0, "draining engine oil", "oil drains from the engine",
           tmp_path, seconds=1.5)
    shot = {"slot_id": 1, "text": "oil drains from the engine", "line_subjects": [],
            "duration_needed_sec": 10, "video_results": []}
    assert sl.inject_candidates([shot]) == 0


def test_promote_strong_verified():
    weak = {"url": "a"}
    strong = {"url": "b", "segment_trust": "verified", "library_score": 0.9}
    rejected = {"url": "c", "segment_trust": "verified", "library_score": 0.9, "irrelevant": True}
    shot = {"video_results": [weak, rejected, strong]}
    assert sl.promote_strong([shot]) == 1
    assert shot["video_results"][0] is strong


def test_segment_starts_at_zero_in_the_xml():
    c = {"library_segment_id": 3, "verified_in_sec": 40, "url": "u"}
    assert output._preferred_in_frame("u", "x.mp4", 60, 900, 30.0, c) == 0
    assert c["in_rule"] == "segment"


def test_generic_ok_reason_does_not_hurt_the_clip():
    pid = ps.create_project("P")
    ratings.record_candidates(pid, [{"slot_id": 1, "selected_results": [
        {"url": "u", "page_url": "u", "source": "youtube"}]}])
    item = ratings.next_task(501)["items"][0]
    for r in (501, 502):
        ratings.save_rating(r, item["id"], "no", reasons=["generic_ok"])
    lab = ratings.item_labels()[0]
    assert lab["other_line"] == 2


# ── page + bot ───────────────────────────────────────────────────────────────

@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), fileserver._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_library_tab_endpoints(server, tmp_path):
    sid = _ready("https://yt/a", "", None, "", "", tmp_path, trust="used")
    auth = rating_page.rate_link(501, server).split("?", 1)[1]
    with urllib.request.urlopen(f"{server}/rate/api/library/next?{auth}", timeout=5) as r:
        task = json.loads(r.read())
    assert task["segment_id"] == sid and task["shot_types"]
    req = urllib.request.Request(f"{server}/rate/media/segment/{sid}.mp4?{auth}",
                                 headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 206 and r.read() == b"x"
    body = json.dumps({"segment_id": sid, "usable": "yes", "identifiable": "0",
                       "description": "hand removes the drain plug and oil pours into a pan",
                       "generic_use": "draining engine oil"}).encode()
    req = urllib.request.Request(f"{server}/rate/api/library/submit?{auth}", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.loads(r.read())["ok"]
    assert sl.get_segment(sid)["trust"] == "verified"
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"{server}/rate/media/segment/{sid}.mp4?r=501&t=nope", timeout=5)
    assert e.value.code == 403


def test_library_stats_message(tmp_path):
    assert "empty" in tb.format_library(sl.stats())
    _ready("https://yt/a", "Toyota Camry", 1, "", "x", tmp_path)
    msg = tb.format_library(sl.stats())
    assert "verified by reviewers: 1" in msg and "Toyota Camry (1)" in msg
    assert "Last " not in msg                        # no delivered videos yet


def test_library_stats_report_share_and_repeats(monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "_exported_placements", lambda path: ({}, set()))
    sid = _ready("https://yt/a", "", 0, "oil", "oil drains from the engine into a pan", tmp_path)
    for n in range(2):
        pid = ps.create_project(f"P{n}")
        shots = [{"slot_id": 1, "selected_results": [
            {"url": "https://yt/a", "page_url": "https://yt/a", "source": "youtube",
             "local_path": f"/d/{n}-1-1.mp4", "library_segment_id": sid},
            {"url": f"https://yt/f{n}", "source": "youtube", "local_path": f"/d/{n}-1-2.mp4"}]}]
        ps.record_delivery(pid, shots)
        sl.mark_used(pid, shots)
    msg = tb.format_library(sl.stats())
    assert "Last 2 video(s): 2 of 4 clip(s) came from the library (50%)" in msg
    assert 'Most repeated: "oil drains from the engine into a pan" — in 2 of them' in msg


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_rating_page_script_parses(tmp_path):
    """The page's JS is a Python string, so nothing else catches a syntax slip."""
    html = rating_page.PAGE_HTML
    js = tmp_path / "page.js"
    js.write_text(html[html.index("<script>") + 8:html.index("</script>")], encoding="utf-8")
    res = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# ── vision drafters: Gemini, then OpenRouter ────────────────────────────────

@pytest.mark.skipif(not HAVE_FFMPEG, reason="needs ffmpeg")
def test_openrouter_takes_over_when_gemini_fails(monkeypatch, tmp_path):
    pid = _learned_project(monkeypatch, tmp_path)
    sl.ingest_from_import(pid)

    def gemini_down(*a):
        raise RuntimeError("HTTP 403")
    seen = {}

    def glm(frames, lines, topic):
        seen["frames"], seen["lines"] = len(frames), lines
        return {"description": "Hand unscrews an oil drain plug under a car and oil flows",
                "subject": "", "identifiable": False, "generic_use": "draining engine oil"}
    monkeypatch.setattr(sl, "_draft_vision", gemini_down)
    monkeypatch.setattr(sl, "_draft_openrouter", glm)
    monkeypatch.setattr(sl, "_draft_text", lambda *a: pytest.fail("text fallback used"))
    sl.process_pending()
    seg = sl.get_segment(1)
    assert seg["draft_source"] == "vision" and seg["generic_use"] == "draining engine oil"
    assert seen == {"frames": 3, "lines": ["drain the old oil from the Camry"]}


def test_openrouter_request_shape_and_429_retry(monkeypatch, tmp_path):
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"\xff\xd8jpeg")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k1")
    monkeypatch.delenv("OPENROUTER_API_KEY_2", raising=False)
    monkeypatch.setattr(sl.time, "sleep", lambda s: None)
    calls = []

    class Resp:
        def __init__(self, code, body=None):
            self.status_code, self._body, self.headers = code, body, {"Retry-After": "1"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

        def json(self):
            return self._body

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        if len(calls) == 1:
            return Resp(429)
        return Resp(200, {"choices": [{"message": {"content": '{"description": "d"}'}}]})
    import requests
    monkeypatch.setattr(requests, "post", post)
    out = sl._draft_openrouter([str(frame)], ["line"], "topic")
    assert out == {"description": "d"} and len(calls) == 2
    body = calls[1]
    assert body["model"] == "z-ai/glm-5.3-flash"
    parts = body["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    monkeypatch.setenv("SEGMENT_VISION_MODEL", "z-ai/glm-5v-turbo")
    assert sl.vision_fallback_model() == "z-ai/glm-5v-turbo"


def test_check_vision_reports_each_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(sl, "gemini_keys", lambda: ["g"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(sl, "_draft_vision", lambda *a: (_ for _ in ()).throw(ValueError("bad key")))
    monkeypatch.setattr(sl, "_draft_openrouter", lambda *a: {"description": "a test pattern"})
    res = {n: (ok, d) for n, ok, d in sl.check_vision("x.jpg")}
    assert res["gemini"][0] is False and "bad key" in res["gemini"][1]
    assert res["openrouter"][0] is True and "glm-5.3-flash" in res["openrouter"][1]


def test_glm_is_tried_before_gemini(monkeypatch, tmp_path):
    order = []
    monkeypatch.setattr(sl, "_draft_openrouter", lambda *a: order.append("openrouter") or {
        "description": "hand unscrews an oil drain plug under a car"})
    monkeypatch.setattr(sl, "_draft_vision", lambda *a: order.append("gemini") or {})
    assert [n for n, _f in sl._VISION_DRAFTERS] == ["openrouter", "gemini"]
    for _name, fn in sl._VISION_DRAFTERS:
        if sl._clean_draft(fn([], [], ""))["description"]:
            break
    assert order == ["openrouter"]


def test_page_serves_library_images(server, tmp_path):
    sid, _ = sl.add_image("https://img/x.png", page="https://site/p")
    f = tmp_path / "seg.png"
    f.write_bytes(b"\x89PNG....")
    with ps._conn() as c:
        c.execute("UPDATE segments SET file_status='ready', file_path=? WHERE id=?", (str(f), sid))
    auth = rating_page.rate_link(501, server).split("?", 1)[1]
    with urllib.request.urlopen(f"{server}/rate/media/segment/{sid}.png?{auth}", timeout=5) as r:
        assert r.headers["Content-Type"] == "image/png" and r.read() == b"\x89PNG...."
    assert sl.add_image("https://img/x.png") == (sid, False)       # deduped by URL

"""Optional storyboard footage check (core.storyboard).

Everything runs offline: the yt-dlp storyboard fetch, the sheet downloads and the
vision model are faked. What matters is the contract — off means the old pipeline
is untouched; on means presenter-only / off-subject videos are rejected on a COPY
of the (shared) candidate dict, in-points move to the on-subject footage, and any
failure leaves candidates exactly as the ranker left them.
"""

import io
import re

import pytest
from PIL import Image

import core.director_rank as dr
import core.output as output
import core.storyboard as sb


@pytest.fixture(autouse=True)
def _no_jitter(monkeypatch):
    monkeypatch.setenv("RANK_JITTER_MAX", "0")


# ── helpers ──────────────────────────────────────────────────────────────────

TIMES = [i * 5.0 for i in range(20)]     # 20 sampled frames, 5 s apart


def _level(frags=6, rows=3, cols=3, w=32, h=18, fps=0.2, last=12.0):
    per = rows * cols / fps
    fr = [{"url": f"https://i.ytimg.com/sb/x/M{i}.jpg", "duration": per} for i in range(frags)]
    fr[-1]["duration"] = last
    return {"format_id": "sb0", "width": w, "height": h, "rows": rows, "columns": cols,
            "fps": fps, "fragments": fr}


def _fake_sheets(monkeypatch, level, fail=False):
    """Serve a solid-colour sheet per fragment; returns the list of fetched URLs."""
    w, h, rows, cols = level["width"], level["height"], level["rows"], level["columns"]
    fetched = []

    def _get(url):
        fetched.append(url)
        if fail:
            raise OSError("boom")
        i = int(re.search(r"M(\d+)", url).group(1))
        im = Image.new("RGB", (cols * w, rows * h), (20 * i, 60, 200 - 20 * i))
        buf = io.BytesIO()
        im.save(buf, "JPEG")
        return buf.getvalue()

    monkeypatch.setattr(sb, "_download_sheet", _get)
    return fetched


def _yt(vid, title="clip", **extra):
    return dict({"source": "youtube", "url": f"https://www.youtube.com/watch?v={vid * 11}",
                 "title": title}, **extra)


def _px(n):
    return {"source": "pexels", "url": f"https://videos.pexels.com/{n}.mp4",
            "page_url": f"https://www.pexels.com/video/stock-{n}/", "title": f"stock {n}"}


def _shot(slot, cands, dur=3.0):
    return {"slot_id": slot, "priority": "medium", "text": f"narration {slot}",
            "shot_intent": "engine bay of a car", "duration_needed_sec": dur,
            "video_results": list(cands), "selected_results": []}


def _turn_on(monkeypatch, verdicts, prepared_times=TIMES):
    """Switch the check on with a fake fetch and a fake vision model.
    ``verdicts`` maps a candidate title → {"on_subject": [...], "presenter": [...]}.
    Returns the list of (slot, [titles]) the model was asked about."""
    monkeypatch.setenv("ENABLE_STORYBOARD_CHECK", "true")
    monkeypatch.setattr(sb, "vision_configured", lambda: True)
    monkeypatch.setattr(sb, "prepare_video", lambda url: {
        "video_id": sb._vid({"url": url}), "jpeg": b"jpg", "times": list(prepared_times),
        "duration": 100})
    asked = []

    def _vision(system, user_text, jpegs):
        titles = re.findall(r'CANDIDATE \d+: "(.*?)"', user_text)
        slot = int(re.search(r"narration (\d+)", user_text).group(1))
        asked.append((slot, titles))
        assert len(jpegs) == len(titles)
        return {"candidates": [dict(verdicts.get(t, {}), id=n + 1, note=f"note {t}")
                               for n, t in enumerate(titles)]}

    monkeypatch.setattr(sb, "_vision_json", _vision)
    return asked


# ── switch ───────────────────────────────────────────────────────────────────

def test_off_by_default():
    assert sb.enabled() is False


def test_off_means_nothing_runs(monkeypatch):
    monkeypatch.setattr(sb, "prepare_video", lambda url: pytest.fail("must not fetch"))
    monkeypatch.setattr(sb, "_vision_json", lambda *a: pytest.fail("must not call the model"))
    shots = [_shot(1, [_yt("a")])]
    sb.check_shots(shots)
    assert "sb_checked" not in shots[0]["video_results"][0]


# ── storyboard geometry ──────────────────────────────────────────────────────

def test_pick_level_prefers_320_and_ignores_non_storyboards():
    lv320, lv160 = _level(w=320, h=180), dict(_level(w=160, h=90), format_id="sb1")
    formats = [{"format_id": "137", "height": 1080}, lv160, lv320]
    assert sb._pick_level(formats) is lv320
    assert sb._pick_level([lv160]) is lv160
    assert sb._pick_level([{"format_id": "sb0", "width": 320, "rows": 3, "columns": 3}]) is None
    assert sb._pick_level([]) is None


def test_frame_layout_times_and_partial_last_sheet():
    layout = sb.frame_layout(_level(frags=3, last=12.0))
    assert len(layout) == 9 + 9 + 2                 # last sheet holds 2 frames
    assert layout[0] == (0.0, 0, 0, 0)
    assert layout[4] == (20.0, 0, 1, 1)             # 5 s apart, row-major
    assert layout[9][:2] == (45.0, 1)               # second sheet starts at 45 s


def test_sample_indices_even_and_distinct():
    assert sb.sample_indices(5, 20) == [0, 1, 2, 3, 4]
    idx = sb.sample_indices(160, 20)
    assert len(idx) == 20 and idx[0] == 0 and idx[-1] == 159 and idx == sorted(set(idx))
    assert sb.sample_indices(0, 20) == []


def test_contact_sheet_fetches_only_needed_sheets(monkeypatch):
    monkeypatch.setenv("STORYBOARD_FRAMES", "6")
    monkeypatch.setenv("STORYBOARD_COLUMNS", "5")
    level = _level(frags=6)                          # 47 frames over 6 sheets
    fetched = _fake_sheets(monkeypatch, level)
    jpeg, times = sb.build_contact_sheet(level)
    assert len(times) == 6 and times[0] == 0.0
    assert times == sorted(times)
    img = Image.open(io.BytesIO(jpeg))
    cell_h = round(256 * 18 / 32)
    assert img.size == (5 * 256, 2 * cell_h)         # 6 frames → 5 columns × 2 rows
    assert len(set(fetched)) == len(fetched) <= 6


def test_contact_sheet_gives_up_when_sheets_fail(monkeypatch):
    level = _level()
    _fake_sheets(monkeypatch, level, fail=True)
    assert sb.build_contact_sheet(level) is None


def test_prepare_video_caches_on_disk(monkeypatch):
    level = _level()
    _fake_sheets(monkeypatch, level)
    calls = []
    monkeypatch.setattr(sb, "_fetch_spec", lambda url: calls.append(url) or
                        {"level": level, "duration": 200})
    url = "https://www.youtube.com/watch?v=" + "q" * 11
    first = sb.prepare_video(url)
    assert first and first["video_id"] == "q" * 11 and first["duration"] == 200
    monkeypatch.setattr(sb, "_fetch_spec", lambda url: pytest.fail("cache miss"))
    again = sb.prepare_video(url)
    assert again["jpeg"] == first["jpeg"] and again["times"] == first["times"]
    assert len(calls) == 1


def test_prepare_video_none_without_storyboard(monkeypatch):
    monkeypatch.setattr(sb, "_fetch_spec", lambda url: {})
    assert sb.prepare_video("https://www.youtube.com/watch?v=" + "z" * 11) is None
    assert sb.prepare_video("not a youtube url") is None


# ── verdicts ─────────────────────────────────────────────────────────────────

def test_analyze_bad_when_almost_all_presenter():
    a = sb.analyze({"presenter": list(range(1, 18)), "on_subject": [18, 19]}, TIMES)
    assert a["verdict"] == "bad" and a["presenter"] == 0.85


def test_analyze_bad_when_nothing_on_subject():
    assert sb.analyze({"presenter": [1, 2], "on_subject": []}, TIMES)["verdict"] == "bad"
    assert sb.analyze({}, TIMES)["verdict"] == "bad"


def test_analyze_sparse_sample_demotes_instead_of_rejecting_an_empty_result():
    """Frames 40 s apart can easily skip a short stretch of footage: not seeing
    the subject is then weak evidence, so the video is demoted, not thrown out."""
    sparse = [i * 40.0 for i in range(20)]
    assert sb.analyze({"on_subject": []}, sparse)["verdict"] == "weak"
    assert sb.analyze({"on_subject": []}, TIMES)["verdict"] == "bad"      # 5 s apart
    # A presenter-dominated video is rejected however sparse the sample is.
    assert sb.analyze({"presenter": list(range(1, 19))}, sparse)["verdict"] == "bad"
    assert sb.analyze({"on_subject": []}, sparse, )["start_sec"] is None


def test_analyze_weak_and_good_and_window_start():
    weak = sb.analyze({"on_subject": [3]}, TIMES)
    assert weak["verdict"] == "weak" and weak["start_sec"] == 10.0
    good = sb.analyze({"on_subject": [12, 5, 6, 7], "presenter": [1, 2]}, TIMES)
    assert good["verdict"] == "good"
    assert good["start_sec"] == 20.0                 # longest run is frames 5-7
    assert good["relevant"] == 0.2 and good["presenter"] == 0.1


def test_analyze_tie_takes_earliest_and_ignores_junk():
    a = sb.analyze({"on_subject": [8, 9, 2, 3, "x", None, 99, 0]}, TIMES)
    assert a["start_sec"] == 5.0


def test_analyze_frame_in_both_lists_counts_as_presenter():
    a = sb.analyze({"on_subject": [1, 2, 3, 4], "presenter": [1]}, TIMES)
    assert a["presenter"] == 0.05 and a["relevant"] == 0.15


# ── the stage ────────────────────────────────────────────────────────────────

def test_bad_video_marked_irrelevant_on_a_copy(monkeypatch):
    talker, real = _yt("a", "talker"), _yt("b", "real")
    _turn_on(monkeypatch, {"talker": {"presenter": list(range(1, 20)), "on_subject": [20]},
                           "real": {"on_subject": [4, 5, 6, 7, 8, 9]}})
    shared = [talker, real]
    shots = [_shot(1, shared)]
    stats = sb.check_shots(shots)
    vr = shots[0]["video_results"]
    by_title = {c["title"]: c for c in vr}
    assert by_title["talker"]["irrelevant"] is True and by_title["talker"]["sb_verdict"] == "bad"
    assert "irrelevant" not in by_title["real"] and by_title["real"]["sb_verdict"] == "good"
    # The candidate dicts are shared between shots by the query cache: the
    # originals must not carry this shot's verdict.
    assert "irrelevant" not in talker and "sb_checked" not in talker
    assert "sb_checked" not in real
    assert stats["bad"] == 1 and stats["good"] == 1 and stats["checked"] == 2


def test_good_video_gets_in_point_and_note(monkeypatch):
    _turn_on(monkeypatch, {"real": {"on_subject": [4, 5, 6, 7, 8, 9]}})
    shots = [_shot(1, [_yt("b", "real")])]
    sb.check_shots(shots)
    c = shots[0]["video_results"][0]
    assert c["storyboard_in_sec"] == 15.0            # frame 4 → 15 s
    assert c["sb_relevant"] == 0.3 and c["sb_note"] == "note real"
    assert sb.run_stats()["retimed"] == 1


def test_window_at_the_very_start_sets_no_in_point(monkeypatch):
    _turn_on(monkeypatch, {"real": {"on_subject": [1, 2, 3, 4, 5]}})
    shots = [_shot(1, [_yt("b", "real")])]
    sb.check_shots(shots)
    assert "storyboard_in_sec" not in shots[0]["video_results"][0]


def test_weak_video_goes_to_the_back_but_stays_eligible(monkeypatch):
    _turn_on(monkeypatch, {"thin": {"on_subject": [3]},
                           "solid": {"on_subject": [3, 4, 5, 6, 7, 8]}})
    shots = [_shot(1, [_yt("a", "thin"), _yt("b", "solid")])]
    sb.check_shots(shots)
    vr = shots[0]["video_results"]
    assert [c["title"] for c in vr] == ["solid", "thin"]
    assert vr[1]["sb_verdict"] == "weak" and not vr[1].get("irrelevant")


def test_only_youtube_non_library_non_irrelevant_candidates_are_sent(monkeypatch):
    asked = _turn_on(monkeypatch, {"ok": {"on_subject": [2, 3, 4, 5, 6, 7]}})
    cands = [{"source": "pexels", "url": "https://pexels/1", "title": "stock"},
             _yt("a", "seg", library_segment_id=7),
             _yt("b", "rejected", irrelevant=True),
             _yt("c", "ok")]
    sb.check_shots([_shot(1, cands, dur=6.0)])          # quota 2: the segment fills one
    assert asked == [(1, ["ok"])]


def test_library_segment_counts_toward_the_quota(monkeypatch):
    asked = _turn_on(monkeypatch, {})
    sb.check_shots([_shot(1, [_yt("a", "seg", library_segment_id=7), _yt("c", "other")])])
    assert asked == []                                  # quota 1 already met by the segment


def test_weak_top_pick_does_not_fill_the_quota(monkeypatch):
    """A weak clip is demoted, so the check must keep going until one passes —
    otherwise an unchecked (maybe presenter-only) clip is what gets picked."""
    bad = {"presenter": list(range(1, 21))}
    asked = _turn_on(monkeypatch, {"thin": {"on_subject": [10, 11]}, "talker": bad,
                                   "real": {"on_subject": [2, 3, 4, 5, 6, 7]}})
    shots = [_shot(1, [_yt("a", "thin"), _yt("b", "talker"), _yt("c", "real"), _px(1)])]
    sb.check_shots(shots)
    assert asked == [(1, ["thin", "talker"]), (1, ["real"])]
    dr.auto_select_top_candidates(shots)
    assert [c["title"] for c in shots[0]["selected_results"]] == ["real", "stock 1"]


def test_weak_stays_ahead_of_unchecked_candidates(monkeypatch):
    _turn_on(monkeypatch, {"thin": {"on_subject": [10, 11]},
                           "talker": {"presenter": list(range(1, 21))}})
    monkeypatch.setenv("STORYBOARD_WAVES", "1")
    shots = [_shot(1, [_yt("a", "thin"), _yt("b", "talker"), _yt("c", "unseen")])]
    sb.check_shots(shots)
    assert [c["title"] for c in shots[0]["video_results"]] == ["thin", "talker", "unseen"]


def test_second_wave_checks_next_candidates_when_first_were_bad(monkeypatch):
    bad = {"presenter": list(range(1, 21))}
    asked = _turn_on(monkeypatch, {"a": bad, "b": bad, "c": {"on_subject": [2, 3, 4, 5, 6, 7]}})
    shots = [_shot(1, [_yt("a", "a"), _yt("b", "b"), _yt("c", "c"), _yt("d", "d")])]
    sb.check_shots(shots)
    assert asked == [(1, ["a", "b"]), (1, ["c", "d"])]   # quota 1 → check 2 per wave


def test_enough_good_clips_stops_checking(monkeypatch):
    asked = _turn_on(monkeypatch, {"a": {"on_subject": [2, 3, 4, 5, 6, 7]},
                                   "b": {"on_subject": [2, 3, 4, 5, 6, 7]}})
    shots = [_shot(1, [_yt("a", "a"), _yt("b", "b"), _yt("c", "c")])]
    sb.check_shots(shots)
    assert asked == [(1, ["a", "b"])]                   # a good one exists → no wave 2
    sb.check_shots(shots)                               # nothing new to check on a re-run
    assert len(asked) == 1


def test_videos_are_prepared_once_across_shots(monkeypatch):
    _turn_on(monkeypatch, {})
    prepared = []
    real = sb.prepare_video
    monkeypatch.setattr(sb, "prepare_video", lambda url: prepared.append(url) or real(url))
    same = _yt("a", "shared")
    sb.check_shots([_shot(1, [same]), _shot(2, [same])])
    assert len(prepared) == 1


def test_model_failure_fails_open_once(monkeypatch):
    _turn_on(monkeypatch, {})
    monkeypatch.setattr(sb, "_vision_json", lambda *a: (_ for _ in ()).throw(RuntimeError("429")))
    errors = []
    shots = [_shot(1, [_yt("a", "a")]), _shot(2, [_yt("b", "b")])]
    sb.check_shots(shots, errors=errors)
    for s in shots:
        c = s["video_results"][0]
        assert not c.get("irrelevant") and c["sb_verdict"] == "unknown"
    assert len([e for e in errors if "storyboard check failed" in e]) == 1
    assert sb.run_stats()["failed"] == 2


def test_no_storyboard_leaves_the_candidate_alone(monkeypatch):
    _turn_on(monkeypatch, {})
    monkeypatch.setattr(sb, "prepare_video", lambda url: None)
    monkeypatch.setattr(sb, "_vision_json", lambda *a: pytest.fail("nothing to show the model"))
    shots = [_shot(1, [_yt("a", "a")])]
    sb.check_shots(shots)
    c = shots[0]["video_results"][0]
    assert not c.get("irrelevant") and c["sb_verdict"] == "unknown"
    assert sb.run_stats()["no_storyboard"] == 1


def test_model_forgetting_a_candidate_fails_open(monkeypatch):
    _turn_on(monkeypatch, {})
    monkeypatch.setattr(sb, "_vision_json", lambda *a: {"candidates": []})
    shots = [_shot(1, [_yt("a", "a")])]
    sb.check_shots(shots)
    assert shots[0]["video_results"][0]["sb_verdict"] == "unknown"


def test_non_dict_model_answer_fails_open(monkeypatch):
    _turn_on(monkeypatch, {})
    monkeypatch.setattr(sb, "_vision_json", lambda *a: ["not", "a", "dict"])
    shots = [_shot(1, [_yt("a", "a")])]
    sb.check_shots(shots)
    assert not shots[0]["video_results"][0].get("irrelevant")


def test_no_vision_key_is_reported_and_changes_nothing(monkeypatch):
    monkeypatch.setenv("ENABLE_STORYBOARD_CHECK", "true")
    monkeypatch.setattr(sb, "vision_configured", lambda: False)
    monkeypatch.setattr(sb, "prepare_video", lambda url: pytest.fail("must not fetch"))
    errors = []
    shots = [_shot(1, [_yt("a")])]
    sb.check_shots(shots, errors=errors)
    assert errors and "no vision key" in errors[0]
    assert "sb_checked" not in shots[0]["video_results"][0]


def test_video_cap_is_reported(monkeypatch):
    _turn_on(monkeypatch, {"a": {"on_subject": [2, 3, 4, 5, 6, 7]}})
    monkeypatch.setenv("STORYBOARD_MAX_VIDEOS", "1")
    errors = []
    shots = [_shot(1, [_yt("a", "a"), _yt("b", "b")])]
    sb.check_shots(shots, errors=errors)
    assert any("STORYBOARD_MAX_VIDEOS" in e for e in errors)
    assert sb.run_stats()["videos"] == 1


def test_verdicts_survive_a_rerank(monkeypatch):
    """The ranker rebuilds video_results and can clear ``irrelevant``; the stored
    verdict must put it back, without paying for another look."""
    asked = _turn_on(monkeypatch, {"talker": {"presenter": list(range(1, 21))},
                                   "real": {"on_subject": [2, 3, 4, 5, 6, 7]}})
    shots = [_shot(1, [_yt("a", "talker"), _yt("b", "real")])]
    sb.check_shots(shots)
    shot = shots[0]
    shot["video_results"] = [dict(c) for c in shot["video_results"]]
    for c in shot["video_results"]:
        c.pop("irrelevant", None)                       # what a fresh ranking may do
    shot["video_results"].reverse()
    sb.check_shots(shots)
    by_title = {c["title"]: c for c in shot["video_results"]}
    assert by_title["talker"]["irrelevant"] is True
    assert len(asked) == 1


def test_progress_and_cancel(monkeypatch):
    _turn_on(monkeypatch, {"a": {"on_subject": [2, 3, 4, 5, 6, 7]}})
    ticks = []
    sb.check_shots([_shot(1, [_yt("a", "a")])], progress=lambda d, t: ticks.append((d, t)))
    assert ticks and ticks[-1][0] == ticks[-1][1]
    shots = [_shot(1, [_yt("z", "z")])]
    sb.check_shots(shots, should_cancel=lambda: True)
    assert "sb_checked" not in shots[0]["video_results"][0]


# ── the model call ───────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.headers = payload, status, {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._p


def test_vision_json_sends_prompt_and_images_to_openrouter(monkeypatch):
    import requests
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("OPENROUTER_API_KEY_2", raising=False)
    sent = {}

    def _post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json, headers=headers)
        return _Resp({"choices": [{"message": {"content": '```json\n{"candidates": [{"id": 1}]}\n```'}}],
                      "usage": {}})

    monkeypatch.setattr(requests, "post", _post)
    out = sb._vision_json("SYS", "USER", [b"aa", b"bb"])
    assert out == {"candidates": [{"id": 1}]}
    msgs = sent["body"]["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    imgs = [p for p in msgs[1]["content"] if p["type"] == "image_url"]
    assert len(imgs) == 2 and imgs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert sent["headers"]["Authorization"] == "Bearer or-key"


def test_reasoning_hint_defaults_low_and_can_be_dropped(monkeypatch):
    assert sb._openrouter_body("s", "u", ["aa"])["reasoning"] == {"effort": "low"}
    monkeypatch.setenv("STORYBOARD_REASONING_EFFORT", "none")
    assert "reasoning" not in sb._openrouter_body("s", "u", ["aa"])


def test_vision_json_retries_without_the_reasoning_hint_when_rejected(monkeypatch):
    import requests
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("OPENROUTER_API_KEY_2", raising=False)
    seen = []

    def _post(url, json=None, headers=None, timeout=None):
        seen.append("reasoning" in json)
        if "reasoning" in json:
            return _Resp({"error": "unsupported"}, status=400)
        return _Resp({"choices": [{"message": {"content": '{"candidates": []}'}}], "usage": {}})

    monkeypatch.setattr(requests, "post", _post)
    assert sb._vision_json("SYS", "USER", [b"aa"]) == {"candidates": []}
    assert seen == [True, False]


def test_vision_json_falls_back_to_gemini(monkeypatch):
    import requests
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    urls = []

    def _post(url, json=None, headers=None, timeout=None):
        urls.append(url)
        if "openrouter" in url:
            return _Resp({}, status=400)
        assert json["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/jpeg"
        return _Resp({"candidates": [{"content": {"parts": [{"text": '{"candidates": []}'}]}}]})

    monkeypatch.setattr(requests, "post", _post)
    assert sb._vision_json("SYS", "USER", [b"aa"]) == {"candidates": []}
    assert any("openrouter" in u for u in urls) and any("generativelanguage" in u for u in urls)


def test_vision_json_raises_without_any_key(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "GEMINI_API_KEY",
              "GEMINI_API_KEY_2", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError):
        sb._vision_json("SYS", "USER", [b"aa"])


# ── hooks into the rest of the app ───────────────────────────────────────────

def _rank_shot(slot):
    return {"slot_id": slot, "priority": "medium", "text": "t", "shot_intent": "x",
            "video_results": [{"url": "a"}, {"url": "b"}], "selected_results": []}


def test_rank_hook_only_runs_when_enabled(monkeypatch):
    calls = []
    monkeypatch.setattr(dr, "_call_llm_json", lambda *a, **k: {"shots": []})
    monkeypatch.setattr(sb, "check_shots", lambda shots, **k: calls.append((shots, k)))
    dr.rank_shot_candidates([_rank_shot(1)], api_key="k", video_topic="cars")
    assert calls == []
    monkeypatch.setenv("ENABLE_STORYBOARD_CHECK", "true")
    dr.rank_shot_candidates([_rank_shot(1)], api_key="k", video_topic="cars",
                            storyboard_progress="P", should_cancel="C")
    assert len(calls) == 1
    shots, kw = calls[0]
    assert kw["video_topic"] == "cars" and kw["progress"] == "P" and kw["should_cancel"] == "C"


def test_rank_survives_a_crashing_storyboard_stage(monkeypatch):
    monkeypatch.setenv("ENABLE_STORYBOARD_CHECK", "true")
    monkeypatch.setattr(dr, "_call_llm_json", lambda *a, **k: {"shots": []})
    monkeypatch.setattr(sb, "check_shots", lambda *a, **k: 1 / 0)
    shots = [_rank_shot(1)]
    assert dr.rank_shot_candidates(shots, api_key="k") is shots


def test_in_point_rule_ranks_below_human_rules_above_habit():
    fps, slot, media = 30.0, 90, 30 * 600
    c = {"source": "youtube", "url": "https://y/x", "storyboard_in_sec": 42.0}
    assert output._preferred_in_frame("https://y/x", "x.mp4", slot, media, fps, c) == 1260
    assert c["in_rule"] == "storyboard"

    both = dict(c, verified_in_sec=10.0)
    assert output._preferred_in_frame("https://y/x", "x.mp4", slot, media, fps, both) == 300
    assert both["in_rule"] == "verified"

    late = dict(c, storyboard_in_sec=599.0)               # no room left for the slot
    # ...so the clip starts past its channel intro instead (YT_MIN_IN_SEC = 8 s = 240 frames)
    assert output._preferred_in_frame("https://y/x", "x.mp4", slot, media, fps, late) == 240
    assert late["in_rule"] == "intro"

    seg = dict(c, library_segment_id=3)
    assert output._preferred_in_frame("https://y/x", "x.mp4", slot, media, fps, seg) == 0
    assert seg["in_rule"] == "segment"


def test_qa_label_mentions_the_check_only_when_it_ran():
    plain = {"source": "youtube", "title": "T", "matched_query": "q"}
    assert "footage check" not in dr._selected_clip_label({"selected_results": [plain]})
    checked = dict(plain, sb_checked=True, sb_verdict="good", sb_relevant=0.55,
                   sb_presenter=0.3, storyboard_in_sec=130.0, sb_note="engine and interior shots")
    label = dr._selected_clip_label({"selected_results": [checked]})
    assert "55% of sampled frames on-subject" in label
    assert "30% presenter to camera" in label and "cut starts at 2:10" in label
    assert "engine and interior shots" in label
    unknown = dict(plain, sb_checked=True, sb_verdict="unknown")
    assert "footage check" not in dr._selected_clip_label({"selected_results": [unknown]})


def test_reviewer_is_told_about_footage_notes_only_when_they_exist(monkeypatch):
    prompts = []
    monkeypatch.setattr(dr, "_call_llm_json",
                        lambda client, system, user, **k: prompts.append(system) or
                        {"overall": "ok", "issues": []})

    def _tl(**extra):
        clip = dict({"title": "T", "source": "youtube"}, **extra)
        return [{"slot_id": i, "priority": "medium", "shot_intent": "x", "duration_needed_sec": 3.0,
                 "selected_results": [dict(clip)]} for i in (1, 2)]

    dr.review_timeline(_tl(), api_key="k")
    dr.review_timeline(_tl(sb_checked=True, sb_verdict="good", sb_relevant=0.4,
                           sb_presenter=0.1), api_key="k")
    assert "footage check" not in prompts[0]           # old path: prompt untouched
    assert "footage check" in prompts[1]


def test_setting_is_off_by_default_and_maps_to_the_env_flag():
    from bot import settings as bs
    assert bs.DEFAULTS["storyboard_check"] is False
    assert bs.env_overrides(bs.DEFAULTS)["ENABLE_STORYBOARD_CHECK"] == "false"
    on = dict(bs.DEFAULTS, storyboard_check=True)
    assert bs.env_overrides(on)["ENABLE_STORYBOARD_CHECK"] == "true"
    assert sb.enabled() is False
    with bs.apply_env(on):
        assert sb.enabled() is True
    assert sb.enabled() is False


def test_setting_toggles_from_the_menu(tmp_path, monkeypatch):
    from bot import settings as bs
    monkeypatch.setattr(bs, "_SETTINGS_PATH", str(tmp_path / "s.json"))
    assert bs.toggle(7, "storyboard_check")["storyboard_check"] is True
    assert bs.toggle(7, "storyboard_check")["storyboard_check"] is False


def test_review_message_line():
    from bot.telegram_bot import format_storyboard_line
    assert format_storyboard_line({}) is None
    assert format_storyboard_line({"attempts": {"storyboard": {"checked": 0}}}) is None
    line = format_storyboard_line({"attempts": {"storyboard": {
        "checked": 42, "bad": 8, "weak": 3, "retimed": 20, "no_storyboard": 2, "failed": 1}}})
    assert "42 clip(s) checked" in line and "8 rejected" in line and "3 demoted" in line
    assert "20 cut where the footage starts" in line and "2 without a storyboard" in line
    assert "1 not checked" in line


# ── the clips actually picked ────────────────────────────────────────────────

def test_check_selected_covers_picks_past_the_checked_ones(monkeypatch):
    """Three shots share one candidate list (query cache); "fresh YouTube first"
    sends shot 3 past the two clips the rank-time check looked at."""
    bad = {"presenter": list(range(1, 21))}
    good = {"on_subject": [2, 3, 4, 5, 6, 7]}
    _turn_on(monkeypatch, {"d": good, "e": good, "f": bad, "g": good})
    shared = [_yt("d", "d"), _yt("e", "e"), _yt("f", "f"), _yt("g", "g")]
    shots = [_shot(i, shared + [_px(i)]) for i in (1, 2, 3)]
    sb.check_shots(shots)
    dr.auto_select_top_candidates(shots)
    assert shots[2]["selected_results"][0]["title"] == "f"     # the gap
    swapped = sb.check_selected(shots)
    assert swapped == 1
    pick = shots[2]["selected_results"][0]
    assert pick["title"] == "g" and pick["sb_verdict"] == "good"
    assert pick["storyboard_in_sec"] == 5.0
    assert sb.run_stats()["replaced"] == 1
    for s in shots:                           # YouTube picks carry their verdicts
        assert all(c.get("sb_checked") for c in s["selected_results"] if dr._is_youtube(c))


def test_check_selected_keeps_a_lone_bad_pick_and_skips_manual_ones(monkeypatch):
    bad = {"presenter": list(range(1, 21))}
    _turn_on(monkeypatch, {"only": bad, "mine": bad})
    auto = _shot(1, [_yt("a", "only")])
    auto.update(selected_results=[auto["video_results"][0]], auto_selected=True)
    manual = _shot(2, [_yt("b", "mine")])
    manual["selected_results"] = [manual["video_results"][0]]
    assert sb.check_selected([auto, manual]) == 0
    assert auto["selected_results"][0]["title"] == "only"      # never leave a slot empty
    assert auto["selected_results"][0]["sb_verdict"] == "bad"
    assert "sb_checked" not in manual["selected_results"][0]


def test_check_selected_off_does_nothing(monkeypatch):
    monkeypatch.setattr(sb, "_vision_json", lambda *a: pytest.fail("must not call the model"))
    s = _shot(1, [_yt("a")])
    s.update(selected_results=list(s["video_results"]), auto_selected=True)
    assert sb.check_selected([s]) == 0


# ── the lookup must not stop at an EMPTY answer (the first server run: 32 of 32 "no storyboard") ──

def _sb_level(n=3):
    return [{"format_id": f"sb{i}", "fragments": [{"url": "u", "duration": 10}], "width": 160 + i * 80,
             "height": 90, "rows": 5, "columns": 5, "fps": 0.2} for i in range(n)]


class _Lookup:
    """core.youtube._extract_info_with_backoff stand-in: answers per configuration."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, opts, url, **k):
        label = ("cookies" if opts.get("cookiefile") else
                 "web clients" if opts.get("extractor_args") else "no cookies")
        self.calls.append(label)
        a = self.answers[label]
        if isinstance(a, Exception):
            raise a
        return a


def _patch_lookup(monkeypatch, answers, cookies=True):
    import core.youtube as yt
    look = _Lookup(answers)
    monkeypatch.setattr(yt, "_extract_info_with_backoff", look)
    monkeypatch.setattr(yt, "_get_cookie_opts", lambda: {"cookiefile": "c.txt"} if cookies else {})
    monkeypatch.delenv("YT_DOWNLOAD_NO_COOKIES", raising=False)
    sb.reset_run_stats()
    return look


EMPTY = {"formats": []}                                  # what a flagged IP gets back
WITH_SB = {"formats": _sb_level(), "duration": 300}
NO_SB = {"formats": [{"format_id": "18", "url": "x"}], "duration": 19}      # answered, but no storyboard


def test_an_empty_cookie_lookup_falls_through_to_the_next_configuration(monkeypatch):
    look = _patch_lookup(monkeypatch, {"cookies": EMPTY, "no cookies": WITH_SB, "web clients": EMPTY})
    spec = sb._fetch_spec("https://www.youtube.com/watch?v=" + "a" * 11)
    assert spec["level"]["format_id"] == "sb2" and spec["duration"] == 300      # 320 px wide is the closest
    assert look.calls == ["cookies", "no cookies"]


def test_a_config_that_was_empty_while_a_later_one_worked_is_skipped_afterwards(monkeypatch):
    look = _patch_lookup(monkeypatch, {"cookies": EMPTY, "no cookies": WITH_SB, "web clients": EMPTY})
    sb._fetch_spec("https://www.youtube.com/watch?v=" + "a" * 11)
    sb._fetch_spec("https://www.youtube.com/watch?v=" + "b" * 11)
    assert look.calls == ["cookies", "no cookies", "no cookies"]                # second video: straight to the one that works


def test_a_video_that_really_has_no_storyboard_stops_at_the_first_answer(monkeypatch):
    look = _patch_lookup(monkeypatch, {"cookies": NO_SB, "no cookies": WITH_SB, "web clients": WITH_SB})
    assert sb._fetch_spec("https://www.youtube.com/watch?v=" + "z" * 11) == {}   # YouTube answered: it has none
    assert look.calls == ["cookies"]


def test_when_nothing_answers_the_result_says_blocked(monkeypatch):
    look = _patch_lookup(monkeypatch, {"cookies": EMPTY, "no cookies": EMPTY, "web clients": EMPTY})
    spec = sb._fetch_spec("https://www.youtube.com/watch?v=" + "a" * 11)
    assert spec == {"blocked": True, "tried": ["cookies", "no cookies", "web clients"]}
    assert look.calls == ["cookies", "no cookies", "web clients"]


def test_it_raises_only_when_every_lookup_raised(monkeypatch):
    boom = RuntimeError("Sign in to confirm you're not a bot")
    _patch_lookup(monkeypatch, {"cookies": boom, "no cookies": boom, "web clients": boom})
    with pytest.raises(RuntimeError, match="not a bot"):
        sb._fetch_spec("https://www.youtube.com/watch?v=" + "a" * 11)
    _patch_lookup(monkeypatch, {"cookies": boom, "no cookies": EMPTY, "web clients": boom})
    assert sb._fetch_spec("https://www.youtube.com/watch?v=" + "a" * 11)["blocked"]   # a mix is "blocked"


def test_downloads_told_to_skip_cookies_make_the_lookup_skip_them_too(monkeypatch):
    look = _patch_lookup(monkeypatch, {"cookies": EMPTY, "no cookies": WITH_SB, "web clients": EMPTY})
    monkeypatch.setenv("YT_DOWNLOAD_NO_COOKIES", "1")
    assert sb._fetch_spec("https://www.youtube.com/watch?v=" + "a" * 11)["level"]
    assert look.calls == ["no cookies"]
    look = _patch_lookup(monkeypatch, {"no cookies": WITH_SB, "web clients": EMPTY}, cookies=False)
    sb._fetch_spec("https://www.youtube.com/watch?v=" + "a" * 11)
    assert look.calls == ["no cookies"]                                          # nothing configured: no cookie try


def test_blocked_lookups_are_counted_and_explained(monkeypatch):
    sb.reset_run_stats()
    monkeypatch.setattr(sb, "_fetch_spec", lambda url: {"blocked": True, "tried": ["cookies", "no cookies"]})
    assert sb.prepare_video("https://www.youtube.com/watch?v=" + "c" * 11) is None
    assert sb.run_stats()["blocked"] == 1


def test_undownloadable_storyboard_images_are_counted(monkeypatch):
    sb.reset_run_stats()
    level = dict(_sb_level(1)[0], fragments=[{"url": f"u{i}", "duration": 10} for i in range(3)])
    monkeypatch.setattr(sb, "_download_sheet", lambda url: (_ for _ in ()).throw(OSError("403 Forbidden")))
    assert sb.build_contact_sheet(level) is None
    assert sb.run_stats()["sheet_failed"] == 1


def test_the_review_line_says_why_nothing_was_checked():
    import bot.telegram_bot as tb
    line = tb.format_storyboard_line({"attempts": {"storyboard": {
        "checked": 0, "no_storyboard": 32, "blocked": 32, "sheet_failed": 0}}})
    assert "32 without a storyboard (32 blocked by YouTube)" in line
    plain = tb.format_storyboard_line({"attempts": {"storyboard": {"checked": 3, "no_storyboard": 2}}})
    assert "2 without a storyboard" in plain and "blocked" not in plain


def test_the_checker_reports_the_block_in_the_runs_errors(monkeypatch):
    sb.reset_run_stats()
    monkeypatch.setenv("ENABLE_STORYBOARD_CHECK", "true")
    monkeypatch.setattr(sb, "_fetch_spec", lambda url: {"blocked": True, "tried": ["no cookies"]})
    monkeypatch.setattr(sb, "vision_configured", lambda: True)
    shots = [{"slot_id": 1, "priority": "medium", "text": "x", "video_results": [
        {"url": "https://www.youtube.com/watch?v=" + "d" * 11, "source": "youtube", "title": "t"}]}]
    errors = []
    sb.check_shots(shots, "topic", errors)
    assert any("returned no storyboard data" in e and "cookies" in e for e in errors)

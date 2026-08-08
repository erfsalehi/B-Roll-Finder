"""Stage 7b — Gemini watching the shortlisted YouTube candidates.

The expensive mistakes this guards against: burning tokens (wrong resolution /
fps / a call per shot instead of per video), and leaking one shot's verdict onto
another shot that happens to share a candidate dict.
"""

import json
import os

import pytest

from core import visual_verify as vv


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Every test starts with the stage on, a key set, and an empty cache."""
    for var in list(os.environ):
        if var.startswith("VERIFY_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ENABLE_VISUAL_VERIFY", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY_2", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(vv, "_CACHE_PATH", str(tmp_path / "verify.json"))
    vv.clear_cache()
    yield
    vv.clear_cache()


def _shot(slot_id, urls, intent="engine bay close-up", dur=6.0):
    return {
        "slot_id": slot_id,
        "priority": "medium",
        "shot_intent": intent,
        "text": "the timing chain is what fails first",
        "duration_needed_sec": dur,
        "video_results": [
            {"source": "youtube", "url": u, "title": "clip", "duration": 300}
            for u in urls
        ],
    }


def _reply(*entries):
    """A Gemini-shaped response wrapping the given shot verdicts."""
    return {
        "candidates": [{"content": {"parts": [{"text": json.dumps({"shots": list(entries)})}]}}],
        "usageMetadata": {"promptTokenCount": 40000, "candidatesTokenCount": 120},
    }


# ── enabling ──────────────────────────────────────────────────────────────────

def test_enabled_requires_flag_and_key(monkeypatch):
    assert vv.enabled() is True
    monkeypatch.setenv("ENABLE_VISUAL_VERIFY", "0")
    assert vv.enabled() is False
    monkeypatch.setenv("ENABLE_VISUAL_VERIFY", "1")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert vv.enabled() is False


def test_stage_is_noop_when_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_VISUAL_VERIFY", "0")
    shots = [_shot(1, ["https://www.youtube.com/watch?v=aaaaaaaaaaa"])]
    before = list(shots[0]["video_results"])
    stats = vv.verify_shot_candidates(shots)
    assert stats["skipped"] == "disabled"
    assert shots[0]["video_results"] == before


# ── video identity + grouping ─────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://example.com/video.mp4", ""),
])
def test_video_id_extraction(url, expected):
    assert vv.video_id({"url": url}) == expected


# ── the cost levers ───────────────────────────────────────────────────────────

def test_payload_uses_low_resolution_low_fps_and_no_thinking():
    body = vv.build_payload("https://www.youtube.com/watch?v=aaaaaaaaaaa",
                            "PROMPT", [{"shot_id": 1, "needs": "n", "narration": "x",
                                        "seconds": 5}])
    gc = body["generationConfig"]
    assert gc["mediaResolution"] == "MEDIA_RESOLUTION_LOW"   # 66 tokens/frame
    assert gc["thinkingConfig"]["thinkingBudget"] == 0
    assert gc["responseMimeType"] == "application/json"

    parts = body["contents"][0]["parts"]
    # Video first, text last (recommended order for a single-video prompt).
    assert parts[0]["fileData"]["fileUri"].endswith("v=aaaaaaaaaaa")
    assert parts[0]["videoMetadata"]["fps"] == 0.2             # 1 frame / 5s
    assert "PROMPT" in parts[1]["text"] and "SHOT ID 1" in parts[1]["text"]


def test_resolution_and_fps_are_overridable(monkeypatch):
    monkeypatch.setenv("VERIFY_MEDIA_RESOLUTION", "medium")
    monkeypatch.setenv("VERIFY_FPS", "1")
    body = vv.build_payload("u", "P", [])
    assert body["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_MEDIUM"
    assert body["contents"][0]["parts"][0]["videoMetadata"]["fps"] == 1.0


def test_long_video_watch_window_is_capped(monkeypatch):
    monkeypatch.setenv("VERIFY_MAX_WATCH_SECONDS", "600")
    body = vv.build_payload("u", "P", [], duration=1800)
    assert body["contents"][0]["parts"][0]["videoMetadata"]["endOffset"] == "600s"
    # A video inside the cap carries no offset at all.
    body = vv.build_payload("u", "P", [], duration=120)
    assert "endOffset" not in body["contents"][0]["parts"][0]["videoMetadata"]


# ── verdict parsing ───────────────────────────────────────────────────────────

def test_parse_verdicts_validates_and_clamps():
    data = {"shots": [
        {"shot_id": 1, "match": 9, "segments": [{"start": 10, "end": 20}], "why": "good"},
        {"shot_id": 2, "match": 99, "segments": []},              # clamped to 10
        {"shot_id": 3, "match": 5},                              # hallucinated id
        {"shot_id": 4, "match": "nope"},                         # unparseable
    ]}
    out = vv.parse_verdicts(data, {1, 2, 4}, duration=600)
    assert out[1]["match"] == 9 and out[1]["segments"] == [{"start": 10.0, "end": 20.0}]
    assert out[2]["match"] == 10
    assert 3 not in out and 4 not in out


def test_parse_verdicts_drops_impossible_segments():
    data = {"shots": [{"shot_id": 1, "match": 8, "segments": [
        {"start": 50, "end": 40},        # reversed
        {"start": 700, "end": 720},      # past the end of the video
        {"start": 10, "end": 11},        # too short to cut inside
        {"start": 100, "end": 130},      # keeper
    ]}]}
    out = vv.parse_verdicts(data, {1}, duration=600)
    assert out[1]["segments"] == [{"start": 100.0, "end": 130.0}]


def test_parse_verdicts_clamps_segment_end_to_video_length():
    data = {"shots": [{"shot_id": 1, "match": 7,
                       "segments": [{"start": 580, "end": 900}]}]}
    out = vv.parse_verdicts(data, {1}, duration=600)
    assert out[1]["segments"] == [{"start": 580.0, "end": 600.0}]


def test_parse_verdicts_tolerates_timestamp_strings():
    data = {"shots": [{"shot_id": 1, "match": 7,
                       "segments": [{"start": "2:10", "end": "2:20"}]}]}
    out = vv.parse_verdicts(data, {1}, duration=600)
    assert out[1]["segments"] == [{"start": 130.0, "end": 140.0}]


def test_short_shot_accepts_a_short_segment():
    data = {"shots": [{"shot_id": 1, "match": 8,
                       "segments": [{"start": 5, "end": 6.5}]}]}
    # Global floor is 3s, but a 1.5s shot must not be denied a 1.5s window.
    assert vv.parse_verdicts(data, {1}, duration=60, needs={1: 1.5})[1]["segments"]
    assert not vv.parse_verdicts(data, {1}, duration=60, needs={1: 8.0})[1]["segments"]


# ── the stage end to end (HTTP stubbed) ───────────────────────────────────────

def _stub_post(monkeypatch, handler):
    calls = []

    class _Resp:
        status_code = 200
        reason = "OK"

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def _post(url, json=None, timeout=None, headers=None):
        calls.append({"url": url, "body": json, "headers": headers})
        return _Resp(handler(json))

    monkeypatch.setattr(vv.requests, "post", _post)
    return calls


def test_one_call_per_video_even_when_shots_share_it(monkeypatch):
    """The batching that makes this affordable: 3 shots, 1 video, 1 request."""
    url = "https://www.youtube.com/watch?v=sharedvideo"
    shots = [_shot(i, [url]) for i in (1, 2, 3)]

    def handler(body):
        ids = [line.split()[-1] for line in body["contents"][0]["parts"][1]["text"].splitlines()
               if line.startswith("SHOT ID ")]
        return _reply(*[{"shot_id": int(i), "match": 8,
                         "segments": [{"start": 30, "end": 45}], "why": "seen"}
                        for i in ids])

    calls = _stub_post(monkeypatch, handler)
    stats = vv.verify_shot_candidates(shots, video_topic="BMW M3")

    assert len(calls) == 1 and stats["calls"] == 1
    assert stats["applied"] == 3
    for s in shots:
        c = s["video_results"][0]
        assert c["visual_match"] == 8 and c["verified_in_sec"] == 30.0


def test_shots_beyond_one_call_are_chunked_not_dropped(monkeypatch):
    """More shots share a video than one call takes → every shot still gets a
    verdict, via a second viewing rather than silent truncation."""
    monkeypatch.setenv("VERIFY_MAX_SHOTS_PER_CALL", "2")
    url = "https://www.youtube.com/watch?v=sharedvideo"
    shots = [_shot(i, [url]) for i in (1, 2, 3, 4, 5)]

    def handler(body):
        text = body["contents"][0]["parts"][1]["text"]
        ids = [int(l.split()[-1]) for l in text.splitlines() if l.startswith("SHOT ID ")]
        assert len(ids) <= 2
        return _reply(*[{"shot_id": i, "match": 8, "segments": []} for i in ids])

    calls = _stub_post(monkeypatch, handler)
    vv.verify_shot_candidates(shots)

    assert len(calls) == 3                    # 2 + 2 + 1
    assert all(s["video_results"][0]["visual_match"] == 8 for s in shots)


def test_output_budget_scales_with_group_size():
    small = vv.build_payload("u", "P", [{"shot_id": 1, "needs": "", "narration": "",
                                         "seconds": 5}])
    big = vv.build_payload("u", "P", [{"shot_id": i, "needs": "", "narration": "",
                                       "seconds": 5} for i in range(12)])
    assert big["generationConfig"]["maxOutputTokens"] > \
        small["generationConfig"]["maxOutputTokens"]


def test_key_is_sent_as_header_not_in_url(monkeypatch):
    shots = [_shot(1, ["https://www.youtube.com/watch?v=aaaaaaaaaaa"])]
    calls = _stub_post(monkeypatch, lambda body: _reply({"shot_id": 1, "match": 7}))
    vv.verify_shot_candidates(shots)
    assert calls[0]["headers"]["x-goog-api-key"] == "test-key"
    assert "test-key" not in calls[0]["url"]


def test_low_scorer_is_flagged_irrelevant_and_demoted(monkeypatch):
    good = "https://www.youtube.com/watch?v=goodgoodgoo"
    bad = "https://www.youtube.com/watch?v=badbadbadba"
    shots = [_shot(1, [bad, good])]        # ranker put the bad one first

    verdicts = {"badbadbadba": 1, "goodgoodgoo": 9}

    def handler(body):
        vid = body["contents"][0]["parts"][0]["fileData"]["fileUri"].split("v=")[1]
        return _reply({"shot_id": 1, "match": verdicts[vid], "segments": []})

    _stub_post(monkeypatch, handler)
    stats = vv.verify_shot_candidates(shots)

    ordered = shots[0]["video_results"]
    assert ordered[0]["url"] == good          # promoted: actually contains it
    assert ordered[1]["irrelevant"] is True   # demoted and skipped by auto-select
    assert stats["rejected"] == 1


def test_verdicts_do_not_leak_between_shots_sharing_a_candidate(monkeypatch):
    """The query cache hands the SAME dict to every shot that ran that query, so
    a per-shot verdict must never be written onto the shared object."""
    url = "https://www.youtube.com/watch?v=sharedvideo"
    shared = {"source": "youtube", "url": url, "title": "clip", "duration": 300}
    shot_a = _shot(1, [])
    shot_b = _shot(2, [])
    shot_a["video_results"] = [shared]
    shot_b["video_results"] = [shared]      # identical object, as in production

    def handler(body):
        text = body["contents"][0]["parts"][1]["text"]
        return _reply(*[{"shot_id": int(line.split()[-1]),
                         "match": 9 if line.endswith(" 1") else 0, "segments": []}
                        for line in text.splitlines() if line.startswith("SHOT ID ")])

    _stub_post(monkeypatch, handler)
    vv.verify_shot_candidates([shot_a, shot_b])

    assert shot_a["video_results"][0]["visual_match"] == 9
    assert shot_b["video_results"][0]["visual_match"] == 0
    assert shared.get("visual_match") is None    # the shared dict is untouched


def test_second_run_is_served_from_cache(monkeypatch):
    url = "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    handler = lambda body: _reply({"shot_id": 1, "match": 8, "segments": []})

    calls = _stub_post(monkeypatch, handler)
    vv.verify_shot_candidates([_shot(1, [url])])
    assert len(calls) == 1

    stats = vv.verify_shot_candidates([_shot(1, [url])])
    assert len(calls) == 1                  # no second request
    assert stats["cached"] == 1 and stats["applied"] == 1


def test_budget_caps_how_many_videos_are_watched(monkeypatch):
    monkeypatch.setenv("VERIFY_MAX_VIDEOS", "2")
    urls = [f"https://www.youtube.com/watch?v=vid{i:08d}" for i in range(5)]
    shots = [_shot(i + 1, [u]) for i, u in enumerate(urls)]
    calls = _stub_post(monkeypatch, lambda body: _reply({"shot_id": 1, "match": 8}))

    stats = vv.verify_shot_candidates(shots)
    assert len(calls) == 2
    assert stats["budget_skipped"] == 3


def test_top_k_limits_how_deep_we_watch(monkeypatch):
    """A 6s shot binds ~2 YouTube clips, so watching all 6 candidates is waste."""
    monkeypatch.setenv("VERIFY_TOP_K_MAX", "3")
    urls = [f"https://www.youtube.com/watch?v=vid{i:08d}" for i in range(6)]
    calls = _stub_post(monkeypatch, lambda body: _reply({"shot_id": 1, "match": 8}))
    vv.verify_shot_candidates([_shot(1, urls)])
    assert len(calls) == 3


def test_api_failure_leaves_shots_untouched(monkeypatch):
    monkeypatch.setattr(vv.time, "sleep", lambda *_: None)

    def _boom(*a, **k):
        raise RuntimeError("gemini exploded")

    monkeypatch.setattr(vv.requests, "post", _boom)
    shots = [_shot(1, ["https://www.youtube.com/watch?v=aaaaaaaaaaa"])]
    errors = []
    stats = vv.verify_shot_candidates(shots, errors=errors)

    assert stats["errors"] == 1 and stats["applied"] == 0
    assert "visual_match" not in shots[0]["video_results"][0]
    assert errors and "gemini exploded" in errors[0]


def test_non_youtube_candidates_are_never_touched(monkeypatch):
    shots = [_shot(1, ["https://www.youtube.com/watch?v=aaaaaaaaaaa"])]
    pexels = {"source": "pexels", "url": "https://pexels.com/v/1.mp4",
              "page_url": "https://pexels.com/video/1"}
    shots[0]["video_results"].append(pexels)
    _stub_post(monkeypatch, lambda body: _reply({"shot_id": 1, "match": 9}))

    vv.verify_shot_candidates(shots)
    # Stock keeps its position and gains no verdict — auto-select's Pexels quota
    # behaves exactly as it would without this stage.
    assert shots[0]["video_results"][1] is pexels
    assert "visual_match" not in pexels


def test_usage_is_recorded_for_cost_reporting(monkeypatch):
    from core import usage
    usage.reset()
    _stub_post(monkeypatch, lambda body: _reply({"shot_id": 1, "match": 8}))
    vv.verify_shot_candidates([_shot(1, ["https://www.youtube.com/watch?v=aaaaaaaaaaa"])])

    summary = usage.summary()
    gem = summary["by_provider"].get("gemini")
    assert gem and gem["prompt_tokens"] == 40000
    assert summary["total_usd"] > 0        # priced, so the bot can show a $ line


# ── the payoff: a verified segment becomes the timeline in-point ───────────────

def test_verified_segment_drives_the_clip_in_point(monkeypatch):
    """A 12-min upload whose subject starts at 4:20 must not be cut at its intro."""
    from core import output

    class _NoLibrary:
        @staticmethod
        def find_clip_by_path_or_url(**_kw):
            return None

    monkeypatch.setattr(output, "clip_library", _NoLibrary, raising=False)
    candidate = {"url": "https://www.youtube.com/watch?v=aaaaaaaaaaa",
                 "verified_in_sec": 260.0}

    in_frame = output._preferred_in_frame(
        candidate["url"], "1-1-engine.mp4", duration_frames=120,
        media_dur_frames=20000, fps=23.976, candidate=candidate)
    assert in_frame == output.sec_to_frames(260.0, 23.976)

    # No verdict → unchanged behaviour: start at frame 0.
    assert output._preferred_in_frame(
        candidate["url"], "1-1-engine.mp4", duration_frames=120,
        media_dur_frames=20000, fps=23.976, candidate={"url": "x"}) == 0


def test_verified_in_point_is_ignored_when_the_slot_would_overrun(monkeypatch):
    from core import output

    class _NoLibrary:
        @staticmethod
        def find_clip_by_path_or_url(**_kw):
            return None

    monkeypatch.setattr(output, "clip_library", _NoLibrary, raising=False)
    # Verified start near the very end of the media: in + slot > media → fall back.
    assert output._preferred_in_frame(
        "u", "f.mp4", duration_frames=240, media_dur_frames=250, fps=24.0,
        candidate={"verified_in_sec": 9.0}) == 0


# ── pipeline wiring ───────────────────────────────────────────────────────────

def test_pipeline_source_places_verify_between_rank_and_autoselect():
    """Order matters: candidates must already be ranked (so the shortlist is the
    one that counts) and nothing bound yet (so rejecting a clip is free)."""
    import inspect
    import core.pipeline as pipeline

    src = inspect.getsource(pipeline.run_pipeline_headless)
    i_rank = src.index("rank_shot_candidates(shots")
    i_verify = src.index("visual_verify.verify_shot_candidates")
    i_select = src.index("auto_select_top_candidates(shots)")
    assert i_rank < i_verify < i_select

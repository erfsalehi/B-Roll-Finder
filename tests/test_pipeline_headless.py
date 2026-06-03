"""Headless pipeline orchestration (no Streamlit) + shared clip naming."""

import os
import pytest
import core.pipeline as pipeline
from core.output import clip_filename, clip_base_dir


# ── shared filename helpers (downloader ↔ exporter must agree) ────────────────

def test_clip_filename_dedupes():
    seen = set()
    a = clip_filename(1, 1, "city street", seen)
    b = clip_filename(1, 1, "city street", seen)
    assert a == b.rsplit("-", 1)[0] + ".mp4"   # same base
    assert b.endswith("-2.mp4")                # deduped against the first
    assert a != b


def test_clip_filename_blank_query_falls_back():
    assert clip_filename(3, 2, "", set()) == "3-2-clip.mp4"


def test_clip_base_dir_is_project_scoped():
    d = clip_base_dir("My Video!")
    # _safe_for_fs slugifies; just assert the project-scoped shape.
    assert d.replace("\\", "/").endswith("/director")
    assert "downloads" in d


# ── orchestration (every stage mocked; no network, no Streamlit) ──────────────

@pytest.fixture
def _mock_stages(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("ENABLE_CONTEXT_AWARE_KEYWORDS", raising=False)

    segs = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
    shots = [
        {"slot_id": 1, "priority": "medium", "duration_needed_sec": 5.0,
         "video_results": [{"url": "a"}, {"url": "b"}], "selected_results": []},
        {"slot_id": 2, "priority": "medium", "duration_needed_sec": 5.0,
         "video_results": [{"url": "c"}], "selected_results": []},
    ]

    import core.transcription, core.director, core.keywords, core.director_search
    import core.director_youtube, core.director_rank

    monkeypatch.setattr(core.transcription, "transcribe_audio", lambda *a, **k: segs)
    monkeypatch.setattr(core.keywords, "generate_video_topic", lambda *a, **k: "cars")
    monkeypatch.setattr(core.director, "generate_shot_list_from_transcription",
                        lambda *a, **k: [dict(s) for s in shots])
    monkeypatch.setattr(core.director_youtube, "seed_youtube_keywords", lambda s: s)
    monkeypatch.setattr(core.director_search, "fetch_director_footage", lambda *a, **k: None)
    monkeypatch.setattr(core.director_rank, "rank_shot_candidates", lambda *a, **k: None)
    monkeypatch.setattr(core.director_rank, "review_timeline",
                        lambda *a, **k: {"overall": "ok", "issues": []})
    # Real auto_select runs (it's pure) — leave it. Stub the FCPXML write target.
    monkeypatch.chdir(tmp_path)
    return shots


def test_pipeline_runs_all_stages_and_writes_xml(_mock_stages):
    seen = []
    res = run = pipeline.run_pipeline_headless(
        "voice.mp3", project_name="proj", download=False,
        progress_callback=lambda step, total, label: seen.append((step, label)),
    )
    assert res["n_shots"] == 2
    assert res["n_selected"] == 2            # auto-select bound both shots
    assert res["topic"] == "cars"
    assert res["qa"]["overall"] == "ok"
    assert os.path.exists(res["xml_path"])   # FCPXML written to disk
    assert seen[0][0] == 1 and "Transcrib" in seen[0][1]   # progress fired


def test_pipeline_cancels_before_work(monkeypatch, tmp_path):
    from core.pipeline import PipelineCancelled
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PipelineCancelled):
        pipeline.run_pipeline_headless("v.mp3", should_cancel=lambda: True)


def test_download_cancels(monkeypatch, tmp_path):
    from core.pipeline import PipelineCancelled
    monkeypatch.chdir(tmp_path)
    shots = [{"slot_id": 1, "priority": "medium",
              "selected_results": [{"url": "http://x/a.mp4", "source": "pexels"}]}]
    with pytest.raises(PipelineCancelled):
        pipeline.download_selected_clips(shots, "p", should_cancel=lambda: True)


def test_pipeline_skips_qa_when_disabled(_mock_stages, monkeypatch):
    import core.director_rank
    monkeypatch.setattr(core.director_rank, "review_timeline",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("QA must not run")))
    res = pipeline.run_pipeline_headless("voice.mp3", project_name="noqa",
                                         download=False, run_qa=False)
    assert res["qa"]["overall"] == "QA review skipped."


def test_repair_empty_shots_refetches_and_selects(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("AUTO_USE_LIBRARY", "false")   # isolate from the real library

    import core.director, core.director_youtube, core.director_search, core.director_rank

    # A failed-block shot (no queries, no candidates) + an already-selected shot.
    empty = {"slot_id": 2, "priority": "medium", "shot_intent": "engine",
             "search_queries": [], "video_results": [], "selected_results": []}
    done = {"slot_id": 1, "priority": "medium", "video_results": [{"url": "x"}],
            "selected_results": [{"url": "x"}]}
    shots = [done, empty]

    monkeypatch.setattr(core.director, "ensure_shot_queries",
                        lambda ts, topic="": [s.__setitem__("search_queries", ["q"]) for s in ts])
    monkeypatch.setattr(core.director_youtube, "seed_youtube_keywords", lambda ts: ts)
    # Re-fetch fills the empty shot with a candidate.
    def _fetch(ts, **k):
        for s in ts:
            if not s.get("video_results"):
                s["video_results"] = [{"url": "recovered"}]
    monkeypatch.setattr(core.director_search, "fetch_with_retries", _fetch)
    monkeypatch.setattr(core.director_rank, "rank_shot_candidates", lambda ts, **k: None)

    recovered = pipeline.repair_empty_shots(shots, groq_key="k")
    assert recovered == 1
    assert empty["selected_results"]                       # now has a pick
    assert empty["selected_results"][0]["url"] == "recovered"
    assert done["selected_results"] == [{"url": "x"}]      # untouched


def test_repair_empty_shots_noop_when_all_selected():
    shots = [{"slot_id": 1, "priority": "medium", "selected_results": [{"url": "x"}]}]
    assert pipeline.repair_empty_shots(shots) == 0


def test_pipeline_raises_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        pipeline.run_pipeline_headless("voice.mp3", groq_key="")


def test_pipeline_calls_download_when_enabled(_mock_stages, monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(pipeline, "download_selected_clips",
                        lambda *a, **k: calls.update(n=calls["n"] + 1) or {"ok": 3, "failed": 0})
    res = pipeline.run_pipeline_headless("voice.mp3", project_name="p2", download=True)
    assert calls["n"] == 1 and res["download"]["ok"] == 3


def test_download_routes_by_source(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    routed = []

    import core.direct_downloader, core.youtube

    def _direct(url, out, ts, **k):
        routed.append(("direct", url)); ts["status"] = "completed"
        open(out, "wb").write(b"x")
    def _yt(url, out, q, ts, **k):
        routed.append(("yt", url)); ts["status"] = "completed"
        open(out, "wb").write(b"x")

    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _direct)
    monkeypatch.setattr(core.youtube, "download_video", _yt)

    shots = [{"slot_id": 1, "priority": "medium", "selected_results": [
        {"url": "http://x/a.mp4", "source": "pexels", "matched_query": "city"},
        {"url": "https://youtu.be/abc", "source": "youtube", "matched_query": "road"},
    ]}]
    res = pipeline.download_selected_clips(shots, "proj")
    assert res["ok"] == 2
    assert ("direct", "http://x/a.mp4") in routed
    assert ("yt", "https://youtu.be/abc") in routed

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
    # Keep these orchestration tests hermetic: overlays + the YouTube-coverage
    # search both reach out to LLMs / yt-dlp by default now, so stub them off.
    monkeypatch.setenv("ENABLE_TEXT_OVERLAYS", "false")

    segs = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
    # Distinct, monotonic timestamps as the real director assigns — the
    # pre-export evaluation gate rejects shots that collide on the same start.
    shots = [
        {"slot_id": 1, "priority": "medium", "duration_needed_sec": 5.0,
         "timestamp": 0.0, "end_timestamp": 2.0,
         "video_results": [{"url": "a"}, {"url": "b"}], "selected_results": []},
        {"slot_id": 2, "priority": "medium", "duration_needed_sec": 5.0,
         "timestamp": 2.0, "end_timestamp": 4.0,
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
    # ensure_youtube_coverage() searches YouTube for shots lacking a YT pick;
    # stub it so the orchestration tests don't fire real yt-dlp queries.
    monkeypatch.setattr(core.director_search, "search_youtube_classic", lambda *a, **k: [])
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


def test_pipeline_clears_query_cache_per_job(_mock_stages):
    # A long-lived bot process must not reuse a previous job's search results.
    import core.director_search as ds
    ds._query_cache[("pexels", "stale query", 3, 0)] = [{"url": "stale"}]
    pipeline.run_pipeline_headless("voice.mp3", project_name="fresh", download=False)
    assert ds._query_cache == {}   # cleared at job start; mocked fetch adds nothing


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
    # The pipeline downloads via download_and_repair (download + reselect-on-fail).
    calls = {"n": 0}
    monkeypatch.setattr(pipeline, "download_and_repair",
                        lambda *a, **k: calls.update(n=calls["n"] + 1)
                        or {"ok": 3, "failed": 0, "skipped": 0, "repaired": 0, "dropped": 0})
    res = pipeline.run_pipeline_headless("voice.mp3", project_name="p2", download=True)
    assert calls["n"] == 1 and res["download"]["ok"] == 3


def test_download_dedupes_same_url_across_shots(monkeypatch, tmp_path):
    """The same URL selected for two shots downloads ONCE; the second file is
    materialized as a hardlink/copy of the first."""
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()

    calls = []
    import core.direct_downloader

    def _direct(url, out, ts, **k):
        calls.append(url); ts["status"] = "completed"
        open(out, "wb").write(b"shared-bytes")
    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _direct)

    shots = [
        {"slot_id": 1, "priority": "medium", "selected_results": [
            {"url": "http://x/same.mp4", "source": "pexels", "matched_query": "city"}]},
        {"slot_id": 2, "priority": "medium", "selected_results": [
            {"url": "http://x/same.mp4", "source": "pexels", "matched_query": "city"}]},
    ]
    res = pipeline.download_selected_clips(shots, "proj")

    assert calls == ["http://x/same.mp4"]          # one network fetch
    assert res["ok"] == 2 and res["failed"] == 0   # but both shots got a file
    mp4s = [f for f in os.listdir(res["dir"]) if f.endswith(".mp4")]
    assert len(mp4s) == 2
    for f in mp4s:
        assert open(os.path.join(res["dir"], f), "rb").read() == b"shared-bytes"
    download_cache._reset_for_tests()


def test_download_reuses_cross_session_cache(monkeypatch, tmp_path):
    """A URL already in core.download_cache is linked from disk, not re-fetched."""
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()

    cached = tmp_path / "old_project_clip.mp4"
    cached.write_bytes(b"cached-bytes")
    download_cache.register("http://x/a.mp4", str(cached))

    import core.direct_downloader
    monkeypatch.setattr(core.direct_downloader, "download_direct_video",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not re-download a cached URL")))

    shots = [{"slot_id": 1, "priority": "medium", "selected_results": [
        {"url": "http://x/a.mp4", "source": "pexels", "matched_query": "city"}]}]
    res = pipeline.download_selected_clips(shots, "proj")

    assert res["skipped"] == 1 and res["failed"] == 0
    mp4s = [f for f in os.listdir(res["dir"]) if f.endswith(".mp4")]
    assert mp4s and open(os.path.join(res["dir"], mp4s[0]), "rb").read() == b"cached-bytes"
    download_cache._reset_for_tests()


def test_download_registers_fresh_downloads_in_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()

    import core.direct_downloader

    def _direct(url, out, ts, **k):
        ts["status"] = "completed"
        open(out, "wb").write(b"x")
    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _direct)

    shots = [{"slot_id": 1, "priority": "medium", "selected_results": [
        {"url": "http://x/new.mp4", "source": "pexels", "matched_query": "road"}]}]
    res = pipeline.download_selected_clips(shots, "proj")

    assert res["ok"] == 1
    path = download_cache.lookup_path("http://x/new.mp4")
    assert path and os.path.exists(path)
    download_cache._reset_for_tests()


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

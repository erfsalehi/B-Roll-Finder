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


def test_pipeline_storyboard_toggle_off_runs_the_old_path(_mock_stages, monkeypatch):
    import core.director_rank
    seen = {}
    monkeypatch.setattr(core.director_rank, "rank_shot_candidates",
                        lambda shots, **k: seen.update(k))
    res = pipeline.run_pipeline_headless("voice.mp3", project_name="sb_off", download=False)
    assert "storyboard" not in res["attempts"]
    assert seen["errors"] is None            # rank's own error list, exactly as before


def test_pipeline_storyboard_toggle_on_wires_progress_cancel_and_stats(_mock_stages, monkeypatch):
    import core.director_rank
    import core.storyboard as sb
    monkeypatch.setenv("ENABLE_STORYBOARD_CHECK", "true")
    seen, labels = {}, []

    def _rank(shots, **k):
        seen.update(k)
        sb._bump("checked", 3)
        sb._bump("bad", 1)
        k["storyboard_progress"](2, 5)        # the check reporting into the stage line
    monkeypatch.setattr(core.director_rank, "rank_shot_candidates", _rank)
    res = pipeline.run_pipeline_headless(
        "voice.mp3", project_name="sb_on", download=False,
        progress_callback=lambda step, total, label: labels.append(label),
        should_cancel=lambda: False)
    assert isinstance(seen["errors"], list) and callable(seen["should_cancel"])
    assert "Checking footage · 2/5" in labels
    assert res["attempts"]["storyboard"]["checked"] == 3 and res["attempts"]["storyboard"]["bad"] == 1


def test_pipeline_resets_the_storyboard_tally_per_job(_mock_stages, monkeypatch):
    import core.storyboard as sb
    monkeypatch.setenv("ENABLE_STORYBOARD_CHECK", "true")
    sb._bump("checked", 99)                   # left over from a previous job
    res = pipeline.run_pipeline_headless("voice.mp3", project_name="sb_reset", download=False)
    assert res["attempts"]["storyboard"]["checked"] == 0


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


def test_pipeline_fetches_related_images_when_configured(_mock_stages, monkeypatch):
    """Related images (like extras) run automatically on every full job — but
    only once Google Custom Search is configured."""
    import core.related_images as ri
    monkeypatch.setattr(ri, "cse_configured", lambda: True)
    monkeypatch.setattr(ri, "fetch_related_images",
                        lambda script, key, errors=None, per_query=None: [
                            {"url": "https://x/a.jpg", "query": "toyota logo", "kind": "brand"}])
    calls = []
    monkeypatch.setattr(pipeline, "_store_related_images",
                        lambda images, proj, errors, **kw: (
                            calls.append((images, proj)) or (1, ["toyota logo"])))
    res = pipeline.run_pipeline_headless("voice.mp3", project_name="imgproj", download=False)
    assert calls and calls[0][0][0]["url"] == "https://x/a.jpg"
    assert calls[0][1] == "imgproj"
    assert res["attempts"]["related_images"] == 1


def test_pipeline_skips_related_images_without_credentials(_mock_stages, monkeypatch):
    """No Google credentials configured → silently skipped, no error noise."""
    import core.related_images as ri
    monkeypatch.setattr(ri, "cse_configured", lambda: False)
    called = []
    monkeypatch.setattr(pipeline, "_store_related_images",
                        lambda *a, **k: called.append(1) or (0, []))
    res = pipeline.run_pipeline_headless("voice.mp3", project_name="noimg", download=False)
    assert called == []                       # never reached
    assert "related_images" not in res["attempts"]
    assert not any("related images" in e for e in res["errors"])


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


def _resuming_direct(fail_urls=()):
    """A fake direct downloader that, like the real one, resumes a leftover
    ``.part``, and leaves one behind when it fails. Each clip's bytes are its URL."""
    def _direct(url, out, ts, **k):
        part = out + ".part"
        prev = open(part, "rb").read() if os.path.exists(part) else b""
        if url in fail_urls:
            open(part, "wb").write(b"half-of-" + url.encode())
            ts["status"], ts["error_msg"] = "error", "connection reset"
            return
        open(out, "wb").write(prev + url.encode())
        if os.path.exists(part):
            os.remove(part)
        ts["status"] = "completed"
    return _direct


def _clip(url):
    return {"url": url, "source": "pexels", "matched_query": "red car"}


def _assert_each_clip_has_its_own_bytes(shot):
    paths = [c["local_path"] for c in shot["selected_results"]]
    assert len(set(paths)) == len(paths)
    for c in shot["selected_results"]:
        assert open(c["local_path"], "rb").read() == c["url"].encode()


def test_repair_round_never_hands_a_new_clip_another_clips_file(monkeypatch, tmp_path):
    """Dropping a failed clip shifts the later clips' indexes, so a fresh
    filename can equal a file already holding a different clip."""
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()
    import core.direct_downloader
    monkeypatch.setattr(core.direct_downloader, "download_direct_video",
                        _resuming_direct(fail_urls={"http://x/a.mp4"}))

    shot = {"slot_id": 5, "priority": "medium", "selected_results": [
        _clip("http://x/b.mp4"), _clip("http://x/a.mp4"), _clip("http://x/c.mp4")]}
    pipeline.download_selected_clips([shot], "proj")
    # What download_and_repair does: drop the failure, top up with a new pick.
    shot["selected_results"] = [c for c in shot["selected_results"]
                                if not c.get("_dl_failed")] + [_clip("http://x/d.mp4")]
    res = pipeline.download_selected_clips([shot], "proj")

    assert res["failed"] == 0
    _assert_each_clip_has_its_own_bytes(shot)
    download_cache._reset_for_tests()


def test_repair_round_never_resumes_another_clips_partial(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()
    import core.direct_downloader
    monkeypatch.setattr(core.direct_downloader, "download_direct_video",
                        _resuming_direct(fail_urls={"http://x/a.mp4"}))

    shot = {"slot_id": 5, "priority": "medium", "selected_results": [
        _clip("http://x/b.mp4"), _clip("http://x/a.mp4")]}
    pipeline.download_selected_clips([shot], "proj")
    shot["selected_results"] = [shot["selected_results"][0], _clip("http://x/d.mp4")]
    pipeline.download_selected_clips([shot], "proj")

    _assert_each_clip_has_its_own_bytes(shot)
    download_cache._reset_for_tests()


def test_download_replaces_a_leftover_file_from_another_project(monkeypatch, tmp_path):
    """A file that merely has the clip's name (an earlier project with the same
    folder) is not this clip."""
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()
    import core.direct_downloader
    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _resuming_direct())

    base = os.path.join("downloads", "proj", "director")
    os.makedirs(base)
    open(os.path.join(base, "5-1-red-car.mp4"), "wb").write(b"someone else's clip")
    shot = {"slot_id": 5, "priority": "medium", "selected_results": [_clip("http://x/b.mp4")]}
    res = pipeline.download_selected_clips([shot], "proj")

    assert res["ok"] == 1
    _assert_each_clip_has_its_own_bytes(shot)
    download_cache._reset_for_tests()


def test_library_clip_from_youtube_downloads_through_ytdlp(monkeypatch, tmp_path):
    """A Clip Library hit whose file is gone is re-fetched from its source URL —
    a YouTube watch page, which only yt-dlp can download."""
    monkeypatch.chdir(tmp_path)
    routed = []
    import core.direct_downloader, core.youtube

    def _direct(url, out, ts, **k):
        routed.append("direct"); ts["status"] = "completed"
        open(out, "wb").write(b"<html>")
    def _yt(url, out, q, ts, **k):
        routed.append("yt"); ts["status"] = "completed"
        open(out, "wb").write(b"video")

    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _direct)
    monkeypatch.setattr(core.youtube, "download_video", _yt)
    shots = [{"slot_id": 1, "priority": "medium", "selected_results": [
        {"url": "https://www.youtube.com/watch?v=abc", "source": "library",
         "original_source": "youtube", "local_path": "/gone/old.mp4",
         "matched_query": "road"}]}]
    pipeline.download_selected_clips(shots, "proj")
    assert routed == ["yt"]


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


def test_qa_reviews_the_settled_timeline(_mock_stages, monkeypatch):
    """Boundary enforcement and the YouTube guarantee can swap a shot's lead
    clip, so they must run BEFORE the QA review, not after it."""
    import core.director_rank
    order = []
    real_enforce = pipeline.enforce_timeline
    monkeypatch.setattr(pipeline, "enforce_timeline",
                        lambda *a, **k: order.append("enforce") or real_enforce(*a, **k))
    monkeypatch.setattr(pipeline, "ensure_youtube_coverage",
                        lambda *a, **k: order.append("youtube") or 0)
    monkeypatch.setattr(core.director_rank, "review_timeline",
                        lambda *a, **k: order.append("qa") or {"overall": "ok", "issues": []})
    pipeline.run_pipeline_headless("voice.mp3", project_name="order", download=False)
    assert order.index("enforce") < order.index("qa")
    assert order.index("youtube") < order.index("qa")


# ── downloaded files must be playable videos ─────────────────────────────────

from core.pipeline import _verify_clip_file as _real_verify   # conftest stubs the module attr


def test_unplayable_download_is_a_failure_not_a_clip(monkeypatch, tmp_path):
    """A failed/truncated download that leaves a non-empty file used to count
    as a good clip — shipped in the zip, cached for later projects, and stored
    in the Clip Library."""
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()
    import core.direct_downloader

    def _direct(url, out, ts, **k):
        ts["status"] = "error"; ts["error_msg"] = "EOFError: 8 bytes missing"
        open(out, "wb").write(b"half a video")          # leftover bytes
    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _direct)
    monkeypatch.setattr(pipeline, "_verify_clip_file",
                        lambda p: (False, "moov atom not found"))
    stored = []
    import core.clip_library
    monkeypatch.setattr(core.clip_library, "store_clip", lambda *a, **k: stored.append(a))

    clip = {"url": "http://x/broken.mp4", "source": "pexels", "matched_query": "ac"}
    shots = [{"slot_id": 74, "priority": "medium", "selected_results": [clip]}]
    res = pipeline.download_selected_clips(shots, "proj")
    assert res["ok"] == 0 and res["failed"] == 1
    assert clip.get("_dl_failed") and "not a playable video" in clip["_dl_error"]
    assert not any(f.endswith(".mp4") for f in os.listdir(res["dir"]))   # discarded
    assert download_cache.lookup_path("http://x/broken.mp4") is None      # not cached
    assert stored == []                                                  # not in library
    download_cache._reset_for_tests()


def test_broken_cached_copy_is_forgotten_and_redownloaded(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()
    bad = tmp_path / "old-project-clip.mp4"
    bad.write_bytes(b"corrupt")
    download_cache.register("http://x/c.mp4", str(bad))

    import core.direct_downloader
    calls = []

    def _direct(url, out, ts, **k):
        calls.append(url); ts["status"] = "completed"
        open(out, "wb").write(b"good video")
    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _direct)
    monkeypatch.setattr(pipeline, "_verify_clip_file",
                        lambda p: (b"good" in open(p, "rb").read(), "corrupt"))

    clip = {"url": "http://x/c.mp4", "source": "pexels", "matched_query": "q"}
    res = pipeline.download_selected_clips([{"slot_id": 1, "priority": "medium",
                                             "selected_results": [clip]}], "proj")
    assert calls == ["http://x/c.mp4"]              # cache copy rejected → fresh download
    assert res["ok"] == 1 and clip.get("_dl_ok")
    assert download_cache.lookup_path("http://x/c.mp4") != str(bad)
    download_cache._reset_for_tests()


def test_verify_clip_file_rejects_garbage_and_accepts_a_real_video(tmp_path):
    import shutil, subprocess
    if not (shutil.which("ffprobe") and shutil.which("ffmpeg")):
        pytest.skip("ffmpeg/ffprobe not installed")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 4096)
    ok, why = _real_verify(str(junk))
    assert not ok and why

    real = tmp_path / "real.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(real)], check=True)
    assert _real_verify(str(real)) == (True, "")


def test_missing_media_warnings_collapse_to_one_line(tmp_path, capsys):
    """Before /download nearly every clip is missing; that used to print one
    '[timeline] warning' line per clip on every XML write (~35 per /refine)."""
    from core.output import generate_fcpxml
    shots = [{"slot_id": i, "priority": "medium", "timestamp": (i - 1) * 3.0,
              "end_timestamp": i * 3.0, "text": f"line {i}",
              "selected_results": [{"url": f"https://x/{i}.mp4", "source": "pexels",
                                    "matched_query": "q", "title": f"clip {i}"}]}
             for i in range(1, 6)]
    xml = generate_fcpxml(shots, project_name="p")
    pipeline._evaluate_and_repair_xml(xml, str(tmp_path))
    out = capsys.readouterr().out
    assert "5 clip(s) not on disk yet (expected before /download)" in out
    assert out.count("media not found on disk") == 1    # only the one example

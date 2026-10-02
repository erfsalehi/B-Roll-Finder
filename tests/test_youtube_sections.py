"""Section downloads (YT_SECTION_DOWNLOAD): fetch only the window of a YouTube video the timeline
will play, not the whole upload. Covers the window planning (pipeline.plan_youtube_section),
how the XML maps the in-point into a partial file (output._pick_in_frame, rule "section"), and
the download behaviour: success, a URL shared by several shots, the fall-back to the whole
video when the cut fails, and "off unless enabled". yt-dlp is faked with real tiny ffmpeg videos;
the real thing needs a network where yt-dlp's ffmpeg hand-off works (see scratch/live_section.py
and the /test "YouTube section download" check)."""

import shutil
import subprocess

import pytest

import core.output as output
import core.pipeline as pl
import core.youtube

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg/ffprobe not installed")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("YT_SECTION_DOWNLOAD", "YT_SECTION_HANDLE_SEC", "YT_SECTION_MIN_VIDEO_SEC",
              "YT_SECTION_MAX_SHARE", "YT_MIN_IN_SEC"):
        monkeypatch.delenv(k, raising=False)


def _res(url="https://www.youtube.com/watch?v=aaa", duration=600, **kw):
    return dict({"url": url, "source": "youtube", "duration": duration, "matched_query": "q"}, **kw)


# ── planning the window ──────────────────────────────────────────────────────

def test_the_window_starts_at_the_planned_in_point_minus_a_handle():
    # no human rule: the intro skip puts the in-point at 8 s; 4 s handle each side + 1 s of slack
    plan = pl.plan_youtube_section(_res(), [5.0])
    assert plan == {"start": 4.0, "end": 18.0, "in_sec": 8.0, "rule": "intro"}


def test_a_storyboard_in_point_moves_the_window_there():
    plan = pl.plan_youtube_section(_res(storyboard_in_sec=120.0), [6.0])
    assert plan["rule"] == "storyboard" and plan["start"] == 116.0 and plan["end"] == 131.0


def test_shots_sharing_a_video_widen_the_window():
    one = pl.plan_youtube_section(_res(), [5.0])
    three = pl.plan_youtube_section(_res(), [5.0, 5.0, 6.0])
    assert three["start"] == one["start"] and three["end"] == one["end"] + 11.0


def test_no_window_for_short_unknown_or_mostly_wanted_videos():
    assert pl.plan_youtube_section(_res(duration=30), [5.0]) is None          # small anyway
    assert pl.plan_youtube_section({"url": "u", "source": "youtube"}, [5.0]) is None   # length unknown
    assert pl.plan_youtube_section(_res(duration=45), [30.0]) is None         # window is ~87% of it
    assert pl.plan_youtube_section(_res(duration="n/a"), [5.0]) is None


def test_the_window_is_tunable(monkeypatch):
    monkeypatch.setenv("YT_SECTION_HANDLE_SEC", "10")
    plan = pl.plan_youtube_section(_res(), [5.0])
    assert plan["start"] == 0.0 and plan["end"] == 24.0                       # clamped at the start
    monkeypatch.setenv("YT_SECTION_MIN_VIDEO_SEC", "700")
    assert pl.plan_youtube_section(_res(), [5.0]) is None


def test_the_enable_flag():
    assert not pl.section_downloads_enabled()


# ── the XML maps the in-point into the partial file ──────────────────────────

def test_a_partial_file_plays_from_where_the_window_put_the_in_point():
    c = _res(section_start=116.0, section_in_sec=4.0, section_end=131.0)
    frame = output._preferred_in_frame("u", "x.mp4", 120, 15 * 24, 24.0, c)
    assert frame == 96 and c["in_rule"] == "section"                           # 4 s into a 15 s file


def test_the_in_point_never_overruns_a_partial_file():
    c = _res(section_start=116.0, section_in_sec=14.0, section_end=131.0)
    frame = output._preferred_in_frame("u", "x.mp4", 120, 15 * 24, 24.0, c)
    assert frame + 120 <= 15 * 24


def test_plan_in_point_ignores_stale_section_fields():
    stale = _res(section_start=50.0, section_in_sec=4.0)
    assert output.plan_in_point(stale, 5.0, 600.0)[0] == "intro"               # as for a whole file


# ── the download ─────────────────────────────────────────────────────────────

def _video(path, seconds):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"testsrc2=s=160x90:r=10:d={max(1.0, seconds):.2f}", "-c:v", "libx264",
                    "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True)


class _Net:
    """A stand-in for core.youtube.download_video that records calls and writes real videos."""

    def __init__(self, fail_sections=False, short_sections=False, full_seconds=120):
        self.calls, self.fail_sections, self.short_sections, self.full = [], fail_sections, short_sections, full_seconds

    def __call__(self, url, out, quality, ts, no_audio=True, section=None, **k):
        self.calls.append((url, section))
        ts["status"] = "downloading"
        if section and self.fail_sections:
            ts["status"] = "error"
            ts["error_msg"] = "ffmpeg exited with code 3436169992"
            return
        seconds = self.full if not section else (2.0 if self.short_sections else section[1] - section[0])
        _video(out, seconds)
        ts["status"] = "completed"


def _run(monkeypatch, tmp_path, net, shots, enable=True, workers=None):
    monkeypatch.chdir(tmp_path)
    if enable:
        monkeypatch.setenv("YT_SECTION_DOWNLOAD", "true")
    from core import download_cache
    download_cache._reset_for_tests()
    monkeypatch.setattr(core.youtube, "download_video", net)
    import core.clip_library as library
    monkeypatch.setattr(library, "store_clip", lambda *a, **k: None)
    return pl.download_selected_clips(shots, "proj", max_workers=workers), download_cache


def _shot(slot, res, dur=5.0):
    return {"slot_id": slot, "priority": "medium", "duration_needed_sec": dur, "selected_results": [res]}


@needs_ffmpeg
def test_only_the_window_is_downloaded_and_recorded(monkeypatch, tmp_path):
    net, clip = _Net(), _res()
    rep, cache = _run(monkeypatch, tmp_path, net, [_shot(1, clip)])
    assert net.calls == [(clip["url"], (4.0, 18.0))]
    assert rep["ok"] == 1 and rep["sections"] == 1 and clip["_dl_ok"]
    assert clip["section_start"] == 4.0 and clip["section_in_sec"] == 4.0 and clip["section_end"] == 18.0
    assert cache.lookup_path(clip["url"]) is None          # a window is not the video: never cached
    cache._reset_for_tests()


@needs_ffmpeg
def test_two_shots_sharing_a_video_share_one_file_covering_both(monkeypatch, tmp_path):
    net = _Net()
    a, b = _res(), _res()
    rep, cache = _run(monkeypatch, tmp_path, net, [_shot(1, a, 5.0), _shot(2, b, 6.0)])
    assert len(net.calls) == 1 and net.calls[0][1] == (4.0, 8.0 + 11.0 + 4.0 + 1.0)
    for r in (a, b):
        assert r["_dl_ok"] and r["section_start"] == 4.0 and r["section_in_sec"] == 4.0
    cache._reset_for_tests()


@needs_ffmpeg
def test_a_failed_cut_falls_back_to_the_whole_video_and_stops_trying(monkeypatch, tmp_path):
    net = _Net(fail_sections=True)
    a, b = _res("https://www.youtube.com/watch?v=aaa"), _res("https://www.youtube.com/watch?v=bbb")
    # one worker so the order is fixed: with several, each video already in flight tries its cut once
    rep, cache = _run(monkeypatch, tmp_path, net, [_shot(1, a), _shot(2, b)], workers=1)
    sections = [c for c in net.calls if c[1]]
    assert len(sections) == 1                                # one failure ends the attempts for the run
    assert rep["ok"] == 2 and rep["sections"] == 0 and rep["failed"] == 0
    assert a["_dl_ok"] and "section_start" not in a and "section_start" not in b
    assert sum("section download unavailable" in e for e in rep["errors"]) == 1
    assert cache.lookup_path(a["url"]) or cache.lookup_path(b["url"])        # whole videos are cached
    cache._reset_for_tests()


@needs_ffmpeg
def test_a_cut_that_comes_out_far_too_short_is_not_trusted(monkeypatch, tmp_path):
    net, clip = _Net(short_sections=True), _res()
    rep, cache = _run(monkeypatch, tmp_path, net, [_shot(1, clip)])
    assert [c[1] is not None for c in net.calls] == [True, False]            # tried the cut, then the video
    assert clip["_dl_ok"] and "section_start" not in clip
    cache._reset_for_tests()


@needs_ffmpeg
def test_off_by_default_downloads_the_whole_video(monkeypatch, tmp_path):
    net, clip = _Net(), _res()
    rep, cache = _run(monkeypatch, tmp_path, net, [_shot(1, clip)], enable=False)
    assert net.calls == [(clip["url"], None)] and rep["sections"] == 0
    assert "section_start" not in clip and cache.lookup_path(clip["url"])
    cache._reset_for_tests()


@needs_ffmpeg
def test_a_library_clip_is_never_cut(monkeypatch, tmp_path):
    net = _Net()
    clip = _res(library_segment_id=4)                         # a human already chose that cut
    rep, cache = _run(monkeypatch, tmp_path, net, [_shot(1, clip)])
    assert net.calls and net.calls[0][1] is None
    cache._reset_for_tests()


@needs_ffmpeg
def test_a_clip_with_an_unknown_length_is_downloaded_whole(monkeypatch, tmp_path):
    net, clip = _Net(), _res()
    clip.pop("duration")
    rep, cache = _run(monkeypatch, tmp_path, net, [_shot(1, clip)])
    assert net.calls == [(clip["url"], None)]
    cache._reset_for_tests()

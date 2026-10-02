"""Footage-quality filters that need no model: the black-bar check on downloaded clips
(core.clip_geometry, wired into pipeline.download_selected_clips) and the intro-skip in-point
rule (output._intro_skip_frames). Both come from the first real bot output: two channel
logo/bumper clips played from frame 0 of their source videos, and three shots sat inside black
bars. The bar check is run against real synthetic videos made with ffmpeg."""

import os
import shutil
import subprocess

import pytest

import core.clip_geometry as cg
import core.output as output
import core.pipeline as pipeline

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg/ffprobe not installed")


@pytest.fixture(autouse=True)
def _bar_check_on(monkeypatch):
    monkeypatch.setenv("ENABLE_BAR_CHECK", "true")           # conftest turns it off for fake bytes
    monkeypatch.delenv("CLIP_MIN_PICTURE", raising=False)
    monkeypatch.delenv("YT_MIN_IN_SEC", raising=False)


def _make(path, size, vf="", src="testsrc2", dur=4):
    """A real H.264 video: ``src`` at ``size`` then the ``vf`` filter chain."""
    sep = ":" if "=" in src else "="                         # "color=c=black" already has options
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{src}{sep}s={size}:r=24:d={dur}"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)
    return str(path)


# ── the bar detector ─────────────────────────────────────────────────────────

@needs_ffmpeg
def test_a_full_frame_clip_is_fine(tmp_path):
    a = cg.analyse(_make(tmp_path / "full.mp4", "1920x1080"))
    assert a["picture"] > 0.97 and cg.problem(str(tmp_path / "full.mp4")) is None


@needs_ffmpeg
def test_pillarbox_inside_the_file_is_caught(tmp_path):
    # a 4:3 picture on a 16:9 canvas: two thick black sides (the Dodge in the driveway)
    p = _make(tmp_path / "pillar.mp4", "1440x1080", "pad=1920:1080:240:0:black")
    a = cg.analyse(p)
    assert a["bars"]["left"] == pytest.approx(0.125, abs=0.02) and a["picture"] == pytest.approx(0.75, abs=0.03)
    assert "black bars" in cg.problem(p)


@needs_ffmpeg
def test_letterbox_and_inset_pictures_are_caught(tmp_path):
    lb = _make(tmp_path / "lb.mp4", "1920x800", "pad=1920:1080:0:140:black")
    assert cg.analyse(lb)["picture"] == pytest.approx(0.74, abs=0.04) and cg.problem(lb)
    inset = _make(tmp_path / "inset.mp4", "960x540", "pad=1920:1080:480:270:black")     # in the middle
    assert cg.analyse(inset)["picture"] == pytest.approx(0.25, abs=0.04) and cg.problem(inset)


@needs_ffmpeg
def test_a_portrait_file_is_caught_by_its_shape(tmp_path):
    p = _make(tmp_path / "short.mp4", "608x1080")                 # a YouTube Short, no bars inside
    a = cg.analyse(p)
    assert a["native"] < 0.4 and a["inner"] > 0.95
    assert cg.problem(p).startswith("portrait")


@needs_ffmpeg
def test_a_dark_but_textured_scene_is_not_mistaken_for_bars(tmp_path):
    # a night shot: every pixel dark, but with grain, so no row is flat
    p = _make(tmp_path / "night.mp4", "1920x1080", "noise=alls=22:allf=t,eq=brightness=-0.35")
    assert cg.problem(p) is None


@needs_ffmpeg
def test_a_blank_black_clip_is_inconclusive_not_a_failure(tmp_path):
    p = _make(tmp_path / "black.mp4", "1920x1080", src="color=c=black")
    assert cg.analyse(p) is None and cg.problem(p) is None


@needs_ffmpeg
def test_the_check_can_be_switched_off_and_tuned(tmp_path, monkeypatch):
    p = _make(tmp_path / "pillar.mp4", "1440x1080", "pad=1920:1080:240:0:black")
    monkeypatch.setenv("CLIP_MIN_PICTURE", "0.7")
    assert cg.problem(p) is None                                  # 75% now acceptable
    monkeypatch.delenv("CLIP_MIN_PICTURE")
    monkeypatch.setenv("ENABLE_BAR_CHECK", "false")
    assert cg.problem(p) is None


def test_it_fails_open_on_junk_and_missing_files(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 2048)
    assert cg.problem(str(junk)) is None and cg.problem(str(tmp_path / "nope.mp4")) is None
    assert cg.problem("") is None


def test_native_fraction_math():
    assert cg.native_fraction(1920, 1080) == 1.0
    assert cg.native_fraction(1280, 720) == 1.0                   # scales up to fill
    assert cg.native_fraction(1440, 1080) == pytest.approx(0.75)
    assert cg.native_fraction(1080, 1920) == pytest.approx(0.316, abs=0.01)
    assert cg.native_fraction(0, 0) == 1.0


# ── wired into the download: a barred clip is a failed download ──────────────

def _download(monkeypatch, tmp_path, clip, make):
    monkeypatch.chdir(tmp_path)
    from core import download_cache
    download_cache._reset_for_tests()
    import core.direct_downloader

    def _direct(url, out, ts, **k):
        ts["status"] = "completed"
        make(out)
    monkeypatch.setattr(core.direct_downloader, "download_direct_video", _direct)
    import core.clip_library
    monkeypatch.setattr(core.clip_library, "store_clip", lambda *a, **k: None)
    res = pipeline.download_selected_clips([{"slot_id": 3, "priority": "medium",
                                             "selected_results": [clip]}], "proj")
    return res, download_cache


@needs_ffmpeg
def test_a_barred_download_is_discarded_uncached_and_repickable(monkeypatch, tmp_path):
    clip = {"url": "http://x/barred.mp4", "source": "pexels", "matched_query": "ac"}
    res, cache = _download(monkeypatch, tmp_path, clip,
                           lambda out: _make(out, "1440x1080", "pad=1920:1080:240:0:black"))
    assert res["ok"] == 0 and res["failed"] == 1
    assert clip["_dl_failed"] and "black bars" in clip["_dl_error"]
    assert not any(f.endswith(".mp4") for f in os.listdir(res["dir"]))          # discarded
    assert cache.lookup_path("http://x/barred.mp4") is None                      # not cached
    cache._reset_for_tests()


@needs_ffmpeg
def test_a_clean_download_still_passes(monkeypatch, tmp_path):
    clip = {"url": "http://x/clean.mp4", "source": "pexels", "matched_query": "ac"}
    res, cache = _download(monkeypatch, tmp_path, clip, lambda out: _make(out, "1920x1080"))
    assert res["ok"] == 1 and clip.get("_dl_ok") and not clip.get("_dl_failed")
    cache._reset_for_tests()


@needs_ffmpeg
def test_a_library_segment_is_exempt_because_a_human_cut_it(tmp_path):
    p = _make(tmp_path / "pillar.mp4", "1440x1080", "pad=1920:1080:240:0:black")
    assert pipeline._geometry_problem(p, {"url": "u"}) != ""
    assert pipeline._geometry_problem(p, {"url": "u", "library_segment_id": 7}) == ""
    assert pipeline._geometry_problem(p, {"url": "u", "segment_path": p}) == ""


# ── the intro skip ───────────────────────────────────────────────────────────

FPS, SLOT = 30.0, 120                      # a 4 s slot at 30 fps


def _in(c, media_sec):
    return output._preferred_in_frame("https://y/x", "x.mp4", SLOT, int(media_sec * FPS), FPS, c)


def test_a_youtube_clip_starts_past_its_channel_intro():
    c = {"source": "youtube", "url": "https://y/x"}
    assert _in(c, 600) == 8 * 30 and c["in_rule"] == "intro"      # logo bumpers sit in the first seconds


def test_a_short_upload_skips_proportionally_and_never_overruns():
    c = {"source": "youtube", "url": "https://y/x"}
    assert _in(c, 20) == int(0.2 * 20 * 30)                       # 4 s of a 20 s video
    tight = {"source": "youtube", "url": "https://y/x"}
    assert _in(tight, 4.5) == 0 and tight["in_rule"] == "default" # a 0.5 s skip isn't worth it
    room = {"source": "youtube", "url": "https://y/x"}
    assert _in(room, 5.5) + SLOT <= int(5.5 * FPS)                # always leaves the slot's length


def test_other_sources_and_other_rules_are_untouched():
    for src in ("pexels", "pixabay", ""):
        c = {"source": src, "url": "u"}
        assert _in(c, 600) == 0 and c["in_rule"] == "default"
    seg = {"source": "youtube", "library_segment_id": 3}
    assert _in(seg, 600) == 0 and seg["in_rule"] == "segment"
    sb = {"source": "youtube", "url": "https://y/x", "storyboard_in_sec": 42.0}
    assert _in(sb, 600) == 1260 and sb["in_rule"] == "storyboard"  # a better guess outranks it


def test_the_skip_is_tunable_and_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("YT_MIN_IN_SEC", "12")
    assert _in({"source": "youtube", "url": "u"}, 600) == 12 * 30
    monkeypatch.setenv("YT_MIN_IN_SEC", "0")
    c = {"source": "youtube", "url": "u"}
    assert _in(c, 600) == 0 and c["in_rule"] == "default"


def test_a_library_clip_that_came_from_youtube_is_not_skipped():
    # source 'library' keeps its origin in original_source; its stored segment is already the good part
    c = {"source": "library", "original_source": "youtube", "library_segment_id": 9}
    assert _in(c, 600) == 0 and c["in_rule"] == "segment"

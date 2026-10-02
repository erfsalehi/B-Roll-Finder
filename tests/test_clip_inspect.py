"""The clip inspector (core.clip_inspect): a vision look at frames of each downloaded clip, run
inside download_and_repair so a logo bumper / watermark / unrelated clip is replaced by the repair
loop. The vision model is faked at clip_inspect._vision (the one seam); the clips are real tiny
videos made with ffmpeg, so frame grabbing, the file deletion and the repair loop are exercised
for real. The prompt itself was validated on the first real output (scratch/inspect_frames.py)."""

import io
import os
import shutil
import subprocess

import pytest

import core.clip_inspect as ci
import core.pipeline as p

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg/ffprobe not installed")


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv("ENABLE_CLIP_INSPECT", "true")
    monkeypatch.setattr(ci, "_vision_ready", lambda: True)
    for k in ("CLIP_INSPECT_MIN_RELEVANCE", "CLIP_INSPECT_MAX_CALLS", "CLIP_INSPECT_FRAMES",
              "CLIP_INSPECT_WORKERS"):
        monkeypatch.delenv(k, raising=False)


def _f(issues=(), rel=0.8, note="a car"):
    return {"issues": list(issues), "relevance": rel, "note": note}


# ── the verdict ──────────────────────────────────────────────────────────────

def test_a_logo_or_a_watermark_on_any_frame_rejects_the_clip():
    v = ci.decide([_f(["logo_bumper"], 0.2), _f()])
    assert v["reject"] and "logo" in v["reason"] and v["issues"] == ["logo_bumper"]
    v = ci.decide([_f(), _f(["watermark"], 0.6)])
    assert v["reject"] and "watermark" in v["reason"]


def test_a_presenter_only_rejects_when_it_is_on_every_frame():
    assert ci.decide([_f(["presenter"]), _f(["presenter"])])["reject"]
    assert not ci.decide([_f(["presenter"]), _f()])["reject"]        # b-roll that cuts to a person
    assert not ci.decide([_f(["presenter"])])["reject"]              # one frame is not enough to say


def test_low_relevance_rejects_and_the_threshold_is_tunable(monkeypatch):
    v = ci.decide([_f(rel=0.3), _f(rel=0.4)])                       # the Corvette badge under "ice cold air"
    assert v["reject"] and "not about the narration" in v["reason"] and v["relevance"] == 0.35
    assert not ci.decide([_f(rel=0.5), _f(rel=0.5)])["reject"]
    monkeypatch.setenv("CLIP_INSPECT_MIN_RELEVANCE", "0.6")
    assert ci.decide([_f(rel=0.5), _f(rel=0.5)])["reject"]


def test_text_graphics_alone_is_not_a_reason_and_nothing_usable_is_none():
    assert not ci.decide([_f(["text_graphics"]), _f(["text_graphics"])])["reject"]
    assert ci.decide([None, None]) is None and ci.decide([]) is None
    assert ci.decide([_f(rel=None)])["relevance"] is None


def test_frames_are_taken_inside_the_stretch_that_will_play():
    assert ci.frame_times(8.0, 5.0, 600.0, 2) == [8.75, 12.25]
    assert ci.frame_times(8.0, 5.0, 600.0, 3) == [8.6, 10.5, 12.4]
    assert ci.frame_times(0.0, 4.0, 3.0, 2) == [0.6, 2.7]            # never past the end of a short file
    assert ci.frame_times(0.0, 30.0, 600.0, 1) == [3.0]               # at most 6 s of the clip is looked at


# ── judging in batches ───────────────────────────────────────────────────────

def _fake_vision(calls, answer):
    def _v(system, user, jpegs):
        calls.append((user, len(jpegs)))
        return {"frames": [dict(id=i, **answer(i)) for i in range(1, len(jpegs) + 1)]}
    return _v


def test_clips_are_judged_in_batches_and_matched_back_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(ci, "_vision", _fake_vision(calls, lambda i: {"issues": [], "relevance": i / 10, "note": f"n{i}"}))
    items = [{"frames": [b"a", b"b"], "narration": f"line {k}"} for k in range(5)]   # 10 frames
    out = ci.judge(items, "car AC")
    assert [n for _, n in calls] == [8, 2]                           # 4 clips x 2 frames, then 1 clip
    assert "VIDEO TOPIC: car AC" in calls[0][0] and 'narration: "line 3"' in calls[0][0]
    assert "the editor wanted" not in calls[0][0]                      # only when the shot says so
    assert all(o is not None and len(o) == 2 for o in out)
    assert out[4][0]["relevance"] == 0.1                             # numbering restarts in each call


def test_what_the_director_wanted_goes_to_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(ci, "_vision", _fake_vision(calls, lambda i: {"issues": [], "relevance": 0.9}))
    ci.judge([{"frames": [b"a"], "narration": "it costs nothing at all", "intent": "hands opening an empty wallet"}])
    assert 'the editor wanted: "hands opening an empty wallet"' in calls[0][0]
    assert "empty wallet" in ci._SYSTEM                                  # and the model is told to honour it


def test_a_clip_the_model_did_not_answer_for_stays_unjudged(monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"frames": [{"id": 1, "issues": [], "relevance": 0.9}]})
    out = ci.judge([{"frames": [b"a"], "narration": "x"}, {"frames": [b"b"], "narration": "y"}])
    assert out[0] and out[1] is None
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: None)
    assert ci.judge([{"frames": [b"a"], "narration": "x"}]) == [None]
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"frames": "junk"})
    assert ci.judge([{"frames": [b"a"], "narration": "x"}]) == [None]


def test_the_number_of_vision_calls_is_capped(monkeypatch):
    calls = []
    monkeypatch.setattr(ci, "_vision", _fake_vision(calls, lambda i: {"issues": [], "relevance": 0.9}))
    monkeypatch.setenv("CLIP_INSPECT_MAX_CALLS", "1")
    out = ci.judge([{"frames": [b"a", b"b"], "narration": "x"} for _ in range(10)])
    assert len(calls) == 1 and sum(o is not None for o in out) == 4  # only the first call's clips


def test_off_without_the_switch_or_without_a_vision_provider(monkeypatch):
    monkeypatch.setenv("ENABLE_CLIP_INSPECT", "false")
    assert not ci.enabled()
    monkeypatch.setenv("ENABLE_CLIP_INSPECT", "true")
    monkeypatch.setattr(ci, "_vision_ready", lambda: False)
    assert not ci.enabled()


# ── the pass over downloaded clips ───────────────────────────────────────────

def _video(path, colour, seconds=8):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c={colour}:s=320x180:r=10:d={seconds}", "-c:v", "libx264", "-preset",
                    "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True)
    return str(path)


def _is_red(jpg: bytes) -> bool:
    from PIL import Image
    r, g, b = Image.open(io.BytesIO(jpg)).convert("RGB").resize((1, 1)).getpixel((0, 0))
    return r > 150 and g < 90 and b < 90


def _logo_if_red(system, user, jpegs):
    """A stand-in model: a red frame is a logo bumper, anything else is a fine on-topic frame."""
    return {"frames": [{"id": i, "issues": ["logo_bumper"] if _is_red(j) else [],
                        "relevance": 0.2 if _is_red(j) else 0.85, "note": "x"}
                       for i, j in enumerate(jpegs, 1)]}


def _shot(slot, clips, dur=6.0):
    return {"slot_id": slot, "priority": "medium", "duration_needed_sec": dur, "text": "ice cold air",
            "selected_results": clips}


@needs_ffmpeg
def test_a_bad_clip_is_deleted_failed_and_forgotten_a_good_one_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(ci, "_vision", _logo_if_red)
    forgotten = []
    import core.clip_library as library
    monkeypatch.setattr(library, "forget_clip", lambda u: forgotten.append(u) or True)
    bad = {"url": "https://y/bad", "source": "pexels", "local_path": _video(tmp_path / "bad.mp4", "red")}
    ok = {"url": "https://y/ok", "source": "pexels", "local_path": _video(tmp_path / "ok.mp4", "blue")}
    errors = []
    st = ci.inspect_clips([_shot(1, [bad, ok])], "car AC", errors)
    assert st == {"checked": 2, "rejected": 1, "skipped": 0}
    assert bad["_dl_failed"] and "logo" in bad["_dl_error"] and not os.path.exists(bad["local_path"])
    assert forgotten == ["https://y/bad"]
    assert any("clip check" in e and "bad.mp4" in e for e in errors)
    assert ok["inspect"]["relevance"] == 0.85 and ok["_inspected"] and os.path.exists(ok["local_path"])
    assert "_dl_failed" not in ok


@needs_ffmpeg
def test_a_clip_is_only_inspected_once_and_library_cuts_are_exempt(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: seen.append(len(j)) or _logo_if_red(s, u, j))
    clip = {"url": "u1", "source": "pexels", "local_path": _video(tmp_path / "a.mp4", "blue")}
    seg = {"url": "u2", "source": "library", "library_segment_id": 5,
           "local_path": _video(tmp_path / "s.mp4", "red")}               # a human chose this cut
    shots = [_shot(1, [clip, seg])]
    assert ci.inspect_clips(shots)["checked"] == 1
    assert ci.inspect_clips(shots) == {"checked": 0, "rejected": 0, "skipped": 0}
    assert len(seen) == 1 and os.path.exists(seg["local_path"]) and "inspect" not in seg


@needs_ffmpeg
def test_it_looks_inside_a_partial_download_at_the_recorded_in_point(monkeypatch, tmp_path):
    grabbed = []
    real = ci._grab
    monkeypatch.setattr(ci, "_grab", lambda path, t, width=512: grabbed.append(t) or real(path, t, width))
    monkeypatch.setattr(ci, "_vision", _logo_if_red)
    clip = {"url": "u", "source": "youtube", "local_path": _video(tmp_path / "w.mp4", "blue", 14),
            "section_start": 100.0, "section_in_sec": 4.0, "section_end": 114.0}
    ci.inspect_clips([_shot(1, [clip], dur=5.0)])
    assert grabbed and min(grabbed) >= 4.0                      # in the window, not the handle before it


@needs_ffmpeg
def test_it_fails_open(monkeypatch, tmp_path):
    clip = {"url": "u", "source": "pexels", "local_path": _video(tmp_path / "a.mp4", "blue")}
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: (_ for _ in ()).throw(RuntimeError("model down")))
    monkeypatch.setattr(ci, "judge", lambda items, topic="": (_ for _ in ()).throw(RuntimeError("boom")))
    assert ci.inspect_clips([_shot(1, [clip])]) == {"checked": 0, "rejected": 0, "skipped": 0}
    assert os.path.exists(clip["local_path"]) and "_dl_failed" not in clip
    monkeypatch.setenv("ENABLE_CLIP_INSPECT", "false")
    assert ci.inspect_clips([_shot(1, [clip])])["checked"] == 0


# ── inside the download loop ─────────────────────────────────────────────────

@needs_ffmpeg
def test_download_and_repair_replaces_a_clip_the_inspector_rejects(monkeypatch, tmp_path):
    monkeypatch.setattr(ci, "_vision", _logo_if_red)
    shot = _shot(1, [{"url": "bad", "source": "youtube"}], dur=4.0)

    def fake_download(shots, project, quality="1080", progress=None, should_cancel=None, max_workers=None):
        for s in shots:
            for c in s.get("selected_results") or []:
                if not (c.get("local_path") and os.path.exists(c["local_path"])):
                    c["local_path"] = _video(tmp_path / f"{c['url']}.mp4", "red" if c["url"] == "bad" else "blue")
                    c["_dl_ok"] = True
                    c.pop("_dl_failed", None)
        return {"ok": 1, "failed": 0, "skipped": 0, "dir": str(tmp_path), "errors": []}

    def fake_repick(shots, slot_ids, groq_key=None, video_topic="", errors=None, blacklist=None):
        assert blacklist == {"bad"}                                   # the logo clip is never offered again
        for s in shots:
            if s["slot_id"] in set(slot_ids):
                s["selected_results"] = [{"url": "good", "source": "youtube"}]
        return 1

    monkeypatch.setattr(p, "download_selected_clips", fake_download)
    monkeypatch.setattr(p, "repick_failed_shots", fake_repick)
    import core.clip_library as library
    monkeypatch.setattr(library, "forget_clip", lambda u: True)

    report = p.download_and_repair([shot], "proj", rounds=2)
    assert [c["url"] for c in shot["selected_results"]] == ["good"]
    assert report["repaired"] == 1 and report["dropped"] == 0 and report["ok"] == 1
    assert report["inspected"] == 2 and report["inspect_rejected"] == 1
    assert not os.path.exists(tmp_path / "bad.mp4")
    assert any("clip check" in e for e in report["errors"])


# ── what the rest of the system does with it ─────────────────────────────────

def test_weak_relevance_lowers_a_clips_confidence_and_strong_raises_it():
    import core.confidence as conf
    base = {"url": "u", "source": "pexels"}
    plain = conf.clip_fit({}, dict(base))[0]
    weak, why = conf.clip_fit({}, dict(base, inspect={"relevance": 0.42}))
    strong = conf.clip_fit({}, dict(base, inspect={"relevance": 0.95}))[0]
    assert weak < plain < strong
    assert any("relevance 0.42" in r for r in why)


def test_the_bot_has_a_clip_check_setting_on_by_default():
    from bot import settings
    assert settings.DEFAULTS["clip_check"] is True
    assert any(o["key"] == "clip_check" and o["env"] == "ENABLE_CLIP_INSPECT" for o in settings.OPTIONS)


def test_forget_clip_removes_a_row(monkeypatch, tmp_path):
    import core.clip_library as lib
    monkeypatch.setattr(lib, "_db_path", lambda: str(tmp_path / "lib.db"))
    monkeypatch.setattr(lib, "_embed", lambda text: __import__("numpy").zeros(4, dtype="float32"))
    assert lib.store_clip("ice cold air", {"url": "https://y/bad", "source": "youtube"})
    assert lib.forget_clip("https://y/bad") is True
    assert lib.forget_clip("https://y/bad") is False and lib.forget_clip("") is False

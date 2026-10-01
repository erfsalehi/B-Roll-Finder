"""Motion cards end to end, minus the network and Chrome: the XML card track, the
render step (Remotion faked at the subprocess seam), the plan-with-pictures step,
the review-message line and cards.txt, and the pipeline's finalize hook."""

import json
import os
import re
import xml.dom.minidom as minidom

import pytest
from PIL import Image

import core.card_images as ci
import core.motion_cards as mc
import core.output as output
import core.pipeline as pl
import core.scene_plan as sp
from core.output import evaluate_fcpxml, generate_fcpxml


# ── helpers ──────────────────────────────────────────────────────────────────

def _file(path, data=b"x" * 64):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def _jpg(path, size=(1000, 724), color=(120, 40, 40)):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    Image.new("RGB", size, color).save(path, "JPEG")
    return str(path)


def _beat(slot=5, start=10.0, end=14.0, trigger="step", label="CLEAN EXTERIOR", left=None, **kw):
    b = {"id": f"b{slot:03d}", "template": "split_card", "trigger": trigger, "slot_ids": [slot],
         "start": start, "end": end, "enabled": True, "label": label,
         "step": {"number": 1, "kind": "METHOD", "name": label} if trigger == "step" else None,
         "right": {"kind": "number", "text": "1", "sub": "METHOD"} if trigger == "step" else {"kind": "image"},
         "confidence": {"footage": 0.31, "tier": "low"}, "reasons": ["ranked #7 of 8"],
         "assets": {"left": {"kind": "google", "path": left, "fit": 0.8}} if left else {}}
    b.update(kw)
    return b


@pytest.fixture
def media(monkeypatch):
    """ffprobe replaced: a media file is as long as the slot asked of it."""
    monkeypatch.setattr(output, "_get_media_duration", lambda p, fallback_duration=3600.0: fallback_duration)


def _shots():
    return [{"slot_id": i, "timestamp": float(i * 5), "end_timestamp": float(i * 5 + 5),
             "selected_results": [{"url": f"https://x/{i}.mp4", "matched_query": "q"}]}
            for i in range(0, 6)]


# ── the XML track ────────────────────────────────────────────────────────────

def test_cards_are_the_top_video_track_and_validate(tmp_path, media):
    card = _file(tmp_path / "cards" / "c1.mp4")
    ov = _file(tmp_path / "overlays" / "o1.mov")
    xml = generate_fcpxml(_shots(), "demo",
                          overlays=[{"filepath": ov, "start_sec": 2.0, "end_sec": 5.0, "is_video": True}],
                          cards=[{"filepath": card, "start_sec": 10.0, "end_sec": 14.0,
                                  "name": "CARD shot 2 - CLEAN EXTERIOR",
                                  "comment": "step · picture: google"}],
                          xml_dir=str(tmp_path))
    minidom.parseString(xml)
    rep = evaluate_fcpxml(xml, xml_dir=str(tmp_path), check_media=True)
    assert rep["ok"], rep["errors"]
    tracks = re.findall(r"<track>", xml.split("<audio>")[0])
    assert len(tracks) == 3                                    # footage, overlays, cards
    cards_part = xml.split("<track>")[3]
    assert "CARD shot 2 - CLEAN EXTERIOR" in cards_part and "<name>Motion card</name>" in cards_part
    assert "step · picture: google" in cards_part
    assert "<alphatype>" not in cards_part                     # opaque, no alpha handling
    assert "o1.mov" not in cards_part and "c1.mp4" in cards_part
    # a 4 s card at 23.976: 10 s → frame 240, 14 s → 336
    assert "<start>240</start>" in cards_part and "<end>336</end>" in cards_part


def test_cards_without_text_overlays_still_sit_on_v3(tmp_path, media):
    card = _file(tmp_path / "c.mp4")
    xml = generate_fcpxml(_shots(), "demo", cards=[{"filepath": card, "start_sec": 3.0, "end_sec": 6.0}],
                          xml_dir=str(tmp_path))
    assert len(re.findall(r"<track>", xml.split("<audio>")[0])) == 3
    assert evaluate_fcpxml(xml, xml_dir=str(tmp_path))["ok"]


def test_no_cards_leaves_the_xml_as_it_was(tmp_path, media):
    a = generate_fcpxml(_shots(), "demo", xml_dir=str(tmp_path))
    b = generate_fcpxml(_shots(), "demo", cards=None, xml_dir=str(tmp_path))
    assert a == b and len(re.findall(r"<track>", a.split("<audio>")[0])) == 1


def test_missing_card_files_and_early_cards_are_skipped(tmp_path, media):
    real = _file(tmp_path / "c.mp4")
    xml = generate_fcpxml(_shots(), "demo", time_offset=8.0, xml_dir=str(tmp_path), cards=[
        {"filepath": str(tmp_path / "gone.mp4"), "start_sec": 12.0, "end_sec": 15.0},
        {"filepath": real, "start_sec": 5.0, "end_sec": 8.0},           # before the chunk starts
        {"filepath": real, "start_sec": 12.0, "end_sec": 15.0}])        # kept, shifted by -8 s
    assert xml.count("<clipitem id=\"clip-card-") == 1
    assert evaluate_fcpxml(xml, xml_dir=str(tmp_path))["ok"]


def test_a_card_cannot_outrun_its_media(tmp_path, monkeypatch):
    monkeypatch.setattr(output, "_get_media_duration", lambda p, fallback_duration=3600.0: 2.0)
    card = _file(tmp_path / "c.mp4")
    xml = generate_fcpxml(_shots(), "demo", cards=[{"filepath": card, "start_sec": 10.0, "end_sec": 16.0}],
                          xml_dir=str(tmp_path))
    part = xml.split("<track>")[3]
    start, end = (int(re.search(rf"<{t}>(\d+)</{t}>", part).group(1)) for t in ("start", "end"))
    assert end - start <= 50                                    # ~2 s, not 6
    assert evaluate_fcpxml(xml, xml_dir=str(tmp_path))["ok"]


def test_overlapping_cards_would_be_rejected_by_the_validator(tmp_path, media):
    """The planner removes overlap; this pins that the validator really would catch it."""
    card = _file(tmp_path / "c.mp4")
    xml = generate_fcpxml(_shots(), "demo", xml_dir=str(tmp_path), cards=[
        {"filepath": card, "start_sec": 10.0, "end_sec": 14.0},
        {"filepath": card, "start_sec": 12.0, "end_sec": 16.0}])
    assert not evaluate_fcpxml(xml, xml_dir=str(tmp_path))["ok"]


# ── remotion props + cache key ───────────────────────────────────────────────

def test_props_inline_pictures_but_the_cache_key_uses_their_content(tmp_path):
    left = _jpg(tmp_path / "l.jpg")
    right = _jpg(tmp_path / "r.jpg", color=(10, 90, 30))
    b = _beat(left=left, right={"kind": "image"})
    b["assets"]["right"] = {"kind": "google", "path": right}
    props = mc.remotion_props(b)
    assert props["image"].startswith("data:image/jpeg;base64,")
    assert props["right"]["kind"] == "image" and props["right"]["image"].startswith("data:")
    assert props["durationSec"] == pytest.approx(4.0, abs=0.04) and props["fps"] == 30
    key = mc.cache_key(b)
    assert mc.cache_key(dict(b)) == key
    assert mc.cache_key(_beat(left=left, right={"kind": "image"}, label="OTHER",
                              assets=b["assets"])) != key
    _jpg(tmp_path / "l.jpg", color=(1, 2, 3))                  # same path, new picture
    assert mc.cache_key(b) != key


def test_text_card_props_have_no_picture(tmp_path):
    b = _beat()
    props = mc.remotion_props(b)
    assert props["image"] is None and props["right"] == {"kind": "number", "text": "1", "sub": "METHOD"}
    assert mc.renderable(b)                                    # a step with no picture is a text card
    assert not mc.renderable(_beat(trigger="low_confidence"))  # a plain beat needs its picture


# ── rendering ────────────────────────────────────────────────────────────────

class FakeRemotion:
    def __init__(self, tmp_path, monkeypatch, fail=False):
        self.calls = []
        self.fail = fail
        import core.overlays_remotion as ov
        monkeypatch.setattr(ov, "_remotion_bin", lambda: "remotion-fake")
        monkeypatch.setattr(mc, "_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(mc.subprocess, "run", self._run)

    def _run(self, cmd, **kw):
        self.calls.append(cmd)
        if not self.fail:
            with open(cmd[4], "wb") as f:
                f.write(b"mp4")
        return type("R", (), {"returncode": 1 if self.fail else 0, "stderr": "boom", "stdout": ""})()


def test_render_plan_renders_caches_and_returns_xml_entries(tmp_path, monkeypatch):
    fake = FakeRemotion(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    left = _jpg(tmp_path / "l.jpg")
    beats = [_beat(slot=8, start=30.0, end=34.0, left=left, trigger="low_confidence"),
             _beat(slot=2, start=10.0, end=14.0),                                   # text card
             _beat(slot=4, start=20.0, end=24.0, left=left, enabled=False),         # off
             _beat(slot=6, start=25.0, end=28.0, trigger="low_confidence")]         # no picture
    seen = []
    entries = mc.render_plan(beats, "demo", progress=lambda d, t: seen.append((d, t)))
    assert [e["slot_ids"] for e in entries] == [[2], [8]]                # time order, only renderable
    cmd = fake.calls[0]
    assert "ImageCard" in cmd and "--config=remotion.card.config.ts" in cmd
    assert all(os.path.exists(e["filepath"]) for e in entries)
    assert entries[0]["name"].startswith("CARD shot 2") and "step" in entries[0]["comment"]
    assert len(fake.calls) == 2 and seen[-1] == (2, 2)
    # the same plan again is served from the cache, with no new render
    mc.render_plan(beats, "demo")
    assert len(fake.calls) == 2


def test_a_failed_render_skips_that_card_and_says_so(tmp_path, monkeypatch):
    FakeRemotion(tmp_path, monkeypatch, fail=True)
    monkeypatch.chdir(tmp_path)
    errors = []
    assert mc.render_plan([_beat()], "demo", errors=errors) == []
    assert errors and "couldn't render the card for shot 5" in errors[0]


def test_no_remotion_is_reported_not_raised(tmp_path, monkeypatch):
    import core.overlays_remotion as ov
    monkeypatch.setattr(ov, "_remotion_bin", lambda: None)
    monkeypatch.chdir(tmp_path)
    errors = []
    assert mc.render_plan([_beat()], "demo", errors=errors) == []
    assert "Remotion isn't installed" in errors[0]


def test_browser_override_is_passed_to_remotion(tmp_path, monkeypatch):
    fake = FakeRemotion(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REMOTION_BROWSER_EXECUTABLE", "C:/chrome.exe")
    mc.render_plan([_beat()], "demo")
    assert "--browser-executable=C:/chrome.exe" in fake.calls[0]


# ── plan + pictures ──────────────────────────────────────────────────────────

def _sourced(beat, shot, pdir, topic, errors):
    """What card_images.source_images would return for a good generic photo."""
    p = _jpg(os.path.join(pdir, "images", "cards", f"{beat['id']}-left.jpg"))
    return {"left": {"kind": "google", "path": p, "fit": 0.82}, "right": None, "ai": False, "fit": 0.82}


def _plan(monkeypatch, beats, sourcer=_sourced):
    monkeypatch.setattr(sp, "plan_scene", lambda *a, **k: beats)
    monkeypatch.setattr(ci, "source_images", sourcer)


def test_build_plan_sources_a_picture_for_each_beat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    beats = [_beat(slot=5), _beat(slot=9, trigger="low_confidence", start=40.0, end=45.0)]
    _plan(monkeypatch, beats)
    diag = {}
    out = mc.build_plan([{"slot_id": 5}, {"slot_id": 9}], "demo", "topic", None, diag=diag)
    assert all(b["assets"]["left"]["fit"] == 0.82 and b["sourced"] for b in out)
    assert out[0]["confidence"]["image"] == 0.82 and diag["sourced"] == 2


def test_pictures_are_not_sourced_twice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    _plan(monkeypatch, [_beat()], lambda *a, **k: calls.append(1) or _sourced(*a, **k))
    beats = mc.build_plan([{"slot_id": 5}], "demo")
    mc.build_plan([{"slot_id": 5}], "demo")                    # sp.plan_scene returns the same dicts
    assert len(calls) == 1 and beats[0]["sourced"]


def test_a_list_step_without_a_picture_becomes_a_text_card(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    none = lambda *a, **k: {"left": None, "right": None, "ai": False, "fit": None}
    b = _beat(right={"kind": "image"})
    _plan(monkeypatch, [b], none)
    out = mc.build_plan([{"slot_id": 5}], "demo")[0]
    assert out["enabled"] and out["right"] == {"kind": "number", "text": "1", "sub": "METHOD"}
    assert mc.renderable(out) and any("text card" in r for r in out["reasons"])


def test_a_weak_footage_card_without_a_picture_keeps_the_footage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    none = lambda *a, **k: {"left": None, "right": None, "ai": False, "fit": None}
    _plan(monkeypatch, [_beat(trigger="low_confidence")], none)
    out = mc.build_plan([{"slot_id": 5}], "demo")[0]
    assert not out["enabled"] and any("footage kept" in r for r in out["reasons"])


def test_a_card_never_replaces_better_footage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def meh(beat, shot, pdir, topic, errors):
        r = _sourced(beat, shot, pdir, topic, errors)
        r["left"]["fit"] = r["fit"] = 0.20                      # worse than the footage's 0.31
        return r
    _plan(monkeypatch, [_beat(trigger="low_confidence", right={"kind": "stat", "text": "5%"})], meh)
    out = mc.build_plan([{"slot_id": 5}], "demo")[0]
    assert not out["enabled"] and any("below the footage" in r for r in out["reasons"])
    # …but a list step is structural and always stays
    _plan(monkeypatch, [_beat(trigger="step")], meh)
    assert mc.build_plan([{"slot_id": 5}], "demo")[0]["enabled"]


def test_a_missing_second_picture_falls_back_so_the_right_half_is_never_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _plan(monkeypatch, [_beat(right={"kind": "image"})])           # a step, sourcer returns no right
    out = mc.build_plan([{"slot_id": 5}], "demo")[0]
    assert out["enabled"] and out["right"]["kind"] == "number"
    _plan(monkeypatch, [_beat(trigger="low_confidence")])          # not a step: nothing to put there
    out = mc.build_plan([{"slot_id": 5}], "demo")[0]
    assert not out["enabled"] and any("second picture" in r for r in out["reasons"])


def test_a_crashing_sourcer_only_drops_its_own_card(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def boom(beat, shot, pdir, topic, errors):
        if beat["id"] == "b009":
            raise RuntimeError("vision exploded")
        return _sourced(beat, shot, pdir, topic, errors)
    _plan(monkeypatch, [_beat(slot=5), _beat(slot=9, trigger="low_confidence", start=40.0, end=44.0)], boom)
    errors = []
    out = mc.build_plan([{"slot_id": 5}, {"slot_id": 9}], "demo", errors=errors)
    assert out[0]["enabled"] and not out[1]["enabled"]
    assert any("pictures for shot 9 failed" in e for e in errors)


def test_pictures_deleted_with_the_project_are_found_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    _plan(monkeypatch, [_beat()], lambda *a, **k: calls.append(1) or _sourced(*a, **k))
    beats = mc.build_plan([{"slot_id": 5}], "demo")
    os.remove(beats[0]["assets"]["left"]["path"])
    mc.build_plan([{"slot_id": 5}], "demo")
    assert len(calls) == 2


# ── reporting ────────────────────────────────────────────────────────────────

def test_review_line_and_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    beats = [_beat(slot=2), _beat(slot=7, trigger="low_confidence", start=30.0, end=35.0,
                                  left=_jpg(tmp_path / "a.jpg")),
             _beat(slot=9, trigger="low_confidence", start=50.0, end=54.0, enabled=False)]
    beats[1]["assets"]["left"] = {"kind": "ai", "path": beats[1]["assets"]["left"]["path"],
                                  "model": "microsoft/mai-image-2.6-flash", "prompt": "a wheel", "fit": 0.7}
    line = mc.format_line(beats, {"ai_usd": 0.0147})
    assert line.startswith("🃏 Motion cards: 2 (1 list step(s) · 1 low-confidence)")
    assert "1 AI image(s) ≈ $0.01" in line and "1 skipped" in line
    assert mc.format_line([], {}) is None
    shots = [{"slot_id": 7, "timestamp": 30.0, "confidence": {"score": 0.31, "tier": "low",
                                                              "reasons": ["QA flagged (high) -0.35"]}},
             {"slot_id": 1, "timestamp": 0.0, "confidence": {"score": None, "tier": "skip", "reasons": []}}]
    path = mc.write_manifest(beats, shots, "demo")
    text = open(path, encoding="utf-8").read()
    assert "[on] shot 2" in text and "[OFF] shot 9" in text
    assert "generated by microsoft/mai-image-2.6-flash — prompt: a wheel" in text
    assert "   7    30.0   0.31  low" in text and "QA flagged (high)" in text
    assert "   1" not in text.split("FOOTAGE CONFIDENCE")[1]            # skipped shots aren't listed


def test_card_cache_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_CACHE_DIR", str(tmp_path / "cc"))
    assert mc.card_cache_size() == 0 and mc.clear_card_cache() == (0, 0)
    _file(tmp_path / "cc" / "a.mp4", b"x" * 100)
    assert mc.card_cache_size() == 100 and mc.clear_card_cache() == (1, 100)


# ── the pipeline hooks ───────────────────────────────────────────────────────

def test_plan_motion_cards_is_best_effort(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOTION_CARDS_MODE", "auto")
    errors, attempts = [], {}

    def boom(*a, **k):
        raise RuntimeError("llm down")
    monkeypatch.setattr(mc, "build_plan", boom)
    prev = [_beat()]
    assert pl.plan_motion_cards([], "demo", existing=prev, errors=errors) == prev    # keeps the old plan
    assert "planning failed" in errors[0]
    monkeypatch.setattr(mc, "build_plan", lambda *a, **k: ([_beat()], k["diag"].update(sourced=1))[0])
    out = pl.plan_motion_cards([], "demo", attempts=attempts)
    assert len(out) == 1 and attempts["cards"]["sourced"] == 1 and attempts["cards"]["total"] == 1
    monkeypatch.setenv("MOTION_CARDS_MODE", "off")
    assert pl.plan_motion_cards([], "demo") == []


def test_finalize_replans_after_the_download_and_renders_into_the_xml(tmp_path, monkeypatch, media):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOTION_CARDS_MODE", "auto")
    order, written = [], {}
    monkeypatch.setattr(pl, "enforce_timeline", lambda *a, **k: order.append("enforce") or {"ok": True})
    monkeypatch.setattr(pl, "ensure_youtube_coverage", lambda *a, **k: 0)
    monkeypatch.setattr(pl, "download_and_repair", lambda *a, **k: order.append("download") or {"ok": 1})
    card_file = _file(tmp_path / "cards" / "c.mp4")

    def fake_plan(shots, proj, topic, key, qa=None, existing=None, **k):
        order.append(("plan", len(existing or []), qa))
        return [_beat()]

    def fake_render(beats, shots, proj, **k):
        order.append("render")
        return [{"filepath": card_file, "start_sec": 10.0, "end_sec": 14.0, "name": "CARD"}]

    def fake_write(shots, proj, overlays=None, sfx_list=None, cards=None):
        order.append("write")
        written["cards"] = cards
        return "x.xml"
    monkeypatch.setattr(pl, "plan_motion_cards", fake_plan)
    monkeypatch.setattr(pl, "render_motion_cards", fake_render)
    monkeypatch.setattr(pl, "write_fcpxml", fake_write)
    fin = pl.finalize_project(_shots(), "demo", scene_plan=[_beat(), _beat(slot=7)], qa={"issues": []})
    assert order == ["enforce", "download", ("plan", 2, {"issues": []}), "render", "write"]
    assert fin["cards"][0]["name"] == "CARD" and written["cards"] == fin["cards"]
    assert fin["xml_path"] == "x.xml" and len(fin["scene_plan"]) == 1


def test_finalize_with_cards_off_never_touches_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOTION_CARDS_MODE", "off")
    monkeypatch.setattr(pl, "enforce_timeline", lambda *a, **k: {})
    monkeypatch.setattr(pl, "ensure_youtube_coverage", lambda *a, **k: 0)
    monkeypatch.setattr(pl, "download_and_repair", lambda *a, **k: {})
    monkeypatch.setattr(pl, "plan_motion_cards", lambda *a, **k: pytest.fail("planned"))
    monkeypatch.setattr(pl, "write_fcpxml", lambda *a, **k: "x.xml")
    fin = pl.finalize_project(_shots(), "demo")
    assert fin["cards"] == [] and fin["scene_plan"] == []


def test_the_plan_travels_in_the_result_and_the_snapshot():
    st = pl.PipelineState(scene_plan=[_beat()], cards=[{"filepath": "x"}])
    res = st.to_result()
    assert res["scene_plan"][0]["label"] == "CLEAN EXTERIOR" and res["cards"][0]["filepath"] == "x"
    import core.project_store as ps
    assert "scene_plan" in ps._SNAPSHOT_KEYS
    json.dumps(res["scene_plan"])                                    # plain JSON for the pending store

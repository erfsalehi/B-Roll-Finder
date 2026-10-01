"""Card pictures (core.card_images): cropping around the detail, judging candidates
with a (faked) vision model, and the real-photo-first / generate-if-generic /
never-generate-a-named-thing ordering."""

import os

import numpy as np
import pytest
from PIL import Image

import core.ai_images as ai
import core.card_images as ci


def _photo(path, size=(1000, 700), busy=None, seed=0):
    """A flat grey picture, optionally with a noisy 'busy' box (x0, y0, x1, y1)."""
    rng = np.random.default_rng(seed)
    arr = np.full((size[1], size[0], 3), 90, dtype=np.uint8)
    if busy:
        x0, y0, x1, y1 = busy
        arr[y0:y1, x0:x1] = rng.integers(0, 255, (y1 - y0, x1 - x0, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, "JPEG", quality=95)
    return str(path)


# ── cropping ─────────────────────────────────────────────────────────────────

def test_wide_picture_is_cropped_toward_the_detail(tmp_path):
    im = Image.open(_photo(tmp_path / "a.jpg", (1600, 700), busy=(1300, 100, 1580, 600)))
    x0, y0, x1, y1 = ci.crop_box(im)
    assert (y0, y1) == (0, 700)                              # full height kept
    assert abs((x1 - x0) / (y1 - y0) - ci.RATIO) < 0.01
    assert x0 > 500                                          # slid right, toward the busy area


def test_tall_picture_prefers_the_top(tmp_path):
    im = Image.open(_photo(tmp_path / "t.jpg", (800, 1400)))        # nothing to see anywhere
    x0, y0, x1, y1 = ci.crop_box(im)
    assert (x0, x1) == (0, 800) and y0 <= 1400 - (y1 - y0) / 2      # not pinned to the bottom


def test_prepare_outputs_a_card_shaped_jpeg(tmp_path):
    out = ci.prepare(_photo(tmp_path / "a.jpg", (1600, 900)), str(tmp_path / "o.jpg"))
    assert out["mode"] == "crop" and out["width"] == ci.OUT_W
    assert abs(out["width"] / out["height"] - ci.RATIO) < 0.01
    assert Image.open(out["path"]).size == (out["width"], out["height"])


def test_prepare_rejects_small_and_unreadable_files(tmp_path):
    assert ci.prepare(_photo(tmp_path / "s.jpg", (300, 200)), str(tmp_path / "o.jpg")) is None
    bad = tmp_path / "x.jpg"
    bad.write_bytes(b"not an image")
    assert ci.prepare(str(bad), str(tmp_path / "o.jpg")) is None
    assert ci.prepare(str(tmp_path / "missing.jpg"), str(tmp_path / "o.jpg")) is None


def test_extreme_shapes_are_shown_whole_over_a_blur(tmp_path):
    out = ci.prepare(_photo(tmp_path / "pano.jpg", (2400, 600)), str(tmp_path / "o.jpg"))
    assert out["mode"] == "fit"
    assert abs(out["width"] / out["height"] - ci.RATIO) < 0.01
    out2 = ci.prepare(_photo(tmp_path / "tall.jpg", (600, 1600)), str(tmp_path / "o2.jpg"))
    assert out2["mode"] == "fit"


def test_detail_is_a_closer_crop_and_refuses_soft_sources(tmp_path):
    prep = ci.prepare(_photo(tmp_path / "a.jpg", (1600, 900), busy=(200, 100, 600, 500)),
                      str(tmp_path / "p.jpg"))
    d = ci.detail(prep["path"], str(tmp_path / "d.jpg"))
    assert d and d["width"] == ci.OUT_W and d["mode"] == "detail"
    small = ci.prepare(_photo(tmp_path / "b.jpg", (700, 480)), str(tmp_path / "q.jpg"))
    assert ci.detail(small["path"], str(tmp_path / "e.jpg")) is None


# ── judging ──────────────────────────────────────────────────────────────────

def _cands(tmp_path, n=3):
    return [{"local_path": _photo(tmp_path / f"c{i}.jpg", (900, 640), seed=i),
             "url": f"https://x/{i}.jpg", "page": f"https://page/{i}", "title": f"t{i}"}
            for i in range(n)]


def test_rate_maps_verdicts_and_applies_issue_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [
        {"id": 1, "fit": 0.9, "issues": [], "note": "wheel"},
        {"id": 2, "fit": 0.9, "issues": ["watermark"]},
        {"id": 3, "fit": 0.7, "issues": ["logo_heavy"]}]})
    v = ci.rate(_cands(tmp_path), {"topic": "t", "narration": "n"})
    assert v[0]["usable"] and v[0]["fit"] == 0.9 and v[0]["rated"]
    assert not v[1]["usable"]                      # a watermark is a hard reject
    assert v[2]["fit"] == pytest.approx(0.55) and not v[2]["usable"]     # 0.7 - 0.15 < 0.6


def test_retail_banners_and_multipacks_are_hard_rejects(tmp_path, monkeypatch):
    # live run: a Gunk six-pack listing banner (blue header bar, product pushed to one side)
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [
        {"id": 1, "fit": 0.9, "issues": ["ad_layout"]},
        {"id": 2, "fit": 0.9, "issues": ["collage"]},
        {"id": 3, "fit": 0.9, "issues": []}]})
    v = ci.rate(_cands(tmp_path), {})
    assert [x["usable"] for x in v] == [False, False, True]
    assert "ad_layout" in ci._RATE_SYSTEM and "multi-pack" in ci._RATE_SYSTEM    # the model is told what they are


def test_rate_without_a_vision_verdict_is_unrated(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: None)
    v = ci.rate(_cands(tmp_path, 2), {})
    assert [x["rated"] for x in v] == [False, False] and all(x["fit"] is None for x in v)


def test_rate_flags_unreadable_files_and_skips_them_in_the_call(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: seen.setdefault("n", len(j)) and
                        {"images": [{"id": 1, "fit": 0.8, "issues": []}]})
    cands = _cands(tmp_path, 1) + [{"local_path": str(tmp_path / "gone.jpg")}]
    v = ci.rate(cands, {})
    assert seen["n"] == 1 and v[0]["usable"] and v[1]["issues"] == ["unreadable"]


# ── choosing the picture ─────────────────────────────────────────────────────

def _beat(**kw):
    return dict({"id": "b05", "label": "CLEAN EXTERIOR", "image_query": "wheel rim wipe",
                 "image_prompt": "a gloved hand wiping an alloy wheel", "subject_kind": "generic",
                 "image_style": "photo", "right": {"kind": "number", "text": "1"}}, **kw)


def _shot(tmp_path, n=3):
    return {"slot_id": 5, "text": "First, clean the exterior.", "images": _cands(tmp_path, n),
            "image_query": "wheel rim wipe"}


def _no_ai(monkeypatch):
    calls = []
    monkeypatch.setattr(ai, "enabled", lambda: True)
    monkeypatch.setattr(ai, "generate", lambda *a, **k: calls.append(a) or None)
    return calls


def test_picks_the_best_google_image_and_a_second_for_the_right_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [
        {"id": 1, "fit": 0.65, "issues": []}, {"id": 2, "fit": 0.9, "issues": []},
        {"id": 3, "fit": 0.8, "issues": []}]})
    calls = _no_ai(monkeypatch)
    beat = _beat(right={"kind": "image"})
    out = ci.source_images(beat, _shot(tmp_path), str(tmp_path / "proj"))
    assert out["left"]["kind"] == "google" and out["left"]["fit"] == 0.9
    assert out["left"]["page"] == "https://page/1"                   # image #2 (index 1)
    assert out["right"]["kind"] == "google" and out["right"]["fit"] == 0.8
    assert os.path.exists(out["left"]["path"]) and not calls and not out["ai"]


def test_no_second_photo_falls_back_to_a_detail_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [{"id": 1, "fit": 0.9, "issues": []}]})
    _no_ai(monkeypatch)
    shot = _shot(tmp_path, 1)
    shot["images"][0]["local_path"] = _photo(tmp_path / "big.jpg", (1800, 1200), busy=(100, 100, 700, 600))
    out = ci.source_images(_beat(right={"kind": "image"}), shot, str(tmp_path / "proj"))
    assert out["right"]["kind"] == "detail"


def test_generic_subject_with_no_usable_photo_is_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [
        {"id": i, "fit": 0.2, "issues": ["wrong_subject"]} for i in (1, 2, 3)]})
    gen_calls = []

    def fake_generate(prompt, out_path, **kw):
        gen_calls.append((prompt, kw))
        _photo(out_path, (1024, 768))
        return {"path": out_path, "model": "microsoft/mai-image-2.6-flash", "cost": 0.0147,
                "width": 1024, "height": 768, "prompt": prompt, "cached": False}

    monkeypatch.setattr(ai, "enabled", lambda: True)
    monkeypatch.setattr(ai, "generate", fake_generate)
    out = ci.source_images(_beat(), _shot(tmp_path), str(tmp_path / "proj"), allow_search=False)
    assert out["ai"] and out["left"]["kind"] == "ai" and out["left"]["cost"] == 0.0147
    assert gen_calls[0][0] == "a gloved hand wiping an alloy wheel"
    assert gen_calls[0][1]["style"] == "photo"


def test_a_named_subject_is_never_generated(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [
        {"id": i, "fit": 0.2, "issues": ["wrong_subject"]} for i in (1, 2, 3)]})
    calls = _no_ai(monkeypatch)
    out = ci.source_images(_beat(subject_kind="identifiable"), _shot(tmp_path),
                           str(tmp_path / "proj"), allow_search=False)
    assert out["left"] is None and not calls


def test_ai_disabled_leaves_the_card_without_a_picture(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [
        {"id": i, "fit": 0.1, "issues": []} for i in (1, 2, 3)]})
    monkeypatch.setattr(ai, "enabled", lambda: False)
    out = ci.source_images(_beat(), _shot(tmp_path), str(tmp_path / "proj"), allow_search=False)
    assert out["left"] is None


def test_without_vision_the_top_result_is_accepted_unrated(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: None)
    calls = _no_ai(monkeypatch)
    out = ci.source_images(_beat(), _shot(tmp_path), str(tmp_path / "proj"))
    assert out["left"]["kind"] == "google" and out["left"]["rated"] is False and not calls


def test_library_stills_keep_their_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [{"id": 1, "fit": 0.9, "issues": []}]})
    shot = _shot(tmp_path, 1)
    shot["images"][0]["library_segment_id"] = 4
    out = ci.source_images(_beat(), shot, str(tmp_path / "proj"))
    assert out["left"]["kind"] == "library"


def test_searches_again_with_the_cards_own_query_when_shot_images_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, "_vision", lambda s, u, j: {"images": [
        {"id": 1, "fit": 0.1, "issues": []}, {"id": 2, "fit": 0.1, "issues": []},
        {"id": 3, "fit": 0.1, "issues": []}]} if len(j) == 3 else
        {"images": [{"id": 1, "fit": 0.85, "issues": []}]})
    fresh = [{"local_path": _photo(tmp_path / "fresh.jpg", (900, 640)), "url": "https://f/1.jpg",
              "page": "https://f/p", "title": "fresh"}]
    seen = []
    monkeypatch.setattr(ci, "_search_more", lambda q, d, t, e, num=4: seen.append(q) or fresh)
    _no_ai(monkeypatch)
    beat = _beat(image_query="alloy wheel being polished")
    out = ci.source_images(beat, _shot(tmp_path), str(tmp_path / "proj"))
    assert seen == ["alloy wheel being polished"]
    assert out["left"]["url"] == "https://f/1.jpg"

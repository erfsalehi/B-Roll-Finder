"""Extra contextual B-roll: brand/model/part keyword rules, YouTube-only fetch,
and placement after the timeline (named 'Extra - <keyword>')."""

import re
import xml.etree.ElementTree as ET

import core.extras as extras
from core.output import generate_fcpxml, generate_shots_srt


# ── keyword rules ─────────────────────────────────────────────────────────────

def test_build_extra_keywords_rules():
    ents = {"brands": ["Toyota"], "models": ["Camry"], "parts": ["brake pads"]}
    kws = [k["keyword"] for k in extras.build_extra_keywords(ents)]
    # brand -> company / logo / manufacturing line
    assert any("toyota" in k.lower() and "logo" in k.lower() for k in kws)
    assert any("manufacturing" in k.lower() for k in kws)
    # model -> pov driving / test drive / review
    assert any("camry" in k.lower() and "pov" in k.lower() for k in kws)
    assert any("camry" in k.lower() and "test drive" in k.lower() for k in kws)
    # part -> explained / how it works
    assert any("brake pads explained" == k.lower() for k in kws)
    assert any(k.lower().startswith("how brake pads") for k in kws)


def test_build_extra_keywords_caps_and_dedupes():
    ents = {"brands": ["Toyota", "Toyota"], "models": [], "parts": []}
    kws = extras.build_extra_keywords(ents, max_keywords=2)
    assert len(kws) == 2                       # capped
    assert len({k["keyword"].lower() for k in kws}) == 2   # deduped


# ── fetch -> extra shots ──────────────────────────────────────────────────────

def _yt(title):
    return {"url": f"https://youtu.be/{title}", "source": "youtube", "title": title,
            "is_short": False, "width": 1920, "height": 1080, "duration": 120}


def test_fetch_extra_shots_builds_placed_extras(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(extras, "extract_extra_entities",
                        lambda script, key: {"brands": ["Toyota"], "models": [], "parts": []})
    import core.director_search as ds
    monkeypatch.setattr(ds, "search_youtube_classic",
                        lambda kw, num_results=3, errors=None: [_yt(f"{kw}-{i}") for i in range(num_results)])

    shots = extras.fetch_extra_shots("script about Toyota", api_key="k",
                                     start_slot_id=50, start_sec=100.0, clip_sec=6.0)
    assert shots, "expected extra shots"
    # All tagged + labelled + YouTube + placed back-to-back from start_sec.
    assert all(s["is_extra"] and s["extra_label"].startswith("Extra - ") for s in shots)
    assert all(s["selected_results"][0]["source"] == "youtube" for s in shots)
    assert shots[0]["timestamp"] == 100.0
    assert shots[1]["timestamp"] == 106.0          # +clip_sec, contiguous
    # 3 brand keywords x 2 per keyword (default) = 6 clips.
    assert len(shots) == 6


# ── run_extras_only (the /extras bot path) ────────────────────────────────────

def test_run_extras_only_downloads_and_writes_xml(monkeypatch, tmp_path):
    """Extras-only run: transcribe → fetch_extra_shots → download → XML, counting
    only clips that landed a file on disk."""
    import os
    import core.pipeline as pipeline

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)

    monkeypatch.setattr("core.transcription.transcribe_audio",
                        lambda path, key: [{"start": 0.0, "end": 1.0, "text": "Toyota Camry brakes"}])

    extra = {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 6.0, "priority": "low",
             "is_extra": True, "extra_label": "Extra - toyota logo",
             "selected_results": [{"url": "https://youtu.be/x", "source": "youtube",
                                   "matched_query": "toyota logo"}]}
    monkeypatch.setattr("core.extras.fetch_extra_shots",
                        lambda *a, **k: [extra])

    def _fake_download(shots, project_name, **kw):
        # Pretend the clip downloaded: write a real file + point local_path at it.
        d = os.path.join("downloads", project_name, "director")
        os.makedirs(d, exist_ok=True)
        fp = os.path.join(d, "clip.mp4")
        with open(fp, "wb") as f:
            f.write(b"x" * 16)
        shots[0]["selected_results"][0]["local_path"] = fp
        return {"ok": 1, "failed": 0, "skipped": 0, "dir": d, "errors": []}

    monkeypatch.setattr(pipeline, "download_selected_clips", _fake_download)
    monkeypatch.setattr(pipeline, "write_fcpxml", lambda shots, proj, *a, **k: f"downloads/{proj}/{proj}.xml")

    res = pipeline.run_extras_only("audio.mp3", project_name="extras_test")
    assert res["n_clips"] == 1
    assert res["shots"] and res["shots"][0]["is_extra"]
    assert res["xml_path"].endswith("extras_test.xml")


def test_run_extras_only_no_entities_returns_zero(monkeypatch, tmp_path):
    """No named brands/models/parts → no clips, no XML, no crash."""
    import core.pipeline as pipeline
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr("core.transcription.transcribe_audio",
                        lambda path, key: [{"start": 0.0, "end": 1.0, "text": "generic talk"}])
    monkeypatch.setattr("core.extras.fetch_extra_shots", lambda *a, **k: [])
    res = pipeline.run_extras_only("audio.mp3", project_name="empty")
    assert res["n_clips"] == 0
    assert res["shots"] == []
    assert res["xml_path"] is None


# ── placement + naming + SRT ──────────────────────────────────────────────────

def _video_names_and_spans(xml: str):
    root = ET.fromstring(xml)
    track = next(root.iter("track"))
    out = []
    for ci in track.findall("clipitem"):
        s, e = ci.findtext("start"), ci.findtext("end")
        out.append((ci.findtext("name"), int(s), int(e)))
    return out


def test_extras_appended_after_timeline_with_label():
    shots = [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 5.0,
         "selected_results": [{"url": "https://x/a.mp4", "matched_query": "a"}]},
        {"slot_id": 2, "timestamp": 5.0, "end_timestamp": 11.0, "priority": "low",
         "is_extra": True, "extra_keyword": "toyota logo",
         "extra_label": "Extra - toyota logo",
         "selected_results": [{"url": "https://x/b.mp4", "matched_query": "toyota logo"}]},
    ]
    xml = generate_fcpxml(shots, project_name="demo")
    items = _video_names_and_spans(xml)
    assert any(n == "Extra - toyota logo" for n, _s, _e in items)
    # The extra sits after the narration clip (later start frame).
    narr = [it for it in items if it[0] != "Extra - toyota logo"][0]
    extra = [it for it in items if it[0] == "Extra - toyota logo"][0]
    assert extra[1] >= narr[2]


def test_srt_labels_extras():
    shots = [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 5.0, "selected_results": [{"url": "x"}]},
        {"slot_id": 2, "timestamp": 5.0, "end_timestamp": 11.0, "is_extra": True,
         "extra_keyword": "camry test drive", "extra_label": "Extra - camry test drive",
         "selected_results": [{"url": "y"}]},
    ]
    srt = generate_shots_srt(shots)
    assert "Shot 1" in srt
    assert "Extra - camry test drive" in srt

"""The whole learning loop on real files, with only the network mocked:

project delivered (real FCPXML) → editor's XML read back → labels → used cut
becomes a library segment (really cut with ffmpeg) → reviewer verifies it →
a NEW project's similar line is offered that segment → the downloader copies
it (no download) → the new XML starts it at 0 on the stored file.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from core import output, pipeline
from core import project_store as ps
from core import ratings
from core import segment_library as sl
from core.xml_reimport import parse_fcpxml

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def fake_embed(text):
    v = np.zeros(256, dtype=np.float32)
    for w in sl._tokens(text):
        v[sum(map(ord, w)) % 256] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


def _video(path, seconds):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc2=duration={seconds}:size=640x360:rate=30",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def test_learning_loop_end_to_end(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)                     # downloads/ lives in tmp
    monkeypatch.setenv("SEGMENT_LIBRARY_DIR", str(tmp_path / "lib"))
    monkeypatch.setattr(sl, "_embed", fake_embed)
    monkeypatch.setattr(sl, "tag_lines", lambda shots, topic="": None)
    monkeypatch.setattr(sl, "kick", lambda: None)
    monkeypatch.setattr(sl, "_draft_openrouter", lambda frames, lines, topic: {
        "description": "Close-up of a hand unscrewing the oil drain plug; oil pours out",
        "subject": "", "identifiable": False, "generic_use": "draining engine oil from a car",
        "shot_type": "close_up"})

    # ── 1. First project, delivered with a real FCPXML ──────────────────────
    proj = "Oil change basics"
    pid = ps.create_project(proj, proj, operator_id=1, operator_name="op")
    clip_dir = output.clip_base_dir(proj)
    os.makedirs(clip_dir)
    shots, files = [], []
    for i, (line, secs) in enumerate([("drain the old engine oil into a pan", 20),
                                      ("fit a brand new oil filter", 20)], 1):
        f = os.path.join(clip_dir, f"{i}-1-clip{i}.mp4")
        _video(f, secs)
        files.append(f)
        shots.append({"slot_id": i, "timestamp": (i - 1) * 5.0, "end_timestamp": i * 5.0,
                      "text": line, "duration_needed_sec": 5, "priority": "medium",
                      "selected_results": [{"url": f"https://yt/{i}", "page_url": f"https://yt/{i}",
                                            "source": "youtube", "title": f"clip {i}",
                                            "local_path": f, "_dl_ok": True}],
                      "video_results": []})
    xml_path = pipeline.write_fcpxml(shots, proj)
    ps.record_delivery(pid, shots, xml_path=xml_path)
    ratings.record_candidates(pid, shots)

    # ── 2. Editor's XML: kept clip 1 but started it 6s in; dropped clip 2 ───
    items = parse_fcpxml(xml_path)
    edited = []
    for it in items:
        if it["name"].startswith("1-1-"):
            it = dict(it, in_frame=int(6 * it["fps"]), in_seconds=6.0,
                      out_seconds=6.0 + (it["out_frame"] - it["in_frame"]) / it["fps"])
            edited.append(it)
    summary = ps.analyze_import(pid, edited, imported_by=1, filename="final.xml")
    assert summary["clips"] == {"used_here": 1, "unused": 1}

    # ── 3. The used cut becomes a library segment, really cut ───────────────
    assert sl.ingest_from_import(pid) == 1
    assert sl.process_pending() == 1
    seg = sl.get_segment(1)
    assert seg["file_status"] == "ready", seg["file_error"]
    assert seg["src_in"] == 6.0 and os.path.isfile(seg["file_path"])
    assert seg["draft_source"] == "vision"

    # ── 4. A reviewer verifies it ───────────────────────────────────────────
    sl.save_review(501, seg["id"], usable="yes", fits_line="yes", identifiable="0",
                   description="Close-up of a hand unscrewing the oil drain plug; oil pours out",
                   generic_use="drain the engine oil from a car")
    assert sl.get_segment(seg["id"])["trust"] == "verified"

    # ── 5. A new project's similar line gets it, copied not downloaded ──────
    proj2 = "Civic service"
    pid2 = ps.create_project(proj2, proj2)
    ps.create_project("filler 1")                   # push project 1 out of "recent"
    shot = {"slot_id": 1, "timestamp": 0, "end_timestamp": 4, "priority": "medium",
            "text": "first drain the engine oil from your Civic", "line_subjects": ["Honda Civic"],
            "duration_needed_sec": 4, "video_results": []}
    assert sl.inject_candidates([shot]) == 1
    cand = shot["video_results"][0]
    assert cand["library_segment_id"] == seg["id"]
    shot["selected_results"] = [cand]
    import core.direct_downloader as dd
    import core.youtube as yt
    for mod, fn in ((yt, "download_video"), (dd, "download_direct_video")):
        monkeypatch.setattr(mod, fn,
                            lambda *a, **k: pytest.fail("a library clip must not be downloaded"))
    dl = pipeline.download_and_repair([shot], proj2, rounds=0)
    assert dl.get("failed", 0) == 0
    assert os.path.isfile(cand["local_path"])

    # ── 6. …and the new XML starts it at 0 on the stored file ───────────────
    xml2 = pipeline.write_fcpxml([shot], proj2)
    placed = [i for i in parse_fcpxml(xml2) if i["name"].endswith(".mp4")]
    assert len(placed) == 1 and placed[0]["in_frame"] == 0
    assert cand["in_rule"] == "segment"

    # ── 7. Delivery records the use (so it rests next time) ─────────────────
    assert sl.mark_used(pid2, [shot]) == 1
    assert sl.get_segment(seg["id"])["times_used"] == 1


def test_library_segments_survive_the_shorts_filter():
    seg = {"url": "https://yt/1", "source": "youtube", "duration": 6.0, "library_segment_id": 3}
    plain = {"url": "https://yt/2", "source": "youtube", "duration": 6.0}
    shot = {"video_results": [seg, plain]}
    assert pipeline.drop_shorts([shot]) == 1
    assert shot["video_results"] == [seg]
    assert pipeline.validate_timeline([{"slot_id": 1, "selected_results": [seg]}])["ok"]


def test_image_loop_end_to_end(monkeypatch, tmp_path):
    """Per-shot image the editor put in the edit → library still (stored,
    described, reviewed) → offered first in a new project's images folder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEGMENT_LIBRARY_DIR", str(tmp_path / "lib"))
    monkeypatch.setattr(sl, "_embed", fake_embed)
    monkeypatch.setattr(sl, "tag_lines", lambda shots, topic="": None)
    monkeypatch.setattr(sl, "kick", lambda: None)
    seen = {}

    def draft(frames, lines, topic):
        seen["frames"] = frames
        return {"description": "Diagram of a car engine oil filter cut in half, labelled parts",
                "subject": "", "identifiable": False, "generic_use": "how an oil filter works",
                "shot_type": "screen", "problems": ["text_logo"]}
    monkeypatch.setattr(sl, "_draft_openrouter", draft)

    proj = "Filter guide"
    pid = ps.create_project(proj, proj)
    img_dir = tmp_path / "downloads" / "filter-guide" / "images" / "shots" / "shot_01"
    img_dir.mkdir(parents=True)
    img = img_dir / "01-1-oil-filter.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=800x600:duration=1", "-frames:v", "1", str(img)], check=True)
    shots = [{"slot_id": 1, "timestamp": 0, "end_timestamp": 5, "priority": "medium",
              "text": "the oil filter traps dirt before it reaches the engine",
              "selected_results": [],
              "images": [{"url": "https://img.example/filter.png", "local_path": str(img),
                          "title": "oil filter diagram", "page": "https://example.com/filters"}]}]
    monkeypatch.setattr(ps, "_exported_placements", lambda path: ({}, set()))
    ps.record_delivery(pid, shots)

    # The editor dragged the still onto the timeline over the first line.
    item = {"name": img.name, "local_path": f"C:/edit/{img.name}", "fps": 30.0,
            "start_frame": 30, "end_frame": 120, "in_frame": 0, "out_frame": 90,
            "in_seconds": 0.0, "out_seconds": 3.0}
    s = ps.analyze_import(pid, [item])
    assert s["images"] == {"used_here": 1}

    assert sl.ingest_from_import(pid) == 1
    assert sl.process_pending() == 1
    seg = sl.get_segment(1)
    assert seg["kind"] == "image" and seg["file_status"] == "ready", seg["file_error"]
    assert seg["file_path"].endswith(".png") and seg["origin_page"] == "https://example.com/filters"
    assert len(seen["frames"]) == 1 and seen["frames"][0].endswith(".jpg")
    assert seg["problems"] == ["text_logo"]
    task = sl.next_review_task(501)
    assert task["kind"] == "image" and task["media_ext"] == "png"
    sl.save_review(501, seg["id"], usable="yes", identifiable="0",
                   description="Diagram of a car engine oil filter cut in half, labelled parts",
                   generic_use="how an oil filter works")

    # A new project: the library still goes first into the shot's images folder.
    proj2 = "Engine care"
    pid2 = ps.create_project(proj2, proj2)
    ps.create_project("filler")
    shot = {"slot_id": 3, "text": "a clogged oil filter lets dirt into the engine",
            "line_subjects": [], "priority": "medium",
            "images": [{"url": "https://google/x.jpg", "local_path": "/g/x.jpg"}]}
    assert sl.add_library_images([shot], proj2) == 1
    first = shot["images"][0]
    assert first["library_segment_id"] == seg["id"]
    assert os.path.basename(first["local_path"]).startswith("03-L1-library-")
    assert os.path.isfile(first["local_path"])
    assert shot["images"][1]["url"] == "https://google/x.jpg"

    # Delivery links it back to the library entry and rests it next time.
    ps.record_delivery(pid2, [shot])
    with ps._conn() as c:
        row = c.execute("SELECT source, segment_id FROM project_assets WHERE project_id=? "
                        "AND kind='image' AND position=0", (pid2,)).fetchone()
    assert tuple(row) == ("library", seg["id"])
    assert sl.mark_used(pid2, [shot]) == 1
    fresh = {"slot_id": 3, "text": "a clogged oil filter lets dirt into the engine",
             "line_subjects": [], "images": []}
    assert sl.add_library_images([fresh], "Another") == 0

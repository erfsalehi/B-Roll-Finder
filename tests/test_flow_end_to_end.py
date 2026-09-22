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

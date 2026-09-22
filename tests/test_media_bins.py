"""Reference images and extras go into searchable Premiere bins — in the
Project panel, never on the timeline — without upsetting the XML checks or
the XML read-back."""

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest

from core import output, pipeline
from core.xml_reimport import parse_fcpxml

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _media(path, still=False, seconds=12):
    args = ["-f", "lavfi", "-i", f"testsrc2=size=640x360:duration={seconds}"]
    args += ["-frames:v", "1"] if still else ["-pix_fmt", "yuv420p"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, str(path)], check=True)
    return str(path)


@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    proj = "Oil change basics"
    clips = output.clip_base_dir(proj)
    os.makedirs(clips)
    img_dir = os.path.join(os.path.dirname(clips), "images", "shots", "shot_01")
    os.makedirs(img_dir)
    shots = [
        {"slot_id": 1, "timestamp": 0, "end_timestamp": 5, "priority": "medium",
         "text": "drain the old engine oil into a pan",
         "selected_results": [{"url": "https://yt/1", "source": "youtube", "_dl_ok": True,
                               "local_path": _media(os.path.join(clips, "1-1-a.mp4"))}],
         "images": [
             {"url": "https://lib/1.png", "local_path": _media(os.path.join(img_dir, "01-L1-library-pan.png"), True),
              "title": "[Library] Oil pouring into a drain pan", "library_segment_id": 4},
             {"url": "https://g/2.jpg", "local_path": _media(os.path.join(img_dir, "01-1-drain-plug.jpg"), True),
              "title": "drain plug close up", "width": 640, "height": 360},
             {"url": "https://g/3.jpg", "local_path": os.path.join(img_dir, "missing.jpg")}]},
        {"slot_id": 9, "is_extra": True, "extra_label": "Extra - Toyota Camry",
         "timestamp": 10.0, "end_timestamp": 14.0,
         "selected_results": [{"url": "https://yt/x", "source": "youtube", "_dl_ok": True,
                               "title": "Camry walkaround", "matched_query": "Toyota Camry",
                               "local_path": _media(os.path.join(clips, "9-1-camry.mp4"))}]},
    ]
    return proj, shots


def test_bins_hold_images_and_extras_but_not_the_timeline(project):
    proj, shots = project
    xml_path = pipeline.write_fcpxml(shots, proj)          # runs the evaluator + repair gate
    root = ET.parse(xml_path).getroot()
    bins = {b.findtext("name"): b for b in root.find("project/children").findall("bin")}
    assert set(bins) == {"Reference images", "Extras"}

    shot_bin = bins["Reference images"].find("children/bin")
    assert shot_bin.findtext("name") == "Shot 01 — drain the old engine oil into a pan"
    names = [c.findtext("name") for c in shot_bin.iter("clip")]
    assert names == ["01 · Library · Oil pouring into a drain pan",
                     "01 · Google · drain plug close up"]              # missing file skipped
    first = next(shot_bin.iter("clip"))
    assert first.findtext("logginginfo/lognote") == "drain the old engine oil into a pan"
    assert first.find(".//pathurl").text.endswith("images/shots/shot_01/01-L1-library-pan.png")
    assert first.find(".//samplecharacteristics/width").text == "640"
    assert first.find(".//start") is None                               # never a timeline clip

    extra = next(bins["Extras"].iter("clip"))
    assert extra.findtext("name") == "Extra - Toyota Camry · Camry walkaround"

    # The timeline itself is unchanged: one b-roll clip, and reading the XML
    # back sees only that — bin contents aren't "used".
    items = parse_fcpxml(xml_path)
    assert [i["name"] for i in items] == ["1-1-a.mp4"]
    assert output.evaluate_fcpxml(open(xml_path, encoding="utf-8").read())["ok"]


def test_bins_can_be_switched_off(project, monkeypatch):
    proj, shots = project
    monkeypatch.setenv("FCPXML_MEDIA_BINS", "0")
    root = ET.parse(pipeline.write_fcpxml(shots, proj)).getroot()
    assert root.find("project/children").findall("bin") == []


def test_repair_leaves_bins_alone(project):
    proj, shots = project
    xml = output.generate_fcpxml(shots, project_name=proj,
                                 xml_dir=os.path.dirname(output.clip_base_dir(proj)))
    broken = xml.replace("<end>", "<end>9999", 1)              # force a repair pass
    repaired = output.repair_fcpxml(broken)
    assert repaired.count("<clip id=\"bin-") == xml.count("<clip id=\"bin-") == 3

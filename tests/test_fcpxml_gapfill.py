"""The b-roll video track must never have a gap: a shot that ends up empty
(failed fetch / all irrelevant / every download failed / priority=none) is
covered by stretching the neighbouring clip."""

import re
import xml.etree.ElementTree as ET

from core.output import generate_fcpxml


def _video_clip_spans(xml: str):
    """(start, end) frame pairs for video clips on the first track, time-ordered."""
    root = ET.fromstring(xml)
    track = next(root.iter("track"))
    spans = []
    for ci in track.findall("clipitem"):
        s, e = ci.findtext("start"), ci.findtext("end")
        if s is not None and e is not None and int(s) >= 0:
            spans.append((int(s), int(e)))
    return sorted(spans)


def test_empty_middle_shot_leaves_no_gap():
    shots = [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 4.0,
         "selected_results": [{"url": "https://x/a.mp4", "matched_query": "a"}]},
        {"slot_id": 2, "timestamp": 4.0, "end_timestamp": 9.0,  # DROPPED — empty
         "selected_results": []},
        {"slot_id": 3, "timestamp": 9.0, "end_timestamp": 13.0,
         "selected_results": [{"url": "https://x/c.mp4", "matched_query": "c"}]},
    ]
    spans = _video_clip_spans(generate_fcpxml(shots, project_name="demo"))
    assert len(spans) == 2
    # No gap: clip 1 must stretch across the empty shot 2, meeting clip 3's start.
    assert spans[0][1] == spans[1][0], f"gap between {spans[0]} and {spans[1]}"


def test_empty_lead_shot_anchors_first_clip_at_zero():
    shots = [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 5.0,  # empty intro
         "selected_results": []},
        {"slot_id": 2, "timestamp": 5.0, "end_timestamp": 9.0,
         "selected_results": [{"url": "https://x/b.mp4", "matched_query": "b"}]},
    ]
    spans = _video_clip_spans(generate_fcpxml(shots, project_name="demo"))
    assert spans and spans[0][0] == 0, f"first clip should start at 0, got {spans}"


def test_gapfill_can_be_disabled(monkeypatch):
    monkeypatch.setenv("FCPXML_FILL_GAPS", "0")
    shots = [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 4.0,
         "selected_results": [{"url": "https://x/a.mp4", "matched_query": "a"}]},
        {"slot_id": 2, "timestamp": 4.0, "end_timestamp": 9.0, "selected_results": []},
        {"slot_id": 3, "timestamp": 9.0, "end_timestamp": 13.0,
         "selected_results": [{"url": "https://x/c.mp4", "matched_query": "c"}]},
    ]
    spans = _video_clip_spans(generate_fcpxml(shots, project_name="demo"))
    # With fill off, the honest gap (clip 1 ends at shot 2 start) remains.
    assert spans[0][1] < spans[1][0]

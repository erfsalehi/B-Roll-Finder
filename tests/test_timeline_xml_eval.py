"""The pre-finalize evaluation round: structurally check a generated FCPXML,
auto-repair the recoverable corruption (bad clip ends + the start-cascade they
trigger), and re-verify before the timeline is allowed to ship."""

import xml.etree.ElementTree as ET

import pytest

from core.output import (generate_fcpxml, evaluate_fcpxml, repair_fcpxml,
                         generate_shots_srt, evaluate_shot_timings, evaluate_srt,
                         TimelineXMLError, ProjectValidationError)


def _shots():
    return [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 4.0, "shot_intent": "open",
         "selected_results": [{"url": "https://x/a.mp4", "matched_query": "a"},
                              {"url": "https://x/b.mp4", "matched_query": "b"}]},
        {"slot_id": 2, "timestamp": 4.0, "end_timestamp": 9.0,
         "selected_results": [{"url": "https://x/c.mp4", "matched_query": "c"}]},
        {"slot_id": 3, "timestamp": 9.0, "end_timestamp": 14.0,
         "selected_results": [{"url": "https://x/d.mp4", "matched_query": "d"},
                              {"url": "https://x/e.mp4", "matched_query": "e"}]},
    ]


def _break_last_clip_ends(xml: str) -> str:
    """Reproduce the real-world corruption: force the last sub-clip of each shot
    to end=0, which a chaining writer then propagates into the next clip's start.
    Here we only zero the ends; the evaluator must still catch them and repair
    must restore a clean, gap-free, valid timeline."""
    root = ET.fromstring(xml)
    track = next(root.iter("track"))
    clips = track.findall("clipitem")
    # zero the end of every clip whose name marks it the 2nd pick (…-2-…)
    broken = 0
    for ci in clips:
        s = int(ci.findtext("start"))
        # break a couple of interior clips
        if s > 0 and broken < 2:
            ci.find("end").text = "0"
            broken += 1
    return ET.tostring(root, encoding="unicode")


def test_clean_output_passes():
    rep = evaluate_fcpxml(generate_fcpxml(_shots(), project_name="demo"))
    assert rep["ok"], rep["errors"]
    assert rep["stats"]["clips"] == 5


def test_detects_end_before_start():
    broken = _break_last_clip_ends(generate_fcpxml(_shots(), project_name="demo"))
    rep = evaluate_fcpxml(broken)
    assert not rep["ok"]
    assert any("end (0)" in e for e in rep["errors"])


def test_detects_same_track_overlap():
    # Two b-roll clips occupying the same frames on one track.
    xml = (
        '<xmeml version="4"><project><name>p</name><children>'
        '<sequence id="s"><name>S</name><duration>100</duration>'
        '<media><video><track>'
        '<clipitem id="c1"><start>0</start><end>50</end><in>0</in><out>50</out>'
        '<file id="f1"><pathurl>a.mp4</pathurl></file></clipitem>'
        '<clipitem id="c2"><start>20</start><end>100</end><in>0</in><out>80</out>'
        '<file id="f2"><pathurl>b.mp4</pathurl></file></clipitem>'
        '</track></video></media></sequence></children></project></xmeml>'
    )
    rep = evaluate_fcpxml(xml)
    assert not rep["ok"]
    assert any("overlap" in e for e in rep["errors"])


def test_detects_wrong_sequence_duration():
    xml = generate_fcpxml(_shots(), project_name="demo").replace(
        "<duration>", "<duration>", 1)
    # bump only the sequence duration (first <duration> in the doc)
    import re
    xml = re.sub(r"<duration>\d+</duration>", "<duration>999999</duration>", xml, count=1)
    rep = evaluate_fcpxml(xml)
    assert not rep["ok"]
    assert any("sequence <duration>" in e for e in rep["errors"])


def test_detects_orphan_file_reference():
    xml = (
        '<xmeml version="4"><project><name>p</name><children>'
        '<sequence id="s"><name>S</name><duration>50</duration>'
        '<media><video><track>'
        '<clipitem id="c1"><start>0</start><end>50</end><in>0</in><out>50</out>'
        '<file id="f1"/></clipitem>'   # referenced, never defined
        '</track></video></media></sequence></children></project></xmeml>'
    )
    rep = evaluate_fcpxml(xml)
    assert not rep["ok"]
    assert any("never fully defined" in e for e in rep["errors"])


def test_repair_fixes_broken_ends_and_revalidates():
    broken = _break_last_clip_ends(generate_fcpxml(_shots(), project_name="demo"))
    assert not evaluate_fcpxml(broken)["ok"]
    repaired = repair_fcpxml(broken)
    rep = evaluate_fcpxml(repaired)
    assert rep["ok"], rep["errors"]
    # every clip now has positive length and the track is gap-free from 0
    root = ET.fromstring(repaired)
    spans = sorted((int(c.findtext("start")), int(c.findtext("end")))
                   for c in next(root.iter("track")).findall("clipitem"))
    assert spans[0][0] == 0
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        assert e0 == s1, f"gap/overlap between {(s0, e0)} and {(s1, e1)}"
        assert e1 > s1


def test_warnings_do_not_fail_the_gate():
    # check_media on, but the referenced files aren't on disk -> warnings, still ok.
    xml = generate_fcpxml(_shots(), project_name="demo")
    rep = evaluate_fcpxml(xml, xml_dir="/nonexistent", check_media=True)
    assert rep["ok"]
    assert rep["warnings"]


# ── shot timings + SRT ───────────────────────────────────────────────────────

def _timed_shots():
    return [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 3.0, "selected_results": [{"url": "a"}]},
        {"slot_id": 2, "timestamp": 3.0, "end_timestamp": 7.0, "selected_results": [{"url": "b"}]},
        {"slot_id": 3, "timestamp": 7.0, "end_timestamp": 11.0, "selected_results": [{"url": "c"}]},
    ]


def test_clean_timings_and_srt_pass():
    shots = _timed_shots()
    assert evaluate_shot_timings(shots)["ok"]
    assert evaluate_srt(generate_shots_srt(shots))["ok"]


def test_extras_with_cursor_timestamps_are_not_flagged():
    shots = _timed_shots() + [
        {"slot_id": 4, "timestamp": 11.0, "end_timestamp": 15.0, "is_extra": True,
         "extra_label": "Extra - x", "selected_results": [{"url": "e"}]},
        {"slot_id": 5, "timestamp": 15.0, "end_timestamp": 19.0, "is_extra": True,
         "extra_label": "Extra - y", "selected_results": [{"url": "f"}]},
    ]
    assert evaluate_shot_timings(shots)["ok"]


def test_timing_collision_is_an_error():
    # The real-world corruption: a block of shots collapses onto timestamp 0.
    shots = [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 1.0, "selected_results": [{"url": "a"}]},
        {"slot_id": 21, "timestamp": 0.0, "end_timestamp": 1.0, "selected_results": [{"url": "b"}]},
        {"slot_id": 22, "timestamp": 0.0, "end_timestamp": 1.0, "selected_results": [{"url": "c"}]},
        {"slot_id": 2, "timestamp": 7.0, "end_timestamp": 11.0, "selected_results": [{"url": "d"}]},
    ]
    rep = evaluate_shot_timings(shots)
    assert not rep["ok"]
    assert any("share start timestamp" in e for e in rep["errors"])
    # and the SRT those shots render into is correspondingly broken
    assert not evaluate_srt(generate_shots_srt(shots))["ok"]


def test_missing_timestamp_is_an_error():
    shots = [{"slot_id": 1, "selected_results": [{"url": "a"}]},
             {"slot_id": 2, "timestamp": 3.0, "end_timestamp": 5.0, "selected_results": [{"url": "b"}]}]
    rep = evaluate_shot_timings(shots)
    assert not rep["ok"]
    assert any("timestamp" in e for e in rep["errors"])


def test_evaluate_srt_flags_overlap_and_bad_duration():
    overlap = ("1\n00:00:00,000 --> 00:00:02,000\nA\n\n"
               "2\n00:00:01,000 --> 00:00:03,000\nB\n")
    rep = evaluate_srt(overlap)
    assert not rep["ok"] and any("overlap" in e for e in rep["errors"])
    bad = "1\n00:00:05,000 --> 00:00:05,000\nA\n"
    assert any("end <= start" in e for e in evaluate_srt(bad)["errors"])


def test_export_gate_blocks_corrupt_timing():
    from core.pipeline import _evaluate_project_before_export
    bad = [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 1.0, "selected_results": [{"url": "a"}]},
        {"slot_id": 2, "timestamp": 0.0, "end_timestamp": 1.0, "selected_results": [{"url": "b"}]},
    ]
    with pytest.raises(ProjectValidationError):
        _evaluate_project_before_export(bad)
    # a clean project passes the gate silently
    _evaluate_project_before_export(_timed_shots())

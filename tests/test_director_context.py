"""Hierarchical context pre-pass + decoupled auto-selection.

Offline unit coverage (no API calls) for:
  - subject_for_timestamp / subjects_in_span (Phase 1 state machine)
  - _render_director_system_prompt placeholder stripping (Phase 2, Mode A regression)
  - auto_select_top_candidates (Phase 3)
"""

from core.director import (
    subject_for_timestamp,
    subjects_in_span,
    load_director_prompt,
    _render_director_system_prompt,
)
from core.director_rank import auto_select_top_candidates


ROADMAP = {
    "video_global_subject": "10 cars you should not buy",
    "segments": [
        {"subject": "BMW M3 E90", "start_time": 75.0, "end_time": 165.0},
        {"subject": "Audi RS4 B7", "start_time": 165.0, "end_time": 240.0},
    ],
}


# ── Phase 1: timestamp → subject state machine ────────────────────────────────

def test_subject_inside_segment():
    assert subject_for_timestamp(ROADMAP, 120.0) == "BMW M3 E90"
    assert subject_for_timestamp(ROADMAP, 200.0) == "Audi RS4 B7"


def test_subject_on_boundary_is_inclusive():
    # 165.0 is the shared edge; the first matching segment wins.
    assert subject_for_timestamp(ROADMAP, 165.0) == "BMW M3 E90"


def test_subject_before_first_segment_falls_back_to_nearest():
    # Intro (t < first start) has no segment — nearest segment is the M3.
    assert subject_for_timestamp(ROADMAP, 10.0) == "BMW M3 E90"


def test_subject_empty_roadmap_returns_blank():
    assert subject_for_timestamp({}, 100.0) == ""
    assert subject_for_timestamp(None, 100.0) == ""


def test_subject_no_segments_uses_global():
    rm = {"video_global_subject": "topic", "segments": []}
    assert subject_for_timestamp(rm, 100.0) == "topic"


def test_subjects_in_span_collects_straddled_subjects():
    # A block spanning the 165s boundary should surface both subjects in order.
    subs = subjects_in_span(ROADMAP, 150.0, 200.0)
    assert subs == ["BMW M3 E90", "Audi RS4 B7"]


def test_subjects_in_span_single_subject_no_dupes():
    assert subjects_in_span(ROADMAP, 100.0, 140.0) == ["BMW M3 E90"]


# ── Phase 2: prompt rendering (Mode A regression + context block) ─────────────

def test_prompt_strips_both_blocks_when_empty():
    rendered = _render_director_system_prompt(load_director_prompt(), "", "")
    assert "{custom_instructions_block}" not in rendered
    assert "{segment_context_block}" not in rendered
    assert "SEGMENT CONTEXT AWARENESS" not in rendered  # not injected in Mode A


def test_prompt_injects_segment_block_when_context_aware():
    rendered = _render_director_system_prompt(
        load_director_prompt(), "cars", "", context_aware=True
    )
    assert "{segment_context_block}" not in rendered
    assert "SEGMENT CONTEXT AWARENESS" in rendered
    assert "OVERALL VIDEO TOPIC: cars" in rendered  # custom block still rendered


# ── Phase 3: auto-selection ───────────────────────────────────────────────────

def _shot(**kw):
    base = {"slot_id": 1, "priority": "medium", "video_results": [], "selected_results": []}
    base.update(kw)
    return base


def test_auto_select_binds_top_non_irrelevant():
    shot = _shot(video_results=[
        {"url": "a", "irrelevant": True},
        {"url": "b"},
        {"url": "c"},
    ])
    auto_select_top_candidates([shot])
    assert [r["url"] for r in shot["selected_results"]] == ["b"]
    assert shot["auto_selected"] is True


def test_auto_select_all_irrelevant_falls_back_to_first():
    shot = _shot(video_results=[
        {"url": "a", "irrelevant": True},
        {"url": "b", "irrelevant": True},
    ])
    auto_select_top_candidates([shot])
    assert [r["url"] for r in shot["selected_results"]] == ["a"]
    assert shot["auto_selected"] is True


def test_auto_select_never_overwrites_manual_pick():
    shot = _shot(
        video_results=[{"url": "a"}, {"url": "b"}],
        selected_results=[{"url": "b"}],
    )
    auto_select_top_candidates([shot])
    assert [r["url"] for r in shot["selected_results"]] == ["b"]
    assert "auto_selected" not in shot


def test_auto_select_skips_none_priority_and_skipped():
    none_shot = _shot(priority="none", video_results=[{"url": "a"}])
    skipped_shot = _shot(skipped=True, video_results=[{"url": "a"}])
    auto_select_top_candidates([none_shot, skipped_shot])
    assert none_shot["selected_results"] == []
    assert skipped_shot["selected_results"] == []


def test_auto_select_skips_shots_without_candidates():
    shot = _shot(video_results=[])
    auto_select_top_candidates([shot])
    assert shot["selected_results"] == []
    assert "auto_selected" not in shot


def test_auto_select_is_idempotent():
    shot = _shot(video_results=[{"url": "a"}, {"url": "b"}])
    auto_select_top_candidates([shot])
    first = list(shot["selected_results"])
    auto_select_top_candidates([shot])  # re-run must not change the pick
    assert shot["selected_results"] == first


def test_auto_select_start_slot_id_skips_earlier_shots():
    shots = [
        _shot(slot_id=1, video_results=[{"url": "a1"}]),
        _shot(slot_id=2, video_results=[{"url": "a2"}]),
        _shot(slot_id=3, video_results=[{"url": "a3"}]),
    ]
    auto_select_top_candidates(shots, start_slot_id=2)
    # Shots before #2 are left for manual review …
    assert shots[0]["selected_results"] == []
    assert "auto_selected" not in shots[0]
    # … shots at/after #2 are auto-selected.
    assert shots[1]["selected_results"][0]["url"] == "a2"
    assert shots[2]["selected_results"][0]["url"] == "a3"


def test_auto_select_start_slot_id_none_covers_all():
    shots = [
        _shot(slot_id=1, video_results=[{"url": "a1"}]),
        _shot(slot_id=2, video_results=[{"url": "a2"}]),
    ]
    auto_select_top_candidates(shots, start_slot_id=None)
    assert all(s["selected_results"] for s in shots)

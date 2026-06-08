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
from core.director_rank import auto_select_top_candidates, clear_auto_selections


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


def test_auto_select_binds_top_non_irrelevant_first():
    # Default min_clips=2 → top two non-irrelevant, best first, irrelevant skipped.
    shot = _shot(video_results=[
        {"url": "a", "irrelevant": True},
        {"url": "b"},
        {"url": "c"},
    ])
    auto_select_top_candidates([shot])
    assert [r["url"] for r in shot["selected_results"]] == ["b", "c"]
    assert shot["auto_selected"] is True


def test_auto_select_all_irrelevant_falls_back_to_first():
    shot = _shot(video_results=[
        {"url": "a", "irrelevant": True},
        {"url": "b", "irrelevant": True},
    ])
    auto_select_top_candidates([shot])
    assert shot["selected_results"][0]["url"] == "a"  # best (least-bad) first
    assert shot["auto_selected"] is True


def test_auto_select_scales_clip_count_with_duration():
    # YouTube count scales with duration (~1 every AUTO_SELECT_YT_SECONDS=4.5s).
    shot = _shot(slot_id=1, duration_needed_sec=30.0,
                 video_results=[{"url": f"u{i}", "source": "youtube"} for i in range(10)])
    auto_select_top_candidates([shot])
    assert len(shot["selected_results"]) == 7   # ceil(30 / 4.5)


def test_auto_select_short_shot_still_meets_floor():
    # A short shot can't scale up, but the floor top-up keeps it non-empty:
    # max(min_clips=2, min_pexels+1=3) = 3 distinct clips when available.
    shot = _shot(slot_id=1, duration_needed_sec=4.0,
                 video_results=[{"url": "a"}, {"url": "b"}, {"url": "c"}])
    auto_select_top_candidates([shot])
    assert len(shot["selected_results"]) == 3   # floor = max(2, min_pexels+1)


def test_auto_select_caps_at_available_distinct_candidates():
    shot = _shot(slot_id=1, duration_needed_sec=60.0,
                 video_results=[{"url": "a"}, {"url": "b"}])
    auto_select_top_candidates([shot])
    assert len(shot["selected_results"]) == 2   # only 2 distinct available


def test_auto_select_respects_max_clips():
    # ceil(100 / 4.5) = 23 YouTube clips wanted, but capped at max_clips.
    shot = _shot(slot_id=1, duration_needed_sec=100.0,
                 video_results=[{"url": f"u{i}", "source": "youtube"} for i in range(20)])
    auto_select_top_candidates([shot], max_clips=8)
    assert len(shot["selected_results"]) == 8


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


def test_director_block_size_env_controls_call_count(monkeypatch):
    # Each block is one LLM call; DIRECTOR_BLOCK_SIZE sets segments-per-call.
    import core.director as d

    calls = {"n": 0}

    def _fake_llm(client, system_prompt, user_msg, temperature=0.4, max_tokens=3000):
        calls["n"] += 1
        return {"shots": [{"script_chunk": "x", "start": 0.0, "end": 1.0,
                           "search_queries": ["a"]}]}

    monkeypatch.setattr(d, "_call_llm_json", _fake_llm)
    segments = [{"start": float(i), "end": float(i) + 1, "text": f"seg {i}"} for i in range(40)]

    # Default block size 20 → 40 segments = 2 calls.
    d.generate_shot_list_from_transcription(segments, api_key="k")
    assert calls["n"] == 2

    # Doubling the block size halves the calls.
    calls["n"] = 0
    monkeypatch.setenv("DIRECTOR_BLOCK_SIZE", "40")
    d.generate_shot_list_from_transcription(segments, api_key="k")
    assert calls["n"] == 1


def test_auto_select_avoids_back_to_back_duplicate():
    shots = [
        _shot(slot_id=1, video_results=[{"url": "same"}, {"url": "alt1"}]),
        _shot(slot_id=2, video_results=[{"url": "same"}, {"url": "alt2"}]),
    ]
    auto_select_top_candidates(shots)
    assert shots[0]["selected_results"][0]["url"] == "same"
    assert shots[1]["selected_results"][0]["url"] == "alt2"  # avoided the repeat


def test_auto_select_allows_duplicate_when_no_alternative():
    # Graceful degradation: a slot is never left empty just to avoid a repeat.
    shots = [
        _shot(slot_id=1, video_results=[{"url": "same"}]),
        _shot(slot_id=2, video_results=[{"url": "same"}]),
    ]
    auto_select_top_candidates(shots)
    assert shots[1]["selected_results"][0]["url"] == "same"


def test_auto_select_variety_window_spans_multiple_shots():
    shots = [
        _shot(slot_id=1, video_results=[{"url": "A"}, {"url": "x1"}]),
        _shot(slot_id=2, video_results=[{"url": "B"}, {"url": "x2"}]),
        _shot(slot_id=3, video_results=[{"url": "A"}, {"url": "C"}]),  # A still in window
    ]
    auto_select_top_candidates(shots, lookback=3)
    assert shots[2]["selected_results"][0]["url"] == "C"


def test_auto_select_variety_avoids_repeating_manual_neighbor():
    shots = [
        _shot(slot_id=1, selected_results=[{"url": "manual"}], video_results=[{"url": "x"}]),
        _shot(slot_id=2, video_results=[{"url": "manual"}, {"url": "fresh"}]),
    ]
    auto_select_top_candidates(shots)
    assert shots[1]["selected_results"][0]["url"] == "fresh"  # avoided the manual neighbor


def test_auto_select_lookback_zero_disables_variety():
    shots = [
        _shot(slot_id=1, video_results=[{"url": "same"}, {"url": "alt"}]),
        _shot(slot_id=2, video_results=[{"url": "same"}, {"url": "alt2"}]),
    ]
    auto_select_top_candidates(shots, lookback=0)
    assert shots[1]["selected_results"][0]["url"] == "same"  # no variety penalty


def test_clear_auto_selections_only_clears_auto_picks():
    auto = _shot(slot_id=1, selected_results=[{"url": "a"}], auto_selected=True)
    manual = _shot(slot_id=2, selected_results=[{"url": "b"}])  # no auto flag
    clear_auto_selections([auto, manual])
    assert auto["selected_results"] == [] and "auto_selected" not in auto
    assert manual["selected_results"] == [{"url": "b"}]  # untouched


def test_reapply_with_new_start_preserves_manual_picks():
    # #1 manual, #2 + #3 previously auto. Re-apply from #3 only.
    shots = [
        _shot(slot_id=1, video_results=[{"url": "a1"}], selected_results=[{"url": "manual1"}]),
        _shot(slot_id=2, video_results=[{"url": "a2"}], selected_results=[{"url": "a2"}], auto_selected=True),
        _shot(slot_id=3, video_results=[{"url": "a3"}], selected_results=[{"url": "a3"}], auto_selected=True),
    ]
    clear_auto_selections(shots)
    auto_select_top_candidates(shots, start_slot_id=3)
    assert shots[0]["selected_results"] == [{"url": "manual1"}]  # manual kept
    assert shots[1]["selected_results"] == []                    # auto cleared, below start
    assert shots[2]["selected_results"][0]["url"] == "a3"        # re-selected at start

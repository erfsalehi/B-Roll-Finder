"""Selection fixes: no irrelevant picks, fresh queries for empty shots, and
YouTube videos not repeated across shots when a fresh one exists."""

import core.director
import core.director_rank
import core.director_search
import core.director_youtube
import core.pipeline as pipeline
from core.director_rank import auto_select_top_candidates


def _yt(url):
    return {"url": url, "source": "youtube"}


# ── auto-select: YouTube variety across the whole video ──────────────────────

def test_auto_select_prefers_youtube_unused_elsewhere():
    # "popular" tops every shot's list; beyond the 3-shot look-back the old
    # selector picked it again. Each later shot should lead with its own video.
    shots = [{"slot_id": i, "priority": "medium", "duration_needed_sec": 3.0,
              "video_results": [_yt("popular"), _yt(f"fresh{i}a"), _yt(f"fresh{i}b")]}
             for i in range(1, 7)]
    auto_select_top_candidates(shots)
    leads = [s["selected_results"][0]["url"] for s in shots]
    assert leads == ["popular"] + [f"fresh{i}a" for i in range(2, 7)]
    used = [c["url"] for s in shots for c in s["selected_results"]]
    assert used.count("popular") == 1


def test_auto_select_repeats_youtube_when_no_fresh_option():
    shots = [{"slot_id": i, "priority": "medium", "duration_needed_sec": 3.0,
              "video_results": [_yt("only")]} for i in (1, 2, 3, 4, 5)]
    auto_select_top_candidates(shots)
    # Outside the look-back window the same video is allowed back (a repeat
    # beats an empty shot; the XML starts it at a different in-point).
    assert shots[4]["selected_results"][0]["url"] == "only"


# ── repair: new queries, not the same failed searches again ──────────────────

def _stub_repair(monkeypatch):
    monkeypatch.setenv("AUTO_USE_LIBRARY", "false")
    monkeypatch.setattr(pipeline, "_inject_segments", lambda *a, **k: 0)
    monkeypatch.setattr(core.director_rank, "rank_shot_candidates", lambda ts, **k: None)
    notes, fetched = [], {}

    def _regen(shots, slot_ids, custom_instructions="", **k):
        notes.append(custom_instructions)
        for s in shots:
            if s["slot_id"] in slot_ids:
                s["search_queries"] = [f"new q{len(notes)}"]
    monkeypatch.setattr(core.director, "regenerate_shot_queries", _regen)

    def _fetch(ts, **k):
        for s in ts:
            fetched[s["slot_id"]] = (list(s["search_queries"]),
                                     list(s.get("youtube_keywords") or []))
    monkeypatch.setattr(core.director_search, "fetch_with_retries", _fetch)
    return notes, fetched


def test_repair_writes_new_queries_avoiding_tried_ones(monkeypatch):
    notes, fetched = _stub_repair(monkeypatch)
    shot = {"slot_id": 3, "priority": "medium", "text": "the gearbox whines under load",
            "search_queries": ["gearbox whine"], "youtube_keywords": ["gearbox noise"],
            "video_results": [{"url": "junk", "source": "youtube", "irrelevant": True}],
            "selected_results": []}

    pipeline.repair_empty_shots([shot], groq_key="k")
    assert "gearbox whine" in notes[0] and "gearbox noise" in notes[0]
    # Searched with the new queries — YouTube re-seeded from them, not the old ones.
    assert fetched[3] == (["new q1"], ["new q1"])
    assert shot["selected_results"] == []          # junk is still irrelevant

    pipeline.repair_empty_shots([shot], groq_key="k")
    # The second attempt is told about everything tried so far.
    assert all(q in notes[1] for q in ("gearbox whine", "gearbox noise", "new q1"))
    assert fetched[3][0] == ["new q2"]


def test_repair_skips_regen_for_shots_without_narration(monkeypatch):
    notes, fetched = _stub_repair(monkeypatch)
    extra = {"slot_id": 9, "priority": "medium", "is_extra": True,
             "search_queries": ["bmw e90"], "video_results": [], "selected_results": []}
    pipeline.repair_empty_shots([extra], groq_key="k")
    assert notes == []                              # no LLM call
    assert fetched[9][0] == ["bmw e90"]             # still re-fetched


# ── refine: irrelevant clips only where any valid clip beats the current one ──

def test_refine_irrelevant_pool_only_replaces_a_failing_clip(monkeypatch):
    monkeypatch.setenv("AUTO_USE_LIBRARY", "false")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "_inject_segments", lambda *a, **k: 0)
    monkeypatch.setattr(core.director, "regenerate_shot_queries", lambda *a, **k: None)
    monkeypatch.setattr(core.director_youtube, "seed_youtube_keywords", lambda ts, **k: ts)
    monkeypatch.setattr(core.director_rank, "rank_shot_candidates", lambda ts, **k: None)

    def _fetch(ts, **k):
        for s in ts:
            s["video_results"] = [{"url": "weak", "source": "youtube", "irrelevant": True}]
    monkeypatch.setattr(core.director_search, "fetch_with_retries", _fetch)

    old = [{"url": "vertical", "source": "youtube", "width": 1080, "height": 1920}]
    qa = {"issues": [{"slot_id": 4, "severity": "high", "problem": "p", "suggestion": "s"}]}

    # Boundary enforcement (real auto-select): the weak clip replaces the failing one.
    shots = [{"slot_id": 4, "priority": "high", "text": "x", "selected_results": list(old)}]
    assert pipeline.refine_flagged_shots(shots, qa, groq_key="k", errors=[],
                                         only_slots={4}, qa_driven=False) == 1
    assert shots[0]["selected_results"][0]["url"] == "weak"

    # QA-driven: an all-irrelevant pool never replaces the reviewed pick.
    shots = [{"slot_id": 4, "priority": "high", "text": "x", "selected_results": list(old)}]
    assert pipeline.refine_flagged_shots(shots, qa, groq_key="k", errors=[]) == 0
    assert shots[0]["selected_results"] == old

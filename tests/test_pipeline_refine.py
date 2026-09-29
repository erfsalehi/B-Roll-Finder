"""QA-driven refine: targets only flagged shots and re-picks them."""

import core.director
import core.director_youtube
import core.director_search
import core.director_rank
from core.pipeline import refine_flagged_shots
from core.director_youtube import seed_youtube_keywords as _real_seed


def _stub_deps(monkeypatch):
    """Replace the heavy pipeline calls refine uses with no-ops / fakes."""
    regen_calls = []
    monkeypatch.setattr(core.director, "regenerate_shot_queries",
                        lambda shots, slot_ids, **k: regen_calls.append(set(slot_ids)))
    monkeypatch.setattr(core.director_youtube, "seed_youtube_keywords", lambda shots, **k: shots)
    monkeypatch.setattr(core.director_search, "fetch_with_retries", lambda shots, **k: None)
    monkeypatch.setattr(core.director_rank, "rank_shot_candidates", lambda shots, **k: None)

    def _fake_select(shots, **k):
        for s in shots:
            if not s.get("selected_results"):
                s["selected_results"] = [{"url": "new", "source": "pexels"}]
    monkeypatch.setattr(core.director_rank, "auto_select_top_candidates", _fake_select)
    # Skip the optional library + HD-filter branches.
    monkeypatch.setenv("AUTO_USE_LIBRARY", "false")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    return regen_calls


def test_refine_targets_only_high_medium(monkeypatch):
    regen_calls = _stub_deps(monkeypatch)
    shots = [
        {"slot_id": 1, "priority": "high", "selected_results": [{"url": "keep"}]},
        {"slot_id": 2, "priority": "high", "selected_results": [{"url": "old2"}]},
        {"slot_id": 3, "priority": "low",  "selected_results": [{"url": "old3"}]},
    ]
    qa = {"issues": [
        {"slot_id": 2, "severity": "high", "problem": "x", "suggestion": "y"},
        {"slot_id": 3, "severity": "low",  "problem": "z", "suggestion": "w"},  # ignored
    ]}
    n = refine_flagged_shots(shots, qa, groq_key="k", video_topic="t", errors=[])
    assert n == 1                                   # only slot 2 refreshed
    assert regen_calls == [{2}]                     # low-severity slot 3 untouched
    assert shots[1]["selected_results"] == [{"url": "new", "source": "pexels"}]
    assert shots[2]["selected_results"] == [{"url": "old3"}]   # unchanged
    assert shots[0]["selected_results"] == [{"url": "keep"}]   # not flagged


def test_refine_noop_without_issues(monkeypatch):
    regen_calls = _stub_deps(monkeypatch)
    shots = [{"slot_id": 1, "priority": "high", "selected_results": [{"url": "a"}]}]
    assert refine_flagged_shots(shots, {"issues": []}, groq_key="k") == 0
    assert regen_calls == []


def test_refine_never_repicks_the_rejected_clip(monkeypatch):
    _stub_deps(monkeypatch)
    old = {"url": "yt/cat-bank", "source": "youtube", "title": "Cute Cat Coin Bank"}
    fresh = {"url": "yt/cash", "source": "youtube", "title": "Counting cash"}

    # The re-fetch brings the flagged clip straight back alongside a new one.
    def _fetch(shots, **k):
        for s in shots:
            s["video_results"] = [dict(old), fresh]
    monkeypatch.setattr(core.director_search, "fetch_with_retries", _fetch)

    def _select(shots, **k):
        for s in shots:
            if not s.get("selected_results") and s.get("video_results"):
                s["selected_results"] = [s["video_results"][0]]
    monkeypatch.setattr(core.director_rank, "auto_select_top_candidates", _select)

    shots = [{"slot_id": 37, "priority": "high", "selected_results": [old]}]
    qa = {"issues": [{"slot_id": 37, "severity": "medium", "problem": "cat bank", "suggestion": "cash"}]}
    assert refine_flagged_shots(shots, qa, groq_key="k", errors=[]) == 1
    assert shots[0]["selected_results"][0]["url"] == "yt/cash"
    assert "yt/cat-bank" in shots[0]["qa_rejected"]


def test_refine_low_severity_when_asked(monkeypatch):
    regen_calls = _stub_deps(monkeypatch)
    shots = [{"slot_id": 15, "priority": "high", "selected_results": [{"url": "tesla"}]}]
    qa = {"issues": [{"slot_id": 15, "severity": "low", "problem": "p", "suggestion": "s"}]}
    n = refine_flagged_shots(shots, qa, groq_key="k", errors=[],
                             severities=("high", "medium", "low"))
    assert n == 1 and regen_calls == [{15}]


def test_regen_context_skips_extras(monkeypatch):
    """The last narration shots sit next to extras, which have no text/shot_type —
    that used to KeyError and silently leave the old queries in place."""
    from core import director
    seen = {}

    def _fake_llm(client, system, user, **k):
        seen["user"] = user
        return {"shots": [{"search_queries": ["stack of cash garage counter"]}]}
    monkeypatch.setattr(director, "_call_llm_json", _fake_llm)
    monkeypatch.setattr(director, "Groq", lambda api_key: None)
    monkeypatch.setattr(director, "load_director_prompt", lambda: "{video_topic}")
    monkeypatch.setattr(director, "_render_director_system_prompt", lambda t, v, c: "sys")

    shots = [
        {"slot_id": 36, "text": "tint it", "shot_type": "broll", "shot_intent": "tint",
         "search_queries": ["window tint"]},
        {"slot_id": 37, "text": "save money", "shot_type": "broll", "shot_intent": "savings",
         "search_queries": ["cat coin bank"]},
        {"slot_id": 39, "is_extra": True, "priority": "low", "search_queries": ["ac vent"],
         "shot_intent": "extra B-roll: ac vent", "selected_results": [{"url": "x"}]},
    ]
    director.regenerate_shot_queries(shots, {37}, api_key="k")
    assert shots[1]["search_queries"] == ["stack of cash garage counter"]
    assert "ac vent" not in seen["user"]


def test_timeline_summary_excludes_extras():
    from core.director_rank import build_timeline_summary
    shots = [
        {"slot_id": 1, "text": "a", "selected_results": [{"title": "Main clip"}]},
        {"slot_id": 2, "is_extra": True, "selected_results": [{"title": "Extra clip"}]},
    ]
    out = build_timeline_summary(shots)
    assert "Main clip" in out and "Extra clip" not in out


def _fetching(monkeypatch, pools):
    """Stub the fetch so each target shot gets ``pools[slot_id]`` as candidates."""
    def _fetch(shots, **k):
        for s in shots:
            s["video_results"] = [dict(c) for c in pools.get(s["slot_id"], [])]
    monkeypatch.setattr(core.director_search, "fetch_with_retries", _fetch)

    def _select(shots, **k):
        for s in shots:
            if not s.get("selected_results") and s.get("video_results"):
                s["selected_results"] = [s["video_results"][0]]
    monkeypatch.setattr(core.director_rank, "auto_select_top_candidates", _select)


def test_refine_reseeds_youtube_keywords_from_new_queries(monkeypatch):
    """YouTube searches run on youtube_keywords — they must follow the new
    queries, not keep re-running the original search."""
    _stub_deps(monkeypatch)
    monkeypatch.setattr(core.director_youtube, "seed_youtube_keywords", _real_seed)
    seen = {}

    def _regen(shots, slot_ids, **k):
        for s in shots:
            if s["slot_id"] in slot_ids:
                s["search_queries"] = ["mechanic handing customer large invoice"]
    monkeypatch.setattr(core.director, "regenerate_shot_queries", _regen)

    def _fetch(shots, **k):
        seen["yt"] = [s.get("youtube_keywords") for s in shots]
    monkeypatch.setattr(core.director_search, "fetch_with_retries", _fetch)

    shots = [{"slot_id": 8, "priority": "high", "search_queries": ["mechanic customer"],
              "youtube_keywords": ["mechanic customer"], "selected_results": [{"url": "handshake"}]}]
    qa = {"issues": [{"slot_id": 8, "severity": "high", "problem": "p", "suggestion": "s"}]}
    refine_flagged_shots(shots, qa, groq_key="k", errors=[])
    assert seen["yt"] == [["mechanic handing customer large invoice"]]


def test_refine_uses_youtube_queries_written_by_the_regen(monkeypatch):
    _stub_deps(monkeypatch)
    monkeypatch.setattr(core.director_youtube, "seed_youtube_keywords", _real_seed)
    seen = {}

    def _regen(shots, slot_ids, **k):
        for s in shots:
            if s["slot_id"] in slot_ids:
                s["search_queries"] = ["mechanic handing customer large invoice"]
                s["youtube_keywords"] = ["auto repair shop bill shock"]
    monkeypatch.setattr(core.director, "regenerate_shot_queries", _regen)

    def _fetch(shots, **k):
        seen["yt"] = [s.get("youtube_keywords") for s in shots]
    monkeypatch.setattr(core.director_search, "fetch_with_retries", _fetch)

    shots = [{"slot_id": 8, "priority": "high", "search_queries": ["mechanic customer"],
              "youtube_keywords": ["mechanic customer"], "selected_results": [{"url": "handshake"}]}]
    qa = {"issues": [{"slot_id": 8, "severity": "high", "problem": "p", "suggestion": "s"}]}
    refine_flagged_shots(shots, qa, groq_key="k", errors=[])
    assert seen["yt"] == [["auto repair shop bill shock"]]


def test_refine_keeps_previous_pick_when_nothing_found(monkeypatch):
    _stub_deps(monkeypatch)
    _fetching(monkeypatch, {})                     # re-fetch returns nothing
    old = [{"url": "vent", "source": "youtube"}, {"url": "px", "source": "pexels"}]
    shots = [{"slot_id": 5, "priority": "high", "selected_results": list(old)}]
    qa = {"issues": [{"slot_id": 5, "severity": "medium", "problem": "p", "suggestion": "s"}]}
    errors = []
    assert refine_flagged_shots(shots, qa, groq_key="k", errors=errors) == 0
    assert shots[0]["selected_results"] == old     # not left empty
    assert any("kept its previous clip" in e for e in errors)


def test_refine_keeps_previous_pick_over_irrelevant_one(monkeypatch):
    _stub_deps(monkeypatch)
    _fetching(monkeypatch, {5: [{"url": "junk", "source": "youtube", "irrelevant": True}]})
    old = [{"url": "vent", "source": "youtube"}]
    qa = {"issues": [{"slot_id": 5, "severity": "medium", "problem": "p", "suggestion": "s"}]}

    shots = [{"slot_id": 5, "priority": "high", "selected_results": list(old)}]
    assert refine_flagged_shots(shots, qa, groq_key="k", errors=[]) == 0
    assert shots[0]["selected_results"] == old

    # Boundary enforcement: a valid-but-weak clip beats the failing one, and the
    # lead is NOT banned (the failing clip may be a different one).
    shots = [{"slot_id": 5, "priority": "high", "selected_results": list(old)}]
    assert refine_flagged_shots(shots, qa, groq_key="k", errors=[], only_slots={5},
                                qa_driven=False) == 1
    assert shots[0]["selected_results"][0]["url"] == "junk"
    assert "qa_rejected" not in shots[0]


def test_refine_avoids_clips_other_shots_use(monkeypatch):
    _stub_deps(monkeypatch)
    _fetching(monkeypatch, {27: [{"url": "tint-closeup", "source": "youtube"},
                                 {"url": "hot-cabin", "source": "youtube"}]})
    shots = [
        {"slot_id": 27, "priority": "high", "selected_results": [{"url": "old"}]},
        {"slot_id": 28, "priority": "high", "selected_results": [{"url": "tint-closeup"}]},
    ]
    qa = {"issues": [{"slot_id": 27, "severity": "medium", "problem": "p", "suggestion": "s"}]}
    assert refine_flagged_shots(shots, qa, groq_key="k", errors=[]) == 1
    assert shots[0]["selected_results"][0]["url"] == "hot-cabin"


def test_ranker_sees_reviewer_feedback():
    from core.director_rank import _format_shot_block
    shot = {"slot_id": 8, "text": "t", "shot_intent": "i", "refine_note": "handshake reads as trust",
            "video_results": [{"title": "a", "source": "youtube"}]}
    assert "REVIEWER FEEDBACK" in _format_shot_block(shot)
    shot.pop("refine_note")
    assert "REVIEWER FEEDBACK" not in _format_shot_block(shot)


def test_timeline_summary_shows_clip_count():
    from core.director_rank import build_timeline_summary
    shots = [{"slot_id": 20, "text": "a", "duration_needed_sec": 13.5,
              "selected_results": [{"title": "Traffic stop"}, {"title": "b"}, {"title": "c"}]}]
    assert "+2 more clips" in build_timeline_summary(shots)


# ── refine_and_review: settle the changed shots, then verify only those ──────

def test_refine_and_review_verifies_only_changed_shots(monkeypatch):
    import core.pipeline as pl
    calls = []

    def _refine(shots, qa, **k):
        shots[1]["selected_results"] = [{"url": "new-8"}]    # slot 8 got a new clip
        return 1
    monkeypatch.setattr(pl, "refine_flagged_shots", _refine)
    monkeypatch.setattr(pl, "enforce_timeline", lambda shots, **k: calls.append("enforce") or {})
    monkeypatch.setattr(pl, "ensure_youtube_coverage",
                        lambda shots, **k: calls.append(("yt", [s["slot_id"] for s in shots])) or 0)
    seen = {}

    def _review(shots, **k):
        seen.update(k)
        return {"overall": "ok", "issues": []}
    monkeypatch.setattr(core.director_rank, "review_timeline", _review)

    shots = [{"slot_id": 5, "priority": "high", "selected_results": [{"url": "vent"}]},
             {"slot_id": 8, "priority": "high", "selected_results": [{"url": "handshake"}]}]
    qa = {"issues": [{"slot_id": 5, "severity": "low", "problem": "p"},
                     {"slot_id": 8, "severity": "high", "problem": "q"}]}
    n, new_qa = pl.refine_and_review(shots, qa, groq_key="k", errors=[])
    assert n == 1 and new_qa["refined"] == 1
    assert calls == ["enforce", ("yt", [8])]                 # only the changed shot re-covered
    assert seen["changed_slots"] == {8} and seen["previous"] is qa


def test_refine_and_review_skips_review_when_nothing_changed(monkeypatch):
    import core.pipeline as pl
    monkeypatch.setattr(pl, "refine_flagged_shots", lambda shots, qa, **k: 0)
    monkeypatch.setattr(pl, "enforce_timeline", lambda shots, **k: {})
    monkeypatch.setattr(core.director_rank, "review_timeline",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no re-review")))
    shots = [{"slot_id": 5, "priority": "high", "selected_results": [{"url": "vent"}]}]
    qa = {"issues": [{"slot_id": 5, "severity": "medium", "problem": "p"}]}
    assert pl.refine_and_review(shots, qa, groq_key="k", errors=[]) == (0, qa)


def test_refine_regenerates_queries_in_parallel(monkeypatch):
    """Each flagged shot's query regeneration is its own LLM call; they must run
    concurrently. A 3-party barrier only passes if all 3 calls overlap."""
    import threading
    _stub_deps(monkeypatch)
    monkeypatch.delenv("REFINE_WORKERS", raising=False)
    barrier = threading.Barrier(3, timeout=5)
    notes = {}

    def _regen(shots, slot_ids, custom_instructions="", **k):
        barrier.wait()                                  # BrokenBarrierError if serial
        notes[next(iter(slot_ids))] = custom_instructions
    monkeypatch.setattr(core.director, "regenerate_shot_queries", _regen)

    shots = [{"slot_id": i, "priority": "high", "selected_results": [{"url": f"u{i}"}]}
             for i in (1, 2, 3)]
    qa = {"issues": [{"slot_id": i, "severity": "high", "problem": f"bad {i}", "suggestion": "s"}
                     for i in (1, 2, 3)]}
    errors = []
    refine_flagged_shots(shots, qa, groq_key="k", errors=errors)
    assert errors == []
    assert all(f"bad {i}" in notes[i] for i in (1, 2, 3))   # each call got its own notes


def test_regen_failure_is_reported(monkeypatch):
    from core import director
    monkeypatch.setattr(director, "_call_llm_json",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout")))
    monkeypatch.setattr(director, "Groq", lambda api_key: None)
    monkeypatch.setattr(director, "load_director_prompt", lambda: "p")
    monkeypatch.setattr(director, "_render_director_system_prompt", lambda t, v, c: "sys")
    shots = [{"slot_id": 4, "text": "t", "search_queries": ["old query"]}]
    errors = []
    director.regenerate_shot_queries(shots, {4}, api_key="k", errors=errors)
    assert shots[0]["search_queries"] == ["old query"]
    assert errors and "kept old queries" in errors[0] and "timeout" in errors[0]

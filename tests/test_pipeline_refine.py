"""QA-driven refine: targets only flagged shots and re-picks them."""

import core.director
import core.director_youtube
import core.director_search
import core.director_rank
from core.pipeline import refine_flagged_shots


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

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

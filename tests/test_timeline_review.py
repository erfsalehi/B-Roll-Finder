"""Stage 5 — holistic 'executive producer' timeline review."""

import core.director_rank as dr
from core.director_rank import build_timeline_summary, review_timeline


def _shot(slot_id, *, selected=True, priority="medium", skipped=False, title="Engine bay"):
    return {
        "slot_id": slot_id,
        "priority": priority,
        "skipped": skipped,
        "shot_intent": f"intent {slot_id}",
        "duration_needed_sec": 3.0,
        "timestamp_start_str": "00:00:00",
        "timestamp_end_str": "00:00:03",
        "selected_results": [{"title": title, "source": "pexels", "matched_query": "engine"}] if selected else [],
    }


def test_summary_includes_only_bound_shots():
    shots = [
        _shot(1),
        _shot(2, selected=False),         # no clip → excluded
        _shot(3, priority="none"),        # talking head → excluded
        _shot(4, skipped=True),           # skipped → excluded
        _shot(5, title="Transmission"),
    ]
    summary = build_timeline_summary(shots)
    assert "Shot 1" in summary and "Shot 5" in summary
    assert "Shot 2" not in summary and "Shot 3" not in summary and "Shot 4" not in summary
    assert "Transmission" in summary  # clip title surfaced


def test_review_needs_two_selected_shots():
    out = review_timeline([_shot(1)], api_key="k")
    assert out["issues"] == [] and "Not enough" in out["overall"]


def test_review_filters_hallucinated_slot_ids(monkeypatch):
    monkeypatch.setattr(dr, "_call_llm_json", lambda *a, **k: {
        "overall": "Mostly fine.",
        "issues": [
            {"slot_id": 1, "severity": "high", "problem": "Repeats shot 2", "suggestion": "Swap it"},
            {"slot_id": 999, "severity": "low", "problem": "ghost", "suggestion": "x"},  # not in timeline
        ],
    })
    out = review_timeline([_shot(1), _shot(2)], api_key="k")
    assert [i["slot_id"] for i in out["issues"]] == [1]      # 999 dropped
    assert out["reviewed"] == 2


def test_review_sorts_by_severity_and_validates(monkeypatch):
    monkeypatch.setattr(dr, "_call_llm_json", lambda *a, **k: {
        "overall": "ok",
        "issues": [
            {"slot_id": 1, "severity": "low", "problem": "minor"},
            {"slot_id": 2, "severity": "bogus", "problem": "weird sev"},  # → medium
            {"slot_id": 1, "severity": "high", "problem": "big"},
            {"slot_id": 2, "severity": "high", "problem": ""},            # empty problem → dropped
        ],
    })
    out = review_timeline([_shot(1), _shot(2)], api_key="k")
    sevs = [i["severity"] for i in out["issues"]]
    assert sevs == ["high", "medium", "low"]                 # sorted, bogus normalized
    assert all(i["problem"] for i in out["issues"])          # empty-problem entry removed


def test_review_handles_llm_failure_gracefully(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("429")
    monkeypatch.setattr(dr, "_call_llm_json", _boom)
    out = review_timeline([_shot(1), _shot(2)], api_key="k")
    assert out["issues"] == [] and "unavailable" in out["overall"].lower()


def test_review_uses_smart_tier(monkeypatch):
    captured = {}
    def _fake(client, system_prompt, user_msg, **kwargs):
        captured.update(kwargs)
        return {"overall": "ok", "issues": []}
    monkeypatch.setattr(dr, "_call_llm_json", _fake)
    review_timeline([_shot(1), _shot(2)], api_key="k")
    assert captured.get("tier") == "smart"   # reasoning tier for the global pass

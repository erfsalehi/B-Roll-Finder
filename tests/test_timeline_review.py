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


# ── follow-up (verification) review ─────────────────────────────────────────

_PREV = {"overall": "x", "issues": [
    {"slot_id": 1, "severity": "high", "problem": "handshake reads as trust", "suggestion": "invoice"},
    {"slot_id": 2, "severity": "low", "problem": "eye-roll too harsh", "suggestion": "head shake"},
]}


def test_follow_up_rechecks_only_changed_shots(monkeypatch):
    seen = {}

    def _fake(client, system_prompt, user_msg, **k):
        seen["system"], seen["user"] = system_prompt, user_msg
        return {"overall": "Fixes landed.", "issues": [
            {"slot_id": 1, "severity": "medium", "problem": "still friendly"},  # changed → kept
            {"slot_id": 3, "severity": "high", "problem": "brand-new nitpick"},  # unchanged → dropped
        ]}
    monkeypatch.setattr(dr, "_call_llm_json", _fake)
    out = review_timeline([_shot(1), _shot(2), _shot(3)], api_key="k",
                          previous=_PREV, changed_slots={1})
    got = {(i["slot_id"], i["problem"], bool(i.get("carried"))) for i in out["issues"]}
    assert got == {(1, "still friendly", False),            # re-judged
                   (2, "eye-roll too harsh", True)}         # untouched → carried as-is
    assert "FOLLOW-UP REVIEW" in seen["system"]
    assert "Shot 1 [NEW CLIP]" in seen["user"] and "Shot 2 [NEW CLIP]" not in seen["user"]


def test_follow_up_resolves_a_fixed_shot(monkeypatch):
    monkeypatch.setattr(dr, "_call_llm_json", lambda *a, **k: {"overall": "ok", "issues": []})
    out = review_timeline([_shot(1), _shot(2)], api_key="k", previous=_PREV, changed_slots={1, 2})
    assert out["issues"] == []                              # both replaced, both now fine


def test_follow_up_keeps_carried_issues_when_llm_fails(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("429")
    monkeypatch.setattr(dr, "_call_llm_json", _boom)
    out = review_timeline([_shot(1), _shot(2)], api_key="k", previous=_PREV, changed_slots={1})
    assert [i["slot_id"] for i in out["issues"]] == [2]
    assert "not re-checked" in out["overall"]


def test_summary_shows_youtube_description_not_stock_credit():
    yt = {"slot_id": 1, "shot_intent": "cold air from vent", "selected_results": [{
        "title": "INSANE AC HACK!!", "source": "youtube",
        "description": "by Driver's Therapy — AC vent temperature test with a probe"}]}
    px = {"slot_id": 2, "shot_intent": "sunny road", "selected_results": [{
        "title": "Car On Sunny Road", "source": "pexels", "description": "By Jane Doe"}]}
    out = build_timeline_summary([yt, px])
    assert "AC vent temperature test" in out           # YouTube: footage hint shown
    assert "Jane Doe" not in out                         # stock author credit skipped

"""Fallback so every non-'none' shot gets queries even when the LLM omits them."""

from core.director import ensure_shot_queries, _fallback_queries
from core.director_youtube import seed_youtube_keywords


def test_empty_queries_get_filled_and_flagged():
    shots = [{
        "slot_id": 7, "priority": "medium",
        "shot_intent": "introduce the first engine on the list",
        "text": "The first engine is the Saab unit", "search_queries": [],
    }]
    ensure_shot_queries(shots, video_topic="unreliable car engines")
    assert shots[0]["search_queries"]            # no longer empty
    assert shots[0]["queries_fallback"] is True  # flagged for the UI hint


def test_existing_queries_are_preserved_and_not_flagged():
    shots = [{"slot_id": 1, "priority": "high",
              "shot_intent": "hook", "search_queries": ["car graveyard aerial"]}]
    ensure_shot_queries(shots, video_topic="cars")
    assert shots[0]["search_queries"] == ["car graveyard aerial"]
    assert "queries_fallback" not in shots[0]


def test_none_priority_shots_stay_empty():
    shots = [{"slot_id": 6, "priority": "none",
              "shot_intent": "transition", "search_queries": []}]
    ensure_shot_queries(shots, video_topic="cars")
    assert shots[0]["search_queries"] == []
    assert not shots[0].get("queries_fallback")


def test_whitespace_only_queries_treated_as_empty():
    shots = [{"slot_id": 2, "priority": "medium",
              "shot_intent": "show the engine bay", "search_queries": ["  ", ""]}]
    ensure_shot_queries(shots, video_topic="cars")
    assert any(q.strip() for q in shots[0]["search_queries"])
    assert shots[0]["queries_fallback"] is True


def test_fallback_anchors_to_topic():
    q = _fallback_queries({"shot_intent": "describe the overheating issue"},
                          video_topic="car engines")
    assert q and all(isinstance(s, str) and s for s in q)
    assert any("car engines" in s for s in q)


def test_fallback_feeds_youtube_keywords():
    shots = [{"slot_id": 3, "priority": "medium",
              "shot_intent": "explain the heat management issue",
              "text": "", "search_queries": []}]
    ensure_shot_queries(shots, video_topic="car engines")
    seed_youtube_keywords(shots)
    assert shots[0]["youtube_keywords"]  # seeded from the fallback queries

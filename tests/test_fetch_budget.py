"""Pre-fetch request estimate + per-source query caps."""

import core.director_search as ds
from core.director_search import estimate_stock_requests, fetch_director_footage


def _shots(n, q_per_shot=3):
    return [
        {"slot_id": i, "priority": "medium",
         "search_queries": [f"q{i}-{j}" for j in range(q_per_shot)]}
        for i in range(1, n + 1)
    ]


def test_estimate_counts_one_request_per_query():
    shots = _shots(4, q_per_shot=3)
    est = estimate_stock_requests(shots)
    assert est == {"pexels": 12, "pixabay": 12, "shots": 4}


def test_estimate_respects_pexels_cap():
    shots = _shots(4, q_per_shot=3)
    est = estimate_stock_requests(shots, pexels_max_queries=1)
    assert est["pexels"] == 4      # 1 query/shot
    assert est["pixabay"] == 12    # uncapped


def test_estimate_skips_none_priority_and_query_less_shots():
    shots = _shots(2, q_per_shot=2)
    shots.append({"slot_id": 99, "priority": "none", "search_queries": ["x", "y"]})
    shots.append({"slot_id": 100, "priority": "medium", "search_queries": []})
    est = estimate_stock_requests(shots)
    assert est == {"pexels": 4, "pixabay": 4, "shots": 2}


def test_cap_limits_actual_requests(monkeypatch):
    calls = {"pexels": 0, "pixabay": 0}

    def fake_fetch_query(query, source, key, n, errors, min_height=0):
        if source in calls:
            calls[source] += 1
        return []

    monkeypatch.setattr(ds, "_fetch_query", fake_fetch_query)
    monkeypatch.setenv("PEXELS_API_KEY", "fake")
    monkeypatch.setenv("PIXABAY_API_KEY", "fake")

    fetch_director_footage(
        _shots(4, q_per_shot=3),
        use_pexels=True, use_pixabay=True,
        use_youtube_search=False, use_youtube_api=False,
        pexels_max_queries=1, pixabay_max_queries=2,
    )
    assert calls["pexels"] == 4    # 1/shot × 4
    assert calls["pixabay"] == 8   # 2/shot × 4

import core.director_search as director_search
from core.director_search import (fetch_director_footage, search_youtube_classic,
                                  _fetch_query, clear_query_cache)


def test_search_youtube_classic_adapts_results(monkeypatch):
    monkeypatch.setattr(
        "core.director_search.search_youtube_single",
        lambda keyword, num_shorts=0, num_longs=3, errors=None, min_height=0: [
            {"title": "Repair clip", "url": "abc123def45", "is_short": False, "duration": 120}
        ],
    )

    results = search_youtube_classic("engine repair", num_results=1)

    assert results == [
        {
            "title": "Repair clip",
            "url": "https://www.youtube.com/watch?v=abc123def45",
            "page_url": "https://www.youtube.com/watch?v=abc123def45",
            "source": "youtube",
            "thumbnail": "",
            "description": "",
            "duration": 120,
            "is_short": False,
            "width": None,
            "height": None,
            "available_resolutions": [],
            "quality": None,
            "file_size": None,
            "matched_query": "engine repair",
        }
    ]


def test_fetch_director_footage_uses_youtube_keywords_for_classic(monkeypatch):
    calls = []

    def fake_search(keyword, num_results, errors=None, min_height=0):
        calls.append((keyword, num_results))
        return [{
            "title": keyword,
            "url": f"https://youtube.test/{keyword.replace(' ', '-')}",
            "page_url": f"https://youtube.test/{keyword.replace(' ', '-')}",
            "source": "youtube",
            "matched_query": keyword,
        }]

    monkeypatch.setattr("core.director_search.search_youtube_classic", fake_search)

    shots = [{
        "slot_id": 1,
        "priority": "medium",
        "search_queries": ["stock query"],
        "youtube_keywords": ["plain youtube query", "second query"],
    }]

    updated = fetch_director_footage(
        shots,
        use_pexels=False,
        use_pixabay=False,
        use_youtube=True,
        youtube_mode="classic",
        youtube_search_num_results=2,
    )

    assert sorted(calls) == [("plain youtube query", 2), ("second query", 2)]
    assert sorted(r["matched_query"] for r in updated[0]["video_results"]) == [
        "plain youtube query",
        "second query",
    ]


def test_fetch_director_footage_can_use_api_and_classic_together(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "yt-key")

    def fake_api(query, api_key, num_results, errors=None, min_height=0):
        return [{
            "title": "api result",
            "url": "https://youtube.test/api",
            "source": "youtube",
            "matched_query": query,
        }]

    def fake_search(keyword, num_results, errors=None, min_height=0):
        return [{
            "title": "classic result",
            "url": "https://youtube.test/classic",
            "source": "youtube",
            "matched_query": keyword,
        }]

    monkeypatch.setattr("core.director_search.search_youtube_data_api", fake_api)
    monkeypatch.setattr("core.director_search.search_youtube_classic", fake_search)

    shots = [{
        "slot_id": 1,
        "priority": "medium",
        "search_queries": ["director api query"],
        "youtube_keywords": ["classic search query"],
    }]

    updated = fetch_director_footage(
        shots,
        use_pexels=False,
        use_pixabay=False,
        use_youtube_api=True,
        use_youtube_search=True,
        youtube_api_num_results=2,
        youtube_search_num_results=2,
    )

    assert sorted(r["matched_query"] for r in updated[0]["video_results"]) == [
        "classic search query",
        "director api query",
    ]


# ── query cache bounds ─────────────────────────────────────────────────────────

def test_query_cache_caches_and_is_clearable(monkeypatch):
    clear_query_cache()
    calls = []
    monkeypatch.setattr("core.director_search.search_pexels",
                        lambda q, key, n, errors=None: calls.append(q) or [{"url": f"u-{q}"}])

    _fetch_query("city", "pexels", "k", 3, [])
    _fetch_query("city", "pexels", "k", 3, [])      # cache hit — no second call
    assert calls == ["city"]

    clear_query_cache()
    _fetch_query("city", "pexels", "k", 3, [])      # cleared — refetches
    assert calls == ["city", "city"]
    clear_query_cache()


def test_query_cache_evicts_oldest_at_cap(monkeypatch):
    clear_query_cache()
    monkeypatch.setattr(director_search, "_QUERY_CACHE_MAXSIZE", 3)
    monkeypatch.setattr("core.director_search.search_pexels",
                        lambda q, key, n, errors=None: [{"url": f"u-{q}"}])

    for q in ("q1", "q2", "q3", "q4"):
        _fetch_query(q, "pexels", "k", 3, [])

    keys = {k[1] for k in director_search._query_cache}
    assert len(director_search._query_cache) == 3
    assert "q1" not in keys and "q4" in keys        # oldest evicted, newest kept
    clear_query_cache()

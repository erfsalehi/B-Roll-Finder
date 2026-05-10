from core.director_search import fetch_director_footage, search_youtube_classic


def test_search_youtube_classic_adapts_results(monkeypatch):
    monkeypatch.setattr(
        "core.director_search.search_youtube_single",
        lambda keyword, num_shorts, num_longs, errors: [
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
            "quality": None,
            "file_size": None,
            "matched_query": "engine repair",
        }
    ]


def test_fetch_director_footage_uses_youtube_keywords_for_classic(monkeypatch):
    calls = []

    def fake_search(keyword, num_results, errors):
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
        youtube_num_results=2,
    )

    assert sorted(calls) == [("plain youtube query", 2), ("second query", 2)]
    assert sorted(r["matched_query"] for r in updated[0]["video_results"]) == [
        "plain youtube query",
        "second query",
    ]

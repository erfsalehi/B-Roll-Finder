from core.director_youtube import seed_youtube_keywords


def test_seed_youtube_keywords_uses_existing_queries_and_clears_none_priority():
    shots = [
        {
            "priority": "medium",
            "search_queries": ["first stock query", "second stock query", "third"],
        },
        {
            "priority": "none",
            "search_queries": ["talking head"],
            "youtube_keywords": ["old"],
        },
    ]

    seed_youtube_keywords(shots, max_keywords=2)

    assert shots[0]["youtube_keywords"] == ["first stock query", "second stock query"]
    assert shots[1]["youtube_keywords"] == []

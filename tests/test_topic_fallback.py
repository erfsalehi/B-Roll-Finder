"""Topic fallback: shots that fetched nothing get a generic on-topic Pexels clip
so they carry real footage instead of relying on the render-time gap stretch."""

import core.pipeline as pipeline


def _cands(n=2):
    return [{"url": f"https://pex/{i}.mp4", "page_url": f"https://pex/p{i}",
             "source": "pexels", "title": f"clip {i}"} for i in range(n)]


def test_covers_empty_shots_round_robin(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "k")
    monkeypatch.setenv("FILL_EMPTY_WITH_TOPIC", "1")
    import core.keywords, core.stock_apis
    monkeypatch.setattr(core.keywords, "generate_fallback_queries",
                        lambda topic, key, n=4: ["car engine", "garage"])
    monkeypatch.setattr(core.stock_apis, "search_pexels",
                        lambda q, key, n=3, errors=None: _cands(2))

    shots = [
        {"slot_id": 1, "priority": "normal", "selected_results": [{"url": "x"}]},
        {"slot_id": 2, "priority": "normal", "selected_results": []},   # empty
        {"slot_id": 3, "priority": "none",   "selected_results": []},   # talking head
        {"slot_id": 4, "priority": "normal", "selected_results": []},   # empty
    ]
    n = pipeline.cover_empty_shots_with_topic(shots, video_topic="car repair", groq_key="g")
    assert n == 2                                   # only the two priority!=none empties
    assert shots[1]["selected_results"] and shots[1]["_topic_fallback"]
    assert shots[3]["selected_results"] and shots[3]["_topic_fallback"]
    assert shots[2]["selected_results"] == []        # priority=none left alone
    # round-robin gave the two empties different clips
    assert (shots[1]["selected_results"][0]["page_url"]
            != shots[3]["selected_results"][0]["page_url"])


def test_no_pexels_key_is_noop(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    shots = [{"slot_id": 1, "priority": "normal", "selected_results": []}]
    assert pipeline.cover_empty_shots_with_topic(shots, video_topic="t", groq_key="g") == 0
    assert shots[0]["selected_results"] == []


def test_disabled_by_flag(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "k")
    monkeypatch.setenv("FILL_EMPTY_WITH_TOPIC", "0")
    shots = [{"slot_id": 1, "priority": "normal", "selected_results": []}]
    assert pipeline.cover_empty_shots_with_topic(shots, video_topic="t", groq_key="g") == 0

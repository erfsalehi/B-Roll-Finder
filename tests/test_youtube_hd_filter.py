"""Throttled YouTube HD/SD filtering of fetched candidates."""

import core.stock_apis as stock_apis
from core.director_search import filter_youtube_sd_candidates, auto_fetch_plan


def _clear_auto_env(monkeypatch):
    for v in ("AUTO_USE_PEXELS", "AUTO_USE_YOUTUBE", "AUTO_PEXELS_NUM",
              "AUTO_YOUTUBE_NUM", "AUTO_MIN_HEIGHT"):
        monkeypatch.delenv(v, raising=False)


def test_auto_fetch_plan_pexels_on_pixabay_and_ytapi_off(monkeypatch):
    _clear_auto_env(monkeypatch)
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    p = auto_fetch_plan()
    assert p["use_pexels"] is True
    assert p["use_pixabay"] is False          # removed as an auto source
    assert p["use_youtube_api"] is False       # removed as an auto source
    assert p["use_youtube_search"] is True     # Pexels + YouTube classic by default
    assert p["min_height"] == 720


def test_auto_fetch_plan_env_overrides(monkeypatch):
    _clear_auto_env(monkeypatch)
    monkeypatch.setenv("PEXELS_API_KEY", "x")
    monkeypatch.setenv("AUTO_PEXELS_NUM", "5")
    monkeypatch.setenv("AUTO_USE_YOUTUBE", "1")
    monkeypatch.setenv("AUTO_YOUTUBE_NUM", "2")
    p = auto_fetch_plan()
    assert p["pexels_num_results"] == 5
    assert p["use_youtube_search"] is True and p["youtube_search_num_results"] == 2


def test_auto_fetch_plan_pexels_disabled_without_key(monkeypatch):
    _clear_auto_env(monkeypatch)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    p = auto_fetch_plan()
    assert p["use_pexels"] is False and p["use_pixabay"] is False


def _yt(url, definition=None):
    c = {"url": url, "source": "youtube"}
    if definition is not None:
        c["definition"] = definition
        c["definition_checked"] = True
    return c


def _fake_defs(mapping):
    def _f(urls, api_key, errors=None):
        return {u: mapping.get(u, "unknown") for u in urls}
    return _f


def test_drops_sd_keeps_hd_and_unknown(monkeypatch):
    monkeypatch.setattr(stock_apis, "fetch_youtube_definitions_batch",
                        _fake_defs({"a": "hd", "b": "sd", "c": "unknown"}))
    shots = [{"priority": "medium", "video_results": [_yt("a"), _yt("b"), _yt("c"),
                                                      {"url": "p", "source": "pexels"}]}]
    res = filter_youtube_sd_candidates(shots, api_key="k")
    urls = [c["url"] for c in shots[0]["video_results"]]
    assert "b" not in urls                 # SD dropped
    assert {"a", "c", "p"} <= set(urls)    # HD, unknown, and non-YouTube kept
    assert res == {"checked": 3, "hd": 1, "sd": 1, "unknown": 1, "removed": 1}


def test_skips_already_checked(monkeypatch):
    called = {"n": 0}
    def _f(urls, api_key, errors=None):
        called["n"] += 1
        return {u: "hd" for u in urls}
    monkeypatch.setattr(stock_apis, "fetch_youtube_definitions_batch", _f)
    shots = [{"priority": "medium", "video_results": [_yt("a", "hd")]}]  # already checked
    res = filter_youtube_sd_candidates(shots, api_key="k")
    assert res["checked"] == 0 and called["n"] == 0   # nothing to do, no API call


def test_caps_at_max_checks(monkeypatch):
    seen = {}
    def _f(urls, api_key, errors=None):
        seen["count"] = len(urls)
        return {u: "hd" for u in urls}
    monkeypatch.setattr(stock_apis, "fetch_youtube_definitions_batch", _f)
    shots = [{"priority": "medium", "video_results": [_yt(f"u{i}") for i in range(10)]}]
    filter_youtube_sd_candidates(shots, api_key="k", max_checks=4)
    assert seen["count"] == 4   # only the first 4 checked this run


def test_removes_sd_from_selected_results(monkeypatch):
    monkeypatch.setattr(stock_apis, "fetch_youtube_definitions_batch",
                        _fake_defs({"bad": "sd", "good": "hd"}))
    # selected_results holds the SAME dict refs as video_results, as the UI does.
    bad, good = _yt("bad"), _yt("good")
    shot = {"priority": "medium",
            "video_results": [bad, good],
            "selected_results": [bad, good]}
    filter_youtube_sd_candidates([shot], api_key="k")
    assert [c["url"] for c in shot["selected_results"]] == ["good"]


def test_no_api_key_returns_error(monkeypatch):
    res = filter_youtube_sd_candidates([{"video_results": [_yt("a")]}], api_key="")
    assert "error" in res and res["checked"] == 0


def test_drop_sd_false_marks_but_keeps(monkeypatch):
    monkeypatch.setattr(stock_apis, "fetch_youtube_definitions_batch",
                        _fake_defs({"a": "sd"}))
    shots = [{"priority": "medium", "video_results": [_yt("a")]}]
    res = filter_youtube_sd_candidates(shots, api_key="k", drop_sd=False)
    assert res["sd"] == 1 and res["removed"] == 0
    assert shots[0]["video_results"][0]["definition"] == "sd"   # tagged, not removed

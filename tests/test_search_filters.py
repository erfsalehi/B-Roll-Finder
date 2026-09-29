"""Search-time filters (Shorts, over-long YouTube, Pexels landscape/Full HD) and
fresh in-points when the same source is used twice."""

import xml.etree.ElementTree as ET

import core.output as output
import core.stock_apis as sa
import core.youtube as yt


# ── YouTube: Shorts + length ──────────────────────────────────────────────────

def test_short_needs_hashtag_not_just_the_word():
    assert yt._looks_like_short("https://youtube.com/shorts/abc", "Engine", 300)
    assert yt._looks_like_short("u", "Cold start #Shorts", 300)
    assert yt._looks_like_short("u", "cold start", 45)
    for title in ("Short ram intake install", "Short shifter review",
                  "Chip shortage explained", "Short circuit in the alternator"):
        assert not yt._looks_like_short("https://youtube.com/watch?v=x", title, 400), title


def test_search_skips_over_long_videos_and_keeps_short_titles(monkeypatch):
    monkeypatch.setenv("YT_MAX_CLIP_SECONDS", "1000")
    entries = [
        {"url": "https://www.youtube.com/watch?v=long0000000", "title": "24/7 stream",
         "duration": 5000},
        {"url": "https://www.youtube.com/watch?v=intake00000",
         "title": "Short ram intake install", "duration": 420},
        {"url": "https://www.youtube.com/watch?v=tagged00000", "title": "Launch #shorts",
         "duration": 90},
        {"url": "https://www.youtube.com/watch?v=shifter0000", "title": "Short shifter review",
         "duration": 600},
    ]
    monkeypatch.setattr(yt, "_extract_info_with_backoff", lambda opts, q: {"entries": entries})
    items = yt.search_youtube_single("bmw e90 mods", num_longs=3)
    assert [i["title"] for i in items] == ["Short ram intake install", "Short shifter review"]


# ── Pexels: server-side filters ───────────────────────────────────────────────

def test_pexels_asks_for_landscape_full_hd_with_spares(monkeypatch):
    seen = {}

    class _Resp:
        headers = {}

        def json(self):
            return {"videos": []}

    def _get(*a, **k):
        seen.update(k.get("params") or {})
        return _Resp()
    monkeypatch.setattr(sa, "_http_get_with_retry", _get)
    monkeypatch.setattr(sa._PEXELS_GATE, "blocked_for", lambda: 0)

    sa.search_pexels("mechanic engine bay", api_key="k", num_results=3)
    assert seen["orientation"] == "landscape"
    assert seen["size"] == "medium"
    assert seen["per_page"] == 6


# ── FCPXML: a reused source starts after the part already shown ───────────────

def test_unused_in_frame():
    assert output._unused_in_frame(0, 100, 1000, []) == 0
    assert output._unused_in_frame(0, 100, 1000, [(0, 100)]) == 100
    assert output._unused_in_frame(0, 100, 1000, [(0, 100), (100, 250)]) == 250
    assert output._unused_in_frame(500, 100, 1000, [(0, 100)]) == 500   # already clear
    assert output._unused_in_frame(0, 100, 150, [(0, 100)]) is None      # no room left


def _ins(xml: str) -> list:
    track = next(ET.fromstring(xml).iter("track"))
    items = sorted(track.findall("clipitem"), key=lambda ci: int(ci.findtext("start")))
    return [(int(ci.findtext("in")), int(ci.findtext("out"))) for ci in items]


def _reused_shots(**extra):
    first = {"url": "https://yt/same", "source": "youtube", "matched_query": "a", **extra}
    again = {"url": "https://yt/same", "source": "youtube", "matched_query": "c", **extra}
    shots = [
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 4.0, "selected_results": [first]},
        {"slot_id": 2, "timestamp": 4.0, "end_timestamp": 8.0,
         "selected_results": [{"url": "https://yt/other", "source": "youtube",
                               "matched_query": "b"}]},
        {"slot_id": 3, "timestamp": 8.0, "end_timestamp": 12.0, "selected_results": [again]},
    ]
    return shots, first, again


def test_reused_source_gets_fresh_in_point(monkeypatch):
    monkeypatch.setattr(output, "_get_media_duration", lambda p, fallback_duration=0: 600.0)
    monkeypatch.setattr(output, "_preferred_in_frame", lambda *a, **k: 0)
    shots, first, again = _reused_shots()
    spans = _ins(output.generate_fcpxml(shots, project_name="demo"))
    assert spans[0][0] == 0
    assert spans[2][0] == spans[0][1]        # starts where the first use ended
    assert again["in_rule"] == "reuse"
    assert first.get("in_rule") != "reuse"


def test_library_segment_reuse_keeps_its_in_point(monkeypatch):
    monkeypatch.setattr(output, "_get_media_duration", lambda p, fallback_duration=0: 600.0)
    monkeypatch.setattr(output, "_preferred_in_frame", lambda *a, **k: 0)
    shots, _first, _again = _reused_shots(library_segment_id=7)
    spans = _ins(output.generate_fcpxml(shots, project_name="demo"))
    assert spans[0][0] == 0 and spans[2][0] == 0

"""Source/quality selection rules: YouTube priority, no shorts, 1080 cap."""

from core.director_rank import prioritize_youtube, _is_youtube
from core.pipeline import drop_shorts, cap_quality, _is_short


# ── YouTube 70% priority ──────────────────────────────────────────────────────

def _shot(slot, cands):
    return {"slot_id": slot, "priority": "normal", "video_results": cands}


def test_prioritize_youtube_ratio_split():
    # 10 shots, each with a stock clip ranked first and a YouTube clip second.
    shots = [_shot(i, [{"source": "pexels", "url": f"p{i}"},
                       {"source": "youtube", "url": f"y{i}"}]) for i in range(10)]
    prioritize_youtube(shots, ratio=0.7)
    yt_first = sum(1 for s in shots if _is_youtube(s["video_results"][0]))
    assert yt_first == 7                      # exactly 70% lead with YouTube
    # The 30% that stay stock-first keep a valid stock lead.
    stock_first = [s for s in shots if not _is_youtube(s["video_results"][0])]
    assert len(stock_first) == 3


def test_prioritize_youtube_skips_when_no_youtube():
    shots = [_shot(1, [{"source": "pexels", "url": "p"}])]
    prioritize_youtube(shots, ratio=1.0)
    assert shots[0]["video_results"][0]["source"] == "pexels"   # unchanged


def test_prioritize_youtube_ratio_zero_noop():
    shots = [_shot(i, [{"source": "pexels", "url": f"p{i}"},
                       {"source": "youtube", "url": f"y{i}"}]) for i in range(5)]
    prioritize_youtube(shots, ratio=0.0)
    assert all(s["video_results"][0]["source"] == "pexels" for s in shots)


# ── no shorts ─────────────────────────────────────────────────────────────────

def test_is_short_detection():
    assert _is_short({"source": "youtube", "is_short": True})
    assert _is_short({"source": "youtube", "url": "https://youtube.com/shorts/abc"})
    assert _is_short({"source": "youtube", "duration": 30})
    # A short *stock* clip is legitimate and must NOT be treated as a Short.
    assert not _is_short({"source": "pexels", "duration": 8})
    assert not _is_short({"source": "youtube", "duration": 600})


def test_drop_shorts_removes_only_shorts():
    shots = [{"video_results": [
        {"source": "youtube", "url": "https://youtu.be/long", "duration": 300},
        {"source": "youtube", "url": "https://youtube.com/shorts/x", "duration": 20},
        {"source": "pexels", "url": "p", "duration": 6},
    ]}]
    removed = drop_shorts(shots)
    assert removed == 1
    urls = [c["url"] for c in shots[0]["video_results"]]
    assert "https://youtube.com/shorts/x" not in urls and "p" in urls


# ── 1080 download cap ─────────────────────────────────────────────────────────

def test_cap_quality_clamps_to_1080():
    assert cap_quality("2160") == "1080"
    assert cap_quality(1440) == "1080"
    assert cap_quality("1080p") == "1080"
    assert cap_quality("720") == "720"
    assert cap_quality("480p") == "480"
    assert cap_quality("Best") == "1080"   # no digits → default ceiling

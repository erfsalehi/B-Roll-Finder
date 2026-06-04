"""Pexels search returns only 1080p+ files, preferring exactly 1080."""

import core.stock_apis as sa


class _Resp:
    def __init__(self, payload):
        self._p = payload
        self.headers = {}

    def json(self):
        return self._p


def _vid(vid_id, files):
    return {
        "id": vid_id, "url": f"https://pexels.com/video/{vid_id}/", "duration": 12,
        "image": "thumb.jpg", "user": {"name": "A"},
        "video_files": [{"link": f"{vid_id}-{h}", "width": int(h * 16 / 9), "height": h,
                         "quality": "hd" if h >= 720 else "sd"} for h in files],
    }


def test_pexels_picks_1080_and_drops_sub_1080(monkeypatch):
    payload = {"videos": [
        _vid(1, [540, 720, 1080, 2160]),   # has 1080 → keep, pick 1080
        _vid(2, [360, 540, 720]),          # no >=1080 → dropped entirely
        _vid(3, [1080, 1440]),             # pick 1080 (closest qualifying)
    ]}
    monkeypatch.setattr(sa, "_http_get_with_retry", lambda *a, **k: _Resp(payload))
    monkeypatch.setattr(sa._PEXELS_GATE, "blocked_for", lambda: 0)

    results = sa.search_pexels("city", api_key="k", num_results=10)
    page_urls = {r["page_url"] for r in results}
    # video 2 dropped (no >=1080 file); videos 1 and 3 kept
    assert "https://pexels.com/video/1/" in page_urls
    assert "https://pexels.com/video/3/" in page_urls
    assert "https://pexels.com/video/2/" not in page_urls
    # every kept clip is at exactly 1080, and its download link is the 1080 file
    assert all(r["height"] == 1080 and r["width"] == 1920 for r in results)
    assert {r["url"] for r in results} == {"1-1080", "3-1080"}


def test_pexels_returns_empty_when_all_sub_1080(monkeypatch):
    payload = {"videos": [_vid(9, [360, 480, 720])]}
    monkeypatch.setattr(sa, "_http_get_with_retry", lambda *a, **k: _Resp(payload))
    monkeypatch.setattr(sa._PEXELS_GATE, "blocked_for", lambda: 0)
    assert sa.search_pexels("x", api_key="k") == []

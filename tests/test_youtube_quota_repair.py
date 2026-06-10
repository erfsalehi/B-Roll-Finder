"""Per-shot YouTube/Pexels quota + download-aware repair loop.

Covers the rule that replaced the old 70% YouTube bias (1 Pexels + 1 YouTube
for short shots; 2 Pexels + ceil(dur/5) YouTube otherwise, with a guaranteed
YouTube clip per shot) and the download repair loop that re-picks a live clip
when a selected one fails to download.
"""

import core.pipeline as p
from core.director_rank import (shot_source_quota, auto_select_top_candidates,
                                _is_youtube)


# ── quota rule ────────────────────────────────────────────────────────────────

def test_shot_source_quota_rules():
    assert shot_source_quota({"duration_needed_sec": 3}) == (1, 1)   # short
    assert shot_source_quota({"duration_needed_sec": 5}) == (2, 1)   # ceil(5/5)
    assert shot_source_quota({"duration_needed_sec": 6}) == (2, 2)   # ceil(6/5)
    assert shot_source_quota({"duration_needed_sec": 11}) == (2, 3)  # ceil(11/5)


def test_auto_select_guarantees_a_youtube_clip():
    shot = {"slot_id": 1, "priority": "normal", "duration_needed_sec": 3.0,
            "selected_results": [],
            "video_results": [
                {"url": "p1", "source": "pexels"},
                {"url": "p2", "source": "pexels"},
                {"url": "y1", "source": "youtube"},
            ]}
    auto_select_top_candidates([shot])
    assert any(_is_youtube(c) for c in shot["selected_results"])


def test_auto_select_forces_youtube_even_if_filtered_by_variety():
    # The only YouTube candidate was used by the previous shot (so it's in the
    # variety window) — the guarantee must still bind it for this shot.
    shots = [
        {"slot_id": 1, "priority": "normal", "duration_needed_sec": 3.0,
         "selected_results": [], "video_results": [{"url": "y", "source": "youtube"}]},
        {"slot_id": 2, "priority": "normal", "duration_needed_sec": 3.0,
         "selected_results": [],
         "video_results": [{"url": "p", "source": "pexels"},
                           {"url": "y", "source": "youtube"}]},
    ]
    auto_select_top_candidates(shots, lookback=3)
    assert any(_is_youtube(c) for c in shots[1]["selected_results"])


# ── drop undownloaded ─────────────────────────────────────────────────────────

def test_drop_undownloaded_removes_failed(tmp_path):
    good = tmp_path / "a.mp4"
    good.write_bytes(b"x" * 16)
    shot = {"slot_id": 1, "priority": "normal", "selected_results": [
        {"url": "a", "local_path": str(good)},
        {"url": "b", "_dl_failed": True},
        {"url": "c"},   # never produced a file
    ]}
    dropped = p.drop_undownloaded([shot])
    assert dropped == 2
    assert [c["url"] for c in shot["selected_results"]] == ["a"]


def test_purge_clips_by_url_clears_pool_and_selection():
    shot = {"slot_id": 1, "priority": "normal",
            "selected_results": [{"url": "dead"}, {"url": "live"}],
            "video_results": [{"url": "dead"}, {"url": "live"}, {"url": "x"}]}
    removed = p._purge_clips_by_url([shot], {"dead"})
    assert removed == 2
    assert [c["url"] for c in shot["selected_results"]] == ["live"]
    assert [c["url"] for c in shot["video_results"]] == ["live", "x"]


# ── download + repair loop ────────────────────────────────────────────────────

def test_download_and_repair_replaces_failed_youtube(monkeypatch, tmp_path):
    live_file = tmp_path / "good.mp4"
    live_file.write_bytes(b"x" * 32)

    shot = {"slot_id": 1, "priority": "normal", "duration_needed_sec": 3.0,
            "selected_results": [{"url": "dead", "source": "youtube"}]}
    shots = [shot]

    calls = {"n": 0}

    def fake_download(shots, project, quality="1080", progress=None,
                      should_cancel=None, max_workers=None):
        calls["n"] += 1
        for s in shots:
            for c in s.get("selected_results") or []:
                if c.get("url") == "dead":
                    c["_dl_failed"] = True
                    c["_dl_error"] = "Video unavailable"
                    c.pop("local_path", None)
                else:
                    c["local_path"] = str(live_file)
                    c["_dl_ok"] = True
                    c.pop("_dl_failed", None)
        return {"ok": 0, "failed": 0, "skipped": 0, "dir": str(tmp_path), "errors": []}

    def fake_repick(shots, slot_ids, groq_key=None, video_topic="",
                    errors=None, blacklist=None):
        # The dead clip was already purged; bind a fresh live YouTube clip.
        for s in shots:
            if s["slot_id"] in set(slot_ids):
                s["selected_results"] = [{"url": "live", "source": "youtube"}]
                s["auto_selected"] = True
        return 1

    monkeypatch.setattr(p, "download_selected_clips", fake_download)
    monkeypatch.setattr(p, "repick_failed_shots", fake_repick)

    report = p.download_and_repair(shots, "proj", rounds=2)

    assert calls["n"] >= 2               # initial + at least one repair pass
    assert report["repaired"] == 1
    assert report["dropped"] == 0
    assert report["ok"] == 1
    assert shot["selected_results"][0]["url"] == "live"


def test_download_and_repair_drops_unfixable(monkeypatch, tmp_path):
    shot = {"slot_id": 1, "priority": "normal", "duration_needed_sec": 3.0,
            "selected_results": [{"url": "dead", "source": "youtube"}]}

    def fake_download(shots, project, quality="1080", progress=None,
                      should_cancel=None, max_workers=None):
        for s in shots:
            for c in s.get("selected_results") or []:
                c["_dl_failed"] = True
                c["_dl_error"] = "Video unavailable"
        return {"ok": 0, "failed": 1, "skipped": 0, "dir": str(tmp_path), "errors": []}

    # Repick can't find a replacement (returns nothing, selection stays empty
    # after the dead clip is purged).
    monkeypatch.setattr(p, "download_selected_clips", fake_download)
    monkeypatch.setattr(p, "repick_failed_shots",
                        lambda *a, **k: 0)

    report = p.download_and_repair([shot], "proj", rounds=2)
    assert report["ok"] == 0
    assert not shot.get("selected_results")   # unfixable clip dropped, not dangling


def test_ensure_youtube_coverage_adds_youtube(monkeypatch):
    import core.director_search as ds
    import core.director_rank as dr
    import core.director_youtube as dy

    monkeypatch.setattr(dy, "seed_youtube_keywords",
                        lambda shots, **k: [s.setdefault("youtube_keywords", ["kw"])
                                            for s in shots])
    monkeypatch.setattr(ds, "search_youtube_classic",
                        lambda kw, num_results=4, errors=None:
                        [{"url": "yy", "source": "youtube", "title": "t"}])
    monkeypatch.setattr(dr, "rank_shot_candidates", lambda shots, **k: shots)

    shot = {"slot_id": 1, "priority": "normal", "duration_needed_sec": 3.0,
            "search_queries": ["q"],
            "video_results": [{"url": "p1", "source": "pexels"}],
            "selected_results": [{"url": "p1", "source": "pexels"}]}
    secured = p.ensure_youtube_coverage([shot])
    assert secured == 1
    assert any((c.get("source") or "").lower() == "youtube"
               for c in shot["selected_results"])

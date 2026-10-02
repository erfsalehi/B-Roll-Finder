"""Footage density (FOOTAGE_DENSITY): "dense" is today's behaviour (every shot gets its full per-shot
quota, a cut every ~2 s: the first real output had 32 cuts in 64 s). "balanced" / "lean": when the
best clip of a shot is a high-confidence match (core.confidence.clip_fit) AND long enough to cover
the shot, the shot keeps just that clip (balanced: plus one alternative) and the rest are never
downloaded. Anything less certain keeps the whole quota; manual picks are never touched."""

import pytest

import core.director_rank as dr
import core.pipeline as pl


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("FOOTAGE_DENSITY", "LEAN_CLIP_FIT", "LEAN_SECOND_FIT"):
        monkeypatch.delenv(k, raising=False)


def _yt(i, dur=600, **kw):
    return dict({"url": f"https://y/{i}", "source": "youtube", "duration": dur, "title": f"yt {i}"}, **kw)


def _px(i, dur=14, **kw):
    return dict({"url": f"https://p/{i}", "source": "pexels", "duration": dur, "title": f"px {i}",
                 "video_id": i}, **kw)


def _shot(pool, dur=6.0, slot=1):
    """A shot whose ranked candidates are ``pool`` (best first); the auto-selector fills it."""
    return {"slot_id": slot, "priority": "medium", "duration_needed_sec": dur,
            "video_results": pool, "selected_results": []}


def _select(shot, mode=None, monkeypatch=None):
    if mode:
        monkeypatch.setenv("FOOTAGE_DENSITY", mode)
    dr.auto_select_top_candidates([shot])
    return [c["url"] for c in shot["selected_results"]]


def _pool():
    return [_yt(1), _yt(2), _px(3), _px(4), _px(5), _yt(6)]


def test_dense_is_the_default_and_changes_nothing(monkeypatch):
    assert dr.footage_density() == "dense"
    monkeypatch.setenv("FOOTAGE_DENSITY", "nonsense")
    assert dr.footage_density() == "dense"
    urls = _select(_shot(_pool()))
    assert len(urls) == 4                                  # 6 s: 2 stock + 2 YouTube, as before


def test_lean_keeps_only_the_rankers_first_pick_when_it_covers_the_shot(monkeypatch):
    shot = _shot(_pool())
    assert _select(shot, "lean", monkeypatch) == ["https://y/1"]
    assert shot["footage_density"] == "lean: kept 1 of 4"


def test_balanced_keeps_the_best_plus_one_alternative(monkeypatch):
    urls = _select(_shot(_pool()), "balanced", monkeypatch)
    assert len(urls) == 2 and urls[0] == "https://y/1"


def test_a_clip_too_short_for_the_shot_is_not_trusted_alone(monkeypatch):
    pool = [_yt(1, dur=5.0)] + _pool()[1:]                 # 5 s of footage for a 6 s shot
    assert len(_select(_shot(pool), "lean", monkeypatch)) == 4
    pool = [_yt(1, dur=None)] + _pool()[1:]                # length unknown (YouTube Data API results)
    assert len(_select(_shot(pool), "lean", monkeypatch)) == 4


def test_a_best_pick_that_is_not_confident_keeps_the_whole_quota(monkeypatch):
    monkeypatch.setenv("FOOTAGE_DENSITY", "lean")
    pool = [_px(i) for i in range(10, 20)]                 # ten ranked candidates
    shot = _shot(pool)
    deep = [pool[6], pool[7], pool[8]]                     # variety / de-dup pushed the picks far down
    assert dr.lean_trim(shot, deep) == deep                # 0.55 - 0.10: nowhere near 0.70
    top = [pool[0], pool[7], pool[8]]
    assert dr.lean_trim(shot, top) == [pool[0]]            # the #1 pick alone clears it
    monkeypatch.setenv("LEAN_CLIP_FIT", "0.99")            # and the bar is tunable
    assert dr.lean_trim(shot, top) == top


def test_a_storyboard_rejected_best_clip_does_not_qualify(monkeypatch):
    pool = [_yt(1, sb_checked=True, sb_verdict="weak")] + _pool()[1:]
    assert len(_select(_shot(pool), "lean", monkeypatch)) == 4        # 0.60 + 0.15 - 0.15 < 0.70


def test_manual_picks_are_left_alone(monkeypatch):
    monkeypatch.setenv("FOOTAGE_DENSITY", "lean")
    shot = _shot(_pool())
    shot["selected_results"] = [_yt(1), _px(3), _px(4)]               # an editor chose these three
    dr.auto_select_top_candidates([shot])
    assert len(shot["selected_results"]) == 3 and "footage_density" not in shot


def test_a_dropped_stock_clip_stays_available_to_the_next_shot(monkeypatch):
    monkeypatch.setenv("FOOTAGE_DENSITY", "lean")
    a = _shot([_yt(1), _px(3), _px(4)], slot=1)
    b = _shot([_yt(2), _px(3), _px(4)], slot=2)
    dr.auto_select_top_candidates([a, b], lookback=0)
    assert [c["url"] for c in a["selected_results"]] == ["https://y/1"]
    assert [c["url"] for c in b["selected_results"]] == ["https://y/2"]


def test_lean_trim_is_idempotent_and_never_empties_a_shot(monkeypatch):
    monkeypatch.setenv("FOOTAGE_DENSITY", "lean")
    shot = _shot(_pool())
    dr.auto_select_top_candidates([shot])
    once = list(shot["selected_results"])
    assert dr.lean_trim(shot, once) == once
    assert dr.lean_trim(shot, []) == [] and dr.lean_trim(shot, [_px(9)]) == [_px(9)]


def test_a_trusted_library_clip_is_enough_on_its_own(monkeypatch):
    monkeypatch.setenv("FOOTAGE_DENSITY", "lean")
    lib = {"url": "https://lib/1", "source": "library", "library_segment_id": 7, "segment_trust": "used",
           "duration": 9.0, "title": "segment"}
    shot = _shot([lib, _yt(2), _px(3)], dur=5.0)
    assert [c["url"] for c in dr.lean_trim(shot, [lib, _yt(2), _px(3)])] == ["https://lib/1"]


def test_the_repair_loop_applies_the_same_density(monkeypatch):
    """A rejected clip is replaced under the same setting, not by a full quota again."""
    monkeypatch.setenv("FOOTAGE_DENSITY", "lean")
    import core.director_search as ds
    import core.director as director
    import core.director_youtube as dy
    monkeypatch.setattr(director, "ensure_shot_queries", lambda *a, **k: None)
    monkeypatch.setattr(dy, "seed_youtube_keywords", lambda *a, **k: None)
    monkeypatch.setattr(ds, "fetch_with_retries", lambda targets, errors=None: [
        t.update(video_results=[_yt(1), _yt(2), _px(3), _px(4)]) for t in targets])
    monkeypatch.setattr(pl, "drop_shorts", lambda t: None)
    monkeypatch.setattr(pl, "drop_long_videos", lambda t: None)
    monkeypatch.setattr(pl, "drop_vertical", lambda t: None)
    monkeypatch.setattr(dr, "rank_shot_candidates", lambda *a, **k: None)
    monkeypatch.setattr(pl, "_check_picked_footage", lambda *a, **k: None)
    shot = {"slot_id": 1, "priority": "medium", "duration_needed_sec": 6.0, "auto_selected": True,
            "selected_results": [{"url": "https://y/dead", "source": "youtube", "_dl_failed": True}]}
    pl._purge_clips_by_url([shot], {"https://y/dead"})
    pl.repick_failed_shots([shot], {1}, blacklist={"https://y/dead"})
    assert [c["url"] for c in shot["selected_results"]] == ["https://y/1"]


def test_the_bot_setting(monkeypatch):
    from bot import settings
    assert settings.DEFAULTS["footage_density"] == "dense"
    opt = next(o for o in settings.OPTIONS if o["key"] == "footage_density")
    assert opt["env"] == "FOOTAGE_DENSITY" and opt["choices"] == ["dense", "balanced", "lean"]

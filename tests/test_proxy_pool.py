"""Validated proxy pool: research, ensure, round-robin, mark-dead, refresh."""

import core.proxy_pool as pp


def _setup(monkeypatch, good, bad, size=3):
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setenv("PROXY_POOL_SIZE", str(size))
    monkeypatch.setattr(pp, "_raw_proxies", lambda: list(good) + list(bad))
    import core.youtube as yt

    def _fake_probe(url, p, timeout=10, use_cookies=False):
        return (p in good, "ok" if p in good else "dead")
    monkeypatch.setattr(yt, "probe_proxy", _fake_probe)
    pp._reset()


def test_pool_inactive_without_url(monkeypatch):
    monkeypatch.delenv("YT_DLP_PROXY_URL", raising=False)
    pp._reset()
    assert pp.pool_active() is False
    assert pp.get_proxy() == ""
    assert pp.ensure_working() == []


def test_research_keeps_only_working(monkeypatch):
    good = ["http://g1", "http://g2", "http://g3"]
    bad = [f"http://b{i}" for i in range(8)]
    _setup(monkeypatch, good, bad, size=2)
    pp.research(2)
    snap = pp.working_snapshot()
    assert len(snap) >= 2
    assert all(p in good for p in snap)


def test_ensure_working_fills_to_target(monkeypatch):
    good = ["http://g1", "http://g2", "http://g3", "http://g4"]
    _setup(monkeypatch, good, [], size=3)
    out = pp.ensure_working()
    assert len(out) == 3
    assert all(p in good for p in out)


def test_get_proxy_round_robins_working(monkeypatch):
    good = ["http://g1", "http://g2"]
    _setup(monkeypatch, good, [], size=2)
    pp.ensure_working()
    picks = [pp.get_proxy() for _ in range(4)]
    assert set(picks) == set(good)          # both used
    assert picks[0] == picks[2] and picks[1] == picks[3]   # cycles


def test_get_proxy_lazy_researches_when_empty(monkeypatch):
    good = ["http://g1"]
    _setup(monkeypatch, good, [], size=1)
    assert pp.working_snapshot() == []      # nothing validated yet
    assert pp.get_proxy() == "http://g1"    # researched on demand


def test_mark_dead_drops_from_pool(monkeypatch):
    good = ["http://g1", "http://g2", "http://g3"]
    _setup(monkeypatch, good, [], size=3)
    pp.ensure_working()
    victim = pp.working_snapshot()[0]
    pp.mark_dead(victim)
    assert victim not in pp.working_snapshot()


def test_refresh_clears_and_reresearches(monkeypatch):
    good = ["http://g1", "http://g2"]
    _setup(monkeypatch, good, [], size=2)
    pp.ensure_working()
    pp.mark_dead("http://g1")
    n = pp.refresh()                         # forgets dead+working, re-tests all
    assert n == 2
    assert set(pp.working_snapshot()) == set(good)


def test_research_validates_cookieless_by_default(monkeypatch):
    seen = {}
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.delenv("PROXY_VALIDATE_COOKIES", raising=False)
    monkeypatch.setattr(pp, "_raw_proxies", lambda: ["http://g1"])
    import core.youtube as yt

    def _probe(url, p, timeout=10, use_cookies=False):
        seen["uc"] = use_cookies
        return (True, "ok")
    monkeypatch.setattr(yt, "probe_proxy", _probe)
    pp._reset()
    pp.research(1)
    assert seen["uc"] is False          # cookie-less validation (high yield)


def test_research_uses_cookies_when_opted_in(monkeypatch):
    seen = {}
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setenv("PROXY_VALIDATE_COOKIES", "1")
    monkeypatch.setattr(pp, "_raw_proxies", lambda: ["http://g1"])
    import core.youtube as yt

    def _probe(url, p, timeout=10, use_cookies=False):
        seen["uc"] = use_cookies
        return (True, "ok")
    monkeypatch.setattr(yt, "probe_proxy", _probe)
    pp._reset()
    pp.research(1)
    assert seen["uc"] is True


def test_research_stops_on_cancel(monkeypatch):
    bad = [f"http://b{i}" for i in range(100)]
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.delenv("PROXY_RESEARCH_MAX", raising=False)
    monkeypatch.setattr(pp, "_raw_proxies", lambda: bad)
    import core.youtube as yt
    calls = {"n": 0}

    def _probe(url, p, timeout=10, use_cookies=False):
        calls["n"] += 1
        return (False, "dead")
    monkeypatch.setattr(yt, "probe_proxy", _probe)
    pp._reset()
    added = pp.research(5, should_cancel=lambda: True)   # cancel immediately
    assert added == 0
    assert calls["n"] < 100               # did NOT scan the whole list


def test_research_scans_whole_list_by_default(monkeypatch):
    # One good proxy hidden at the end of a long dead list — found because there's
    # no low cap any more (PROXY_RESEARCH_MAX defaults to unlimited).
    bad = [f"http://b{i}" for i in range(60)]
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.delenv("PROXY_RESEARCH_MAX", raising=False)
    monkeypatch.setattr(pp, "_raw_proxies", lambda: bad + ["http://gOOD"])
    import core.youtube as yt
    monkeypatch.setattr(yt, "probe_proxy",
                        lambda url, p, timeout=10, use_cookies=False: (p == "http://gOOD", ""))
    pp._reset()
    pp.research(1)
    assert pp.working_snapshot() == ["http://gOOD"]


def test_lazy_research_is_bounded(monkeypatch):
    monkeypatch.setenv("YT_DLP_PROXY_URL", "http://list")
    monkeypatch.setenv("PROXY_BACKGROUND_MAX", "10")
    bad = [f"http://b{i}" for i in range(50)]
    monkeypatch.setattr(pp, "_raw_proxies", lambda: bad)
    import core.youtube as yt
    calls = {"n": 0}

    def _probe(url, p, timeout=10, use_cookies=False):
        calls["n"] += 1
        return (False, "dead")
    monkeypatch.setattr(yt, "probe_proxy", _probe)
    pp._reset()
    assert pp.get_proxy() == ""            # none work
    assert calls["n"] <= 12                # bounded near PROXY_BACKGROUND_MAX, not 50


def test_stats_shape(monkeypatch):
    good = ["http://g1", "http://g2"]
    bad = ["http://b1"]
    _setup(monkeypatch, good, bad, size=2)
    pp.ensure_working()
    s = pp.stats()
    assert s["active"] is True
    assert s["working"] == 2
    assert s["raw"] == 3

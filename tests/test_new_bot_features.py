"""Pexels key rotation, empty-shot fill, log snapshot, new command predicates."""

import time


# ── Pexels multi-key rotation ─────────────────────────────────────────────────

def test_pexels_key_pool_order_and_dedup(monkeypatch):
    import core.stock_apis as s
    monkeypatch.setenv("PEXELS_API_KEY_2", "k2")
    monkeypatch.setenv("PEXELS_API_KEYS", "k3, k1")   # k1 duplicates the primary
    assert s.pexels_key_pool("k1") == ["k1", "k3", "k2"]


def test_search_pexels_rotates_on_rate_limit(monkeypatch):
    import core.stock_apis as s
    monkeypatch.setenv("PEXELS_API_KEY_2", "key2")
    monkeypatch.delenv("PEXELS_API_KEYS", raising=False)
    s._pexels_gates.clear()
    calls = []

    def fake_one(keyword, api_key, num_results=3, page=1):
        calls.append(api_key)
        if api_key == "key1":
            raise s._RateLimited(time.time() + 3600)   # first key exhausted
        return [{"url": "vid", "source": "pexels"}]

    monkeypatch.setattr(s, "_search_pexels_one", fake_one)
    out = s.search_pexels("car engine", "key1", num_results=2, errors=[])
    assert calls == ["key1", "key2"]          # rotated to the second key
    assert out and out[0]["url"] == "vid"


# ── fill empty shots (purge Shorts, refill, YouTube-first) ───────────────────

def test_fill_empty_shots_purges_short_and_refills(monkeypatch):
    import core.pipeline as P
    shot = {
        "slot_id": 1, "priority": "high", "duration_needed_sec": 4,
        "video_results": [
            {"url": "https://youtube.com/shorts/x", "source": "youtube", "is_short": True},
            {"url": "https://youtube.com/watch?v=y", "source": "youtube", "duration": 120},
        ],
    }
    shots = [shot]

    def fake_repair(shots, **k):
        # Simulate repair selecting the Short (top of the pool) for the empty shot.
        for s in shots:
            if not s.get("selected_results"):
                s["selected_results"] = [s["video_results"][0]]
        return 1

    monkeypatch.setattr(P, "repair_empty_shots", fake_repair)
    n = P.fill_empty_shots(shots, groq_key="k", passes=2)
    sel = shot.get("selected_results") or []
    assert n == 1
    assert sel and all(not P._is_short(c) for c in sel)     # Short purged
    assert sel[0]["url"].endswith("v=y")                     # refilled with the long clip


def test_fill_empty_shots_noop_when_all_filled(monkeypatch):
    import core.pipeline as P
    shots = [{"slot_id": 1, "priority": "high", "selected_results": [{"url": "a"}]}]
    called = {"repair": False}
    monkeypatch.setattr(P, "repair_empty_shots",
                        lambda *a, **k: called.__setitem__("repair", True) or 0)
    assert P.fill_empty_shots(shots) == 0
    assert called["repair"] is False        # nothing empty → repair never called


# ── log snapshot ──────────────────────────────────────────────────────────────

def test_snapshot_logs(tmp_path, monkeypatch):
    import bot.logsetup as L
    monkeypatch.setattr(L, "LOG_PATH", str(tmp_path / "bot.log"))
    with open(L.LOG_PATH, "w", encoding="utf-8") as f:
        f.write("hello log line\n")
    dest = str(tmp_path / "snap.txt")
    assert L.snapshot_logs(dest) == dest
    assert "hello log line" in open(dest, encoding="utf-8").read()


def test_snapshot_logs_none_when_absent(tmp_path, monkeypatch):
    import bot.logsetup as L
    monkeypatch.setattr(L, "LOG_PATH", str(tmp_path / "missing.log"))
    assert L.snapshot_logs(str(tmp_path / "snap.txt")) is None


# ── YouTube cookie-mode status ────────────────────────────────────────────────

def test_cookie_mode_flags_missing_file(tmp_path, monkeypatch):
    import core.youtube as y
    monkeypatch.setattr(y, "_cookies_broken", False)
    monkeypatch.setattr(y, "_cookies_search_root", lambda: str(tmp_path))  # empty
    monkeypatch.setenv("YT_COOKIE_FILE", "/nope/cookies.txt")
    ok, detail = y.cookie_mode()
    assert ok is False and "NOT FOUND" in detail


def test_cookie_mode_flags_browser_on_server(tmp_path, monkeypatch):
    import core.youtube as y
    monkeypatch.setattr(y, "_cookies_broken", False)
    monkeypatch.setattr(y, "_cookies_search_root", lambda: str(tmp_path))  # empty
    monkeypatch.delenv("YT_COOKIE_FILE", raising=False)
    monkeypatch.setenv("YT_COOKIE_BROWSER", "firefox")
    ok, detail = y.cookie_mode()
    assert ok is False and "won't work on a server" in detail


def test_cookie_mode_ok_with_file(tmp_path, monkeypatch):
    import core.youtube as y
    monkeypatch.setattr(y, "_cookies_broken", False)
    cookie = tmp_path / "cookies.txt"
    cookie.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("YT_COOKIE_FILE", str(cookie))
    ok, detail = y.cookie_mode()
    assert ok is True and "file" in detail


def test_cookie_autodetects_folder_and_beats_browser(tmp_path, monkeypatch):
    """A cookies/*.txt is found without YT_COOKIE_FILE, and wins over a leftover
    YT_COOKIE_BROWSER — exactly the 'put cookies.txt in a cookies/ folder' setup."""
    import core.youtube as y
    monkeypatch.setattr(y, "_cookies_broken", False)
    monkeypatch.delenv("YT_COOKIE_FILE", raising=False)
    monkeypatch.setenv("YT_COOKIE_BROWSER", "firefox")
    cdir = tmp_path / "cookies"
    cdir.mkdir()
    (cdir / "www.youtube.com_cookies.txt").write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setattr(y, "_cookies_search_root", lambda: str(tmp_path))
    ok, detail = y.cookie_mode()
    assert ok is True and "www.youtube.com_cookies.txt" in detail
    assert "cookiefile" in y._get_cookie_opts()   # file beats the browser env


# ── new command predicates ────────────────────────────────────────────────────

def test_redo_and_logs_predicates():
    import bot.telegram_bot as tb
    assert tb.is_redo_command("/redo") and tb.is_redo_command("/fill")
    assert tb.is_logs_command("/logs") and tb.is_logs_command("/log@Bot")
    # /redo is no longer an alias of /refine
    assert not tb.is_refine_command("/redo")
    assert not tb.is_redo_command("/refine")

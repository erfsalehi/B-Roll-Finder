"""Signed download-link tokens for the bot file server."""

import os
import time
import importlib


def _fresh(monkeypatch):
    monkeypatch.setenv("BOT_FILE_TOKEN_SECRET", "test-secret")
    monkeypatch.delenv("BOT_LINK_TTL", raising=False)
    from bot import fileserver as fs
    importlib.reload(fs)
    return fs


def test_token_round_trip(monkeypatch):
    fs = _fresh(monkeypatch)
    exp = int(time.time()) + 600
    tok = fs.sign_token("proj.zip", exp)
    assert fs.verify_token("proj.zip", exp, tok) is True


def test_token_rejects_tamper_and_expiry(monkeypatch):
    fs = _fresh(monkeypatch)
    exp = int(time.time()) + 600
    tok = fs.sign_token("proj.zip", exp)
    assert fs.verify_token("other.zip", exp, tok) is False     # different path
    assert fs.verify_token("proj.zip", exp + 1, tok) is False   # different expiry
    assert fs.verify_token("proj.zip", exp, "deadbeef") is False
    past = int(time.time()) - 1
    assert fs.verify_token("proj.zip", past, fs.sign_token("proj.zip", past)) is False


def test_build_link_inside_root_and_verifiable(monkeypatch):
    fs = _fresh(monkeypatch)
    abs_path = os.path.join(fs.DOWNLOADS_ROOT, "myvid.zip")
    url = fs.build_link(abs_path, "host.example", port=8770)
    assert url.startswith("http://host.example:8770/d/myvid.zip?")
    # Pull out e/t and verify they validate for this relpath.
    import urllib.parse
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert fs.verify_token("myvid.zip", int(q["e"][0]), q["t"][0]) is True


def test_build_link_rejects_path_outside_downloads(monkeypatch):
    fs = _fresh(monkeypatch)
    outside = os.path.abspath(os.path.join(fs.DOWNLOADS_ROOT, "..", "secret.txt"))
    assert fs.build_link(outside, "host", 8770) is None


# ── permanent links (the default) ─────────────────────────────────────────────

def test_links_are_permanent_by_default(monkeypatch):
    fs = _fresh(monkeypatch)
    assert fs.link_ttl() == 0
    url = fs.build_link(os.path.join(fs.DOWNLOADS_ROOT, "proj.zip"), "host", 8770)
    assert "e=0&" in url or url.endswith("e=0") or "e=0" in url
    import urllib.parse
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert int(q["e"][0]) == fs.NEVER
    assert fs.verify_token("proj.zip", fs.NEVER, q["t"][0]) is True


def test_permanent_token_still_bound_to_its_path(monkeypatch):
    """No expiry must not mean no authorisation — the signature still gates it."""
    fs = _fresh(monkeypatch)
    tok = fs.sign_token("proj.zip", fs.NEVER)
    assert fs.verify_token("other.zip", fs.NEVER, tok) is False
    assert fs.verify_token("proj.zip", fs.NEVER, "deadbeef") is False


def test_bot_link_ttl_restores_expiring_links(monkeypatch):
    monkeypatch.setenv("BOT_LINK_TTL", "600")
    fs = _fresh(monkeypatch)
    monkeypatch.setenv("BOT_LINK_TTL", "600")
    assert fs.link_ttl() == 600
    url = fs.build_link(os.path.join(fs.DOWNLOADS_ROOT, "proj.zip"), "host", 8770)
    import urllib.parse
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    exp = int(q["e"][0])
    assert exp > int(time.time())
    assert fs.verify_token("proj.zip", exp, q["t"][0]) is True


def test_ttl_env_is_lenient(monkeypatch):
    fs = _fresh(monkeypatch)
    monkeypatch.setenv("BOT_LINK_TTL", "not-a-number")
    assert fs.link_ttl() == fs.DEFAULT_TTL
    monkeypatch.setenv("BOT_LINK_TTL", "-5")
    assert fs.link_ttl() == 0


def test_expiry_note_wording(monkeypatch):
    fs = _fresh(monkeypatch)
    assert "doesn't expire" in fs.expiry_note()
    monkeypatch.setenv("BOT_LINK_TTL", str(24 * 3600))
    assert fs.expiry_note() == "link expires in 24h"


# ── listing what's downloadable ────────────────────────────────────────────────

def test_list_zips_newest_first_and_ignores_other_files(monkeypatch, tmp_path):
    fs = _fresh(monkeypatch)
    (tmp_path / "old.zip").write_bytes(b"x" * 10)
    (tmp_path / "notes.txt").write_text("ignore me")
    nested = tmp_path / "proj"
    nested.mkdir()
    (nested / "new.zip").write_bytes(b"y" * 20)
    os.utime(tmp_path / "old.zip", (1000, 1000))
    os.utime(nested / "new.zip", (2000, 2000))

    found = fs.list_zips(str(tmp_path))
    assert [f["name"] for f in found] == ["new.zip", "old.zip"]
    assert found[0]["rel"] == "proj/new.zip"
    assert found[1]["size"] == 10


def test_list_zips_missing_root_is_empty(monkeypatch, tmp_path):
    fs = _fresh(monkeypatch)
    assert fs.list_zips(str(tmp_path / "nope")) == []

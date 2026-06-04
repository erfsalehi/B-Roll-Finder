"""Signed download-link tokens for the bot file server."""

import os
import time
import importlib


def _fresh(monkeypatch):
    monkeypatch.setenv("BOT_FILE_TOKEN_SECRET", "test-secret")
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

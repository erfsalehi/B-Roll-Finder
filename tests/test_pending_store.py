"""Review-gate state survives a restart: _PENDING round-trips through disk."""

import os

from bot import pending_store


def test_save_load_roundtrip(tmp_path):
    path = os.path.join(tmp_path, "p.json")
    pending = {
        12345: {"project": "demo", "shots": [{"slot_id": 1, "selected_results": [{"url": "u"}]}],
                "qa": {"issues": []}, "topic": "t", "errors": [], "settings": {"quality": 1080}},
    }
    pending_store.save_pending(pending, path)
    out = pending_store.load_pending(path)
    # chat_id keys come back as ints, not strings.
    assert list(out.keys()) == [12345]
    assert out[12345]["project"] == "demo"
    assert out[12345]["shots"][0]["selected_results"][0]["url"] == "u"


def test_load_missing_returns_empty(tmp_path):
    assert pending_store.load_pending(os.path.join(tmp_path, "nope.json")) == {}


def test_load_corrupt_returns_empty(tmp_path):
    path = os.path.join(tmp_path, "bad.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert pending_store.load_pending(path) == {}


def test_save_tolerates_nonserializable(tmp_path):
    # default=str safety net — a stray object must not crash the save.
    path = os.path.join(tmp_path, "p.json")
    pending_store.save_pending({1: {"obj": object()}}, path)
    out = pending_store.load_pending(path)
    assert 1 in out

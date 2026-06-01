"""Export → merge sync for sharing clip libraries across machines."""

import json
import importlib
import numpy as np
import pytest


def _fresh_lib(tmp_path, monkeypatch, name):
    """A clip_library module bound to its own throwaway DB file."""
    from core import clip_library
    lib = importlib.reload(clip_library)
    monkeypatch.setattr(lib, "_DB_PATH", str(tmp_path / name), raising=False)
    lib.init_db()
    return lib


def _insert(lib, url, *, title="", shot="", embedding=True, usage=1):
    emb = np.ones(384, dtype=np.float32).tobytes() if embedding else None
    with lib._conn() as c:
        c.execute(
            """INSERT INTO clips
                 (project, shot_description, source, clip_url, clip_title,
                  embedding, usage_count, last_used_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("p", shot, "pexels", url, title, emb, usage, "", "2026-01-01"),
        )
        return c.execute("SELECT id FROM clips WHERE clip_url=?", (url,)).fetchone()["id"]


def _add_trim(lib, clip_id, shot, in_s, out_s, when):
    with lib._conn() as c:
        c.execute(
            """INSERT INTO clip_preferred_trims
                 (clip_id, shot_description, in_seconds, out_seconds, confirmed_at)
               VALUES (?,?,?,?,?)""",
            (clip_id, shot, in_s, out_s, when),
        )


def test_export_writes_valid_bundle(tmp_path, monkeypatch):
    lib = _fresh_lib(tmp_path, monkeypatch, "a.db")
    cid = _insert(lib, "http://x/1", title="Engine", shot="engine close-up")
    _add_trim(lib, cid, "engine close-up", 1.0, 5.0, "2026-01-02")

    out = tmp_path / "exp.json"
    res = lib.export_library(str(out))
    assert res["clips"] == 1 and res["trims"] == 1
    bundle = json.loads(out.read_text())
    assert bundle["format"] == "broll-clip-library"
    clip = bundle["clips"][0]
    assert clip["clip_url"] == "http://x/1"
    assert clip["embedding_b64"]                 # embedding survives
    assert clip["trims"][0]["in_seconds"] == 1.0  # trim nested under clip


def test_merge_adds_new_clip_and_is_searchable(tmp_path, monkeypatch):
    src = _fresh_lib(tmp_path, monkeypatch, "src.db")
    _insert(src, "http://x/1", title="Engine", shot="engine close-up")
    exp = tmp_path / "exp.json"
    src.export_library(str(exp))

    dst = _fresh_lib(tmp_path, monkeypatch, "dst.db")  # rebinds _DB_PATH
    res = dst.import_library(str(exp))
    assert res["added"] == 1 and res["updated"] == 0
    with dst._conn() as c:
        row = c.execute("SELECT clip_url, embedding FROM clips").fetchone()
    assert row["clip_url"] == "http://x/1"
    assert row["embedding"] is not None   # imported with embedding → searchable


def test_merge_dedups_known_clip_and_backfills_embedding(tmp_path, monkeypatch):
    # Source has the clip WITH an embedding…
    src = _fresh_lib(tmp_path, monkeypatch, "src.db")
    _insert(src, "http://x/1", title="Engine", shot="engine", usage=5)
    exp = tmp_path / "exp.json"
    src.export_library(str(exp))

    # …destination already has the same URL but NO embedding + lower usage.
    dst = _fresh_lib(tmp_path, monkeypatch, "dst.db")
    _insert(dst, "http://x/1", title="Engine", shot="engine", embedding=False, usage=1)
    res = dst.import_library(str(exp))

    assert res["added"] == 0 and res["updated"] == 1
    with dst._conn() as c:
        row = c.execute("SELECT embedding, usage_count FROM clips").fetchone()
        n = c.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    assert n == 1                       # not duplicated
    assert row["embedding"] is not None  # backfilled from the bundle
    assert row["usage_count"] == 5       # raised to the higher count


def test_merge_is_idempotent(tmp_path, monkeypatch):
    src = _fresh_lib(tmp_path, monkeypatch, "src.db")
    _insert(src, "http://x/1", shot="engine")
    exp = tmp_path / "exp.json"
    src.export_library(str(exp))

    dst = _fresh_lib(tmp_path, monkeypatch, "dst.db")
    dst.import_library(str(exp))
    res2 = dst.import_library(str(exp))  # second merge
    with dst._conn() as c:
        n = c.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
    assert n == 1 and res2["added"] == 0 and res2["updated"] == 1


def test_merge_trim_newest_wins(tmp_path, monkeypatch):
    src = _fresh_lib(tmp_path, monkeypatch, "src.db")
    cid = _insert(src, "http://x/1", shot="engine")
    _add_trim(src, cid, "engine", 2.0, 8.0, "2026-02-01")  # newer
    exp = tmp_path / "exp.json"
    src.export_library(str(exp))

    dst = _fresh_lib(tmp_path, monkeypatch, "dst.db")
    dcid = _insert(dst, "http://x/1", shot="engine")
    _add_trim(dst, dcid, "engine", 1.0, 4.0, "2026-01-01")  # older local
    dst.import_library(str(exp))
    with dst._conn() as c:
        row = c.execute(
            "SELECT in_seconds, out_seconds FROM clip_preferred_trims"
        ).fetchone()
    assert row["in_seconds"] == 2.0 and row["out_seconds"] == 8.0  # newer won


def test_import_rejects_foreign_file(tmp_path, monkeypatch):
    dst = _fresh_lib(tmp_path, monkeypatch, "dst.db")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "something-else", "clips": []}))
    res = dst.import_library(str(bad))
    assert "error" in res

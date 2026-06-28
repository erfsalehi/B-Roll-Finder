"""Bundle a finished project (clips + FCPXML) into a single zip for transfer."""

import os
import zipfile
import pytest
from core.output import zip_project, plan_project_chunks, zip_one_chunk


def test_zip_project_bundles_clips_and_xml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proj_dir = tmp_path / "downloads" / "my-proj" / "director"
    proj_dir.mkdir(parents=True)
    (proj_dir / "1-1-engine.mp4").write_bytes(b"clipdata")
    (tmp_path / "downloads" / "my-proj" / "my-proj.xml").write_text("<xmeml/>")

    res = zip_project("my-proj")
    assert res["files"] == 2 and res["size_bytes"] > 0
    with zipfile.ZipFile(res["path"]) as z:
        names = set(z.namelist())
    # Layout preserved relative to downloads/, so unzip recreates <proj>/...
    assert "my-proj/director/1-1-engine.mp4" in names
    assert "my-proj/my-proj.xml" in names


def test_zip_project_missing_folder_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        zip_project("nope")


def test_zip_project_excludes_itself(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "downloads" / "p" / "director"
    d.mkdir(parents=True)
    (d / "a.mp4").write_bytes(b"x")
    # Pre-existing zip in the project dir must not be packed into the new one.
    res = zip_project("p", out_path=str(tmp_path / "downloads" / "p" / "p.zip"))
    with zipfile.ZipFile(res["path"]) as z:
        assert not any(n.endswith(".zip") for n in z.namelist())


# ── chunked delivery ─────────────────────────────────────────────────────────

def _make_project(tmp_path, n_clips=5, clip_kb=200):
    """A project with a tiny XML, one overlay PNG, and ``n_clips`` clips."""
    proj = tmp_path / "downloads" / "proj"
    (proj / "director").mkdir(parents=True)
    (proj / "overlays").mkdir(parents=True)
    (proj / "proj.xml").write_text("<xmeml/>")
    (proj / "proj.shots.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
    (proj / "overlays" / "cap.png").write_bytes(b"P" * 1024)
    for i in range(1, n_clips + 1):
        (proj / "director" / f"{i}-1-clip.mp4").write_bytes(b"x" * (clip_kb * 1024))
    return proj


def test_plan_chunks_metadata_first_and_capped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path, n_clips=6, clip_kb=400)  # ~400KB clips
    # 1MB cap → ~2 clips per chunk; metadata + overlay lead chunk 0.
    plan = plan_project_chunks("proj", chunk_size_mb=1)
    assert plan["total_files"] == 9          # xml, srt, png, 6 clips
    assert len(plan["chunks"]) >= 3
    # First chunk leads with the project metadata so the XML arrives first.
    first = plan["chunks"][0]["files"]
    assert first[0] == "proj/proj.shots.srt" or first[0].endswith(".xml")
    assert any(f.endswith("proj.xml") for f in first)
    # No chunk exceeds the cap (except a lone oversized file, which we don't have).
    for c in plan["chunks"]:
        assert c["bytes"] <= plan["chunk_size_bytes"]


def test_oversized_clip_gets_its_own_chunk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    proj = tmp_path / "downloads" / "proj"
    (proj / "director").mkdir(parents=True)
    (proj / "proj.xml").write_text("<xmeml/>")
    (proj / "director" / "big.mp4").write_bytes(b"x" * (3 * 1024 * 1024))  # 3MB > 1MB cap
    plan = plan_project_chunks("proj", chunk_size_mb=1)
    big = [c for c in plan["chunks"] if any(f.endswith("big.mp4") for f in c["files"])]
    assert len(big) == 1 and big[0]["n"] == 1   # the oversized clip is alone


def test_zip_one_chunk_preserves_layout_and_purges_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path, n_clips=4, clip_kb=400)
    plan = plan_project_chunks("proj", chunk_size_mb=1)

    seen = set()
    for i in range(len(plan["chunks"])):
        res = zip_one_chunk(plan, i, delete_source=True)
        assert res["index"] == i and res["total"] == len(plan["chunks"])
        with zipfile.ZipFile(res["path"]) as z:
            names = z.namelist()
        # Arcnames stay relative to downloads/, so unzip rebuilds proj/director/*.
        assert all(n.startswith("proj/") for n in names)
        seen.update(names)
        # Source files for this chunk are gone (reclaimed), zip remains.
        for rel in plan["chunks"][i]["files"]:
            assert not os.path.exists(os.path.join("downloads", rel))

    # Every original file ended up in exactly one chunk.
    assert "proj/proj.xml" in seen
    assert sum(1 for n in seen if n.startswith("proj/director/")) == 4


def test_zip_one_chunk_keeps_source_when_not_purging(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_project(tmp_path, n_clips=2, clip_kb=100)
    plan = plan_project_chunks("proj", chunk_size_mb=100)  # all in one chunk
    zip_one_chunk(plan, 0, delete_source=False)
    # Without purge the source clips survive.
    assert os.path.exists(os.path.join("downloads", "proj", "director", "1-1-clip.mp4"))

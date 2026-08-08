"""Bundle a finished project (clips + FCPXML) into a single zip for transfer."""

import os
import zipfile
import pytest
from core.output import zip_project


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

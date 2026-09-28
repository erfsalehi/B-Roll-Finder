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


def _project(tmp_path, monkeypatch, names=("1-1-engine.mp4", "61-1-mechanic-servicing-car-ac.mp4")):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "downloads" / "p" / "director"
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_bytes(b"clip")
    return d


def test_zip_skips_an_unreadable_file_instead_of_failing(tmp_path, monkeypatch):
    """One bad extras clip used to abort the whole delivery with
    '[Errno 22] Invalid argument: .../61-1-mechanic-servicing-car-ac.mp4'."""
    _project(tmp_path, monkeypatch)
    real = zipfile.ZipInfo.from_file

    def _from_file(filename, *a, **k):
        if "61-1-" in filename:
            raise OSError(22, "Invalid argument", filename)
        return real(filename, *a, **k)
    monkeypatch.setattr(zipfile.ZipInfo, "from_file", staticmethod(_from_file))

    res = zip_project("p")
    assert res["files"] == 1
    assert len(res["skipped"]) == 1 and "61-1-mechanic" in res["skipped"][0]
    assert "Invalid argument" in res["skipped"][0]
    with zipfile.ZipFile(res["path"]) as z:
        assert z.testzip() is None
        assert z.namelist() == ["p/director/1-1-engine.mp4"]


def test_zip_tolerates_pre_1980_file_dates(tmp_path, monkeypatch):
    d = _project(tmp_path, monkeypatch, names=("1-1-old.mp4",))
    os.utime(d / "1-1-old.mp4", (200000, 200000))      # early 1970
    res = zip_project("p")
    assert res["files"] == 1 and res["skipped"] == []


def test_failed_zip_leaves_no_partial_file(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)

    def _boom(self, *a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(zipfile.ZipFile, "write", _boom)
    with pytest.raises(RuntimeError):
        zip_project("p")
    left = os.listdir(tmp_path / "downloads")
    assert left == ["p"]                                 # no p.zip, no p.zip.tmp

"""Tests for the FCP7 XML re-import → preferred-trim learning loop."""

import importlib

import pytest

from core import xml_reimport


# A minimal FCP7 XML mimicking what generate_fcpxml() produces after Premiere
# round-trips it: timebase 24 / NTSC (23.976 fps), one trimmed clipitem whose
# source in/out is 24 → 144 frames (1.001 s → 6.006 s).
SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE xmeml>
<xmeml version="4">
  <project><name>Demo</name><children>
    <sequence id="seq">
      <rate><timebase>24</timebase><ntsc>TRUE</ntsc></rate>
      <media><video><track>
        <clipitem id="clip-0-0">
          <name>1-1-engine-oil.mp4</name>
          <rate><timebase>24</timebase><ntsc>TRUE</ntsc></rate>
          <start>0</start><end>120</end>
          <in>24</in><out>144</out>
          <file id="file-1">
            <name>1-1-engine-oil.mp4</name>
            <pathurl>file://localhost/C:/proj/downloads/default/director/1-1-engine-oil.mp4</pathurl>
            <rate><timebase>24</timebase><ntsc>TRUE</ntsc></rate>
          </file>
        </clipitem>
      </track></video></media>
    </sequence>
  </children></project>
</xmeml>
"""


def test_parse_fcpxml_converts_frames_to_seconds_from_str():
    items = xml_reimport.parse_fcpxml(SAMPLE_XML)
    assert len(items) == 1
    it = items[0]
    assert it["name"] == "1-1-engine-oil.mp4"
    assert it["in_frame"] == 24 and it["out_frame"] == 144
    # 24 / 23.976 ≈ 1.001 ; 144 / 23.976 ≈ 6.006
    assert it["in_seconds"] == pytest.approx(1.001, abs=1e-3)
    assert it["out_seconds"] == pytest.approx(6.006, abs=1e-3)
    assert it["local_path"].replace("\\", "/").endswith(
        "proj/downloads/default/director/1-1-engine-oil.mp4"
    )


def test_parse_fcpxml_accepts_bytes():
    # Streamlit uploads arrive as bytes — this used to hit ET.parse(bytes).
    items = xml_reimport.parse_fcpxml(SAMPLE_XML.encode("utf-8"))
    assert len(items) == 1 and items[0]["out_frame"] == 144


@pytest.fixture
def temp_library(tmp_path, monkeypatch):
    """Point the Clip Library at a throwaway SQLite file."""
    from core import clip_library
    importlib.reload(clip_library)
    monkeypatch.setattr(clip_library, "_DB_PATH", str(tmp_path / "lib.db"))
    clip_library.init_db()
    return clip_library


def _insert_clip(lib, url, local_path, shot_description):
    """Insert a clip row directly (bypassing the embedding model)."""
    from datetime import datetime
    with lib._conn() as c:
        cur = c.execute(
            """INSERT INTO clips (clip_url, local_path, shot_description, source,
                                  clip_title, created_at)
               VALUES (?,?,?,?,?,?)""",
            (url, local_path, shot_description, "pexels", "Engine oil",
             datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def test_ingest_records_trim_and_export_applies_it(temp_library, monkeypatch):
    lib = temp_library
    # Library stores a *relative* path; the XML carries an *absolute* one — so
    # this exercises the basename-fallback resolver.
    clip_id = _insert_clip(
        lib,
        url="https://example.com/engine-oil.mp4",
        local_path="downloads/default/director/1-1-engine-oil.mp4",
        shot_description="engine oil warning",
    )

    summary = xml_reimport.ingest_reimported_xml(SAMPLE_XML.encode("utf-8"))
    assert summary["parsed"] == 1
    assert summary["matched"] == 1
    assert summary["recorded"] == 1
    assert summary["unmatched"] == []

    trim = lib.get_preferred_trim(clip_id, "engine oil warning")
    assert trim is not None
    assert trim["in_seconds"] == pytest.approx(1.001, abs=1e-3)

    # Export side: _preferred_in_frame should surface that learned in-point.
    from core import output
    monkeypatch.setattr(output, "clip_library", lib, raising=False)
    in_frame = output._preferred_in_frame(
        clip_url="https://example.com/engine-oil.mp4",
        filename="1-1-engine-oil.mp4",
        duration_frames=48,
        media_dur_frames=1000,
        fps=23.976,
    )
    # 1.001 s * 23.976 ≈ 24 frames
    assert in_frame == 24


def test_preferred_in_frame_clamps_when_slot_exceeds_media(temp_library, monkeypatch):
    lib = temp_library
    _insert_clip(
        lib,
        url="https://example.com/short.mp4",
        local_path="downloads/default/director/short.mp4",
        shot_description="tiny",
    )
    cid = lib.find_clip_by_path_or_url(clip_url="https://example.com/short.mp4")["id"]
    lib.record_trim(cid, "tiny", in_seconds=5.0, out_seconds=6.0)

    from core import output
    monkeypatch.setattr(output, "clip_library", lib, raising=False)
    # in≈120 frames, slot=48, media only 130 → 120+48 > 130 → must fall back to 0.
    in_frame = output._preferred_in_frame(
        clip_url="https://example.com/short.mp4", filename="short.mp4",
        duration_frames=48, media_dur_frames=130, fps=23.976,
    )
    assert in_frame == 0


def test_ingest_reports_unmatched_when_clip_not_in_library(temp_library):
    summary = xml_reimport.ingest_reimported_xml(SAMPLE_XML.encode("utf-8"))
    assert summary["parsed"] == 1
    assert summary["matched"] == 0
    assert summary["recorded"] == 0
    assert summary["unmatched"] == ["1-1-engine-oil.mp4"]

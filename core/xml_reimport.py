"""
FCP7 XML re-import parser.

Reads an XML that originated from generate_fcpxml() (and may have been
round-tripped through Premiere) and returns one dict per <clipitem> with
the source in/out converted to seconds. Read-only — no DB writes here.
"""

import os
import re
import urllib.parse
import xml.etree.ElementTree as ET


def _effective_fps(timebase: int, ntsc: bool) -> float:
    """FCP's NTSC flag means timebase * 1000/1001 (e.g. 24 → 23.976)."""
    if timebase <= 0:
        return 0.0
    return timebase * (1000.0 / 1001.0) if ntsc else float(timebase)


def _read_rate(elem: ET.Element | None) -> tuple[int, bool]:
    """Returns (timebase, ntsc). (0, False) if rate element is missing."""
    if elem is None:
        return 0, False
    rate = elem.find("rate")
    if rate is None:
        return 0, False
    tb_el = rate.find("timebase")
    ntsc_el = rate.find("ntsc")
    timebase = int(tb_el.text.strip()) if tb_el is not None and tb_el.text else 0
    ntsc = bool(ntsc_el is not None and ntsc_el.text and ntsc_el.text.strip().upper() == "TRUE")
    return timebase, ntsc


def _pathurl_to_local(pathurl: str) -> str:
    """
    Converts a file:// URI back to a local OS path. Handles file://localhost/
    and percent-encoding; returns '' if pathurl is empty or not a file URI.
    """
    if not pathurl:
        return ""
    parsed = urllib.parse.urlparse(pathurl)
    if parsed.scheme and parsed.scheme != "file":
        return ""
    path = urllib.parse.unquote(parsed.path or pathurl)
    if not parsed.scheme:
        return os.path.normpath(path)
    # Windows file:///C:/... → path is "/C:/..."; strip the leading slash.
    if re.match(r"^/[A-Za-z]:/", path):
        path = path[1:]
    return os.path.normpath(path)


def _int_or_none(elem: ET.Element | None) -> int | None:
    if elem is None or elem.text is None:
        return None
    try:
        return int(elem.text.strip())
    except ValueError:
        return None


def parse_fcpxml(xml_source: str | bytes | os.PathLike) -> list[dict]:
    """
    Parse an FCP7 XML and return a list of clipitem dicts:

        {
            "clipitem_id":  "clip-0-0",
            "name":         "S01-1-aerial-city.mp4",
            "file_id":      "file-1",
            "pathurl":      "file://localhost/C:/.../aerial-city.mp4",
            "local_path":   "C:\\...\\aerial-city.mp4",
            "timebase":     24,
            "ntsc":         True,
            "fps":          23.976023976023978,
            "start_frame":  0,           # timeline placement
            "end_frame":    144,
            "in_frame":     0,           # source cut points
            "out_frame":    144,
            "in_seconds":   0.0,
            "out_seconds":  6.006,
        }

    ``in_frame`` of -1 (FCP's "no trim" sentinel) is normalised to 0.
    Clipitems without a resolvable <file> are skipped silently.
    """
    if isinstance(xml_source, bytes):
        # Raw XML bytes (e.g. a Streamlit upload's .getvalue()).
        root = ET.fromstring(xml_source)
    elif isinstance(xml_source, str):
        # A markup string starts with '<'; anything else is a file path.
        if xml_source.lstrip().startswith("<"):
            root = ET.fromstring(xml_source)
        else:
            root = ET.parse(xml_source).getroot()
    else:
        # PathLike or an open file object.
        root = ET.parse(xml_source).getroot()

    # Build file-id → pathurl/name lookup from every <file> that has the data.
    # Premiere often inlines the file once and references it by id elsewhere.
    file_lookup: dict[str, dict] = {}
    for f in root.iter("file"):
        fid = f.get("id")
        if not fid:
            continue
        pathurl_el = f.find("pathurl")
        name_el = f.find("name")
        if pathurl_el is None and name_el is None:
            continue
        entry = file_lookup.setdefault(fid, {})
        if pathurl_el is not None and pathurl_el.text and "pathurl" not in entry:
            entry["pathurl"] = pathurl_el.text.strip()
        if name_el is not None and name_el.text and "name" not in entry:
            entry["name"] = name_el.text.strip()

    # Sequence-level rate is the fallback when a clipitem doesn't carry one.
    seq = root.find(".//sequence")
    seq_timebase, seq_ntsc = _read_rate(seq)

    results: list[dict] = []
    for ci in root.iter("clipitem"):
        file_el = ci.find("file")
        if file_el is None:
            continue
        file_id = file_el.get("id") or ""
        info = file_lookup.get(file_id, {})
        pathurl = info.get("pathurl", "")
        name = info.get("name") or (ci.findtext("name") or "")

        # Skip clipitems for which we never saw the file's pathurl — e.g.
        # title generators, color mattes, or referenced-but-missing media.
        if not pathurl:
            continue

        # Clipitem rate overrides sequence rate when present.
        ci_tb, ci_ntsc = _read_rate(ci)
        timebase = ci_tb or seq_timebase
        ntsc = ci_ntsc if ci_tb else seq_ntsc
        fps = _effective_fps(timebase, ntsc)
        if fps <= 0:
            continue

        in_frame = _int_or_none(ci.find("in"))
        out_frame = _int_or_none(ci.find("out"))
        start_frame = _int_or_none(ci.find("start"))
        end_frame = _int_or_none(ci.find("end"))

        # FCP's -1 means "no explicit trim" — treat as 0.
        if in_frame is None or in_frame < 0:
            in_frame = 0
        if out_frame is None or out_frame < 0:
            # Fall back to the placement length if source out is missing.
            if start_frame is not None and end_frame is not None and end_frame > start_frame:
                out_frame = in_frame + (end_frame - start_frame)
            else:
                continue

        results.append({
            "clipitem_id":  ci.get("id") or "",
            "name":         name,
            "file_id":      file_id,
            "pathurl":      pathurl,
            "local_path":   _pathurl_to_local(pathurl),
            "timebase":     timebase,
            "ntsc":         ntsc,
            "fps":          fps,
            "start_frame":  start_frame,
            "end_frame":    end_frame,
            "in_frame":     in_frame,
            "out_frame":    out_frame,
            "in_seconds":   in_frame / fps,
            "out_seconds":  out_frame / fps,
        })

    return results


def ingest_reimported_xml(xml_source: str | bytes | os.PathLike) -> dict:
    """
    Learn preferred trims from a Premiere-edited FCP7 XML.

    Parses the XML, resolves each clipitem back to a Clip Library row (by
    local path / URL / basename), and records the editor's in/out as that
    clip's preferred trim. Idempotent — re-ingesting the same XML just
    overwrites with the latest values.

    Returns a summary::

        {
            "parsed":    <clipitems with usable trims>,
            "matched":   <resolved to a library clip>,
            "recorded":  <trims written>,
            "unmatched": [name, ...],   # couldn't resolve (capped at 50)
        }

    Trims are keyed on the library clip's own stored ``shot_description`` so
    the recording and the export-time lookup always agree, independent of how
    the live shot is worded at export.
    """
    # Local import keeps core.xml_reimport import-light and avoids a cycle.
    from core import clip_library

    items = parse_fcpxml(xml_source)

    # Remember the source path for provenance when it's a real file, not
    # inline XML bytes/markup.
    src_path = ""
    if isinstance(xml_source, (str, os.PathLike)) and not (
        isinstance(xml_source, str) and xml_source.lstrip().startswith("<")
    ):
        src_path = os.fspath(xml_source)

    summary = {"parsed": len(items), "matched": 0, "recorded": 0, "unmatched": []}
    for it in items:
        local_path = it.get("local_path", "")
        row = clip_library.find_clip_by_path_or_url(
            local_path=local_path,
            filename=it.get("name", ""),
        )
        if not row:
            label = it.get("name") or os.path.basename(local_path) or it.get("clipitem_id", "?")
            if len(summary["unmatched"]) < 50:
                summary["unmatched"].append(label)
            continue
        summary["matched"] += 1
        if clip_library.record_trim(
            clip_id=row["id"],
            shot_description=row.get("shot_description", "") or "",
            in_seconds=it["in_seconds"],
            out_seconds=it["out_seconds"],
            source_xml_path=src_path,
        ):
            summary["recorded"] += 1

    return summary

import re
import xml.dom.minidom as minidom

from core.output import generate_fcpxml


def test_fcpxml_marker_value_escapes_quotes_and_xml_chars():
    xml = generate_fcpxml([
        {
            "slot_id": 1,
            "timestamp": 12,
            "shot_intent": 'show "quoted" tool & blade <close>',
            "search_queries": ['macro "steel" & sheath'],
            "selected_results": [
                {"url": "https://example.test/clip.mp4", "matched_query": "steel"}
            ],
        }
    ])

    marker = re.search(r'<comment>([^<]*)</comment>', xml)
    assert marker is not None
    assert "&quot;quoted&quot;" in marker.group(1)
    assert "&amp;" in marker.group(1)
    assert "&lt;close&gt;" in marker.group(1)


def test_fcpxml_sequence_is_fully_formed_so_premiere_builds_timeline():
    """Regression: without a sequence <duration>, <timecode>, and a <rate> inside
    the format, Premiere imports the clips into the bin but never lays them on the
    timeline. Lock those in so the 'empty timeline' bug can't return."""
    xml = generate_fcpxml([
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 4.0,
         "selected_results": [{"url": "https://x/a.mp4", "matched_query": "a"}]},
        {"slot_id": 2, "timestamp": 4.0, "end_timestamp": 9.0,
         "selected_results": [{"url": "https://x/b.mp4", "matched_query": "b"}]},
    ], project_name="demo")

    minidom.parseString(xml)                      # must be well-formed

    # Two clips actually placed on the video track.
    assert xml.count('<clipitem id="clip-') == 2

    # Sequence-level requirements for a buildable timeline.
    seq_dur = re.search(r"<duration>(\d+)</duration>", xml)
    assert seq_dur and int(seq_dur.group(1)) > 0
    assert "<timecode>" in xml and "<displayformat>" in xml

    # The format block (before the first <track>) must carry an editing rate.
    fmt = xml.split("<format>")[1].split("</format>")[0]
    assert "<rate>" in fmt and "<pixelaspectratio>" in fmt


def test_clip_pathurls_are_relative_and_portable():
    """Regression: an absolute pathurl bakes in the generating machine's path
    (e.g. the server's ``/app/downloads/...``), which exists on no editing box —
    so Premiere relinks every import, and on Windows reads ``file://localhost/app``
    as a UNC share and stalls 'locating media' forever. pathurls must be relative
    to the XML's own folder so the unzipped bundle (XML + director/) self-links."""
    xml = generate_fcpxml([
        {"slot_id": 1, "timestamp": 0.0, "end_timestamp": 4.0,
         "selected_results": [{"url": "https://x/a.mp4", "matched_query": "a"}]},
    ], project_name="demo")

    pathurls = re.findall(r"<pathurl>([^<]+)</pathurl>", xml)
    assert pathurls, "expected at least one clip pathurl"
    for u in pathurls:
        assert not u.startswith("file://"), f"pathurl must be relative, got {u}"
        assert not u.startswith("/"), f"pathurl must not be absolute, got {u}"
        assert "/app/" not in u, f"server/container path leaked into pathurl: {u}"
        # Clips live in the project's director/ subfolder, beside the XML.
        assert u.startswith("director/"), f"unexpected relative shape: {u}"


def test_pathurl_is_url_encoded():
    """A raw space in the path (e.g. a project under '.../B-Roll Finder/...')
    makes the file URI malformed and hangs Premiere on 'locating'. The pathurl
    must percent-encode spaces while keeping the scheme and drive colon."""
    from core.output import _get_premiere_safe_pathurl
    u = _get_premiere_safe_pathurl("C:/Users/x/B-Roll Finder/downloads/demo/a clip.mp4")
    assert " " not in u
    assert "%20" in u
    assert u.startswith("file://localhost/C:/")

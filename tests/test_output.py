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

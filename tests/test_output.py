import re

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

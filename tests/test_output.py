import re

from core.output import generate_fcpxml


def test_fcpxml_marker_value_escapes_quotes_and_xml_chars():
    xml = generate_fcpxml([
        {
            "timestamp": 12,
            "shot_intent": 'show "quoted" tool & blade <close>',
            "search_queries": ['macro "steel" & sheath'],
        }
    ])

    marker = re.search(r'<marker [^>]+value="([^"]*)"', xml)
    assert marker is not None
    assert "&quot;quoted&quot;" in marker.group(1)
    assert "&amp;" in marker.group(1)
    assert "&lt;close&gt;" in marker.group(1)

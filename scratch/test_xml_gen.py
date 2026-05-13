import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath("."))

from core.output import generate_fcpxml

# Mock data
shots = [
    {
        "slot_id": 1,
        "timestamp": 0,
        "end_timestamp": 5,
        "shot_intent": "Establishing shot of a city",
        "selected_results": [
            {
                "url": "https://example.com/video1",
                "matched_query": "city skyline",
                "title": "City Skyline"
            }
        ]
    },
    {
        "slot_id": 2,
        "timestamp": 10, # Gap of 5s
        "end_timestamp": 15,
        "shot_intent": "Close up of a person",
        "selected_results": [
            {
                "url": "https://example.com/video2",
                "matched_query": "person face",
                "title": "Person Face"
            }
        ]
    }
]

# We need to mock the files existing so ffprobe doesn't just return defaults or we see the behavior
# But since I'm just checking the XML structure and URI format, it's fine if ffprobe returns defaults.

xml = generate_fcpxml(shots, project_name="test_project")
print(xml)

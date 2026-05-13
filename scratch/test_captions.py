import os
from core.captions import create_text_overlay
from core.output import generate_fcpxml, sec_to_frames

def test_captions_and_xml():
    # 1. Test PNG generation
    ov_dir = "downloads/test_project/overlays"
    os.makedirs(ov_dir, exist_ok=True)
    fname = os.path.join(ov_dir, "test_caption.png")
    
    print(f"Generating test PNG at {fname}...")
    create_text_overlay(
        "TEST $10,000 SAVED",
        fname,
        font_size=150,
        color="#FFD700", # Gold
        shadow_color="#000000",
        y_position=800
    )
    
    if os.path.exists(fname):
        print("✅ PNG generated successfully.")
    else:
        print("❌ PNG generation failed.")
        return

    # 2. Test XML generation with overlays
    shots = [
        {"timestamp": 0, "slot_id": 1, "shot_intent": "Talking head", "selected_results": [{"url": "http://test.com/1", "matched_query": "man talking"}]}
    ]
    overlays = [
        {"start_sec": 1.0, "end_sec": 4.0, "highlight_text": "TEST $10,000 SAVED", "filepath": os.path.abspath(fname), "animation": "Fade In/Out"}
    ]
    
    print("Generating XML with overlays...")
    xml = generate_fcpxml(shots, project_name="test_project", overlays=overlays)
    
    if "<track>" in xml and "test_caption.png" in xml and "opacity" in xml:
        print("✅ XML generated with multi-track and opacity keyframes.")
        # print(xml) # Optional: print a snippet
    else:
        print("❌ XML generation failed to include overlays correctly.")
        # print(xml)

if __name__ == "__main__":
    test_captions_and_xml()

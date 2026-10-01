"""Live check of the Python -> Remotion card render path and the V3 XML track.

Renders three real cards (a picture card with a detail right panel, a numbered step card
and a text-only step card) through motion_cards.render_plan with the real Remotion
binary, then builds the FCPXML around them with synthetic footage and validates it with
the real ffprobe. Run from the repo root:
    venv/Scripts/python.exe scratch/live_render.py <pictures_dir> <work_dir>
<pictures_dir> holds b001-left.jpg / b001-right.jpg / b002-left.jpg (scratch/live_images.py)."""
import json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
os.environ.setdefault("REMOTION_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")

PICS = os.path.abspath(sys.argv[1])
WORK = os.path.abspath(sys.argv[2])
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)                       # project_dir() is cwd-relative ("downloads/")

from core import motion_cards as mc
from core.output import evaluate_fcpxml, generate_fcpxml

def beat(slot, start, end, trigger, label, right, left=None, right_img=None, step=None):
    assets = {}
    if left:
        assets["left"] = {"kind": "ai", "path": left, "model": "microsoft/mai-image-2.6-flash"}
    if right_img:
        assets["right"] = {"kind": "detail", "path": right_img}
    return {"id": f"b{slot:03d}", "template": "split_card", "trigger": trigger, "slot_ids": [slot],
            "start": start, "end": end, "enabled": True, "label": label, "step": step, "right": right,
            "confidence": {"footage": 0.31, "tier": "low"}, "reasons": ["live render check"],
            "assets": assets}

beats = [
    beat(2, 5.0, 9.0, "low_confidence", "CLEAN EXTERIOR", {"kind": "image"},
         left=os.path.join(PICS, "b001-left.jpg"), right_img=os.path.join(PICS, "b001-right.jpg")),
    beat(4, 15.0, 19.0, "step", "WATERPROOF THE ELECTRONICS", {"kind": "number", "text": "2", "sub": "METHOD"},
         left=os.path.join(PICS, "b002-left.jpg"), step={"number": 2, "kind": "METHOD"}),
    beat(6, 25.0, 29.0, "step", "CHECK THE ENGINE IS COOL", {"kind": "number", "text": "3", "sub": "STEP"},
         step={"number": 3, "kind": "STEP"}),
]

errors = []
t0 = time.time()
entries = mc.render_plan(beats, "live-render-test", errors=errors,
                         progress=lambda d, n: print(f"  rendered {d}/{n} ({time.time() - t0:.0f}s)"))
print("errors:", errors)
print("entries:", len(entries))
for e in entries:
    print("  ", os.path.basename(e["filepath"]), os.path.getsize(e["filepath"]) // 1024, "KB", e["start_sec"], e["end_sec"], "|", e["comment"])
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=codec_name,width,height,r_frame_rate,pix_fmt,duration", "-of", "json", e["filepath"]],
                        capture_output=True, text=True)
    print("     ", json.loads(pr.stdout)["streams"][0])

# second pass hits the content-hash cache
t1 = time.time()
mc.render_plan(beats, "live-render-test", errors=errors)
print(f"second render (cache): {time.time() - t1:.1f}s")

# synthetic footage so the XML points at real files
foot = os.path.join(WORK, "footage")
os.makedirs(foot, exist_ok=True)
shots = []
for i in range(0, 8):
    p = os.path.join(foot, f"shot{i}.mp4")
    if not os.path.exists(p):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=1920x1080:rate=24000/1001:duration=8", "-c:v", "libx264", "-pix_fmt", "yuv420p", p], check=True)
    shots.append({"slot_id": i, "timestamp": float(i * 5), "end_timestamp": float(i * 5 + 5),
                  "selected_results": [{"url": f"https://x/{i}.mp4", "matched_query": "q", "local_path": p,
                                        "source": "pexels"}]})
xml_dir = os.path.join(WORK, "xml")
os.makedirs(xml_dir, exist_ok=True)
xml = generate_fcpxml(shots, "live-render-test", cards=entries, xml_dir=xml_dir)
open(os.path.join(xml_dir, "live.xml"), "w", encoding="utf-8").write(xml)
rep = evaluate_fcpxml(xml, xml_dir=xml_dir, check_media=True)
print("XML ok:", rep["ok"], "| errors:", rep["errors"], "| warnings:", rep.get("warnings"))
print("tracks:", xml.split("<audio>")[0].count("<track>"), "| card clipitems:", xml.count('clipitem id="clip-card-'))

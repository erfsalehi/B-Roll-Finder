"""Experiment (2026-10-02): can a vision model catch the logos / watermarks / black bars / off-topic
frames seen in the bot's first real output (B-Roll Sequence.mp4)? Runs the storyboard provider chain
over 2-second frames (f_NNN.jpg, 640 px) with the narration spoken at that moment (narr.json).
    venv/Scripts/python.exe scratch/inspect_frames.py <dir with f_*.jpg and narr.json>
Result: see HANDOVER.md section 13."""
import glob, io, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scratch"))
os.chdir(ROOT)
import curl_shim; curl_shim.install()
from dotenv import load_dotenv; load_dotenv(".env")
from PIL import Image
from core import storyboard

D = sys.argv[1]
narr = json.load(open(os.path.join(D, "narr.json")))
TOPIC = "Car AC tips: 15 secrets that make a car's air conditioning feel twice as cold"
CARDS = {36, 38, 40, 46, 48, 54, 56, 58}          # motion-card frames: not footage
frames = []
for f in sorted(glob.glob(os.path.join(D, "f_*.jpg"))):
    t = (int(os.path.basename(f)[2:5]) - 1) * 2
    if t in CARDS:
        continue
    line = next((s["text"] for s in narr if s["start"] - 0.05 <= t + 1 <= s["end"] + 0.05), "")
    frames.append((t, f, line))

SYSTEM = """You review single frames from b-roll clips that an automated editor picked for a narrated video. For each numbered frame you get the narration line spoken at that moment.

For every frame return:
- "issues": any that apply from
  "logo_bumper" (a channel/studio/brand logo animation, intro or outro bumper, title card or end card — the frame is mostly a logo or branding, not footage),
  "watermark" (a burned-in channel name, TV network bug such as a peacock or channel logo in a corner, stock-agency watermark),
  "presenter" (a person talking to the camera: vlogger, reviewer, mechanic addressing the viewer),
  "black_bars" (the picture sits inside a black frame, or has thick black bars at the sides or top/bottom),
  "text_graphics" (big captions or graphics that are part of the source video),
  "low_quality" (blurry, tiny, very dark or abstract with nothing recognisable).
- "relevance": 0.0-1.0 — how well the frame could illustrate what the narration line is ABOUT in this video (its subject, or an illustration that fits the idea). 1.0 = clearly on topic; 0.5 = loosely related, generic; 0.2 or less = unrelated (a different subject, people posing, random scenery, a different car part).
- "note": at most 10 words on what the frame shows.
Judge only what is visible. Output ONLY valid JSON: {"frames": [{"id": 1, "issues": [], "relevance": 0.8, "note": "..."}]}. Include every id."""


def jpeg(path):
    im = Image.open(path).convert("RGB")
    im.thumbnail((512, 512))
    b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    return b.getvalue()


out = {}
for i in range(0, len(frames), 8):
    chunk = frames[i:i + 8]
    user = f"VIDEO TOPIC: {TOPIC}\n\n" + "\n".join(
        f'Frame {n}: narration: "{line[:160]}"' for n, (t, f, line) in enumerate(chunk, 1))
    data = storyboard._vision_json(SYSTEM, user, [jpeg(f) for _, f, _ in chunk])
    for r in (data or {}).get("frames", []):
        try:
            t = chunk[int(r["id"]) - 1][0]
        except Exception:
            continue
        out[t] = r
for t, f, line in frames:
    r = out.get(t, {})
    print(f"{t//60}:{t%60:02d}  rel={r.get('relevance')}  {','.join(r.get('issues') or []) or '-':32}  {r.get('note','')}")
json.dump({str(k): v for k, v in out.items()}, open(os.path.join(D, "inspect.json"), "w"), indent=1)

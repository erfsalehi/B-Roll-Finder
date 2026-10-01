"""Live check of the motion-card planner on the real VidRush narration (84 Whisper segments
-> shots; scratch/vidrush_transcript.json). Real LLM calls (cheap fast tier), no pictures, no
rendering. Run from the repo root:  venv/Scripts/python.exe scratch/live_plan.py
On the dev box (VPN) the curl shim routes openrouter.ai through curl; harmless elsewhere."""
import json, os, sys, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import curl_shim; curl_shim.install()
from dotenv import load_dotenv
load_dotenv(".env")

HERE = os.path.dirname(os.path.abspath(__file__))
segs = json.load(open(os.path.join(HERE, "vidrush_transcript.json")))

# One shot per segment (the director's shots are similar: 4-8 s of narration).
random.seed(7)
shots = []
for i, s in enumerate(segs, 1):
    pool = [{"url": f"https://y/{i}-{k}", "source": "youtube" if k % 2 == 0 else "pexels",
             "title": f"clip {i}-{k}"} for k in range(8)]
    weak = random.random() < 0.18                      # ~18% of picks came from deep in the ranking
    shot = {"slot_id": i, "text": s["text"], "timestamp": s["start"], "end_timestamp": s["end"],
            "duration_needed_sec": s["end"] - s["start"], "priority": "medium", "shot_type": "medium",
            "shot_intent": s["text"][:60], "selected_results": [pool[6] if weak else pool[0]],
            "video_results": pool}
    if weak:
        shot["tried_queries"] = ["a", "b", "c", "d", "e"]
    shots.append(shot)

qa = {"issues": [{"slot_id": i, "severity": "high"} for i in random.sample(range(1, len(shots) + 1), 5)]}

from core import scene_plan, confidence
errors, diag = [], {}
beats = scene_plan.plan_scene(shots, "Four proven ways to clean your car's engine bay, plus safety steps",
                              os.getenv("GROQ_API_KEY"), qa=qa, errors=errors, diag=diag)
print("errors:", errors)
print("diag:", {k: v for k, v in diag.items() if k != "confidence"}, "| tiers:", diag.get("confidence"))
print()
print("STEPS FOUND:")
for b in beats:
    if b["trigger"] == "step":
        s = next(x for x in shots if x["slot_id"] == b["slot_ids"][0])
        print(f"  #{b['slot_ids'][0]:>2} {b['start']:6.1f}-{b['end']:6.1f}  {b['step']['kind']} {b['step']['number']}  label={b['label']!r:28} right={b['right']}")
        print(f"        \"{s['text'][:90]}\"")
print()
print("OTHER CARDS:")
for b in beats:
    if b["trigger"] != "step":
        s = next(x for x in shots if x["slot_id"] == b["slot_ids"][0])
        print(f"  #{b['slot_ids'][0]:>2} {b['start']:6.1f}-{b['end']:6.1f}  {b['trigger']:15} conf={b['confidence']['footage']}  label={b['label']!r:26} right={b['right']}  "
              f"kind={b['subject_kind']} style={b['image_style']}")
        print(f"        \"{s['text'][:100]}\"  | query={b['image_query']!r}")
json.dump({"beats": beats, "shots": [{k: v for k, v in s.items() if k in ("slot_id", "text", "timestamp", "end_timestamp", "confidence")} for s in shots]},
          open(os.path.join(HERE, "live_plan_out.json"), "w"), indent=1)

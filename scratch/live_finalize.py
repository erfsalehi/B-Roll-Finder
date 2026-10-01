"""Integration check of the motion-card flow through core.pipeline with REAL services
(fast-tier LLM planner, Serper, vision rating, one-to-three AI images, Remotion render) and a
STUBBED footage download (synthetic clips). Covers: stage 9e planning -> review line -> a
simulated /refine re-plan -> finalize_project (re-plan, render, cards.txt, XML V3 track).

    venv/Scripts/python.exe scratch/live_finalize.py <work_dir>

Spend: ~7 Serper credits at most, <= $0.05 of images (AI_IMAGES_MAX_PER_VIDEO=3), a few cents of LLM."""
import json, os, subprocess, sys, time, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import curl_shim; curl_shim.install()
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
os.environ.setdefault("REMOTION_BROWSER_EXECUTABLE", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
os.environ["AI_IMAGES_MAX_PER_VIDEO"] = "3"
os.environ["CARD_MAX_FRACTION"] = "0.2"
os.environ["MOTION_CARDS_MODE"] = "auto"

WORK = os.path.abspath(sys.argv[1])
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)                                   # downloads/ lands here, not in the repo

from core import ai_images, motion_cards, pipeline as pl

PROJECT = "live-final-test"
TOPIC = "Four proven ways to clean your car's engine bay, plus safety steps"
segs = json.load(open(os.path.join(HERE, "vidrush_transcript.json")))[:22]
key = os.getenv("GROQ_API_KEY")

foot = os.path.join(WORK, "footage")
os.makedirs(foot, exist_ok=True)


def clip_file(i):
    p = os.path.join(foot, f"shot{i}.mp4")
    if not os.path.exists(p):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=1280x720:rate=24000/1001:duration=6", "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", p], check=True)
    return p


random.seed(3)
WEAK = {2, 7, 11, 16, 19}                        # shots whose pick came from deep in the ranking
shots = []
for i, s in enumerate(segs, 1):
    pool = [{"url": f"https://y/{i}-{k}.mp4", "source": "youtube" if k % 2 == 0 else "pexels",
             "title": f"clip {i}-{k}", "matched_query": "q", "local_path": clip_file(i)} for k in range(8)]
    shot = {"slot_id": i, "text": s["text"], "timestamp": s["start"], "end_timestamp": s["end"],
            "duration_needed_sec": s["end"] - s["start"], "priority": "medium", "shot_type": "medium",
            "shot_intent": s["text"][:60], "selected_results": [pool[6] if i in WEAK else pool[0]],
            "video_results": pool}
    if i in WEAK:
        shot["tried_queries"] = ["a", "b", "c", "d", "e"]
    shots.append(shot)

ai_images.reset_run()
errors, attempts = [], {}


def table(beats):
    for b in beats:
        a = (b.get("assets") or {})
        l, r = a.get("left"), a.get("right")
        print(f"  shot {b['slot_ids'][0]:>2} {b['start']:6.1f}-{b['end']:6.1f} {b['trigger']:14} "
              f"{'ON ' if b.get('enabled', True) else 'OFF'} {b.get('label')!r:30} right={(b.get('right') or {}).get('kind'):6} "
              f"left={(l or {}).get('kind', '-'):7} fit={(l or {}).get('fit')} kind={b.get('subject_kind')}"
              + ("" if b.get("enabled", True) else f"  <- {(b.get('reasons') or [''])[-1]}"))


# 1. stage 9e ------------------------------------------------------------------------------
t0 = time.time()
beats = pl.plan_motion_cards(shots, PROJECT, TOPIC, key, qa=None, errors=errors, attempts=attempts)
print(f"\n[1] stage 9e plan ({time.time() - t0:.0f}s)  attempts.cards =", attempts.get("cards"))
print("    review line:", motion_cards.format_line(beats, attempts.get("cards")))
table(beats)
print("    errors:", errors)
stats1 = ai_images.run_stats()
print("    AI so far:", stats1)

# 2. simulated /refine ---------------------------------------------------------------------
live_lows = [b for b in beats if b["trigger"] == "low_confidence" and b.get("enabled", True)]
better = live_lows[0]["slot_ids"][0] if live_lows else None
by = {s["slot_id"]: s for s in shots}
if better:                                       # the refine found strong footage for this shot
    by[better]["selected_results"] = [by[better]["video_results"][0]]
    by[better].pop("tried_queries", None)
emptied = next((s["slot_id"] for s in shots if s["slot_id"] not in {b["slot_ids"][0] for b in beats}
                and s["slot_id"] > 14), None)
if emptied:                                      # every download for this shot failed
    by[emptied]["selected_results"] = []
errors2 = []
before_paths = {b["slot_ids"][0]: ((b.get("assets") or {}).get("left") or {}).get("path") for b in beats}
beats2 = pl.plan_motion_cards(shots, PROJECT, TOPIC, key, qa=None, existing=beats, errors=errors2, attempts={})
print(f"\n[2] re-plan after refine: shot {better} got better footage, shot {emptied} lost all footage")
table(beats2)
ids2 = {b["slot_ids"][0]: b for b in beats2}
print("    improved shot still has a card:", better in ids2 and ids2[better].get("enabled", True),
      "| emptied shot has a card:", emptied in ids2, ids2.get(emptied, {}).get("trigger"))
kept_same = [sid for sid, p in before_paths.items() if p and sid in ids2 and
             ((ids2[sid].get("assets") or {}).get("left") or {}).get("path") == p]
print("    cards that kept their picture:", kept_same)
print("    AI spend delta:", round(ai_images.run_stats()["usd"] - stats1["usd"], 4), "| errors:", errors2)

# 3. finalize (download stubbed) -----------------------------------------------------------
pl.enforce_timeline = lambda *a, **k: {"ok": True}
pl.ensure_youtube_coverage = lambda *a, **k: None
pl.download_and_repair = lambda *a, **k: {"ok": True}
errors3 = []
t0 = time.time()
res = pl.finalize_project(shots, PROJECT, groq_key=key, video_topic=TOPIC, errors=errors3,
                          status=lambda m: print("    status:", m), scene_plan=beats2)
print(f"\n[3] finalize_project ({time.time() - t0:.0f}s): cards rendered={len(res['cards'])}, errors={errors3}")
for c in res["cards"]:
    print("   ", os.path.basename(c["filepath"]), round(c["start_sec"], 1), "-", round(c["end_sec"], 1), "|", c["comment"])
xml = open(res["xml_path"], encoding="utf-8").read()
print("    xml:", res["xml_path"])
print("    video tracks:", xml.split("<audio>")[0].count("<track>"), "| card clipitems:", xml.count('clipitem id="clip-card-'))
print("    cards.txt exists:", os.path.exists(os.path.join(WORK, "downloads", PROJECT, "cards.txt")))
print("\nFINAL AI spend:", ai_images.run_stats())
json.dump({"beats": beats2}, open(os.path.join(WORK, "plan.json"), "w"), indent=1, default=str)

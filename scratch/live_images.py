"""Live check of motion-card picture sourcing (Serper -> download -> vision rating -> one AI
generation) against the real services. Costs ~2 Serper credits + ~$0.015 (one image).
Run from the repo root:  venv/Scripts/python.exe scratch/live_images.py <out_dir>
On the dev box (VPN) the curl shim routes openrouter.ai through curl; harmless elsewhere."""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import curl_shim; curl_shim.install()
from dotenv import load_dotenv
load_dotenv(".env")

from core import ai_images, card_images, storyboard

OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "scratch/live_images_out")
os.makedirs(OUT, exist_ok=True)
print("vision configured:", storyboard.vision_configured())
print("ai health:", ai_images.health())
ai_images.reset_run()

TOPIC = "Four proven ways to clean your car's engine bay"


def run(name, beat, shot, **kw):
    errors = []
    t0 = time.time()
    got = card_images.source_images(beat, shot, OUT, topic=TOPIC, errors=errors, **kw)
    print(f"\n=== {name}  ({time.time() - t0:.0f}s)")
    for side in ("left", "right"):
        a = got.get(side)
        if a:
            print(f"  {side}: kind={a['kind']} fit={a.get('fit')} rated={a.get('rated')} "
                  f"note={a.get('note')!r} src={(a.get('page') or a.get('url') or a.get('model') or '')[:90]}")
            print(f"        file={a['path']}")
        else:
            print(f"  {side}: -")
    print("  ai:", got["ai"], "| fit:", got["fit"], "| errors:", errors)
    return got


# A: generic subject, no pre-fetched images -> fresh Serper search, vision rates, best wins
run("A generic + search",
    {"id": "b001", "label": "CLEAN EXTERIOR", "image_query": "engine bay degreaser spray cleaning",
     "image_prompt": "a car engine bay being sprayed with degreaser", "subject_kind": "generic",
     "image_style": "photo", "right": {"kind": "image"}},
    {"text": "Start by spraying degreaser across the whole engine bay and letting it soak for a few minutes.",
     "images": []})

# B: no search allowed, nothing in the shot -> must generate (generic)
run("B generic, generation only",
    {"id": "b002", "label": "WATER EXPOSED RISK",
     "image_query": "water on engine electronics", "subject_kind": "generic", "image_style": "photo",
     "image_prompt": "close photograph of water droplets on the electrical connectors of a car engine",
     "right": {"kind": "image"}},
    {"text": "Water and electronics don't mix, so every connector is a risk.", "images": []},
    allow_search=False)

# C: a named product, no search -> must NOT generate
run("C identifiable, no search (must stay empty, no AI)",
    {"id": "b003", "label": "GUNK DEGREASER", "image_query": "Simple Green Gunk engine degreaser bottle",
     "image_prompt": "a bottle of engine degreaser", "subject_kind": "identifiable", "image_style": "photo",
     "right": {"kind": "image"}},
    {"text": "I used Simple Green Gunk degreaser.", "images": []}, allow_search=False)

# D: a named product WITH search -> real photos only, vision must be strict about the brand
run("D identifiable + search",
    {"id": "b004", "label": "GUNK DEGREASER", "image_query": "Gunk engine degreaser bottle",
     "image_prompt": "a bottle of engine degreaser", "subject_kind": "identifiable", "image_style": "photo",
     "right": {"kind": "image"}},
    {"text": "I used Gunk engine degreaser.", "images": []})

print("\nAI run stats:", ai_images.run_stats())

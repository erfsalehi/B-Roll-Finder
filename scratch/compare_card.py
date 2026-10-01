"""Compare a rendered ImageCard clip with the reference clip's frames (reference frame = mine + 60).

Needs the reference frames, which are NOT in the repo. Make them once (the 4 s clip is
C:/Users/erfsa/Desktop/Downloaded Videos/Export/4 proven engine bay.mp4):

    mkdir -p scratch/ref_frames
    ffmpeg -i "<that clip>" -vf "select='gte(n,56)'" -vsync 0 -start_number 56 scratch/ref_frames/g_%03d.png

and render the template with the reference's own card photo (crop of frame 119, x 187-850, y 214-694)
and label CLEAN EXTERIOR, right={"kind":"none"}, durationSec=4 (see remotion/README or HANDOVER.md).
Run:  venv/Scripts/python.exe scratch/compare_card.py <rendered.mp4>
Last result (2026-10-01, final template): panel edge <=9 px, card width <=4 px, card centre <=4 px,
caption x-range 291-740 vs 292-741, mean |diff| 1.94/255 over the 60 comparable frames."""
import os, subprocess, sys
import numpy as np
from PIL import Image

SP = os.path.dirname(os.path.abspath(__file__))
MP4 = sys.argv[1]
MINE = f"{SP}/ref_mine_frames"
os.makedirs(MINE, exist_ok=True)
for f in os.listdir(MINE):
    os.remove(os.path.join(MINE, f))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", MP4, "-start_number", "0", f"{MINE}/m_%03d.png"], check=True)


def mine(n): return np.asarray(Image.open(f"{MINE}/m_{n:03d}.png").convert("RGB")).astype(int)
def ref(n): return np.asarray(Image.open(f"{SP}/ref_frames/g_{n:03d}.png").convert("RGB")).astype(int)


def measure(im):
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    row = im[100]; gr = row[:, 1] - row[:, 0]; idx = np.where(gr >= 5)[0]
    gx = int(idx[0]) if len(idx) else -1
    lum = 0.2126 * R + 0.7152 * G + 0.0722 * B
    left = lum[:, :(gx if gx > 0 else 1920)]
    ys, xs = np.where(left[:720] > 60)
    card = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) > 20 else None
    cy, cx = np.where(left[760:900] > 120)
    cap = (int(cx.min()), int(cy.min()) + 760, int(cx.max()), int(cy.max()) + 760) if len(cx) > 5 else None
    return gx, card, cap


print("f  | edge ref/mine | card w ref/mine | card cx,cy ref | mine          | caption bbox ref | mine")
for f in (3, 5, 8, 10, 13, 16, 20, 24, 28, 30, 34, 40, 59):
    a, b = ref(f + 60), mine(f)
    ga, ca, pa = measure(a); gb, cb, pb = measure(b)
    w = lambda c: (c[2] - c[0] + 1) if c else None
    c = lambda c: ((c[0] + c[2]) // 2, (c[1] + c[3]) // 2) if c else None
    print(f"{f:2d} | {ga:5d} {gb:5d}   | {str(w(ca)):>4} {str(w(cb)):>4}      | {str(c(ca)):12s} | {str(c(cb)):12s}  | {str(pa):24s} | {pb}")

a, b = ref(119), mine(59)
d = np.abs(a - b)
print("\nfinal frame mean|diff| all:", d.mean().round(2), "left:", d[:, :960].mean().round(2),
      "right panel:", d[:, 960:].mean().round(2), "| caption:", d[795:845, 280:760].mean().round(2),
      "card:", d[214:694, 187:850].mean().round(2))
# whole-sequence similarity (ignoring nothing): mean abs diff per frame
per = [np.abs(ref(f + 60) - mine(f)).mean() for f in range(0, 60)]
print("per-frame mean|diff| over the 60 comparable frames: mean %.2f  max %.2f (frame %d)" % (np.mean(per), np.max(per), int(np.argmax(per))))

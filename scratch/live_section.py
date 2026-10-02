"""Live check of section downloads (core.youtube.download_video(section=...)): fetch ONLY a
window of a YouTube video, compare with the full download, and look at what comes out.
Uses the project's own function with .env loaded (cookies / client settings as in production).
    venv/Scripts/python.exe scratch/live_section.py <out_dir> [url] [start] [end] [--full]
Default video: the Blender Foundation's Big Buck Bunny (public, CC-BY). Downloads a few MB;
--full also downloads the whole video for the size comparison."""
import os, subprocess, sys, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from dotenv import load_dotenv
load_dotenv(".env")
from core import youtube

args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = os.path.abspath(args[0])
URL = args[1] if len(args) > 1 else "https://www.youtube.com/watch?v=aqz-KE-bpKQ"
START = float(args[2]) if len(args) > 2 else 60.0
END = float(args[3]) if len(args) > 3 else 72.0
os.makedirs(OUT, exist_ok=True)


def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,pix_fmt",
                        "-of", "json", p], capture_output=True, text=True)
    d = json.loads(r.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    return {"dur": round(float(d.get("format", {}).get("duration", 0)), 2), "codec": s.get("codec_name"),
            "size": f"{s.get('width')}x{s.get('height')}"}


def run(name, section):
    path = os.path.join(OUT, name)
    for f in os.listdir(OUT):                      # a clean slate (yt-dlp resumes .part files)
        if f.startswith(os.path.splitext(name)[0]):
            os.remove(os.path.join(OUT, f))
    ts = {}
    t0 = time.time()
    youtube.download_video(URL, path, "1080", ts, no_audio=True, section=section)
    dt = time.time() - t0
    files = [f for f in os.listdir(OUT) if f.startswith(os.path.splitext(name)[0])]
    print(f"\n[{name}] status={ts.get('status')} error={str(ts.get('error_msg', ''))[:160]!r} {dt:.0f}s files={files}")
    for f in files:
        fp = os.path.join(OUT, f)
        print(f"   {f}: {os.path.getsize(fp) / 1e6:.1f} MB  {probe(fp)}")
    return ts.get("status") == "completed"


ok = run("section.mp4", (START, END))
print(f"asked for {START}-{END}s ({END - START:.0f}s)")
if "--full" in sys.argv:
    run("full.mp4", None)

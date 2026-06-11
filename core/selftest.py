"""Live preflight self-test for the footage pipeline.

A long voiceover can take many minutes to process — too long to discover only
*then* that yt-dlp can't download, the Groq key is dead, or ffmpeg is missing.
:func:`run_self_test` exercises every external dependency the real run needs,
with tiny throwaway work, so a problem surfaces in ~30s before any real project
starts. It deliberately uses the SAME functions the pipeline calls
(``search_youtube_classic`` / ``download_video`` / ``search_pexels`` /
``download_direct_video`` / ``transcribe_audio`` / ``generate_video_topic``) so a
pass means the actual path works, not a stand-in.

Returns a structured report::

    {"ok": bool,                 # every *critical* check passed
     "results": [{"name", "ok", "critical", "detail"}, ...],
     "errors": [str, ...]}       # detail lines for the checks that failed
"""

import os
import shutil
import subprocess
import tempfile
import time


# A clip must download at least this many bytes to count as a real success
# (guards against a 0-byte/placeholder file that "completed" without data).
_MIN_BYTES = 1024


def _ffmpeg_check():
    path = shutil.which("ffmpeg")
    if not path:
        return False, "MISSING — needed for downloads/normalize/transcription audio"
    return True, f"found ({path})"


def _llm_check():
    """Exercise the chat-LLM path the topic/shot-list/rank stages use."""
    from core.keywords import generate_video_topic
    key = os.getenv("GROQ_API_KEY", "")
    topic = generate_video_topic(
        "A short clip about a morning cup of coffee in a quiet kitchen.", key)
    topic = (topic or "").strip()
    if not topic:
        return False, "LLM returned an empty topic (key invalid or provider down?)"
    return True, f'ok — topic: "{topic[:60]}"'


def _make_test_audio():
    """Generate a 1s mono tone wav with ffmpeg (throwaway). Returns a path or None."""
    if not shutil.which("ffmpeg"):
        return None
    fd, path = tempfile.mkstemp(prefix="selftest_", suffix=".wav")
    os.close(fd)
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i",
           "sine=frequency=440:duration=1", "-ac", "1", "-ar", "16000", path]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=20)
        if r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    except Exception:
        pass
    try:
        os.remove(path)
    except OSError:
        pass
    return None


def _transcription_check():
    """Round-trip a 1s tone through the remote Whisper endpoint. A tone has no
    speech, so empty segments are fine — we only care the endpoint answered
    without an auth/connection error."""
    from core.transcription import transcribe_audio
    audio = _make_test_audio()
    if not audio:
        return False, "couldn't synthesize test audio (ffmpeg missing/failed)"
    try:
        segs = transcribe_audio(audio, os.getenv("GROQ_API_KEY", ""))
        n = len(segs or [])
        return True, f"endpoint reachable ({n} segment(s) from a silent test tone)"
    finally:
        try:
            os.remove(audio)
        except OSError:
            pass


def _pexels_search_check(state):
    from core.stock_apis import search_pexels
    key = os.getenv("PEXELS_API_KEY", "")
    if not key:
        return None, "no PEXELS_API_KEY (skipped)"
    errs: list = []
    results = search_pexels("nature", key, num_results=3, errors=errs)
    state["pexels_results"] = results or []
    if not results:
        return False, "search returned 0 results" + (f" — {errs[0]}" if errs else "")
    return True, f"{len(results)} result(s)"


def _pexels_download_check(state, out_dir):
    from core.direct_downloader import download_direct_video
    results = state.get("pexels_results") or []
    if not results:
        return None, "no Pexels result to download (skipped)"
    cand = results[0]
    url = cand.get("url")
    if not url:
        return None, "Pexels result had no direct URL (skipped)"
    out = os.path.join(out_dir, "pexels_test.mp4")
    ts: dict = {}
    download_direct_video(url, out, ts)
    if os.path.exists(out) and os.path.getsize(out) > _MIN_BYTES:
        return True, f"downloaded {os.path.getsize(out) // 1024} KB"
    return False, f"download failed: {ts.get('error_msg', 'no file produced')}"


def _yt_search_check(state):
    from core.director_search import search_youtube_classic
    errs: list = []
    results = search_youtube_classic("nature documentary", num_results=4, errors=errs)
    # Prefer the shortest videos for the download test (fastest, smallest).
    results = [r for r in results if r.get("url")]
    results.sort(key=lambda r: r.get("duration") or 1e9)
    state["yt_results"] = results
    if not results:
        return False, ("yt-dlp returned 0 results" +
                       (f" — {errs[0]}" if errs else " (search blocked / rate-limited?)"))
    return True, f"{len(results)} result(s) via yt-dlp"


def _yt_download_check(state, out_dir, quality="360", attempts=2):
    """Download the shortest YouTube results — the single most failure-prone
    step (cookies, format selection, Deno nsig, anti-bot). Passes if at least one
    of ``attempts`` clips downloads, mirroring the pipeline's repair philosophy
    (one dead video shouldn't condemn the whole capability)."""
    from core.youtube import download_video
    results = (state.get("yt_results") or [])[:attempts]
    if not results:
        return None, "no YouTube result to download (skipped)"
    oks, fails = [], []
    for i, cand in enumerate(results):
        out = os.path.join(out_dir, f"yt_test_{i}.mp4")
        ts: dict = {}
        title = (cand.get("title") or cand.get("url") or "?")[:40]
        try:
            download_video(cand["url"], out, quality, ts, no_audio=True)
        except Exception as e:
            fails.append(f"{title}: {type(e).__name__}: {e}")
            continue
        if os.path.exists(out) and os.path.getsize(out) > _MIN_BYTES:
            oks.append(f"{title} ({os.path.getsize(out) // 1024} KB)")
        else:
            fails.append(f"{title}: {ts.get('error_msg', 'no file produced')}")
    if oks:
        detail = f"{len(oks)}/{len(results)} downloaded — {oks[0]}"
        if fails:
            detail += f"  ·  {len(fails)} failed (e.g. {fails[0][:120]})"
        return True, detail
    return False, "all download attempts failed — " + ("; ".join(fails)[:300] or "unknown")


def _cookie_check():
    from core.youtube import cookie_mode
    ok, detail = cookie_mode()
    return ok, detail


def run_self_test(do_downloads: bool = True, quality: str = "360",
                  progress=None) -> dict:
    """Run the preflight checks and return a structured report (see module
    docstring). ``progress(label)`` is called before each check so the caller can
    show a live status. ``do_downloads=False`` runs the cheap checks only (no
    actual clip downloads)."""
    results: list = []
    state: dict = {}
    tmp_dir = tempfile.mkdtemp(prefix="brollselftest_")

    def _run(name, fn, critical):
        if progress:
            try:
                progress(name)
            except Exception:
                pass
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        # ok is None → the check was skipped (e.g. no key); never counts as a fail.
        results.append({"name": name, "ok": ok, "critical": critical,
                        "detail": detail, "secs": round(time.time() - t0, 1)})
        return ok

    try:
        _run("ffmpeg", _ffmpeg_check, critical=True)
        _run("LLM (topic/rank)", _llm_check, critical=True)
        _run("Transcription (Whisper)", _transcription_check, critical=True)
        _run("Pexels search", lambda: _pexels_search_check(state),
             critical=bool(os.getenv("PEXELS_API_KEY")))
        _run("YouTube search (yt-dlp)", lambda: _yt_search_check(state), critical=True)
        _run("YouTube cookies", _cookie_check, critical=False)
        if do_downloads:
            _run("Pexels download", lambda: _pexels_download_check(state, tmp_dir),
                 critical=bool(os.getenv("PEXELS_API_KEY")))
            _run("YouTube download (yt-dlp)",
                 lambda: _yt_download_check(state, tmp_dir, quality=quality),
                 critical=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    overall_ok = all(r["ok"] for r in results if r["critical"] and r["ok"] is not None)
    errors = [f"{r['name']}: {r['detail']}" for r in results
              if r["ok"] is False]
    return {"ok": overall_ok, "results": results, "errors": errors}

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


def _ytdlp_version_check():
    """Report the installed yt-dlp version and flag it if it looks stale —
    YouTube changes often break a yt-dlp that's more than a few weeks old, and a
    stale build is a common cause of every download failing at once."""
    import yt_dlp
    v = getattr(getattr(yt_dlp, "version", None), "__version__", "") or "?"
    try:
        from datetime import date
        parts = v.split("-")[0].split(".")
        vd = date(int(parts[0]), int(parts[1]), int(parts[2]))
        age = (date.today() - vd).days
        if age > 45:
            return False, (f"{v} — {age} days old; STALE. YouTube breaks old "
                           "yt-dlp — rebuild/upgrade (pip install -U yt-dlp).")
        return True, f"{v} ({age}d old)"
    except Exception:
        return True, v


def _ytdlp_update_check():
    """Report the daily yt-dlp auto-update marker so a *silently-failing* update
    is obvious. The marker is written even on failure (so a broken pip/network
    doesn't retry every run), which means a persistent ``error`` here is exactly
    the smoking gun for "yt-dlp is stuck stale and that's why downloads die"."""
    try:
        from core.app_utils import _read_update_marker
    except Exception as e:
        return None, f"update marker unreadable ({e})"
    m = _read_update_marker()
    if not m:
        return None, "no auto-update has run yet (marker absent — a restart triggers it)"
    last = m.get("last_check") or "?"
    err = (m.get("error") or "").strip()
    ver = m.get("version") or "?"
    if err:
        return False, f"last auto-update FAILED on {last}: {err[:140]}"
    detail = f"last ran {last}, version {ver}"
    if m.get("changed"):
        detail += " (upgraded — restart to load it)"
    try:
        from datetime import date
        age = (date.today() - date.fromisoformat(last)).days
        if age > 3:
            return False, (detail +
                           f" — but {age} days ago; the daily updater may not be running "
                           "(restart the bot, or the upgrade is erroring earlier)")
    except Exception:
        pass
    return True, detail


def _yt_search_check(state):
    from core.director_search import search_youtube_classic
    errs: list = []
    # A generic B-roll query (returns lots of short, downloadable stock-style
    # clips) rather than "documentary" terms that surface long, often-restricted
    # full episodes — keeps the download test representative of a real run.
    results = search_youtube_classic("city street", num_results=5, errors=errs)
    # Prefer the shortest videos for the download test (fastest, smallest).
    results = [r for r in results if r.get("url")]
    results.sort(key=lambda r: r.get("duration") or 1e9)
    state["yt_results"] = results
    if not results:
        return False, ("yt-dlp returned 0 results" +
                       (f" — {errs[0]}" if errs else " (search blocked / rate-limited?)"))
    return True, f"{len(results)} result(s) via yt-dlp"


def _yt_download_check(state, out_dir, quality="360", attempts=3):
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
    state["yt_download_failed"] = True
    return False, "all download attempts failed — " + ("; ".join(fails)[:300] or "unknown")


def _yt_client_probe_check(state):
    """When downloads fail, resolve formats under several cookie/player-client
    combos so the report shows whether ANY config works (→ force that client) or
    they all fail identically (→ the datacenter IP is blocked / needs a PO-token
    provider or proxy). Skipped when downloads already succeeded."""
    if not state.get("yt_download_failed"):
        return None, "skipped (downloads worked)"
    results = state.get("yt_results") or []
    url = results[0].get("url") if results else None
    if not url:
        return None, "no URL to probe"
    from core.youtube import probe_download_clients
    probe = probe_download_clients(url)
    working = [lbl for lbl, ok, _ in probe if ok]
    lines = []
    for lbl, ok, detail in probe:
        icon = "✅" if ok else ("➖" if ok is None else "❌")
        lines.append(f"   {icon} {lbl}: {detail}")
    from core.youtube import youtube_proxies
    proxies = youtube_proxies()
    if working:
        head = (f"a WORKING config exists → \"{working[0]}\". "
                "Set the matching client (or YT_DOWNLOAD_NO_COOKIES=1 if a "
                "no-cookies row works).")
    elif proxies:
        where = (_mask_proxy(proxies[0]) if len(proxies) == 1
                 else f"all {len(proxies)} YT_DLP_PROXY proxies")
        head = (f"EVERY client failed even via {where} → that proxy IP is ALSO "
                "blocked by YouTube. Use a different residential/mobile proxy. "
                "It is NOT a code bug.")
    else:
        head = ("EVERY client failed the same way → this host's IP is blocked by "
                "YouTube for playback. Fix: set YT_DLP_PROXY to a residential/mobile "
                "proxy (http://user:pass@host:port or socks5://host:port) and re-run "
                "/test. It is NOT a code bug — no client/cookie change helps an IP block.")
    return (bool(working), head + "\n" + "\n".join(lines))


def _cookie_check():
    from core.youtube import cookie_mode
    ok, detail = cookie_mode()
    return ok, detail


def _mask_proxy(url: str) -> str:
    """Hide any user:pass in a proxy URL before showing it in the report."""
    import re
    return re.sub(r"//[^@/]+@", "//***@", url or "")


def _youtube_proxy_check():
    from core.youtube import youtube_proxies
    ps = youtube_proxies()
    if not ps:
        return None, ("not set — fine unless YouTube blocks this host's IP; then set "
                      "YT_DLP_PROXY to a residential proxy")
    shown = ", ".join(_mask_proxy(p) for p in ps[:4])
    if len(ps) > 4:
        shown += f", +{len(ps) - 4} more"
    if len(ps) == 1:
        return True, f"YT_DLP_PROXY = {shown}"
    return True, f"{len(ps)} proxies (round-robin + failover): {shown}"


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
        _run("yt-dlp version", _ytdlp_version_check, critical=False)
        _run("yt-dlp auto-update", _ytdlp_update_check, critical=False)
        _run("YouTube search (yt-dlp)", lambda: _yt_search_check(state), critical=True)
        _run("YouTube cookies", _cookie_check, critical=False)
        _run("YouTube proxy", _youtube_proxy_check, critical=False)
        if do_downloads:
            _run("Pexels download", lambda: _pexels_download_check(state, tmp_dir),
                 critical=bool(os.getenv("PEXELS_API_KEY")))
            _run("YouTube download (yt-dlp)",
                 lambda: _yt_download_check(state, tmp_dir, quality=quality),
                 critical=True)
            # Only does work when the download failed — then it pinpoints whether
            # any client config works (vs. a systemic IP/PO-token block).
            _run("YouTube client probe", lambda: _yt_client_probe_check(state),
                 critical=False)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    overall_ok = all(r["ok"] for r in results if r["critical"] and r["ok"] is not None)
    errors = [f"{r['name']}: {r['detail']}" for r in results
              if r["ok"] is False]
    return {"ok": overall_ok, "results": results, "errors": errors}

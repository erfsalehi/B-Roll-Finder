"""Animated text overlays via Remotion.

Pipeline:
  1. extract_overlay_highlights() — a reasoning LLM (DeepSeek 'smart' tier, with
     Groq/OpenRouter fallback) reads the timestamped transcript and returns the
     headings/titles/stats/money/emphasis worth animating on screen.
  2. render_overlay_clips() — renders each highlight with the Remotion project
     (remotion/) to a transparent ProRes 4444 .mov (true alpha — no green screen),
     with the chosen sound effect baked in.
  3. build_overlays() — convenience wrapper returning overlay dicts in the shape
     core.output.generate_fcpxml already expects: {text, start_sec, end_sec,
     filepath, type, anim, sfx}. They drop straight onto the overlay (V2) track.

Rendering needs Node + the remotion/ deps installed (npm ci) and a headless
Chrome (Remotion fetches one on first run; the Docker image ships the libs).
Everything is best-effort: a failed extraction or render returns/skips quietly so
overlays never break the main job.
"""

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REMOTION_DIR = os.path.join(_REPO_ROOT, "remotion")
_PROMPT_PATH = os.path.join(_REPO_ROOT, "prompts", "overlay_highlights.txt")
# Rendered clips are cached by their props hash so identical overlays (same text,
# style, duration) — common across re-runs and /refine — are never re-rendered.
# Lives under .cache (a persistent volume) and is NOT wiped by the disk-clear.
_CACHE_DIR = os.path.join(_REPO_ROOT, ".cache", "overlay_clips")

_VALID_ANIM = {"title_card", "stat_pop", "money_count", "lower_third", "pop"}
_VALID_SFX = {"swoosh", "ding", "thud", "none"}


def _load_prompt() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def extract_overlay_highlights(segments: list = None, script_text: str = "",
                               groq_key: str = None) -> list:
    """Reasoning pass: pick the overlay-worthy moments from the narration.

    Prefers timestamped ``segments`` (Whisper) so overlay timing matches the
    audio; falls back to bare ``script_text``. Returns a list of dicts with
    ``text, type, anim, sfx, start, end, emphasis``. Empty list on any failure.
    """
    from core.keywords import _call_llm_json
    try:
        from groq import Groq
    except Exception:
        Groq = None

    system_prompt = _load_prompt()
    if segments:
        lines = [
            f"[{float(s['start']):.3f} - {float(s['end']):.3f}]: {s['text'].strip()}"
            for s in segments if s.get("text")
        ]
        user_content = ("Timestamped transcript (use these exact times):\n\n"
                        + "\n".join(lines))
    elif script_text:
        user_content = f"Script for analysis:\n\n{script_text}"
    else:
        return []

    client = Groq(api_key=groq_key) if (groq_key and Groq) else None
    try:
        res = _call_llm_json(client, system_prompt, user_content,
                             temperature=0.4, max_tokens=2000, tier="smart")
    except Exception as e:
        print(f"[overlays] highlight extraction failed: {e}")
        return []

    out = []
    for h in (res.get("overlays") or []):
        text = (h.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(h.get("start") or 0)
            end = float(h.get("end") or (start + 3))
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 3
        anim = h.get("anim") if h.get("anim") in _VALID_ANIM else "title_card"
        sfx = h.get("sfx") if h.get("sfx") in _VALID_SFX else "none"
        out.append({
            "text": text[:80],
            "type": (h.get("type") or "title"),
            "anim": anim,
            "sfx": sfx,
            "start": start,
            "end": end,
            "emphasis": h.get("emphasis") or "normal",
        })
    # Drop time-overlaps and crowding: keep overlays at least OVERLAY_MIN_GAP_SEC
    # apart (start-to-start) so the screen never feels busy.
    try:
        min_gap = float(os.getenv("OVERLAY_MIN_GAP_SEC", "6"))
    except (TypeError, ValueError):
        min_gap = 6.0
    out.sort(key=lambda o: o["start"])
    deduped, last_end, last_start = [], -1.0, -1e9
    for o in out:
        if o["start"] < last_end:                  # time overlap
            continue
        if o["start"] - last_start < min_gap:      # too close to the previous one
            continue
        deduped.append(o)
        last_end, last_start = o["end"], o["start"]
    return deduped


def _remotion_bin() -> str | None:
    """Path to the local remotion CLI binary, or None if deps aren't installed."""
    base = os.path.join(_REMOTION_DIR, "node_modules", ".bin")
    cand = os.path.join(base, "remotion.cmd" if os.name == "nt" else "remotion")
    return cand if os.path.exists(cand) else None


def remotion_available() -> bool:
    return _remotion_bin() is not None


def _sfx_exists(sfx: str) -> bool:
    if not sfx or sfx == "none":
        return False
    return os.path.exists(os.path.join(_REMOTION_DIR, "public", "sfx", f"{sfx}.mp3"))


def _sfx_variants(base: str) -> list:
    """Existing files for an SFX type — ``base.mp3`` and any numbered variants
    ``base<N>.mp3`` (e.g. swoosh, swoosh1, swoosh2). Returns basenames (no
    extension), sorted; empty list if none exist."""
    if not base or base == "none":
        return []
    sfx_dir = os.path.join(_REMOTION_DIR, "public", "sfx")
    out = []
    try:
        for fn in sorted(os.listdir(sfx_dir)):
            name, ext = os.path.splitext(fn)
            if ext.lower() != ".mp3":
                continue
            if name == base or (name.startswith(base) and name[len(base):].isdigit()):
                out.append(name)
    except OSError:
        return []
    return out


def _pick_sfx_variant(base: str, text: str) -> str:
    """Deterministically choose one existing variant for ``base``, keyed on the
    overlay text: different overlays get different sounds (less repetitive across
    a long video) while identical text always maps to the same file (so the
    render cache still hits). Returns 'none' when no file is present."""
    variants = _sfx_variants(base)
    if not variants:
        return "none"
    idx = int(hashlib.sha1((text or "").encode("utf-8")).hexdigest(), 16) % len(variants)
    return variants[idx]


def _props_for(h: dict, fps: int, color: str, accent: str) -> dict:
    dur = max(2.0, min(6.0, float(h["end"]) - float(h["start"])))
    # Pick a (deterministic) variant of the requested SFX, e.g. swoosh→swoosh2.
    sfx = _pick_sfx_variant(h.get("sfx", "none"), h.get("text", ""))
    return {
        "text": h["text"], "type": h.get("type", "title"),
        "anim": h.get("anim", "title_card"), "sfx": sfx,
        "durationSec": round(dur, 3), "fps": fps,
        "color": color, "accent": accent,
    }


def _cache_key(props: dict) -> str:
    return hashlib.sha1(json.dumps(props, sort_keys=True).encode("utf-8")).hexdigest()


def _render_or_reuse(binary: str, props: dict, out_path: str, timeout: int) -> bool:
    """Produce ``out_path`` (a transparent ProRes overlay) from cache if we've
    rendered identical props before, else render and cache it. Returns success."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cached = os.path.join(_CACHE_DIR, f"{_cache_key(props)}.mov")
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        try:
            shutil.copyfile(cached, out_path)
            return True
        except OSError:
            pass  # fall through to a fresh render

    props_fp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as pf:
            json.dump(props, pf)
            props_fp = pf.name
        cmd = [
            binary, "render", "src/index.ts", "Overlay", out_path,
            f"--props={props_fp}",
            "--codec=prores", "--prores-profile=4444",
            "--pixel-format=yuva444p10le", "--image-format=png", "--log=error",
        ]
        r = subprocess.run(cmd, cwd=_REMOTION_DIR, capture_output=True,
                           text=True, timeout=timeout)
        if r.returncode != 0 or not os.path.exists(out_path):
            print(f"[overlays] render failed for '{props['text'][:40]}': "
                  f"{(r.stderr or r.stdout or '')[-300:]}")
            return False
        try:
            shutil.copyfile(out_path, cached)   # populate cache for next time
        except OSError:
            pass
        return True
    except subprocess.TimeoutExpired:
        print(f"[overlays] render timed out for '{props['text'][:40]}'")
        return False
    except Exception as e:
        print(f"[overlays] render error for '{props['text'][:40]}': {e}")
        return False
    finally:
        if props_fp:
            try:
                os.remove(props_fp)
            except OSError:
                pass


def render_overlay_clips(highlights: list, out_dir: str, fps: int = 30,
                         color: str = "#FFFFFF", accent: str = "#FFD400",
                         timeout: int = 240) -> list:
    """Render each highlight to a transparent ProRes 4444 .mov, reusing cached
    renders and rendering several at once. Returns overlay dicts (``filepath``
    set) for the ones that succeeded, in timeline order."""
    binary = _remotion_bin()
    if not binary:
        print("[overlays] remotion deps not installed (run `npm install` in "
              "remotion/) — skipping overlay rendering.")
        return []
    os.makedirs(out_dir, exist_ok=True)
    try:
        workers = max(1, int(os.getenv("OVERLAY_RENDER_WORKERS", "2")))
    except (TypeError, ValueError):
        workers = 2

    def _one(idx_h):
        idx, h = idx_h
        props = _props_for(h, fps, color, accent)
        out_path = os.path.join(out_dir, f"overlay_{idx:03d}.mov")
        if not _render_or_reuse(binary, props, out_path, timeout):
            return None
        return (idx, {
            "text": h["text"], "type": h.get("type"), "anim": h.get("anim"),
            "sfx": props["sfx"], "start_sec": float(h["start"]),
            "end_sec": float(h["end"]), "filepath": out_path, "is_video": True,
        })

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_one, list(enumerate(highlights))):
            if r is not None:
                results.append(r)
    results.sort(key=lambda t: t[0])
    return [ov for _, ov in results]


def build_overlays(out_dir: str, segments: list = None, script_text: str = "",
                   groq_key: str = None, fps: int = 30) -> list:
    """Extract highlights then render them. Returns FCPXML-ready overlay dicts."""
    highlights = extract_overlay_highlights(segments, script_text, groq_key)
    if not highlights:
        return []
    return render_overlay_clips(highlights, out_dir, fps=fps)

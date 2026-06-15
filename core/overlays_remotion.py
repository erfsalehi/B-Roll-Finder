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
import re
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

# A title is the full hook/heading line spoken verbatim — it can run long, so it
# gets a generous cap. Every other overlay is a short phrase/figure.
_TITLE_TYPES = {"title", "heading"}
_TITLE_TEXT_CAP = 200
_OTHER_TEXT_CAP = 80


def _load_prompt() -> str:
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _norm_token(s: str) -> str:
    """Lowercase, strip everything but letters/digits — so the overlay's display
    text ("$4,999", "10,000 MILES", "Title-Case") matches the spoken word tokens
    ("4999", "10000", "miles") regardless of case, punctuation, or currency."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _flatten_words(segments: list) -> list:
    """All per-word timings across segments, time-sorted, normalized token cached.
    Each entry: {'tok': str, 'start': float, 'end': float}. Empty when the
    transcription carried no word-level timestamps."""
    words = []
    for s in (segments or []):
        for w in (s.get("words") or []):
            tok = _norm_token(w.get("word", ""))
            if not tok or w.get("start") is None:
                continue
            try:
                words.append({"tok": tok, "start": float(w["start"]),
                              "end": float(w.get("end", w["start"]))})
            except (TypeError, ValueError):
                continue
    words.sort(key=lambda x: x["start"])
    return words


def _align_to_words(text: str, words: list, hint_start: float,
                    hint_end: float) -> tuple | None:
    """Snap an overlay's (start, end) to the exact span of the spoken words that
    make up ``text``. ``words`` is from :func:`_flatten_words`; ``hint_start`` /
    ``hint_end`` are the LLM's approximate times, used to pick the right
    occurrence when a phrase repeats. Returns ``(start, end)`` or ``None`` when no
    confident match is found (caller then keeps the LLM's times)."""
    toks = [_norm_token(t) for t in text.split()]
    toks = [t for t in toks if t]
    if not toks or not words:
        return None

    n = len(words)
    first, last = toks[0], toks[-1]
    # Candidate start positions: words matching the first token, ranked by how
    # close they sit to the LLM's hinted start (disambiguates a repeated phrase).
    starts = [i for i in range(n) if words[i]["tok"] == first]
    if not starts:
        return None
    i = min(starts, key=lambda k: abs(words[k]["start"] - hint_start))
    start_t = words[i]["start"]

    # End: the last token, searched within a window allowing a few ASR
    # insertions/splits beyond the literal token count.
    span_end = min(n, i + len(toks) + 6)
    end_idxs = [k for k in range(i, span_end) if words[k]["tok"] == last]
    if end_idxs:
        end_t = words[end_idxs[-1]]["end"]
    else:
        # No clean tail match — assume the phrase runs contiguously from i.
        end_t = words[min(n - 1, i + len(toks) - 1)]["end"]

    if end_t <= start_t:
        end_t = start_t + max(0.4, hint_end - hint_start)
    return start_t, end_t


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
        # Overlays are cosmetic — allow_fallback so a DeepSeek/provider hiccup
        # degrades to Groq instead of dropping overlays entirely (paid-only mode).
        # Generous token budget so a long video's many overlays aren't truncated
        # mid-JSON (we'd rather over-caption and prune than miss items).
        res = _call_llm_json(client, system_prompt, user_content,
                             temperature=0.4, max_tokens=6000, tier="smart",
                             allow_fallback=True)
    except Exception as e:
        print(f"[overlays] highlight extraction failed: {e}")
        return []

    # Per-word timings (when available) let us snap each overlay to the exact
    # moment its words are spoken instead of trusting the LLM's eyeballed times.
    words = _flatten_words(segments)

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
        otype = (h.get("type") or "title")
        # Titles/headings keep the full line verbatim; short overlays stay short.
        cap = _TITLE_TEXT_CAP if otype in _TITLE_TYPES else _OTHER_TEXT_CAP
        # Snap to the spoken words so the overlay appears/disappears on-beat.
        aligned = _align_to_words(text, words, start, end)
        if aligned:
            start, end = aligned
        anim = h.get("anim") if h.get("anim") in _VALID_ANIM else "title_card"
        sfx = h.get("sfx") if h.get("sfx") in _VALID_SFX else "none"
        out.append({
            "text": text[:cap],
            "type": otype,
            "anim": anim,
            "sfx": sfx,
            "start": start,
            "end": end,
            "emphasis": h.get("emphasis") or "normal",
        })
    # Cull true time-overlaps always; apply the anti-crowding gap to PROSE only.
    # Numbers/stats/money are the priority — three stats in a row should all stay
    # on screen, so they're exempt from the spacing rule (kept unless they truly
    # overlap another overlay in time).
    try:
        min_gap = float(os.getenv("OVERLAY_MIN_GAP_SEC", "1.5"))
    except (TypeError, ValueError):
        min_gap = 1.5
    keep_close = {"stat", "number", "money"}
    out.sort(key=lambda o: o["start"])
    deduped, last_end, last_start = [], -1.0, -1e9
    for o in out:
        if o["start"] < last_end:                  # true time overlap → drop
            continue
        if o.get("type") not in keep_close and o["start"] - last_start < min_gap:
            continue                               # crowding rule: prose only
        deduped.append(o)
        last_end, last_start = o["end"], o["start"]

    # The fade-out eats the clip's final frames, so without a tail the text
    # starts vanishing while the word is still being said. Hold each overlay a
    # beat past its last word — but never into the next overlay's start, so a
    # number-dense run stays clean.
    try:
        end_pad = float(os.getenv("OVERLAY_END_PAD_SEC", "0.35"))
    except (TypeError, ValueError):
        end_pad = 0.35
    if end_pad > 0:
        for idx, o in enumerate(deduped):
            ceiling = (deduped[idx + 1]["start"] if idx + 1 < len(deduped)
                       else o["end"] + end_pad)
            o["end"] = min(o["end"] + end_pad, max(o["end"], ceiling - 0.05))
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
    # Match the overlay's on-screen length to exactly how long the line is spoken
    # — the fade in/out happens *within* this window (see Overlay.tsx), so the
    # text is fully up while the voice says it. Tiny floor only so a degenerate
    # zero-length highlight still renders to at least one frame.
    dur = max(0.4, float(h["end"]) - float(h["start"]))
    # Pick a (deterministic) variant of the requested SFX, e.g. swoosh→swoosh2.
    sfx = _pick_sfx_variant(h.get("sfx", "none"), h.get("text", ""))
    return {
        "text": h["text"], "type": h.get("type", "title"),
        "anim": h.get("anim", "title_card"), "sfx": sfx,
        "durationSec": round(dur, 3), "fps": fps,
        "color": color, "accent": accent,
    }


# Bump when Overlay.tsx visuals change. The render cache is keyed by props, NOT
# by the component source, so a style change wouldn't otherwise invalidate
# previously-rendered clips — they'd be reused with the OLD look.
_STYLE_VERSION = "5"


def _cache_key(props: dict) -> str:
    payload = json.dumps(props, sort_keys=True) + "|style=" + _STYLE_VERSION
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


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


def _slug(text: str, max_len: int = 40) -> str:
    """Filesystem-safe, lowercase, dash-joined slug of a sentence."""
    cleaned = "".join(c if (c.isalnum() or c in " -_") else " " for c in (text or ""))
    return "-".join(cleaned.split()).lower()[:max_len].strip("-") or "overlay"


def _shot_for_time(t: float, shots: list):
    """slot_id of the shot whose voiceover spans time ``t`` — else the nearest
    preceding shot, else the first shot. None when no shots are available
    (overlay-only mode). Lets each overlay be named for the shot it sits over."""
    if not shots:
        return None
    preceding = None
    first = None
    for s in shots:
        try:
            st = float(s.get("timestamp", 0))
            en = float(s.get("end_timestamp", st))
        except (TypeError, ValueError):
            continue
        if first is None:
            first = s.get("slot_id")
        if st <= t < en or (en <= st and abs(st - t) < 1e-6):
            return s.get("slot_id")
        if st <= t:
            preceding = s.get("slot_id")
    return preceding if preceding is not None else first


def _overlay_basename(idx: int, h: dict, shots: list, seen: set) -> str:
    """Descriptive, unique ``.mov`` name so a clip can be placed by hand if the
    XML import misbehaves. Encodes the shot number it overlays + the start
    timecode + the sentence, e.g. ``shot03_00m12s_our-revenue-tripled.mov``. In
    overlay-only mode (no shots) the shot token is dropped:
    ``00m12s_our-revenue-tripled.mov`` — the timecode still says exactly where it
    goes."""
    start = float(h.get("start", 0) or 0)
    tc = f"{int(start) // 60:02d}m{int(start) % 60:02d}s"
    sid = _shot_for_time(start, shots)
    prefix = f"shot{int(sid):02d}_{tc}" if sid is not None else tc
    base = f"{prefix}_{_slug(h.get('text', ''))}"
    name = f"{base}.mov"
    n = 1
    while name in seen:                 # guard against two overlays slugging alike
        n += 1
        name = f"{base}_{n}.mov"
    seen.add(name)
    return name


def render_overlay_clips(highlights: list, out_dir: str, fps: int = 30,
                         color: str = "#FFFFFF", accent: str = "#FFD400",
                         timeout: int = 240, shots: list = None) -> list:
    """Render each highlight to a transparent ProRes 4444 .mov, reusing cached
    renders and rendering several at once. Returns overlay dicts (``filepath``
    set) for the ones that succeeded, in timeline order. ``shots`` (optional) is
    used only to name each clip after the shot it overlays."""
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

    # Precompute unique filenames sequentially (the dedupe ``seen`` set isn't
    # thread-safe) before rendering in parallel.
    seen_names: set = set()
    names = [_overlay_basename(idx, h, shots, seen_names)
             for idx, h in enumerate(highlights)]

    def _one(idx_h):
        idx, h = idx_h
        props = _props_for(h, fps, color, accent)
        out_path = os.path.join(out_dir, names[idx])
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


_ANIM_FOR_TYPE = {
    "title": "title_card", "heading": "title_card", "stat": "stat_pop",
    "number": "stat_pop", "money": "money_count", "emphasis": "pop",
}
_SFX_FOR_ANIM = {
    "title_card": "swoosh", "stat_pop": "ding", "money_count": "ding", "pop": "thud",
}


def render_one_overlay(text: str, duration_sec: float, out_dir: str,
                       otype: str = None, fps: int = 30) -> dict | None:
    """Render a SINGLE, manually-specified overlay (no LLM extraction) to a
    transparent ProRes .mov. Used by the bot's "give me an overlay for this exact
    text + duration" command. Auto-classifies the animation from the content when
    ``otype`` isn't given (a figure/%/currency → stat pill; else a title card).
    Returns the overlay dict (``filepath`` set) or None on failure."""
    text = (text or "").strip()
    if not text:
        return None
    if otype not in _ANIM_FOR_TYPE:
        # Short text containing a number/%/currency reads best as a stat pop;
        # anything else as a title card.
        if re.search(r"[\$£€%]|\d", text) and len(text.split()) <= 6:
            otype = "stat"
        else:
            otype = "title"
    anim = _ANIM_FOR_TYPE.get(otype, "title_card")
    sfx = _SFX_FOR_ANIM.get(anim, "none")
    try:
        dur = max(0.4, float(duration_sec))
    except (TypeError, ValueError):
        dur = 4.0
    cap = _TITLE_TEXT_CAP if otype in _TITLE_TYPES else _OTHER_TEXT_CAP
    h = {"text": text[:cap], "type": otype, "anim": anim, "sfx": sfx,
         "start": 0.0, "end": dur, "emphasis": "high"}
    color = os.getenv("OVERLAY_TEXT_COLOR", "#FFD60A").strip() or "#FFD60A"
    accent = os.getenv("OVERLAY_ACCENT_COLOR", "#FFD400").strip() or "#FFD400"
    rendered = render_overlay_clips([h], out_dir, fps=fps, color=color, accent=accent)
    return rendered[0] if rendered else None


def build_overlays(out_dir: str, segments: list = None, script_text: str = "",
                   groq_key: str = None, fps: int = 30, shots: list = None) -> list:
    """Extract highlights then render them. Returns FCPXML-ready overlay dicts.
    ``shots`` (optional) names each clip after the shot it overlays."""
    highlights = extract_overlay_highlights(segments, script_text, groq_key)
    if not highlights:
        return []
    # Punchy, non-pale defaults; tunable without a code change. A vivid yellow
    # text with a dark outline (added in Overlay.tsx) reads far better than pale
    # white over footage.
    color = os.getenv("OVERLAY_TEXT_COLOR", "#FFD60A").strip() or "#FFD60A"
    accent = os.getenv("OVERLAY_ACCENT_COLOR", "#FFD400").strip() or "#FFD400"
    return render_overlay_clips(highlights, out_dir, fps=fps,
                                color=color, accent=accent, shots=shots)

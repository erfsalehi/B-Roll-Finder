"""Motion cards: plan → pictures → Remotion render → Premiere XML track.

    build_plan()   core.scene_plan decides the beats; this finds each one's picture
                   (core.card_images: real photo first, generated if none fits) and
                   decides whether the card is actually better than the footage it
                   would cover. Cheap enough to redo after every /refine.
    render_plan()  renders each enabled beat to an opaque H.264 clip with the
                   ``ImageCard`` composition (remotion/src/ImageCard.tsx), cached by
                   content so an unchanged card is never rendered twice, and returns
                   the entries ``core.output.generate_fcpxml`` puts on its own video
                   track ABOVE the footage and the text overlays. Disable that track in
                   Premiere and the footage underneath is exactly as it was.
    write_manifest()  cards.txt in the project folder: every card with where its
                   picture came from (page, or model + prompt) and the footage
                   confidence of every shot with its reasons, for tuning the score.

Everything is best-effort: a failed render or picture drops that card, never the job.
"""

import base64
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile

from core import ai_images, card_images, scene_plan

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_REPO_ROOT, ".cache", "card_clips")
# Bump when ImageCard.tsx changes: the cache is keyed by props, not by the source.
_CARD_VERSION = "2"      # 2: BT.709 limited-range output (was full-range BT.601)
_FPS = 30


def project_dir(project_name: str) -> str:
    from core.output import _safe_for_fs
    return os.path.join(os.path.abspath("downloads"), _safe_for_fs(project_name, 50))


def card_cache_dir() -> str:
    return _CACHE_DIR


def card_cache_size() -> int:
    """Total bytes of cached card clips (0 if the cache doesn't exist)."""
    total = 0
    if os.path.isdir(_CACHE_DIR):
        for fn in os.listdir(_CACHE_DIR):
            try:
                total += os.path.getsize(os.path.join(_CACHE_DIR, fn))
            except OSError:
                pass
    return total


def clear_card_cache() -> tuple:
    """Delete every cached card clip. ``(files, bytes)`` freed."""
    removed = freed = 0
    if os.path.isdir(_CACHE_DIR):
        for fn in os.listdir(_CACHE_DIR):
            fp = os.path.join(_CACHE_DIR, fn)
            try:
                freed += os.path.getsize(fp)
                os.remove(fp)
                removed += 1
            except OSError:
                pass
    return removed, freed


# ── planning with pictures ───────────────────────────────────────────────────

def _source_one(beat: dict, shot: dict, pdir: str, topic: str, errors: list) -> None:
    """Find the picture(s) for one beat and apply the rules about what to do when
    there isn't a good one. Mutates ``beat``."""
    beat["sourced"] = True                    # attempted once; a re-plan won't pay again
    got = card_images.source_images(beat, shot, pdir, topic=topic, errors=errors)
    beat["assets"] = {k: v for k in ("left", "right") if (v := got.get(k))}
    beat["confidence"] = dict(beat.get("confidence") or {}, image=got.get("fit"))
    if not got["left"]:
        if beat.get("step"):
            # A list step always gets its card: with no picture it is a typographic one.
            beat["reasons"] = list(beat.get("reasons") or []) + ["no suitable picture — text card"]
            if beat["right"].get("kind") == "image":
                beat["right"] = {"kind": "number", "text": str(beat["step"]["number"]),
                                 "sub": beat["step"].get("kind", "STEP")}
            return
        beat["enabled"] = False
        beat["reasons"] = list(beat.get("reasons") or []) + ["no suitable picture — footage kept"]
        return
    if beat["right"].get("kind") == "image" and not got["right"]:
        # Promised a second picture and none could be made: fall back so the right
        # half is never empty.
        beat["right"] = ({"kind": "number", "text": str(beat["step"]["number"]),
                          "sub": beat["step"].get("kind", "STEP")} if beat.get("step") else
                         {"kind": "image"})
        if not beat.get("step"):
            beat["enabled"] = False
            beat["reasons"] = list(beat.get("reasons") or []) + ["no second picture — footage kept"]
            return
    # A card must be at least as good as the footage it hides.
    foot = (beat.get("confidence") or {}).get("footage")
    fit = got.get("fit")
    if beat.get("trigger") != "step" and fit is not None and foot is not None and fit < foot:
        beat["enabled"] = False
        beat["reasons"] = list(beat.get("reasons") or []) + [
            f"picture fit {fit:.2f} is below the footage's {foot:.2f} — footage kept"]


def build_plan(shots: list, project_name: str, topic: str = "", api_key: str = None,
               qa: dict = None, existing: list = None, errors: list = None,
               diag: dict = None, should_cancel=None) -> list:
    """Plan the beats and find their pictures. Returns the beat list (JSON-friendly)."""
    if errors is None:
        errors = []
    if diag is None:
        diag = {}
    beats = scene_plan.plan_scene(shots, topic, api_key, qa=qa, existing=existing,
                                  errors=errors, diag=diag)
    if not beats:
        return beats
    pdir = project_dir(project_name)
    by_id = {s.get("slot_id"): s for s in shots}
    todo = [b for b in beats if b.get("enabled", True) and not b.get("sourced")]

    def _run(b):
        if should_cancel and should_cancel():
            return
        try:
            _source_one(b, by_id.get(b["slot_ids"][0], {}), pdir, topic, errors)
        except Exception as e:
            errors.append(f"cards: pictures for shot {b['slot_ids'][0]} failed "
                          f"({type(e).__name__}: {str(e)[:100]})")
            b["sourced"] = True
            b["enabled"] = False
            b["reasons"] = list(b.get("reasons") or []) + ["picture search failed — footage kept"]

    # Beats that kept their pictures from an earlier plan but whose files are gone
    # (the project folder was cleaned up) need them found again.
    for b in beats:
        left = (b.get("assets") or {}).get("left")
        if b.get("enabled", True) and left and not os.path.exists(left.get("path", "")):
            b["assets"], b["sourced"] = {}, False
            if b not in todo:
                todo.append(b)

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            list(ex.map(_run, todo))
    diag["sourced"] = len(todo)
    diag["disabled"] = sum(1 for b in beats if not b.get("enabled", True))
    stats = ai_images.run_stats()
    diag["ai_images"], diag["ai_usd"] = stats["images"], stats["usd"]
    return beats


# ── rendering ────────────────────────────────────────────────────────────────

def _data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def _file_sha(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _duration(beat: dict) -> float:
    return round(max(1.0, float(beat["end"]) - float(beat["start"])) * _FPS) / _FPS


def remotion_props(beat: dict, inline_images: bool = True) -> dict:
    """The input props for the ``ImageCard`` composition. With ``inline_images`` False
    the pictures are replaced by their file hashes — that is what the cache key uses."""
    assets = beat.get("assets") or {}
    left = (assets.get("left") or {}).get("path")
    right = dict(beat.get("right") or {"kind": "none"})
    if right.get("kind") == "image":
        rp = (assets.get("right") or {}).get("path")
        right = {"kind": "image", "image": (_data_uri(rp) if inline_images else _file_sha(rp))} \
            if rp and os.path.exists(rp) else {"kind": "none"}
    img = None
    if left and os.path.exists(left):
        img = _data_uri(left) if inline_images else _file_sha(left)
    return {"label": beat.get("label", ""), "durationSec": _duration(beat), "fps": _FPS,
            "image": img, "right": right, "drift": None}


def cache_key(beat: dict) -> str:
    payload = json.dumps(remotion_props(beat, inline_images=False), sort_keys=True) \
        + f"|v={_CARD_VERSION}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _render_one(beat: dict, out_path: str, timeout: int = 420) -> bool:
    """Render (or copy from cache) one card to ``out_path``. True on success."""
    from core.overlays_remotion import _REMOTION_DIR, _remotion_bin
    binary = _remotion_bin()
    if not binary:
        return False
    os.makedirs(_CACHE_DIR, exist_ok=True)
    cached = os.path.join(_CACHE_DIR, f"{cache_key(beat)}.mp4")
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        try:
            shutil.copyfile(cached, out_path)
            return True
        except OSError:
            pass
    props_fp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as pf:
            json.dump(remotion_props(beat), pf)
            props_fp = pf.name
        cmd = [binary, "render", "src/index.ts", "ImageCard", out_path, f"--props={props_fp}",
               "--config=remotion.card.config.ts", "--log=error"]
        chrome = os.getenv("REMOTION_BROWSER_EXECUTABLE", "").strip()
        if chrome:
            cmd.append(f"--browser-executable={chrome}")
        r = subprocess.run(cmd, cwd=_REMOTION_DIR, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            print(f"[cards] render failed for '{beat.get('label')}': "
                  f"{(r.stderr or r.stdout or '')[-300:]}")
            return False
        try:
            shutil.copyfile(out_path, cached)
        except OSError:
            pass
        return True
    except subprocess.TimeoutExpired:
        print(f"[cards] render timed out for '{beat.get('label')}'")
        return False
    except Exception as e:
        print(f"[cards] render error for '{beat.get('label')}': {e}")
        return False
    finally:
        if props_fp:
            try:
                os.remove(props_fp)
            except OSError:
                pass


def _slug(text: str, n: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:n] or "card"


def renderable(beat: dict) -> bool:
    """Enabled, and either has its picture or is a text card for a list step."""
    if not beat.get("enabled", True):
        return False
    left = (beat.get("assets") or {}).get("left")
    if left:
        return os.path.exists(left.get("path", ""))
    return bool(beat.get("step"))


def render_plan(beats: list, project_name: str, errors: list = None, should_cancel=None,
                progress=None) -> list:
    """Render every renderable beat. Returns the XML track entries, in time order:
    ``[{filepath, start_sec, end_sec, name, comment, is_video, beat_id, slot_ids}]``."""
    from core.overlays_remotion import remotion_available
    if errors is None:
        errors = []
    todo = [b for b in beats if renderable(b)]
    if not todo:
        return []
    if not remotion_available():
        errors.append("cards: Remotion isn't installed here (npm ci in remotion/) — no cards rendered")
        return []
    out_dir = os.path.join(project_dir(project_name), "cards")
    os.makedirs(out_dir, exist_ok=True)
    try:
        workers = max(1, int(os.getenv("CARD_RENDER_WORKERS", "2")))
    except ValueError:
        workers = 2
    done = {"n": 0}

    def _one(b):
        if should_cancel and should_cancel():
            return None
        slot = b["slot_ids"][0]
        out = os.path.join(out_dir, f"{b['id']}_{_slug(b.get('label'))}.mp4")
        ok = _render_one(b, out)
        done["n"] += 1
        if progress:
            try:
                progress(done["n"], len(todo))
            except Exception:
                pass
        if not ok:
            errors.append(f"cards: couldn't render the card for shot {slot} ({b.get('label')})")
            return None
        b["file"] = out
        return b

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        rendered = [b for b in ex.map(_one, todo) if b]
    entries = []
    for b in sorted(rendered, key=lambda x: x["start"]):
        conf = b.get("confidence") or {}
        left = (b.get("assets") or {}).get("left") or {}
        bits = [f"{b.get('trigger', '').replace('_', ' ')}"]
        if conf.get("footage") is not None:
            bits.append(f"footage confidence {conf['footage']:.2f}")
        if left:
            bits.append(f"picture: {left.get('kind')}" + (f" ({left['model']})" if left.get("model") else ""))
        entries.append({
            "filepath": b["file"], "start_sec": float(b["start"]), "end_sec": float(b["end"]),
            "name": f"CARD shot {b['slot_ids'][0]} - {b.get('label', '')}",
            "comment": " · ".join(bits), "is_video": True, "beat_id": b["id"],
            "slot_ids": list(b["slot_ids"])})
    return entries


# ── reporting ────────────────────────────────────────────────────────────────

def format_line(beats: list, diag: dict = None) -> str | None:
    """One line for the review message, or None when there is nothing to report."""
    s = scene_plan.summary(beats or [])
    off = sum(1 for b in beats or [] if not b.get("enabled", True))
    if not s["total"] and not off:
        return None
    parts = []
    if s["steps"]:
        parts.append(f"{s['steps']} list step(s)")
    if s["low"]:
        parts.append(f"{s['low']} low-confidence")
    if s["empty"]:
        parts.append(f"{s['empty']} without footage")
    line = f"🃏 Motion cards: {s['total']}" + (f" ({' · '.join(parts)})" if parts else "")
    if s["ai"]:
        line += f" · {s['ai']} AI image(s)"
        usd = (diag or {}).get("ai_usd")
        if usd:
            line += f" ≈ ${usd:.2f}"
    if off:
        line += f" · {off} skipped (no better than the footage)"
    return line


def write_manifest(beats: list, shots: list, project_name: str) -> str | None:
    """``cards.txt``: each card with where its picture came from, then the footage
    confidence of every shot — the numbers to check the score against."""
    try:
        pdir = project_dir(project_name)
        os.makedirs(pdir, exist_ok=True)
        path = os.path.join(pdir, "cards.txt")
        lines = ["MOTION CARDS", "(layer V3 in the Premiere XML — disable it to see the footage underneath)", ""]
        for b in sorted(beats or [], key=lambda x: x.get("start", 0)):
            conf = b.get("confidence") or {}
            lines.append(f"[{'on' if b.get('enabled', True) else 'OFF'}] shot {b['slot_ids'][0]} · "
                         f"{b['start']:.1f}-{b['end']:.1f}s · {b.get('trigger')} · \"{b.get('label')}\" · "
                         f"right: {(b.get('right') or {}).get('kind')} {(b.get('right') or {}).get('text', '')}")
            for side in ("left", "right"):
                a = (b.get("assets") or {}).get(side)
                if not a:
                    continue
                src = a.get("page") or a.get("url") or a.get("source_path") or ""
                if a.get("kind") == "ai":
                    src = f"generated by {a.get('model')} — prompt: {a.get('prompt')}"
                lines.append(f"      {side}: {a.get('kind')}"
                             + (f" fit {a['fit']:.2f}" if a.get("fit") is not None else "") + f" · {src}")
            if conf.get("footage") is not None:
                lines.append(f"      footage confidence {conf['footage']:.2f} ({conf.get('tier')}): "
                             + "; ".join(b.get("reasons") or []))
            elif b.get("reasons"):
                lines.append("      " + "; ".join(b["reasons"]))
        lines += ["", "FOOTAGE CONFIDENCE BY SHOT", "shot  start   score  tier    reasons"]
        for s in shots or []:
            c = s.get("confidence")
            if not c or c.get("tier") == "skip":
                continue
            score = "-" if c.get("score") is None else f"{c['score']:.2f}"
            lines.append(f"{str(s.get('slot_id')):>4}  {float(s.get('timestamp') or 0):6.1f}  {score:>5}  "
                         f"{c.get('tier'):6s}  " + "; ".join(c.get("reasons") or []))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path
    except OSError:
        return None

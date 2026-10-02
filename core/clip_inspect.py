"""Clip inspector: a vision model looks at frames of the footage the timeline will actually play.

The ranker chooses clips from titles and descriptions, the QA reviewer reads text, and the
storyboard check (opt-in) judges coarse thumbnails of a whole video. Nothing looked at the
frames that end up on the timeline, so the first real output carried two channel-logo bumpers,
a TV-network bug, a presenter talking to the camera and a handful of clips with no connection to
the narration (a Corvette badge under "ice cold air every time").

After a clip is downloaded, this takes 2 frames from the stretch the XML will play
(``output.plan_in_point``, or the in-point recorded when only a window was downloaded), asks the
vision model - the same provider chain as the storyboard check - what is wrong with them
(``logo_bumper`` / ``watermark`` / ``presenter`` / ``low_quality``...) and how well they could
illustrate the narration line (``relevance`` 0-1), and:

  * rejects the clip - logo/bumper, watermark, a presenter on every frame, or a relevance below
    ``CLIP_INSPECT_MIN_RELEVANCE`` (0.40). A rejected clip is treated like a failed download: its
    file is deleted, it leaves the Clip Library, and the existing repair loop re-picks the shot;
  * records ``clip["inspect"] = {relevance, issues, note}`` for every clip it saw, which
    ``core.confidence`` uses (a clip that survives with weak relevance makes its shot a card
    candidate).

Validated on the first real output (2 s frames of a 64 s video): both logos, the network bug,
every black-bar frame and 6 of ~8 unrelated frames were caught (scratch/inspect_frames.py).

Everything fails open: no vision provider, an ffmpeg/vision error or an unparseable answer leaves
the clip exactly as it was. Off unless ``ENABLE_CLIP_INSPECT`` (bot: /settings -> Clip quality
check). At most ``CLIP_INSPECT_MAX_CALLS`` (80) vision calls per run.
"""

import concurrent.futures
import os
import subprocess

_PER_CALL = 8                       # images per vision call
_REJECT_ISSUES = ("logo_bumper", "watermark")

_SYSTEM = """You review single frames from b-roll clips that an automated editor picked for a narrated video. For each numbered frame you get the narration line spoken at that moment and, usually, what the editor wanted to show there.

For every frame return:
- "issues": any that apply from
  "logo_bumper" (a channel/studio/brand logo animation, intro or outro bumper, title card or end card: the frame is mostly a logo or branding, not footage),
  "watermark" (a burned-in channel name, TV network bug such as a peacock or channel logo in a corner, stock-agency watermark),
  "presenter" (a person talking to the camera: vlogger, reviewer, mechanic addressing the viewer),
  "text_graphics" (big captions or graphics that are part of the source video),
  "low_quality" (blurry, tiny, very dark or abstract with nothing recognisable).
- "relevance": 0.0-1.0, how well the frame could illustrate what the narration line is ABOUT in this video (its subject, or an illustration that fits the idea). A frame that shows what the editor wanted is relevant even when it only illustrates the idea (an empty wallet for "costs nothing"). 1.0 = clearly on topic; 0.5 = loosely related, generic; 0.2 or less = unrelated (a different subject, people posing, random scenery, a different car part).
- "note": at most 10 words on what the frame shows.
Judge only what is visible. Output ONLY valid JSON: {"frames": [{"id": 1, "issues": [], "relevance": 0.8, "note": "..."}]}. Include every id."""


def enabled() -> bool:
    if os.getenv("ENABLE_CLIP_INSPECT", "false").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    return _vision_ready()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def min_relevance() -> float:
    return _env_float("CLIP_INSPECT_MIN_RELEVANCE", 0.40)


def max_calls() -> int:
    return max(0, int(_env_float("CLIP_INSPECT_MAX_CALLS", 80)))


def frames_per_clip() -> int:
    return max(1, min(3, int(_env_float("CLIP_INSPECT_FRAMES", 2))))


# ── the model call (one seam, so tests swap it) ──────────────────────────────

def _vision_ready() -> bool:
    try:
        from core import storyboard
        return storyboard.vision_configured()
    except Exception:
        return False


def _vision(system: str, user: str, jpegs: list):
    """The prompt and frames through the storyboard check's provider chain; None on failure."""
    try:
        from core import storyboard
        return storyboard._vision_json(system, user, jpegs)
    except Exception as e:
        print(f"[clip_inspect] vision call failed: {type(e).__name__}: {str(e)[:120]}")
        return None


# ── frames ───────────────────────────────────────────────────────────────────

def frame_times(in_sec: float, need_sec: float, media_sec: float, n: int = 2) -> list:
    """Moments (seconds into the file) inside the stretch the clip will play."""
    span = max(1.0, min(float(need_sec or 0), 6.0))
    last = max(0.0, float(media_sec) - 0.3)
    spots = {1: (0.5,), 2: (0.15, 0.85), 3: (0.12, 0.5, 0.88)}[max(1, min(3, n))]
    return [round(min(last, max(0.0, in_sec + span * f)), 2) for f in spots]


def _grab(path: str, t: float, width: int = 512):
    cmd = ["ffmpeg", "-v", "error", "-threads", "1", "-ss", f"{max(0.0, t):.2f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", "-f", "image2pipe",
           "-vcodec", "mjpeg", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 and r.stdout[:2] == b"\xff\xd8" else None


# ── judging ──────────────────────────────────────────────────────────────────

def judge(items: list, topic: str = "") -> list:
    """Per item ``{"frames": [jpeg...], "narration": str}`` -> a list of per-frame result lists
    (same order; ``None`` for an item the model did not answer for)."""
    out = [None] * len(items)
    if not items:
        return out
    per = max(1, _PER_CALL // max(1, max(len(i["frames"]) for i in items)))
    chunks = [list(range(s, min(s + per, len(items)))) for s in range(0, len(items), per)]
    calls = max_calls()

    def _one(idx):
        imgs, lines, owner = [], [], []
        for k in idx:
            for j, jpg in enumerate(items[k]["frames"]):
                imgs.append(jpg)
                owner.append((k, j))
                want = (items[k].get("intent") or "").strip()
                lines.append(f'Frame {len(imgs)}: narration: "{(items[k].get("narration") or "")[:160]}"'
                             + (f' | the editor wanted: "{want[:120]}"' if want else ""))
        user = f"VIDEO TOPIC: {topic or 'unknown'}\n\n" + "\n".join(lines)
        data = _vision(_SYSTEM, user, imgs)
        rows = (data or {}).get("frames") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            return
        by_id = {}
        for r in rows:
            try:
                by_id[int(r.get("id"))] = r
            except (TypeError, ValueError, AttributeError):
                continue
        for n, (k, j) in enumerate(owner, 1):
            r = by_id.get(n)
            if not r:
                continue
            res = out[k] if out[k] is not None else [None] * len(items[k]["frames"])
            try:
                rel = max(0.0, min(1.0, float(r.get("relevance"))))
            except (TypeError, ValueError):
                rel = None
            res[j] = {"issues": [str(x).lower() for x in (r.get("issues") or [])],
                      "relevance": rel, "note": str(r.get("note") or "")[:80]}
            out[k] = res

    todo = chunks[:calls] if calls else []
    if todo:
        try:
            workers = max(1, int(_env_float("CLIP_INSPECT_WORKERS", 3)))
        except ValueError:
            workers = 3
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            list(ex.map(_one, todo))
    return out


def decide(frames: list) -> dict | None:
    """One clip's verdict from its frames' answers: ``{relevance, issues, note, reject, reason}``,
    or None when the model said nothing usable."""
    got = [f for f in (frames or []) if f]
    if not got:
        return None
    issues = sorted({i for f in got for i in f["issues"]})
    rels = [f["relevance"] for f in got if f["relevance"] is not None]
    relevance = round(sum(rels) / len(rels), 2) if rels else None
    reason = ""
    for bad, why in (("logo_bumper", "a channel logo / bumper"), ("watermark", "a watermark or network bug")):
        if any(bad in f["issues"] for f in got):
            reason = why
            break
    if not reason and len(got) >= 2 and all("presenter" in f["issues"] for f in got):
        reason = "a presenter talking to the camera"
    if not reason and relevance is not None and relevance < min_relevance():
        reason = f"not about the narration (relevance {relevance:.2f})"
    return {"relevance": relevance, "issues": issues, "note": got[0]["note"],
            "reject": bool(reason), "reason": reason}


# ── the pass over a project's downloaded clips ───────────────────────────────

def _media_seconds(path: str) -> float:
    from core.ffmpeg_utils import get_video_metadata
    d = float(get_video_metadata(path).get("duration") or 0)
    return d if 0 < d < 3600.0 else 0.0


def _plan(shot: dict, res: dict, n_clips: int, media: float) -> tuple:
    """``(in_sec, need_sec)`` inside the file for what the timeline will play of this clip."""
    try:
        slot = float(shot.get("duration_needed_sec") or 0)
    except (TypeError, ValueError):
        slot = 0.0
    need = max(2.0, (slot or 4.0) / max(1, n_clips))
    if res.get("section_in_sec") is not None and res.get("section_start") is not None:
        return float(res["section_in_sec"]), need
    from core.output import plan_in_point
    try:
        return plan_in_point(res, slot or need, media)[1], need
    except Exception:
        return 0.0, need


def inspect_clips(shots: list, topic: str = "", errors: list = None, should_cancel=None) -> dict:
    """Look at every downloaded, not-yet-inspected clip of the project's shots; reject the bad
    ones (delete the file, mark them failed so the repair loop re-picks). Returns
    ``{checked, rejected, skipped}``; never raises."""
    stats = {"checked": 0, "rejected": 0, "skipped": 0}
    if errors is None:
        errors = []
    try:
        if not enabled():
            return stats
        work = []
        for shot in shots:
            if shot.get("priority") == "none" or shot.get("is_extra") or shot.get("skipped"):
                continue
            sel = [c for c in shot.get("selected_results") or []
                   if c.get("local_path") and os.path.exists(c["local_path"])]
            for res in sel:
                if res.get("_inspected") or res.get("library_segment_id") or res.get("segment_path"):
                    continue                          # seen already, or a human cut it
                work.append((shot, res, len(sel)))
        if not work:
            return stats
        n = frames_per_clip()

        def _prep(item):
            shot, res, k = item
            path = res["local_path"]
            media = _media_seconds(path)
            if media <= 0:
                return None
            in_sec, need = _plan(shot, res, k, media)
            jpgs = [j for j in (_grab(path, t) for t in frame_times(in_sec, need, media, n)) if j]
            if not jpgs:
                return None
            return {"frames": jpgs, "narration": shot.get("text") or "",
                    "intent": shot.get("shot_intent") or ""}

        workers = max(1, int(_env_float("CLIP_INSPECT_WORKERS", 3)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            prepared = list(ex.map(_prep, work))
        live = [(w, p) for w, p in zip(work, prepared) if p]
        stats["skipped"] = len(work) - len(live)
        if should_cancel and should_cancel():
            return stats
        answers = judge([p for _, p in live], topic)
        for (shot, res, _k), ans in zip((w for w, _ in live), answers):
            v = decide(ans)
            if not v:
                stats["skipped"] += 1
                continue
            stats["checked"] += 1
            res["_inspected"] = True
            res["inspect"] = {"relevance": v["relevance"], "issues": v["issues"], "note": v["note"]}
            if not v["reject"]:
                continue
            stats["rejected"] += 1
            path = res.get("local_path")
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
            res["_dl_failed"] = True
            res["_dl_error"] = f"clip check: {v['reason']}"
            res["inspect"]["rejected"] = v["reason"]
            res.pop("_dl_ok", None)
            try:
                from core import clip_library
                clip_library.forget_clip(res.get("url") or "")
            except Exception:
                pass
            errors.append(f"{os.path.basename(path or '')}: clip check - {v['reason']}"
                          + (f" ({v['note']})" if v["note"] else ""))
    except Exception as e:                            # never let the check break a download
        print(f"[clip_inspect] skipped: {type(e).__name__}: {str(e)[:120]}")
    return stats

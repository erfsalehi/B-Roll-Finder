"""
Storyboard footage check — an optional extra look at YouTube candidates.

The ranker picks YouTube videos from their title and description, and a title says
little about what's on screen: a "Camry review" may be 90% a presenter talking to
the camera. YouTube publishes a *storyboard* for every video — small thumbnails
sampled every few seconds, used by the scrubber — and they can be fetched without
downloading any video. This module turns them into one numbered contact sheet per
candidate and has a vision model say, per frame, whether it shows the footage the
shot needs or a presenter talking to camera.

What it changes (only when ``ENABLE_STORYBOARD_CHECK`` is on; bot: /settings):
    * a video that is (almost) all presenter, or shows nothing of the subject, is
      marked ``irrelevant`` — auto-select then skips it and the existing repair
      loop looks for another;
    * a video with only a sliver of on-subject footage is moved to the back;
    * ``storyboard_in_sec`` is the start of the longest on-subject stretch, which
      ``core.output._pick_in_frame`` uses as the source in-point — so the clip is
      cut where the footage is, not at the intro;
    * ``sb_*`` fields feed the QA reviewer's view of the clip.

Everything fails open: a video with no storyboard, a fetch error or a model error
leaves the candidate exactly as the ranker left it. Segment-library clips are
never checked (a human already chose those cuts).

A storyboard sample is coarse (about one frame per 5 s), so this judges the KIND
of footage, not the exact model or trim; identifying the subject stays with the
ranker and the library.
"""

import base64
import concurrent.futures
import io
import json
import os
import threading
import time

_ROOT = os.path.join(os.path.dirname(__file__), "..", ".cache", "storyboards")


# ── settings ─────────────────────────────────────────────────────────────────

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _n_frames() -> int:
    """Thumbnails on a candidate's contact sheet (STORYBOARD_FRAMES). More frames
    catch shorter stretches of footage but cost more image tokens."""
    return max(6, min(48, _env_int("STORYBOARD_FRAMES", 30)))


def _n_cols() -> int:
    return max(2, min(8, _env_int("STORYBOARD_COLUMNS", 6)))


def enabled() -> bool:
    """The user-facing switch. Off by default: the old pipeline runs unchanged."""
    return os.getenv("ENABLE_STORYBOARD_CHECK", "false").strip().lower() in (
        "1", "true", "yes", "on")


def cache_dir() -> str:
    return os.getenv("STORYBOARD_CACHE_DIR", "").strip() or _ROOT


# ── run tally (shown in the job summary) ─────────────────────────────────────

_tally_lock = threading.Lock()
_tally: dict = {}


def reset_run_stats() -> None:
    with _tally_lock:
        _tally.clear()
        _tally.update(videos=0, checked=0, good=0, weak=0, bad=0, no_storyboard=0,
                      failed=0, retimed=0)


def run_stats() -> dict:
    with _tally_lock:
        return dict(_tally)


def _bump(key: str, n: int = 1) -> None:
    with _tally_lock:
        _tally[key] = _tally.get(key, 0) + n


reset_run_stats()


# ── storyboard spec ──────────────────────────────────────────────────────────

def _pick_level(formats: list):
    """The storyboard level whose thumbnails are closest to STORYBOARD_WIDTH
    (default 320 — the level YouTube calls sb0), or None."""
    sbs = [f for f in formats or []
           if str(f.get("format_id", "")).startswith("sb")
           and f.get("fragments") and f.get("width") and f.get("rows") and f.get("columns")]
    if not sbs:
        return None
    want = _env_int("STORYBOARD_WIDTH", 320)
    return min(sbs, key=lambda f: (abs(f["width"] - want), -f["width"]))


def _fetch_spec(url: str) -> dict:
    """The video's storyboard level (yt-dlp, metadata only — no download). Tries
    the configured cookies first (a datacenter IP is often bot-checked without
    them), then none. ``{}`` when the video has no storyboard."""
    from core import youtube as yt
    base = {
        "logger": yt._QuietLogger(), "quiet": True, "no_warnings": True,
        "skip_download": True, "extract_flat": False, "socket_timeout": 20,
        "ignore_no_formats_error": True,
        **yt._search_proxy_opts(),   # metadata goes direct, like search
    }
    tries = [yt._get_cookie_opts(), {}]
    if not tries[0]:
        tries = [{}]
    last = None
    for ck in tries:
        try:
            info = yt._extract_info_with_backoff({**base, **ck}, url, process=False) or {}
        except Exception as e:
            last = e
            continue
        level = _pick_level(info.get("formats"))
        if level:
            return {"level": level, "duration": info.get("duration")}
        return {}
    if last:
        raise last
    return {}


def frame_layout(level: dict) -> list:
    """``[(t_sec, fragment_idx, row, col), …]`` for every thumbnail in a level."""
    rows, cols = int(level["rows"]), int(level["columns"])
    per = rows * cols
    fps = float(level.get("fps") or 0)
    out, t = [], 0.0
    for fi, fr in enumerate(level["fragments"]):
        dur = float(fr.get("duration") or 0)
        n = min(per, max(1, int(round(dur * fps)))) if fps > 0 else per
        step = (1.0 / fps) if fps > 0 else (dur / n if n else 0.0)
        for k in range(n):
            out.append((t + k * step, fi, k // cols, k % cols))
        t += dur
    return out


def sample_indices(total: int, n: int) -> list:
    """``n`` evenly spaced, distinct indices out of ``total`` (all when total ≤ n)."""
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    if n < 2:
        return [0]
    return sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})


def _download_sheet(url: str) -> bytes:
    import requests
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:      # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def build_contact_sheet(level: dict, n_frames: int = None, cols: int = None) -> tuple:
    """Numbered grid of evenly sampled thumbnails → ``(jpeg_bytes, [t_sec, …])``
    (``times[i]`` is frame number i+1), or None when too few sheets downloaded.
    Only the storyboard sheets that hold a sampled frame are fetched."""
    from PIL import Image, ImageDraw
    n_frames = n_frames or _n_frames()
    cols = cols or _n_cols()
    layout = frame_layout(level)
    picked = [layout[i] for i in sample_indices(len(layout), n_frames)]
    if len(picked) < 4:
        return None

    urls = level["fragments"]
    need = sorted({fi for _t, fi, _r, _c in picked})
    workers = max(1, _env_int("STORYBOARD_SHEET_WORKERS", 4))
    sheets: dict = {}

    def _get(fi):
        try:
            return fi, Image.open(io.BytesIO(_download_sheet(urls[fi]["url"]))).convert("RGB")
        except Exception:
            return fi, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(need))) as ex:
        for fi, im in ex.map(_get, need):
            if im is not None:
                sheets[fi] = im

    w, h = int(level["width"]), int(level["height"])
    frames, times = [], []
    for t, fi, r, c in picked:
        im = sheets.get(fi)
        if im is None or (c + 1) * w > im.width or (r + 1) * h > im.height:
            continue
        frames.append(im.crop((c * w, r * h, (c + 1) * w, (r + 1) * h)))
        times.append(t)
    if len(frames) < 4:
        return None

    cell_w = max(160, min(400, _env_int("STORYBOARD_CELL_WIDTH", 256)))
    cell_h = max(1, round(cell_w * h / w))
    rows = -(-len(frames) // cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h))
    draw = ImageDraw.Draw(sheet)
    font = _font(max(14, cell_w // 11))
    for i, fr in enumerate(frames):
        x, y = (i % cols) * cell_w, (i // cols) * cell_h
        sheet.paste(fr.resize((cell_w, cell_h)), (x, y))
        label, pad = str(i + 1), 3
        box = draw.textbbox((x + pad, y + pad), label, font=font)
        draw.rectangle([x, y, box[2] + pad, box[3] + pad], fill=(0, 0, 0))
        draw.text((x + pad, y + pad), label, fill=(255, 235, 0), font=font)
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=80)
    return buf.getvalue(), times


# ── per-video prep, cached on disk (a storyboard never changes) ──────────────

def _vid(c: dict) -> str:
    from core.youtube import _yt_video_id
    return _yt_video_id(c.get("url") or c.get("page_url") or "")


def _cache_key(vid: str) -> str:
    return (f"{vid}_{_n_frames()}_{_n_cols()}"
            f"_{_env_int('STORYBOARD_CELL_WIDTH', 256)}_{_env_int('STORYBOARD_WIDTH', 320)}")


def _prune(directory: str, keep: int = 600) -> None:
    try:
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg")]
        if len(files) <= keep:
            return
        files.sort(key=os.path.getmtime)
        for f in files[:len(files) - keep]:
            for p in (f, f[:-4] + ".json"):
                try:
                    os.remove(p)
                except OSError:
                    pass
    except OSError:
        pass


def prepare_video(url: str):
    """``{"video_id", "jpeg", "times", "duration"}`` for one video, or None when it
    has no usable storyboard (or fetching failed — the caller fails open)."""
    vid = _vid({"url": url})
    if not vid:
        return None
    d = cache_dir()
    stem = os.path.join(d, _cache_key(vid))
    try:
        with open(stem + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(stem + ".jpg", "rb") as f:
            return {"video_id": vid, "jpeg": f.read(), "times": meta["times"],
                    "duration": meta.get("duration")}
    except Exception:
        pass
    spec = _fetch_spec(url)
    if not spec:
        return None
    built = build_contact_sheet(spec["level"])
    if not built:
        return None
    jpeg, times = built
    try:
        os.makedirs(d, exist_ok=True)
        with open(stem + ".jpg", "wb") as f:
            f.write(jpeg)
        with open(stem + ".json", "w", encoding="utf-8") as f:
            json.dump({"times": times, "duration": spec.get("duration")}, f)
        _prune(d)
    except OSError:
        pass
    return {"video_id": vid, "jpeg": jpeg, "times": times, "duration": spec.get("duration")}


# ── the vision call ──────────────────────────────────────────────────────────

_SYSTEM = """You are checking candidate YouTube videos for a video editor who needs B-ROLL footage for one moment of a script.

For each candidate you get ONE image: a grid of numbered thumbnails sampled evenly across the video (the number is printed in each thumbnail's top-left corner). You are told what time each number is.

For every candidate, list frame numbers:
- "on_subject": frames whose footage actually shows what the SHOT INTENT describes — both its subject AND its action, filmed as footage. Be strict: a parked car is not "driving", a car exterior is not "an engine bay", an interior close-up is not "a road". When unsure, leave the frame out. People appearing inside real footage (hands on a steering wheel, a mechanic working) still count.
- "presenter": frames dominated by a person talking to the camera (vlogger/reviewer/interviewee/anchor at a desk or beside the vehicle, podcast or webcam look). Never put a frame in both lists.
A frame in neither list is anything else: title cards, black or blurry frames, logos, ads, screen recordings, slides, unrelated footage.

Judge only what is visible. Do not guess a specific model, trim or year from a thumbnail — match by KIND of subject. The title is a hint only and is often clickbait.

Output ONLY valid JSON:
{"candidates": [{"id": 1, "on_subject": [2, 3, 4], "presenter": [1, 5], "note": "<=12 words on what the video mostly shows"}]}
Include every candidate id you were given."""


def _openrouter_body(system: str, user_text: str, b64_images: list) -> dict:
    from core import segment_library as sl
    content = [{"type": "text", "text": user_text}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}}
        for b in b64_images]
    body = {"model": os.getenv("STORYBOARD_MODEL", "").strip() or sl.vision_fallback_model(),
            "temperature": 0.1,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": content}],
            "response_format": {"type": "json_object"}}
    effort = os.getenv("STORYBOARD_REASONING_EFFORT", "low").strip().lower()
    if effort in ("low", "medium", "high"):
        # The default model reasons whether asked or not (~1,000 hidden tokens,
        # ≈20 s an answer at its default effort); this is a look-and-list task, so
        # ask for the least. Set STORYBOARD_REASONING_EFFORT=none to send no hint.
        body["reasoning"] = {"effort": effort}
    return body


def _vision_json(system: str, user_text: str, jpegs: list) -> dict:
    """The prompt and images through the vision provider the segment library
    uses: OpenRouter first, Gemini as the backup. Raises when both fail."""
    import requests
    from core import segment_library as sl
    from core.keywords import _loads_llm_json, _record_api_usage

    b64 = [base64.b64encode(j).decode() for j in jpegs]
    last = None

    body = _openrouter_body(system, user_text, b64)
    model = body["model"]
    for key in sl._openrouter_keys():
        for attempt in range(3):
            try:
                r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {key}"}, timeout=(10, 120))
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(min(30, float(r.headers.get("Retry-After") or 10)))
                    continue
                if r.status_code == 400 and "reasoning" in body and attempt < 2:
                    body = {k: v for k, v in body.items() if k != "reasoning"}
                    continue      # this model rejects the reasoning hint
                r.raise_for_status()
                j = r.json()
                try:
                    _record_api_usage("openrouter", model, j.get("usage"))
                except Exception:
                    pass
                return _loads_llm_json(j["choices"][0]["message"]["content"])
            except Exception as e:
                last = e
                break

    parts = [{"text": system + "\n\n" + user_text}] + [
        {"inline_data": {"mime_type": "image/jpeg", "data": b}} for b in b64]
    gbody = {"contents": [{"role": "user", "parts": parts}],
             "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}}
    for key in sl.gemini_keys():
        try:
            r = requests.post(f"{sl.GEMINI_BASE}/{sl.gemini_model()}:generateContent", json=gbody,
                              headers={"x-goog-api-key": key}, timeout=(10, 90))
            r.raise_for_status()
            return _loads_llm_json(sl._gemini_text(r.json()))
        except Exception as e:
            last = e
    raise last or ValueError("no vision key (OPENROUTER_API_KEY or GEMINI_API_KEY)")


def vision_configured() -> bool:
    from core import segment_library as sl
    return bool(sl._openrouter_keys() or sl.gemini_keys())


# ── verdict ──────────────────────────────────────────────────────────────────

def _mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def analyze(entry: dict, times: list) -> dict:
    """Turn one candidate's frame lists into a verdict.

    ``bad``  the video is (almost) all presenter (``STORYBOARD_MAX_PRESENTER``,
             default 0.8 of the frames), or NO frame shows the subject — but only
             when the sample is dense enough to trust that (frames at most
             ``STORYBOARD_REJECT_SPACING`` = 30 s apart); a sparse sample can
             easily skip a short stretch of footage, so that case is ``weak``
    ``weak`` on-subject frames are a sliver (< ``STORYBOARD_WEAK_BELOW``, 0.15)
    ``good`` anything else
    ``start_sec`` is where the longest run of consecutive on-subject frames starts
    (the earliest one on a tie) — the in-point worth cutting from."""
    n = len(times)
    if not n:
        return {"verdict": "unknown", "relevant": 0.0, "presenter": 0.0,
                "start_sec": None, "note": ""}

    def _frames(v):
        out = set()
        for x in v or []:
            try:
                i = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= i <= n:
                out.add(i)
        return out

    presenter = _frames((entry or {}).get("presenter"))
    on = _frames((entry or {}).get("on_subject")) - presenter
    rel, pres = len(on) / n, len(presenter) / n

    spacing = (times[-1] - times[0]) / (n - 1) if n > 1 else 0.0
    dense = spacing <= _env_float("STORYBOARD_REJECT_SPACING", 30.0)
    if pres >= _env_float("STORYBOARD_MAX_PRESENTER", 0.8) or (not on and dense):
        verdict = "bad"
    elif rel < _env_float("STORYBOARD_WEAK_BELOW", 0.15):
        verdict = "weak"
    else:
        verdict = "good"

    start_sec = None
    if on:
        runs, cur = [], [min(on)]
        for i in sorted(on)[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        best = max(runs, key=lambda r: (len(r), -r[0]))
        start_sec = round(float(times[best[0] - 1]), 1)
    return {"verdict": verdict, "relevant": round(rel, 2), "presenter": round(pres, 2),
            "start_sec": start_sec, "note": " ".join(str((entry or {}).get("note") or "").split())[:140]}


def summary_line(c: dict) -> str:
    """One phrase for the QA reviewer, '' when the clip wasn't checked."""
    if not c.get("sb_checked") or c.get("sb_verdict") == "unknown":
        return ""
    bits = [f"{round(c.get('sb_relevant', 0) * 100)}% of sampled frames on-subject",
            f"{round(c.get('sb_presenter', 0) * 100)}% presenter to camera"]
    if c.get("storyboard_in_sec"):
        bits.append(f"cut starts at {_mmss(c['storyboard_in_sec'])}")
    note = c.get("sb_note")
    return f" [footage check: {', '.join(bits)}{'; ' + note if note else ''}]"


# ── the stage ────────────────────────────────────────────────────────────────

def _user_text(shot: dict, video_topic: str, batch: list, prepared: dict) -> str:
    lines = []
    if video_topic and video_topic.strip():
        lines.append(f"OVERALL VIDEO TOPIC: {video_topic.strip()}")
    lines.append(f"NARRATION: \"{shot.get('text', '')}\"")
    lines.append(f"SHOT INTENT: {shot.get('shot_intent', '')}")
    lines.append("")
    for n, (_idx, c) in enumerate(batch, 1):
        p = prepared[_vid(c)]
        lines.append(f"CANDIDATE {n}: \"{c.get('title') or '?'}\""
                     + (f" — channel {c['channel']}" if c.get("channel") else ""))
        lines.append("  frame times: " + ", ".join(
            f"{i}={_mmss(t)}" for i, t in enumerate(p["times"], 1)))
    lines.append(f"\nThe {len(batch)} image(s) follow, in candidate order.")
    return "\n".join(lines)


def _candidates_to_check(shot: dict, want: int, per_call: int, extra: int) -> list:
    """``[(index, candidate)]`` — the next best unchecked YouTube candidates, as
    many as the shot still needs plus a spare. Library segments are never checked."""
    from core.director_rank import _is_youtube
    vr = shot.get("video_results") or []
    good = sum(1 for c in vr if _is_youtube(c) and c.get("sb_checked")
               and c.get("sb_verdict") != "bad")
    if good >= want:
        return []
    take = min(per_call, max(1, want - good) + extra)
    out = []
    for i, c in enumerate(vr):
        if (_is_youtube(c) and not c.get("library_segment_id") and not c.get("irrelevant")
                and not c.get("sb_checked") and _vid(c)):
            out.append((i, c))
            if len(out) >= take:
                break
    return out


def _settle(shot: dict) -> None:
    """Make the stored verdicts stick: a re-rank can clear ``irrelevant`` or
    reorder the list, so re-assert both. Idempotent."""
    vr = shot.get("video_results") or []
    if not any(c.get("sb_checked") for c in vr):
        return
    out = []
    for c in vr:
        if c.get("sb_verdict") == "bad" and not c.get("irrelevant"):
            c = dict(c, irrelevant=True)
        out.append(c)
    shot["video_results"] = ([c for c in out if c.get("sb_verdict") != "weak"]
                             + [c for c in out if c.get("sb_verdict") == "weak"])


def check_shots(shots: list, video_topic: str = "", errors: list = None,
                progress=None, should_cancel=None) -> dict:
    """Run the footage check over the shots' YouTube candidates, in place.

    Waves: check each shot's best unchecked YouTube candidates; a shot that lost
    some to the check gets its next ones (``STORYBOARD_WAVES``, default 2). Videos
    are prepared once and shared between shots; the model is asked once per shot
    (its narration decides what counts as on-subject). Returns the run tally."""
    if not enabled():
        return run_stats()
    if errors is None:
        errors = []
    for s in shots:
        _settle(s)   # verdicts from an earlier pass survive a re-rank
    if not vision_configured():
        errors.append("storyboard check: no vision key (OPENROUTER_API_KEY or GEMINI_API_KEY) — skipped")
        return run_stats()

    from core.director_rank import shot_source_quota
    per_call = max(1, _env_int("STORYBOARD_PER_CALL", 4))
    extra = max(0, _env_int("STORYBOARD_EXTRA", 1))
    waves = max(1, _env_int("STORYBOARD_WAVES", 2))
    max_videos = max(1, _env_int("STORYBOARD_MAX_VIDEOS", 300))
    prep_workers = max(1, _env_int("STORYBOARD_WORKERS", 4))
    llm_workers = max(1, _env_int("STORYBOARD_LLM_WORKERS", 3))

    cancelled = lambda: bool(should_cancel and should_cancel())   # noqa: E731
    prepared: dict = {}     # video id → prepared video, or None (no storyboard)
    done = [0]
    total = [0]

    def _tick(n=1):
        done[0] += n
        if progress:
            try:
                progress(done[0], total[0])
            except Exception:
                pass

    for _wave in range(waves):
        if cancelled():
            break
        work = []
        for s in shots:
            if s.get("priority") == "none" or s.get("skipped"):
                continue
            picks = _candidates_to_check(s, shot_source_quota(s)[1], per_call, extra)
            if picks:
                work.append((s, picks))
        if not work:
            break

        # 1 — prepare each new video once
        new_vids: dict = {}
        for _s, picks in work:
            for _i, c in picks:
                v = _vid(c)
                if v not in prepared and v not in new_vids:
                    new_vids[v] = c.get("url") or c.get("page_url")
        room = max_videos - len(prepared)
        if len(new_vids) > room:
            errors.append(f"storyboard check: only the first {max_videos} videos are checked "
                          f"(STORYBOARD_MAX_VIDEOS)")
            new_vids = dict(list(new_vids.items())[:max(0, room)])
        total[0] += len(new_vids) + len(work)

        def _prep(item):
            v, url = item
            try:
                return v, prepare_video(url)
            except Exception as e:
                print(f"[storyboard] {v}: {type(e).__name__}: {str(e)[:120]}")
                return v, None

        if new_vids:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(prep_workers, len(new_vids))) as ex:
                for v, p in ex.map(_prep, list(new_vids.items())):
                    prepared[v] = p
                    _bump("videos" if p else "no_storyboard")
                    _tick()
        if cancelled():
            break

        # 2 — one model call per shot
        def _judge(item):
            shot, picks = item
            batch = [(i, c) for i, c in picks if prepared.get(_vid(c))]
            if not batch:
                return shot, picks, [], None
            try:
                data = _vision_json(_SYSTEM, _user_text(shot, video_topic, batch, prepared),
                                    [prepared[_vid(c)]["jpeg"] for _i, c in batch])
                return shot, picks, batch, data
            except Exception as e:
                print(f"[storyboard] shot {shot.get('slot_id')}: {type(e).__name__}: {str(e)[:150]}")
                return shot, picks, batch, e

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(llm_workers, len(work))) as ex:
            for shot, picks, batch, data in ex.map(_judge, work):
                _tick()
                if isinstance(data, Exception):
                    _bump("failed", len(batch))
                    if not any("storyboard check failed" in e for e in errors):
                        errors.append(f"storyboard check failed ({type(data).__name__}) "
                                      f"— those clips kept as ranked")
                    # Fail open, and mark as checked so a later wave doesn't retry forever.
                    for i, c in picks:
                        shot["video_results"][i] = dict(c, sb_checked=True, sb_verdict="unknown")
                    continue
                if not isinstance(data, dict):
                    data = {}
                by_id = {}
                for e in (data.get("candidates") or []):
                    if isinstance(e, dict):
                        by_id[e.get("id")] = e
                judged = {i for i, _c in batch}
                for n, (i, c) in enumerate(batch, 1):
                    entry = by_id.get(n)
                    if entry is None:
                        shot["video_results"][i] = dict(c, sb_checked=True, sb_verdict="unknown")
                        _bump("failed")
                        continue
                    a = analyze(entry, prepared[_vid(c)]["times"])
                    new = dict(c, sb_checked=True, sb_verdict=a["verdict"],
                               sb_relevant=a["relevant"], sb_presenter=a["presenter"],
                               sb_note=a["note"])
                    if a["start_sec"]:
                        new["storyboard_in_sec"] = a["start_sec"]
                        _bump("retimed")
                    shot["video_results"][i] = new
                    _bump("checked")
                    _bump(a["verdict"])
                    if a["verdict"] != "good":
                        print(f"[storyboard] shot {shot.get('slot_id')}: {a['verdict']} — "
                              f"{(c.get('title') or '?')[:60]!r} "
                              f"(on-subject {a['relevant']:.0%}, presenter {a['presenter']:.0%})")
                for i, c in picks:   # candidates with no storyboard: don't ask again
                    if i not in judged:
                        shot["video_results"][i] = dict(c, sb_checked=True, sb_verdict="unknown")
                _settle(shot)
        for s, _p in work:
            _settle(s)

    st = run_stats()
    if st.get("checked") or st.get("no_storyboard") or st.get("failed"):
        print(f"[storyboard] {st.get('checked', 0)} clip(s) checked — {st.get('good', 0)} good, "
              f"{st.get('weak', 0)} weak, {st.get('bad', 0)} rejected; "
              f"{st.get('no_storyboard', 0)} without storyboard, {st.get('failed', 0)} failed")
    return st

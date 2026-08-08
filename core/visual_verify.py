"""Stage 7b — Gemini watches the shortlisted YouTube candidates.

Ranking (stage 7) judges a YouTube candidate on its title alone: the yt-dlp
search path returns no description at all, so a clickbait title can outrank a
plainly-named but perfect video, and nothing in the metadata says WHICH part of
a 12-minute upload matches the shot. This stage closes both gaps by handing the
candidate's YouTube URL to Gemini, which actually watches it (sampled frames +
audio) and answers two questions per shot: is the footage really in here, and
where.

Two things land on each verified candidate:

* ``visual_match`` (0-10) + ``visual_why`` — a verdict grounded in pixels, so a
  shot leads with a clip that was *seen* to match. Below ``VERIFY_MIN_MATCH``
  the candidate is flagged ``irrelevant``, which auto-selection already skips.
* ``verified_segments`` + ``verified_in_sec`` — where the usable footage sits,
  so the FCPXML can start the clip on that moment instead of frame 0 of a long
  upload (see ``_preferred_in_frame`` in :mod:`core.output`).

Cost is the whole design constraint, so every lever is pulled:

* ``MEDIA_RESOLUTION_LOW`` — 66 tokens per frame instead of 258.
* One frame every 5s (``VERIFY_FPS``, default 0.2) rather than Gemini's 1 fps.
* Thinking off — this is a perception task, not a reasoning one.
* One request per VIDEO, not per candidate-shot pair: the same upload is usually
  a candidate for several shots, so all of them are asked in a single viewing.
* Verdicts cached on disk by (video, question), so a rerun — or a video that
  recurs across projects — costs nothing.
* A hard per-run budget (``VERIFY_MAX_VIDEOS`` / ``VERIFY_MAX_MINUTES``), spent
  on the videos that serve the most shots first.

Together that puts a 15-minute video around 40k input tokens (~$0.01 on 2.5
Flash) against the ~270k a default-resolution 1 fps viewing would have cost.

Every failure mode is a no-op: no key, the flag off, an API error, a spent
budget, or a malformed reply all leave the shots exactly as ranking left them.
"""

import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time

import requests

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"

# Verdict cache. Keyed by (video id, question fingerprint) so the same video
# asked a *different* question still gets watched.
_CACHE_PATH = os.path.join(".cache", "visual_verify.json")
_CACHE_MAX = 5000
_cache: dict = None
_cache_lock = threading.Lock()


# ── configuration ─────────────────────────────────────────────────────────────

def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        return default


def api_keys() -> list:
    """Gemini keys, in rotation order. Empty ⇒ the stage is off."""
    raw = [os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_API_KEY_2", ""),
           os.getenv("GOOGLE_API_KEY", "")]
    keys, seen = [], set()
    for k in raw:
        k = (k or "").strip()
        if k and k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def requested() -> bool:
    """True when the user asked for the stage, regardless of whether a key is
    set. The preflight uses this to decide whether a dead key is a real problem
    or just an unused feature."""
    return _flag("ENABLE_VISUAL_VERIFY")


def enabled() -> bool:
    """True when the user turned the stage on AND a key is configured."""
    return bool(requested() and api_keys())


def model() -> str:
    return (os.getenv("VERIFY_MODEL", "").strip() or DEFAULT_MODEL)


def media_resolution() -> str:
    """Frame token budget. LOW (66 tokens/frame) is the default and is plenty for
    "is this subject on screen, and when" — MEDIUM/HIGH only pay off when you need
    to read fine on-screen text, at 4x the tokens."""
    raw = (os.getenv("VERIFY_MEDIA_RESOLUTION", "").strip() or "low").upper()
    if raw.startswith("MEDIA_RESOLUTION_"):
        return raw
    return f"MEDIA_RESOLUTION_{raw}" if raw in ("LOW", "MEDIUM", "HIGH") \
        else "MEDIA_RESOLUTION_LOW"


def sample_fps() -> float:
    """Frames sampled per second of video. 0.2 = one frame every 5 seconds."""
    return max(0.01, _env_float("VERIFY_FPS", 0.2))


def _max_watch_seconds() -> int:
    """Guardrail against a pathologically long candidate: only the first N
    seconds are watched. Sits above the ``YT_MAX_CLIP_SECONDS`` candidate cap, so
    normally it never bites."""
    return max(0, _env_int("VERIFY_MAX_WATCH_SECONDS", 1200))


def _min_match() -> int:
    """Score below which a watched candidate is flagged irrelevant."""
    return max(0, min(10, _env_int("VERIFY_MIN_MATCH", 4)))


def _max_segments() -> int:
    return max(1, _env_int("VERIFY_MAX_SEGMENTS", 3))


def _min_segment_seconds() -> float:
    return max(0.5, _env_float("VERIFY_MIN_SEGMENT", 3.0))


def _budget() -> tuple:
    """``(max_videos, max_seconds)`` of footage to watch per run. Either at 0
    means unlimited on that axis."""
    return (max(0, _env_int("VERIFY_MAX_VIDEOS", 40)),
            max(0, int(_env_float("VERIFY_MAX_MINUTES", 90.0) * 60)))


def _top_k(shot: dict) -> int:
    """How many of a shot's YouTube candidates are worth watching.

    The ones that matter are the ones auto-selection would actually bind, plus a
    spare or two so there's something to promote when a pick turns out to be a
    talking head. Watching deeper than that is paying to grade clips nobody will
    ever see."""
    try:
        from core.director_rank import shot_source_quota
        _, want_yt = shot_source_quota(shot)
    except Exception:
        want_yt = 1
    extra = max(0, _env_int("VERIFY_EXTRA_CANDIDATES", 1))
    return max(1, min(want_yt + extra, _env_int("VERIFY_TOP_K_MAX", 4)))


# ── candidate helpers ─────────────────────────────────────────────────────────

def _is_youtube(c: dict) -> bool:
    return (c.get("source") or c.get("original_source") or "").lower() == "youtube"


_YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/|/v/)([A-Za-z0-9_-]{11})")
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def video_id(c: dict) -> str:
    """YouTube video id for a candidate, or '' when it isn't a resolvable watch
    URL. Also the grouping key that makes one request cover many shots."""
    for url in (c.get("url"), c.get("page_url")):
        url = (url or "").strip()
        if not url:
            continue
        m = _YT_ID_RE.search(url)
        if m:
            return m.group(1)
        tail = url.rstrip("/").rsplit("/", 1)[-1]
        if _BARE_ID_RE.match(tail):
            return tail
    return ""


def _watch_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def _candidate_duration(c: dict) -> float:
    try:
        return max(0.0, float(c.get("duration") or 0))
    except (TypeError, ValueError):
        return 0.0


def _to_seconds(val) -> float:
    """Parse a timestamp the model may have written as a number OR, against
    instructions, as 'mm:ss' / 'hh:mm:ss'. Returns -1 when unparseable."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val or "").strip()
    if not s:
        return -1.0
    try:
        return float(s)
    except ValueError:
        pass
    if ":" in s:
        total = 0.0
        try:
            for part in s.split(":"):
                total = total * 60 + float(part or 0)
            return total
        except ValueError:
            return -1.0
    return -1.0


# ── the question asked about each shot ────────────────────────────────────────

def _shot_brief(shot: dict) -> dict:
    return {
        "shot_id": shot.get("slot_id"),
        "needs": (shot.get("shot_intent") or "").strip(),
        "narration": " ".join((shot.get("text") or "").split())[:300],
        "seconds": round(_candidate_seconds_needed(shot), 1),
    }


def _candidate_seconds_needed(shot: dict) -> float:
    try:
        return max(0.0, float(shot.get("duration_needed_sec") or 0))
    except (TypeError, ValueError):
        return 0.0


def _question_fingerprint(brief: dict, topic: str) -> str:
    """Stable hash of what we're asking about a shot, so the cache doesn't serve
    a verdict that answered a different question about the same video."""
    raw = f"{topic}|{brief['needs']}|{brief['narration']}|{brief['seconds']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _render_briefs(briefs: list) -> str:
    out = []
    for b in briefs:
        out.append(
            f"SHOT ID {b['shot_id']}\n"
            f"  NEEDS: {b['needs'] or '(unspecified)'}\n"
            f"  NARRATION: {b['narration'] or '(none)'}\n"
            f"  SECONDS: {b['seconds'] or '?'}"
        )
    return "\n\n".join(out)


def _load_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "..", "prompts", "visual_verify.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_system_prompt(video_topic: str = "", custom_instructions: str = "") -> str:
    ctx = []
    if video_topic and video_topic.strip():
        ctx.append(f"OVERALL VIDEO TOPIC: {video_topic.strip()}\n"
                   "Use it to settle ambiguous subjects — footage from a different "
                   "domain than this topic is a 0, however well shot it is.")
    if custom_instructions and custom_instructions.strip():
        ctx.append(f"USER STYLE NOTES: {custom_instructions.strip()}")
    return (_load_prompt()
            .replace("{context_block}", "\n\n".join(ctx))
            .replace("{min_segment}", f"{_min_segment_seconds():g}")
            .replace("{max_segments}", str(_max_segments())))


# ── verdict cache ─────────────────────────────────────────────────────────────

def _cache_load() -> dict:
    global _cache
    if _cache is None:
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            _cache = loaded if isinstance(loaded, dict) else {}
        except Exception:
            _cache = {}
    return _cache


def _cache_get(vid: str, fp: str):
    with _cache_lock:
        return _cache_load().get(f"{vid}|{fp}")


def _cache_put(vid: str, fp: str, verdict: dict) -> None:
    with _cache_lock:
        c = _cache_load()
        if len(c) >= _CACHE_MAX:
            for k in list(c)[:len(c) - _CACHE_MAX + 1]:
                c.pop(k, None)
        c[f"{vid}|{fp}"] = verdict


def _cache_flush() -> None:
    """Persist the cache. Best-effort: a read-only or full disk must not fail a run."""
    with _cache_lock:
        if not _cache:
            return
        snapshot = dict(_cache)
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH) or ".", exist_ok=True)
        tmp = _CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        os.replace(tmp, _CACHE_PATH)
    except Exception as e:
        print(f"[visual_verify] cache save failed: {e}")


def clear_cache() -> None:
    """Drop cached verdicts (in memory and on disk)."""
    global _cache
    with _cache_lock:
        _cache = {}
    try:
        os.remove(_CACHE_PATH)
    except OSError:
        pass


# ── the Gemini call ───────────────────────────────────────────────────────────

def build_payload(watch_url: str, system_prompt: str, briefs: list,
                  duration: float = 0.0) -> dict:
    """The generateContent body for one video + the shots asked about it.

    The video part comes first and the text last — the recommended order for a
    single-video prompt. ``videoMetadata.fps`` and ``generationConfig
    .mediaResolution`` are the two knobs that decide what this costs.
    """
    meta = {"fps": sample_fps()}
    cap = _max_watch_seconds()
    if cap and duration and duration > cap:
        meta["endOffset"] = f"{int(cap)}s"

    thinking = max(0, _env_int("VERIFY_THINKING_BUDGET", 0))
    # Output scales with how many shots were asked about (a verdict plus up to
    # three segments each), so a big group can't get its JSON truncated.
    out_tokens = _env_int("VERIFY_MAX_OUTPUT_TOKENS", 0) or \
        min(8192, max(1024, 200 * max(1, len(briefs))))
    return {
        "contents": [{
            "role": "user",
            "parts": [
                {"fileData": {"fileUri": watch_url}, "videoMetadata": meta},
                {"text": system_prompt + "\n\nSHOTS:\n" + _render_briefs(briefs)},
            ],
        }],
        "generationConfig": {
            "temperature": _env_float("VERIFY_TEMPERATURE", 0.1),
            "responseMimeType": "application/json",
            "mediaResolution": media_resolution(),
            "maxOutputTokens": out_tokens,
            "thinkingConfig": {"thinkingBudget": thinking},
        },
    }


def extract_text(payload: dict) -> str:
    """First non-empty text part of a generateContent response ('' when none)."""
    for cand in (payload.get("candidates") or []):
        for part in ((cand.get("content") or {}).get("parts") or []):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _record_usage(payload: dict) -> None:
    try:
        from core import usage
        um = payload.get("usageMetadata") or {}
        usage.record_llm("gemini", model(),
                         um.get("promptTokenCount", 0),
                         um.get("candidatesTokenCount", 0))
    except Exception:
        pass


def _request(watch_url: str, system_prompt: str, briefs: list,
             duration: float = 0.0) -> dict:
    """One watched video → the parsed JSON reply. Raises on total failure.

    Keys rotate and 429/5xx back off, matching how the DeepSeek path in
    :mod:`core.keywords` handles a rate-limited provider."""
    from core.keywords import _loads_llm_json

    keys = api_keys()
    if not keys:
        raise ValueError("No Gemini API key (set GEMINI_API_KEY).")

    body = build_payload(watch_url, system_prompt, briefs, duration)
    url = f"{GEMINI_BASE}/{model()}:generateContent"
    backoff = [5, 15, 40]
    last_error = None

    for key in keys:
        for attempt in range(len(backoff) + 1):
            try:
                resp = requests.post(
                    url, json=body, timeout=(10, _env_int("VERIFY_TIMEOUT", 300)),
                    headers={"x-goog-api-key": key,
                             "Content-Type": "application/json"},
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{resp.status_code} {resp.reason}",
                                             response=resp)
                resp.raise_for_status()
                data = resp.json()
                _record_usage(data)

                block = (data.get("promptFeedback") or {}).get("blockReason")
                if block:
                    raise ValueError(f"blocked by Gemini ({block})")
                text = extract_text(data)
                if not text:
                    finish = ((data.get("candidates") or [{}])[0]).get("finishReason", "?")
                    raise ValueError(f"empty response (finishReason={finish})")
                return _loads_llm_json(text)
            except Exception as e:
                last_error = e
                status = getattr(getattr(e, "response", None), "status_code", 0)
                # A bad request / unsupported video won't improve with retries or
                # a different key — surface it immediately.
                if status in (400, 403, 404):
                    raise
                if attempt < len(backoff):
                    time.sleep(backoff[attempt])
    raise last_error or RuntimeError("Gemini request failed")


# ── verdict parsing ───────────────────────────────────────────────────────────

def parse_verdicts(data: dict, valid_ids: set, duration: float = 0.0,
                   needs: dict = None) -> dict:
    """``{shot_id: {match, why, segments}}`` from a reply, dropping anything
    malformed or hallucinated.

    Segments are validated hard: numeric, ordered, inside the video, and long
    enough to be usable. A shot whose segments are all junk still keeps its
    score — the clip then just starts at frame 0, as it did before this stage.

    ``needs`` maps a shot_id to the seconds it actually has to fill, so a 2s shot
    isn't denied a 2.5s segment by the global minimum-length floor.
    """
    out = {}
    max_seg = _max_segments()
    base_floor = _min_segment_seconds()
    needs = needs or {}
    for entry in (data.get("shots") or []):
        if not isinstance(entry, dict):
            continue
        sid = entry.get("shot_id")
        if isinstance(sid, str) and sid.strip().lstrip("-").isdigit():
            sid = int(sid.strip())
        if sid not in valid_ids:
            continue
        try:
            match = int(round(float(entry.get("match", 0))))
        except (TypeError, ValueError):
            continue
        match = max(0, min(10, match))

        # Long enough to cut inside — but never demand more than the shot needs,
        # so a 2s slot can still use a 2.5s window.
        want = needs.get(sid) or 0.0
        floor = min(base_floor, max(1.0, want)) if want else base_floor

        segments = []
        for seg in (entry.get("segments") or [])[: max_seg * 3]:
            if not isinstance(seg, dict):
                continue
            start = _to_seconds(seg.get("start"))
            end = _to_seconds(seg.get("end"))
            if start < 0 or end < 0 or end <= start:
                continue
            if duration:
                if start >= duration:
                    continue
                end = min(end, duration)
            if end - start < floor:
                continue
            segments.append({"start": round(start, 2), "end": round(end, 2)})
            if len(segments) >= max_seg:
                break

        out[sid] = {
            "match": match,
            "why": " ".join(str(entry.get("why", "")).split())[:160],
            "segments": segments,
        }
    return out


# ── applying verdicts to shots ────────────────────────────────────────────────

def _sort_key(min_match: int):
    """Bucket a YouTube candidate: seen-and-good first, unseen next, seen-and-bad
    last; within a bucket, higher score first and otherwise the ranker's order
    (Python's sort is stable, so unseen candidates keep their relative ranking)."""
    def key(c: dict):
        match = c.get("visual_match")
        if match is None:
            return (1, 0)
        return (0 if match >= min_match else 2, -match)
    return key


def _apply_to_shot(shot: dict, verdicts: dict, min_match: int) -> int:
    """Write verdicts onto one shot's candidates and re-order its YouTube picks.

    Candidate dicts are SHARED between shots (the cross-shot query cache in
    ``director_search`` hands the same objects to every shot that ran the same
    query), and a verdict is per-shot — so each annotated candidate is replaced
    by a private shallow copy. Without that, shot 3's "talking head" verdict
    would surface on shot 9, which happens to have found the same video.
    """
    cands = shot.get("video_results") or []
    applied = 0

    for i, c in enumerate(cands):
        if not _is_youtube(c):
            continue
        v = verdicts.get(video_id(c))
        if not v:
            continue
        c = dict(c)
        c["visual_match"] = v["match"]
        c["visual_why"] = v.get("why", "")
        c["visual_checked"] = True
        segments = v.get("segments") or []
        if segments:
            c["verified_segments"] = segments
            c["verified_in_sec"] = segments[0]["start"]
            c["verified_out_sec"] = segments[0]["end"]
        if v["match"] < min_match:
            c["irrelevant"] = True
        else:
            c.pop("irrelevant", None)
        cands[i] = c
        applied += 1

    # Re-order YouTube candidates among themselves, leaving every non-YouTube
    # candidate exactly where it was: auto-selection fills its Pexels and
    # YouTube quotas from separate passes over this list, so preserving the
    # interleaving keeps stock selection identical to a run without this stage.
    if applied:
        yt_positions = [i for i, c in enumerate(cands) if _is_youtube(c)]
        reordered = sorted((cands[i] for i in yt_positions), key=_sort_key(min_match))
        for pos, c in zip(yt_positions, reordered):
            cands[pos] = c
    return applied


# ── the stage ─────────────────────────────────────────────────────────────────

def _plan(shots: list, video_topic: str) -> tuple:
    """Group the candidates worth watching by video.

    Returns ``(plan, briefs)`` where ``plan`` maps a video id to the shots that
    shortlisted it, and ``briefs`` maps a slot_id to its question + fingerprint.
    """
    plan: dict = {}
    briefs: dict = {}
    for shot in shots:
        if shot.get("priority") == "none" or shot.get("skipped"):
            continue
        sid = shot.get("slot_id")
        yt = [c for c in (shot.get("video_results") or [])
              if _is_youtube(c) and video_id(c)]
        if not yt:
            continue
        brief = _shot_brief(shot)
        briefs[sid] = {"brief": brief,
                       "fp": _question_fingerprint(brief, video_topic),
                       "shot": shot}
        for c in yt[: _top_k(shot)]:
            vid = video_id(c)
            entry = plan.setdefault(vid, {"duration": 0.0, "shot_ids": []})
            entry["duration"] = max(entry["duration"], _candidate_duration(c))
            if sid not in entry["shot_ids"]:
                entry["shot_ids"].append(sid)
    return plan, briefs


def verify_shot_candidates(shots: list, video_topic: str = "",
                           custom_instructions: str = "",
                           progress_callback=None, should_cancel=None,
                           errors: list = None) -> dict:
    """Watch the shortlisted YouTube candidates and act on what's actually there.

    Runs between ranking and auto-selection: by then candidates are ordered, so
    the shortlist is small and meaningful, and nothing has been bound or
    downloaded yet — a rejected clip costs nothing.

    Returns a telemetry dict (``videos``, ``cached``, ``calls``, ``applied``,
    ``rejected``, ``segments``, ``watched_seconds``, ``errors``, and ``skipped``
    when the stage did nothing).
    """
    if errors is None:
        errors = []
    stats = {"videos": 0, "cached": 0, "calls": 0, "applied": 0, "rejected": 0,
             "segments": 0, "watched_seconds": 0, "errors": 0}

    if not enabled():
        stats["skipped"] = "no GEMINI_API_KEY" if requested() else "disabled"
        return stats

    plan, briefs = _plan(shots, video_topic)
    if not plan:
        stats["skipped"] = "no YouTube candidates"
        return stats

    system_prompt = render_system_prompt(video_topic, custom_instructions)
    min_match = _min_match()
    max_shots_per_call = max(1, _env_int("VERIFY_MAX_SHOTS_PER_CALL", 12))
    max_videos, max_seconds = _budget()

    # verdicts[vid][slot_id] — filled from cache first, then from live calls.
    verdicts: dict = {vid: {} for vid in plan}
    to_call: list = []
    for vid, entry in plan.items():
        pending = []
        for sid in entry["shot_ids"]:
            fp = briefs[sid]["fp"]
            hit = _cache_get(vid, fp)
            if hit:
                verdicts[vid][sid] = hit
                stats["cached"] += 1
            else:
                pending.append(sid)
        # A shot's brief is ~200 text tokens against ~40k for the video, so
        # asking about more shots in one viewing is nearly free — which is why
        # the group is generous. A video shortlisted by MORE shots than one call
        # takes is chunked rather than truncated: coverage matters more than the
        # rare second viewing, and dropping the overflow would silently leave
        # those shots unverified.
        for i in range(0, len(pending), max_shots_per_call):
            to_call.append((vid, entry, pending[i:i + max_shots_per_call]))

    # Spend the budget where it buys the most: videos that answer for the most
    # shots first, and cheaper (shorter) videos before long ones at equal reach.
    to_call.sort(key=lambda t: (-len(t[2]), t[1]["duration"] or 1e9))

    scheduled, watched = [], 0
    for vid, entry, pending in to_call:
        if max_videos and len(scheduled) >= max_videos:
            break
        dur = entry["duration"] or 0.0
        cap = _max_watch_seconds()
        cost = min(dur, cap) if (cap and dur) else (dur or cap or 0)
        if max_seconds and watched + cost > max_seconds and scheduled:
            continue
        scheduled.append((vid, entry, pending))
        watched += cost
    skipped_budget = len(to_call) - len(scheduled)

    stats["videos"] = len(plan)
    lock = threading.Lock()
    done = [0]
    total = max(1, len(scheduled))

    def _watch(vid, entry, pending):
        if should_cancel and should_cancel():
            return
        group = [briefs[sid]["brief"] for sid in pending]
        valid = {b["shot_id"] for b in group}
        needs = {b["shot_id"]: b["seconds"] for b in group}
        try:
            data = _request(_watch_url(vid), system_prompt, group,
                            duration=entry["duration"])
            parsed = parse_verdicts(data, valid, duration=entry["duration"],
                                    needs=needs)
            with lock:
                stats["calls"] += 1
                stats["watched_seconds"] += int(min(entry["duration"] or 0,
                                                    _max_watch_seconds() or 1e9))
                for sid, verdict in parsed.items():
                    verdicts[vid][sid] = verdict
                    _cache_put(vid, briefs[sid]["fp"], verdict)
        except Exception as e:
            with lock:
                stats["errors"] += 1
                errors.append(f"visual verify ({vid}): {e}")
        finally:
            with lock:
                done[0] += 1
                if progress_callback:
                    try:
                        progress_callback(done[0] / total)
                    except Exception:
                        pass

    if scheduled:
        workers = max(1, min(_env_int("VERIFY_MAX_WORKERS", 3), len(scheduled)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_watch, *item) for item in scheduled]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:      # a worker itself blew up
                    stats["errors"] += 1
                    errors.append(f"visual verify worker: {e}")
        _cache_flush()

    # Apply per shot: reshape each shot's verdicts into {video_id: verdict}.
    for sid, meta in briefs.items():
        per_video = {vid: v[sid] for vid, v in verdicts.items() if sid in v}
        if not per_video:
            continue
        stats["applied"] += _apply_to_shot(meta["shot"], per_video, min_match)
        for v in per_video.values():
            if v["match"] < min_match:
                stats["rejected"] += 1
            if v.get("segments"):
                stats["segments"] += 1

    if skipped_budget:
        stats["budget_skipped"] = skipped_budget
    return stats

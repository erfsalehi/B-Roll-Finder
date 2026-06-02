import os
import math
import time
import random
import concurrent.futures
import threading
from groq import Groq
from core.keywords import _call_llm_json


def _rank_jitter() -> None:
    """Sleep a small random interval before an LLM call.

    A bounded thread pool caps how many ranking calls run *at once*, but not
    how many fire *per minute* — concurrent workers finishing together can
    still burst past a provider's per-minute cap and provoke 429s or abrupt
    TCP resets. A little jitter desynchronizes the workers and spreads the
    traffic out. Tunable via ``RANK_JITTER_MIN`` / ``RANK_JITTER_MAX`` seconds;
    set ``RANK_JITTER_MAX=0`` to disable entirely.
    """
    try:
        lo = float(os.getenv("RANK_JITTER_MIN", "0.5"))
        hi = float(os.getenv("RANK_JITTER_MAX", "1.5"))
    except ValueError:
        lo, hi = 0.5, 1.5
    if hi <= 0:
        return
    if lo > hi:
        lo = hi
    time.sleep(random.uniform(max(0.0, lo), hi))


def _load_rank_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'director_rank.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def _orientation(c: dict) -> str:
    w = c.get('width') or 0
    h = c.get('height') or 0
    if not w or not h:
        return 'unknown'
    if w > h:
        return 'horizontal'
    if h > w:
        return 'vertical'
    return 'square'


def _format_candidate(i: int, c: dict) -> str:
    src = (c.get('source') or '?').upper()
    title = c.get('title') or '?'
    desc = c.get('description') or ''
    query = c.get('matched_query') or ''
    w = c.get('width') or '?'
    h = c.get('height') or '?'
    orient = _orientation(c)
    parts = [f"{i}. [{src}] {title}"]
    if desc:
        parts.append(f"— {desc}")
    parts.append(f"| {w}x{h} ({orient})")
    if query:
        parts.append(f'| query: "{query}"')
    return " ".join(parts)


def _apply_ranked_to_shot(shot: dict, ranked: list) -> None:
    """Apply an LLM ``ranked`` array to one shot, in place.

    Reorders ``shot['video_results']`` best→worst, sets ``rank_reason`` from the
    top pick, and marks ``irrelevant`` candidates. Validates every entry — only
    in-range, non-duplicate integer indices are honored; malformed entries are
    dropped (with a recovery counter) and any candidate the LLM omitted is
    appended at the end so a clip is never lost.
    """
    candidates = shot['video_results']
    if not ranked:
        shot.setdefault('rank_reason', '')
        return

    clean_order = []
    seen = set()
    malformed = 0
    for r in ranked:
        idx = r.get('index') if isinstance(r, dict) else None
        if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
            malformed += 1
            continue
        if idx in seen:
            malformed += 1
            continue
        seen.add(idx)
        clean_order.append(idx)
        if r.get('irrelevant'):
            candidates[idx]['irrelevant'] = True
        else:
            candidates[idx].pop('irrelevant', None)

    # Safety net: any candidate the LLM omitted gets appended at the end.
    for i in range(len(candidates)):
        if i not in seen:
            clean_order.append(i)

    shot['video_results'] = [candidates[i] for i in clean_order]

    # Top pick reason — walk `ranked` in its original order so dedup reordering
    # never mismatches the reason to a different candidate.
    top_reason = ''
    if clean_order:
        top_idx = clean_order[0]
        for r in ranked:
            if isinstance(r, dict) and r.get('index') == top_idx:
                top_reason = r.get('reason', '') or ''
                break
    shot['rank_reason'] = top_reason

    if malformed:
        print(f"Shot {shot.get('slot_id')}: dropped {malformed} malformed/duplicate index(es) from LLM output (recovered).")


def _format_shot_block(shot: dict) -> str:
    """Render one shot (narration + intent + indexed candidates) for a batch."""
    candidate_lines = [_format_candidate(i, c) for i, c in enumerate(shot['video_results'])]
    return (
        f"=== SHOT shot_id={shot.get('slot_id')} ===\n"
        f"NARRATION: \"{shot.get('text', '')}\"\n"
        f"SHOT INTENT: {shot.get('shot_intent', '')}\n"
        f"CANDIDATES:\n" + "\n".join(candidate_lines)
    )


def _rank_one_batch(batch: list, system_prompt: str, client: Groq,
                    errors: list, errors_lock: threading.Lock) -> None:
    """Rank a group of shots in a single LLM call, applying results in place.

    Batching collapses what used to be one request per shot into one request per
    group, which is what keeps free-tier providers (Groq, OpenRouter) under their
    per-minute request caps. Output is keyed by ``shot_id`` so a missing or
    reordered shot in the response is handled robustly.
    """
    rankable = [s for s in batch if s.get('video_results')]
    for s in batch:
        if not s.get('video_results'):
            s.setdefault('rank_reason', '')
    if not rankable:
        return

    user_msg = (
        "Rank the candidates for EACH shot below independently. Candidate "
        "indices are 0-based and local to their own shot.\n\n"
        + "\n\n".join(_format_shot_block(s) for s in rankable)
    )

    try:
        # Space this call out from sibling workers to avoid bursty traffic.
        _rank_jitter()
        # Judging is deterministic; low temperature + JSON response_format keep
        # ranking stable. max_tokens scales with the batch (indices + one reason
        # per shot is compact, but give headroom for larger candidate sets).
        data = _call_llm_json(client, system_prompt, user_msg,
                              temperature=0.1, max_tokens=4000)
        ranked_by_id = {}
        for entry in (data.get('shots') or []):
            if isinstance(entry, dict) and entry.get('shot_id') is not None:
                ranked_by_id[entry.get('shot_id')] = entry.get('ranked', []) or []

        for shot in rankable:
            ranked = ranked_by_id.get(shot.get('slot_id'))
            if ranked is None:
                # Shot missing from the response — leave its candidates in their
                # original order rather than failing it outright.
                shot.setdefault('rank_reason', '')
                continue
            _apply_ranked_to_shot(shot, ranked)
            shot.pop('rank_error', None)
    except Exception as e:
        ids = ", ".join(str(s.get('slot_id')) for s in rankable)
        msg = f"Shots {ids}: {e}"
        print(f"Ranking failed for {msg}")
        with errors_lock:
            errors.append(msg)
        for shot in rankable:
            shot['rank_error'] = str(e)
            shot.setdefault('rank_reason', '')


def _load_review_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'timeline_review.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def _selected_clip_label(shot: dict) -> str:
    """Best one-line description of the visual chosen for a shot."""
    sel = shot.get("selected_results") or []
    if not sel:
        return "(no clip selected)"
    c = sel[0]
    title = (c.get("title") or c.get("clip_title") or "untitled").strip()
    src = (c.get("source") or c.get("original_source") or "?")
    q = c.get("matched_query") or ""
    extra = f', via "{q}"' if q else ""
    return f'"{title}" [{src}{extra}]'


def build_timeline_summary(shots: list) -> str:
    """Render the chronological, selected timeline as text for the reviewer.

    One line per shot that actually has a clip bound (priority != 'none',
    not skipped), showing its time, narration intent, and the chosen visual —
    enough for an LLM to judge thematic flow, repetition, and pacing.
    """
    lines = []
    for s in shots:
        if s.get("priority") == "none" or s.get("skipped"):
            continue
        if not s.get("selected_results"):
            continue
        dur = s.get("duration_needed_sec")
        dur_str = f"{float(dur):.1f}s" if dur is not None else "?s"
        intent = (s.get("shot_intent") or s.get("text") or "").strip()[:120]
        lines.append(
            f"Shot {s.get('slot_id')} "
            f"[{s.get('timestamp_start_str', '?')}–{s.get('timestamp_end_str', '?')}, {dur_str}] "
            f"intent: {intent} | visual: {_selected_clip_label(s)}"
        )
    return "\n".join(lines)


def review_timeline(shots: list, api_key: str, video_topic: str = "",
                    custom_instructions: str = "") -> dict:
    """Stage 5 — a single holistic 'executive producer' pass over the assembled
    timeline (smart/reasoning tier).

    Reads the chronological list of selected clips and flags thematic breaks,
    repetition, continuity/pacing problems, and intent mismatches, each pinned to
    a slot_id with a concrete fix. Returns::

        {"overall": str, "issues": [{"slot_id", "severity", "problem", "suggestion"}],
         "reviewed": <int shots reviewed>}

    Degrades gracefully: fewer than 2 selected shots, a missing key, or an LLM
    failure returns an empty issue list with an explanatory ``overall``.
    """
    selected = [s for s in shots
                if s.get("selected_results") and s.get("priority") != "none" and not s.get("skipped")]
    if len(selected) < 2:
        return {"overall": "Not enough selected shots to review yet.", "issues": [], "reviewed": len(selected)}
    if not api_key:
        return {"overall": "Groq API key required for the review.", "issues": [], "reviewed": 0}

    valid_ids = {s.get("slot_id") for s in selected}

    system_prompt = _load_review_prompt()
    context = []
    if video_topic and video_topic.strip():
        context.append(f"OVERALL VIDEO TOPIC: {video_topic.strip()}")
    if custom_instructions and custom_instructions.strip():
        context.append(f"USER STYLE NOTES: {custom_instructions.strip()}")
    system_prompt = system_prompt.replace("{context_block}", "\n".join(context))

    user_msg = "TIMELINE (chronological):\n" + build_timeline_summary(shots)

    client = Groq(api_key=api_key)
    try:
        data = _call_llm_json(client, system_prompt, user_msg,
                              temperature=0.3, max_tokens=2000, tier="smart")
    except Exception as e:
        print(f"Timeline review failed: {e}")
        return {"overall": f"Review unavailable ({type(e).__name__}).", "issues": [], "reviewed": len(selected)}

    issues = []
    for it in (data.get("issues") or []):
        if not isinstance(it, dict):
            continue
        sid = it.get("slot_id")
        if sid not in valid_ids:          # ignore hallucinated / out-of-range slots
            continue
        sev = str(it.get("severity", "medium")).lower()
        if sev not in ("high", "medium", "low"):
            sev = "medium"
        problem = str(it.get("problem", "")).strip()
        if not problem:
            continue
        issues.append({
            "slot_id": sid,
            "severity": sev,
            "problem": problem,
            "suggestion": str(it.get("suggestion", "")).strip(),
        })

    _order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: _order.get(x["severity"], 1))
    return {
        "overall": str(data.get("overall", "")).strip() or "Review complete.",
        "issues": issues,
        "reviewed": len(selected),
    }


def _asset_ident(c: dict):
    """Stable-ish identifier for a candidate clip, used for variety dedup."""
    return c.get("url") or c.get("clip_url") or c.get("page_url") or c.get("title") or id(c)


def _auto_pick_count(shot: dict, seconds_per_clip: float, min_clips: int, max_clips: int) -> int:
    """How many clips to auto-bind for a shot, scaled by its duration.

    Faceless videos want a fresh visual every few seconds, so a long shot needs
    several clips (laid out sequentially by output.py), while even a short shot
    gets ``min_clips`` so the editor has an alternative to pick from. Clamped to
    ``[min_clips, max_clips]``.
    """
    try:
        dur = float(shot.get("duration_needed_sec") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    n = math.ceil(dur / seconds_per_clip) if seconds_per_clip > 0 else min_clips
    return max(min_clips, min(n, max_clips))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def auto_select_top_candidates(shots: list, start_slot_id=None, lookback=None,
                               seconds_per_clip=None, min_clips=None,
                               max_clips=None) -> list:
    """Phase-3 auto-selection: bind the best ranked clips per shot, scaled by
    duration, with a deterministic look-back variety guard.

    Decoupled post-processing for ``ENABLE_AUTO_SELECTION``. For every shot with
    no manual pick, set ``selected_results`` to its best candidates (the ranker
    already sorted them best→worst) and mark it ``auto_selected``. Two rules:

    * **Count scales with duration** — ``ceil(duration / seconds_per_clip)``,
      clamped to ``[min_clips, max_clips]`` and to the number of distinct
      candidates available. output.py then spreads them across the shot, so a 30s
      shot gets ~6 clips (one every 5s) while a short shot still gets ``min_clips``
      (≥2) as alternatives. Picks within a shot are always distinct.
    * **Cross-shot variety** — a clip used within the last ``lookback`` bound
      shots (manual picks included) is skipped in favour of the next alternative,
      so the same visual never appears back-to-back. If no fresh alternative
      exists the duplicate is allowed (a slot is never left empty).

    Defaults come from AUTO_SELECT_SECONDS_PER_CLIP (5), AUTO_SELECT_MIN_CLIPS
    (2), AUTO_SELECT_MAX_CLIPS (8), AUTO_SELECT_LOOKBACK (3). Never overwrites an
    existing selection (idempotent, safe to re-run). ``start_slot_id`` restricts
    NEW auto-picks to shots at/after that number.
    """
    from collections import deque

    if lookback is None:        lookback = _env_int("AUTO_SELECT_LOOKBACK", 3)
    if seconds_per_clip is None: seconds_per_clip = _env_int("AUTO_SELECT_SECONDS_PER_CLIP", 5)
    if min_clips is None:        min_clips = _env_int("AUTO_SELECT_MIN_CLIPS", 2)
    if max_clips is None:        max_clips = _env_int("AUTO_SELECT_MAX_CLIPS", 8)
    lookback = max(0, lookback)
    min_clips = max(1, min_clips)
    max_clips = max(min_clips, max_clips)

    # Each window entry is the set of clip identifiers picked for one shot, so the
    # look-back spans whole shots (not individual clips) even when shots bind many.
    recent_shots = deque(maxlen=lookback)

    def _recent_ids() -> set:
        out = set()
        for idset in recent_shots:
            out |= idset
        return out

    for shot in shots:
        if shot.get("priority") == "none" or shot.get("skipped"):
            continue
        # Respect any pick the editor (or a previous run) made — but feed it into
        # the variety window so later auto-picks avoid repeating it.
        existing = shot.get("selected_results")
        if existing:
            recent_shots.append({_asset_ident(c) for c in existing})
            continue
        if start_slot_id is not None and shot.get("slot_id", 0) < start_slot_id:
            continue
        candidates = shot.get("video_results") or []
        if not candidates:
            continue

        non_irrelevant = [c for c in candidates if not c.get("irrelevant")]
        pool = non_irrelevant or candidates

        want = _auto_pick_count(shot, seconds_per_clip, min_clips, max_clips)
        recent = _recent_ids()
        chosen, chosen_ids = [], set()
        for _ in range(want):
            avoid = recent | chosen_ids
            cand = next((c for c in pool if _asset_ident(c) not in avoid), None)
            if cand is None:
                break  # no more distinct/fresh candidates for this shot
            chosen.append(cand)
            chosen_ids.add(_asset_ident(cand))
        if not chosen:  # graceful: never leave the slot empty
            chosen = [pool[0]]
            chosen_ids = {_asset_ident(pool[0])}

        shot["selected_results"] = chosen
        shot["auto_selected"] = True
        recent_shots.append(chosen_ids)
    return shots


# Batch-mode schema appended to the single-shot ranking rules. The candidate
# format and relevance rules in director_rank.txt are unchanged; only the
# input/output framing differs so one call can judge several shots at once.
_BATCH_SCHEMA_INSTRUCTION = (
    "\n\n=== BATCH MODE ===\n"
    "You will receive SEVERAL shots in one message, each delimited by a line "
    "'=== SHOT shot_id=<id> ===' and carrying its OWN 0-based candidate list. "
    "Rank each shot's candidates independently using the rules above — never mix "
    "candidates across shots; an index always refers to that shot's own list.\n"
    "Output ONLY valid JSON of this exact shape (one object per shot, keyed by "
    "its shot_id):\n"
    "{\n"
    "  \"shots\": [\n"
    "    {\"shot_id\": <id>, \"ranked\": [{\"index\": 2, \"reason\": \"...\"}, {\"index\": 0}, {\"index\": 1, \"irrelevant\": true}]},\n"
    "    {\"shot_id\": <id>, \"ranked\": [ ... ]}\n"
    "  ]\n"
    "}\n"
    "Include every shot_id you were given. Within each shot's 'ranked' array the "
    "rules are identical to the single-shot schema: order best→worst, 'reason' "
    "required only for that shot's top pick, 'irrelevant': true for non-matches."
)


def _rank_batch_size() -> int:
    """Shots per LLM call (env-tunable). Larger = fewer requests, bigger payloads."""
    try:
        return max(1, int(os.getenv("RANK_BATCH_SIZE", "6")))
    except ValueError:
        return 6


def clear_auto_selections(shots: list) -> list:
    """Undo previous *automatic* picks, leaving manual ones intact.

    Lets the Step 4 control be re-applied with a different start shot without
    re-ranking: only selections flagged ``auto_selected`` are cleared (a manual
    pick drops that flag via the UI, so it survives).
    """
    for shot in shots:
        if shot.get("auto_selected"):
            shot["selected_results"] = []
            shot.pop("auto_selected", None)
    return shots


def rank_shot_candidates(shots: list, api_key: str, custom_instructions: str = "",
                         video_topic: str = "", progress_callback=None,
                         errors: list = None, max_workers: int = 3) -> list:
    """
    Stage 3: Ranks and filters each shot's video_results by relevance.
    - Reorders shot['video_results'] best → worst
    - Sets shot['rank_reason'] with the top pick's one-line reason
    - Marks irrelevant candidates with candidate['irrelevant'] = True
    - On failure, sets shot['rank_error'] and appends a string to ``errors``
      (when provided) so the UI can surface it.

    Shots are ranked in BATCHES (``RANK_BATCH_SIZE`` shots per LLM call,
    default 6) instead of one call per shot — that ~6x cut in request count is
    what keeps free-tier providers (Groq, OpenRouter) under their per-minute
    request caps. Batches are dispatched in parallel (default 3 workers). The
    Groq client and the requests-based OpenRouter fallback are both thread-safe;
    each shot mutates only its own dict, so no per-shot locking is needed beyond
    the shared ``errors`` list.
    """
    if not api_key:
        raise ValueError("Groq API key is missing.")

    if errors is None:
        errors = []

    client = Groq(api_key=api_key)

    system_prompt = _load_rank_prompt()
    custom_block = ""
    if video_topic and video_topic.strip():
        custom_block += f"OVERALL VIDEO TOPIC: {video_topic.strip()}\n"
    if custom_instructions and custom_instructions.strip():
        custom_block += f"USER STYLE NOTES: {custom_instructions.strip()}"
    system_prompt = system_prompt.replace("{custom_instructions_block}", custom_block)
    system_prompt += _BATCH_SCHEMA_INSTRUCTION

    rankable = [s for s in shots if s.get('video_results') and s.get('priority') != 'none']
    if not rankable:
        if progress_callback:
            progress_callback(1.0)
        return shots

    # Group consecutive rankable shots into batches.
    batch_size = _rank_batch_size()
    batches = [rankable[i:i + batch_size] for i in range(0, len(rankable), batch_size)]
    total = len(batches)

    errors_lock = threading.Lock()
    done = 0

    # RANK_MAX_WORKERS is the "semaphore size": how many batch calls run at
    # once. Lower it if a provider is still rate-limiting; raise it for speed
    # when you have the quota. Falls back to the caller's value.
    env_workers = os.getenv("RANK_MAX_WORKERS")
    if env_workers:
        try:
            max_workers = max(1, int(env_workers))
        except ValueError:
            pass

    # Cap workers at the number of batches so we don't spin up idle threads.
    workers = max(1, min(max_workers, total))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_rank_one_batch, batch, system_prompt, client, errors, errors_lock)
            for batch in batches
        ]
        for fut in concurrent.futures.as_completed(futures):
            # _rank_one_batch already records per-shot errors on the shots
            # themselves; surface anything truly unexpected here.
            try:
                fut.result()
            except Exception as e:
                with errors_lock:
                    errors.append(f"Worker exception: {e}")
            done += 1
            if progress_callback:
                progress_callback(done / total)

    return shots

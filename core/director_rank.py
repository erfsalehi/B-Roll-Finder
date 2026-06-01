import os
import concurrent.futures
import threading
from groq import Groq
from core.keywords import _call_llm_json


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


def _rank_one_shot(shot: dict, system_prompt: str, client: Groq,
                   errors: list, errors_lock: threading.Lock) -> None:
    """Rank a single shot in-place. Designed to run in a worker thread."""
    candidates = shot['video_results']
    if not candidates:
        shot.setdefault('rank_reason', '')
        return

    # Always run the judge — even for a single candidate. With one
    # candidate the relevance check (irrelevant flag) is the whole
    # point: an editor wastes time on a single off-topic clip too.
    candidate_lines = [_format_candidate(i, c) for i, c in enumerate(candidates)]

    user_msg = (
        f"NARRATION: \"{shot.get('text', '')}\"\n"
        f"SHOT INTENT: {shot.get('shot_intent', '')}\n\n"
        f"CANDIDATES:\n" + "\n".join(candidate_lines)
    )

    try:
        # Judging is a deterministic task; low temperature + the JSON
        # response_format keep ranking stable across re-runs.
        data = _call_llm_json(client, system_prompt, user_msg,
                              temperature=0.1, max_tokens=2000)
        ranked = data.get('ranked', [])

        if ranked:
            # Validate each entry: must be a non-negative int within
            # range, and only the first occurrence of any given index
            # is honored. Duplicates / out-of-range / non-int entries
            # are dropped with a counter, so a noisy LLM cannot
            # silently duplicate or drop candidates from the output.
            clean_order = []
            seen = set()
            malformed = 0
            for r in ranked:
                idx = r.get('index')
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

            # Safety net: any candidate the LLM omitted gets appended
            # at the end so we never lose a clip.
            for i in range(len(candidates)):
                if i not in seen:
                    clean_order.append(i)

            shot['video_results'] = [candidates[i] for i in clean_order]

            # The top pick is whatever index now leads clean_order.
            # Walk `ranked` (in its original order) for the matching
            # entry's reason — that way reordering during dedup never
            # mismatches the reason to a different candidate.
            top_reason = ''
            if clean_order:
                top_idx = clean_order[0]
                for r in ranked:
                    if r.get('index') == top_idx:
                        top_reason = r.get('reason', '') or ''
                        break
            shot['rank_reason'] = top_reason

            if malformed:
                print(f"Shot {shot.get('slot_id')}: dropped {malformed} malformed/duplicate index(es) from LLM output (recovered).")
        # Successful ranking — clear any error from a previous run.
        shot.pop('rank_error', None)
    except Exception as e:
        msg = f"Shot {shot.get('slot_id')}: {e}"
        print(f"Ranking failed for {msg}")
        with errors_lock:
            errors.append(msg)
        shot['rank_error'] = str(e)
        shot.setdefault('rank_reason', '')


def auto_select_top_candidates(shots: list) -> list:
    """Phase-3 auto-selection: bind the best ranked candidate per shot.

    Decoupled post-processing for ``ENABLE_AUTO_SELECTION``. For every shot that
    still has no manual pick, set ``selected_results`` to the top non-irrelevant
    candidate (falling back to the first candidate when the ranker flagged them
    all) and mark it ``auto_selected`` so the review UI can badge it.

    Designed to run after :func:`rank_shot_candidates`, which has already
    reordered ``video_results`` best→worst. It never overwrites an existing
    selection, so it is idempotent and safe to re-run, and the editor can always
    override an automatic pick by hand in review.
    """
    for shot in shots:
        if shot.get("priority") == "none" or shot.get("skipped"):
            continue
        candidates = shot.get("video_results") or []
        if not candidates:
            continue
        # Respect any pick the editor (or a previous run) already made.
        if shot.get("selected_results"):
            continue

        top = next((c for c in candidates if not c.get("irrelevant")), candidates[0])
        shot["selected_results"] = [top]
        shot["auto_selected"] = True
    return shots


def rank_shot_candidates(shots: list, api_key: str, custom_instructions: str = "",
                         video_topic: str = "", progress_callback=None,
                         errors: list = None, max_workers: int = 4) -> list:
    """
    Stage 3: Ranks and filters each shot's video_results by relevance.
    - Reorders shot['video_results'] best → worst
    - Sets shot['rank_reason'] with the top pick's one-line reason
    - Marks irrelevant candidates with candidate['irrelevant'] = True
    - On per-shot failure, sets shot['rank_error'] and appends a string
      to ``errors`` (when provided) so the UI can surface it.

    Per-shot LLM calls are dispatched in parallel (default 4 workers).
    The Groq client and the requests-based OpenRouter fallback are both
    thread-safe; each shot mutates only its own dict, so no per-shot
    locking is needed beyond the shared ``errors`` list.
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

    rankable = [s for s in shots if s.get('video_results') and s.get('priority') != 'none']
    total = len(rankable)
    if total == 0:
        if progress_callback:
            progress_callback(1.0)
        return shots

    errors_lock = threading.Lock()
    done = 0

    # Cap workers at the number of shots so we don't spin up idle threads.
    workers = max(1, min(max_workers, total))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_rank_one_shot, shot, system_prompt, client, errors, errors_lock)
            for shot in rankable
        ]
        for fut in concurrent.futures.as_completed(futures):
            # _rank_one_shot already records per-shot errors on the shot
            # itself; surface anything truly unexpected here.
            try:
                fut.result()
            except Exception as e:
                with errors_lock:
                    errors.append(f"Worker exception: {e}")
            done += 1
            if progress_callback:
                progress_callback(done / total)

    return shots

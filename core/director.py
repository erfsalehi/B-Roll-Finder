import os
import json
import time
from groq import Groq
from core.keywords import _call_llm_json

def load_director_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'director.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_segmenter_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'segmenter.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def env_flag(name: str) -> bool:
    """Read a boolean feature toggle from the environment.

    Matches the truthy idiom used elsewhere in the app (app.py) so the two
    new pipeline toggles behave consistently with the existing ones.
    """
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def segment_script_structure(segments: list, api_key: str, video_topic: str = "") -> dict:
    """Phase-1 pre-pass: map the whole transcript into subject segments.

    Sends the entire timestamped transcript to one fast LLM pass and returns a
    structural roadmap of the video::

        {"video_global_subject": str,
         "segments": [{"subject": str, "start_time": float, "end_time": float}]}

    This anchors localized narration (e.g. "the transmission is jerky") to the
    macro-subject in play at that moment (e.g. "BMW M3 E90"). Any failure
    returns ``{}`` so the caller transparently degrades to flat (Mode A)
    keyword extraction.
    """
    if not api_key or not segments:
        return {}

    client = Groq(api_key=api_key)
    system_prompt = load_segmenter_prompt()
    if video_topic and video_topic.strip():
        system_prompt += f"\n\nThe video's overall topic (for reference): {video_topic.strip()}"

    transcript = "\n".join(
        f"[{float(s.get('start', 0.0)):.2f} - {float(s.get('end', 0.0)):.2f}]: {s.get('text', '').strip()}"
        for s in segments
        if s.get("text")
    )

    try:
        # Structural pre-pass is a once-per-video global synthesis → smart tier.
        data = _call_llm_json(client, system_prompt, transcript,
                              temperature=0.2, max_tokens=2000, tier="smart")
    except Exception as e:
        print(f"Script segmentation failed: {e}")
        return {}

    if not isinstance(data, dict):
        return {}

    # Sanitize: keep only well-formed segments with numeric, ordered times.
    clean = []
    for seg in data.get("segments", []):
        if not isinstance(seg, dict):
            continue
        subject = str(seg.get("subject", "")).strip()
        try:
            start = float(seg.get("start_time"))
            end = float(seg.get("end_time"))
        except (TypeError, ValueError):
            continue
        if not subject or end < start:
            continue
        clean.append({"subject": subject, "start_time": start, "end_time": end})

    clean.sort(key=lambda s: s["start_time"])
    if not clean:
        return {}
    return {
        "video_global_subject": str(data.get("video_global_subject", "")).strip(),
        "segments": clean,
    }


def subject_for_timestamp(roadmap: dict, t: float) -> str:
    """State-machine lookup: the macro-subject active at time ``t`` (seconds).

    Returns the subject of the roadmap segment whose [start_time, end_time]
    contains ``t``; failing that, the nearest segment by distance, then the
    ``video_global_subject``; finally ``""``. Built defensively so a malformed
    or empty roadmap never raises into the shot-generation loop.
    """
    if not isinstance(roadmap, dict):
        return ""
    segs = roadmap.get("segments") or []
    global_subject = roadmap.get("video_global_subject", "") or ""

    try:
        t = float(t)
    except (TypeError, ValueError):
        return global_subject

    best = None
    best_dist = None
    for seg in segs:
        start = seg.get("start_time", 0.0)
        end = seg.get("end_time", 0.0)
        if start <= t <= end:
            return seg.get("subject", "") or global_subject
        # Track nearest segment as a fallback for times that fall in a gap.
        dist = start - t if t < start else t - end
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = seg
    if best is not None:
        return best.get("subject", "") or global_subject
    return global_subject


def subjects_in_span(roadmap: dict, start: float, end: float) -> list:
    """Distinct macro-subjects overlapping the time window [start, end].

    A Director batch of 20 transcription segments can straddle a subject
    boundary, so we surface every subject the window touches (in order, no
    duplicates) rather than just the one at the window's start.
    """
    if not isinstance(roadmap, dict):
        return []
    out = []
    for subj in (
        subject_for_timestamp(roadmap, start),
        subject_for_timestamp(roadmap, (start + end) / 2.0),
        subject_for_timestamp(roadmap, end),
    ):
        if subj and subj not in out:
            out.append(subj)
    return out

def _build_context_block(video_topic: str, custom_instructions: str) -> str:
    """Compose the {custom_instructions_block} value used by director.txt.

    The video topic is rendered first because it is the most useful frame
    of reference: every query the LLM produces should be consistent with
    it. The custom style notes follow as a softer guidance layer.
    """
    parts = []
    if video_topic and video_topic.strip():
        parts.append(
            f"OVERALL VIDEO TOPIC: {video_topic.strip()}\n"
            "Every search_queries entry MUST be plausibly relevant to this topic. "
            "Words in the script that are ambiguous (e.g. \"tool\", \"market\", \"shot\") "
            "should be disambiguated using this topic — for example, \"tool\" in a car "
            "video means \"wrench\" or \"socket\", not \"saw\" or \"chisel\"."
        )
    if custom_instructions and custom_instructions.strip():
        parts.append(
            f"USER STYLE NOTES: {custom_instructions.strip()}\n"
            "Apply these style preferences to shot framing, mood, and query phrasing."
        )
    return "\n\n".join(parts)


def _detailed_queries_enabled() -> bool:
    """Detailed-queries mode (per-chat /settings toggle → ENABLE_DETAILED_QUERIES).

    When on, the Director is told to give a multi-subject shot one query per named
    subject (not just angles on the dominant one), and a follow-up top-up pass
    (:func:`expand_multisubject_queries`) backfills any subject the single call
    still left uncovered."""
    return os.getenv("ENABLE_DETAILED_QUERIES", "").strip().lower() in ("1", "true", "yes", "on")


_DETAILED_QUERIES_INSTRUCTION = (
    "==================================================\n"
    "DETAILED / MULTI-SUBJECT MODE (ON)\n"
    "==================================================\n"
    "When a single script_chunk names TWO OR MORE distinct physical subjects "
    "(e.g. \"this affects the engine's timing chain, the turbo, AND the chassis\"), "
    "do NOT collapse them onto one dominant subject. Instead:\n"
    "  - Emit ONE dedicated search_query for EACH named subject (subject + part/action "
    "+ setting), anchored to the overall video topic so ambiguous nouns resolve "
    "correctly.\n"
    "  - THEN, if room remains, add 1 angle query (setting / close-up / metaphor) as "
    "usual.\n"
    "  - You MAY return up to 6 search_queries for such a shot (the usual 2-3 cap is "
    "lifted ONLY when the chunk genuinely spans several subjects). Never invent "
    "subjects that aren't in the narration.\n"
    "For an ordinary single-subject shot, behave EXACTLY as the standard rules say "
    "(2-3 different-angle queries) — this mode changes nothing there."
)


_SEGMENT_CONTEXT_INSTRUCTION = (
    "SEGMENT CONTEXT AWARENESS:\n"
    "Some script chunks are prefixed with a line of the form\n"
    "  STRUCTURAL CONTEXT (overarching subject at this moment): <subject>\n"
    "When that line is present, keep the named subject in mind — it is the "
    "macro-topic the narration is discussing right now, even when the local "
    "sentence does not repeat it. Blend the subject into your search_queries "
    "where appropriate, and use it to disambiguate generic nouns: if the subject "
    "is \"BMW M3 E90\" and the line says \"the transmission is jerky\", search for "
    "that car's transmission (\"BMW M3 gearbox\"), not a generic gearbox. Do NOT "
    "force the subject into a query when the line is clearly off-subject."
)


def _render_director_system_prompt(template: str, video_topic: str,
                                   custom_instructions: str,
                                   context_aware: bool = False,
                                   detailed: bool = None) -> str:
    """Fill the director system-prompt template's optional placeholder blocks.

    Handles both ``{custom_instructions_block}`` and ``{segment_context_block}``,
    collapsing the surrounding blank lines when a block is empty so the prompt
    never carries a dangling placeholder or extra whitespace.

    When ``detailed`` is on (defaults to the ENABLE_DETAILED_QUERIES env flag),
    the multi-subject instruction is appended so a shot covering several subjects
    gets one query per subject instead of angles on the dominant one.
    """
    custom_block = _build_context_block(video_topic, custom_instructions)
    if custom_block:
        out = template.replace("{custom_instructions_block}", custom_block)
    else:
        out = template.replace("\n\n{custom_instructions_block}\n\n", "\n\n")

    if context_aware:
        out = out.replace("{segment_context_block}", _SEGMENT_CONTEXT_INSTRUCTION)
    else:
        out = out.replace("\n\n{segment_context_block}\n\n", "\n\n")

    if detailed is None:
        detailed = _detailed_queries_enabled()
    if detailed:
        out = out.rstrip() + "\n\n" + _DETAILED_QUERIES_INSTRUCTION
    return out


# Words that describe narration *intent* rather than anything visual, so they
# make poor stock-search terms. Stripped when synthesizing fallback queries.
import re as _re

_QUERY_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "with", "for", "on", "in", "is", "are",
    "that", "this", "into", "as", "at", "by", "its", "it", "be", "or",
    "introduce", "introducing", "explain", "explaining", "describe", "describing",
    "show", "showing", "set", "setting", "tone", "preview", "previewing", "list",
    "emphasize", "emphasizing", "importance", "transition", "transitioning",
    "overview", "topic", "hook", "intro", "outro", "summarize", "summarizing",
    "discuss", "discussing", "mention", "highlight", "highlighting", "first",
    "second", "third", "next", "then", "begin", "start", "end", "point", "idea",
}


def _fallback_queries(shot: dict, video_topic: str = "") -> list:
    """Synthesize plausible visual search queries when the LLM returned none.

    Builds them from the shot's concrete nouns (intent, then narration) anchored
    to the overall video topic, so an empty-query shot still gets *something*
    searchable instead of silently yielding zero candidates.
    """
    topic = (video_topic or "").strip()

    def _keywords(s: str) -> list:
        return [
            w for w in _re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", s or "")
            if w.lower() not in _QUERY_STOPWORDS and len(w) > 2
        ]

    intent_kw = _keywords(shot.get("shot_intent", ""))
    text_kw = _keywords(shot.get("text", ""))

    out = []
    # Primary: topic + the most salient intent nouns (most on-topic / visual).
    primary = " ".join(([topic] if topic else []) + intent_kw[:3]).strip()
    if primary:
        out.append(primary)
    # Secondary: narration nouns, or the topic alone.
    if text_kw:
        out.append(" ".join(([topic] if topic else []) + text_kw[:3]).strip())
    elif topic:
        out.append(topic)

    seen, res = set(), []
    for q in out:
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            res.append(q)
    return res[:2] or ([topic] if topic else ["cinematic b-roll footage"])


def ensure_shot_queries(shots: list, video_topic: str = "") -> list:
    """Guarantee every non-'none' shot has at least one search query.

    The LLM occasionally omits ``search_queries`` for a shot; left empty, that
    shot fetches nothing AND (because youtube_keywords seed from queries) shows
    no keywords either. Fill those in deterministically and flag them with
    ``queries_fallback`` so the UI can hint that a Regenerate would improve them.
    """
    for s in shots:
        if s.get("priority") == "none":
            continue
        qs = [str(q).strip() for q in (s.get("search_queries") or []) if str(q).strip()]
        if qs:
            s["search_queries"] = qs
            s.pop("queries_fallback", None)
        else:
            s["search_queries"] = _fallback_queries(s, video_topic)
            s["queries_fallback"] = True
    return shots


# Boundaries that separate co-ordinate subjects within one chunk
# ("the engine, the turbo, and the chassis").
_SUBJECT_SPLIT_RE = _re.compile(r",|;|/|&|\band\b|\bplus\b", _re.IGNORECASE)


def _count_distinct_subjects(text: str) -> int:
    """Rough count of co-ordinate noun phrases in a chunk (split on commas / "and").

    A part counts only if it carries a content word (>3 chars, not a stopword), so
    "big, bold, bright" inflates less than a real subject list would. This is just
    a cheap gate for the top-up pass — the LLM does the real multi-subject judgement
    and only ADDS queries for genuinely uncovered subjects."""
    parts = [p.strip() for p in _SUBJECT_SPLIT_RE.split(text or "") if p.strip()]
    n = 0
    for p in parts:
        if any(w.lower() not in _QUERY_STOPWORDS and len(w) > 3
               for w in _re.findall(r"[A-Za-z][A-Za-z'-]*", p)):
            n += 1
    return n


def _detailed_max_queries() -> int:
    try:
        return max(2, int(os.getenv("DETAILED_MAX_QUERIES", "6")))
    except ValueError:
        return 6


def expand_multisubject_queries(shots: list, api_key: str, video_topic: str = "",
                                custom_instructions: str = "",
                                progress_callback=None) -> list:
    """Detailed-mode top-up: backfill a query for each subject a multi-subject shot
    still has no footage angle for.

    A cheap heuristic (:func:`_count_distinct_subjects`) flags shots whose narration
    names MORE co-ordinate subjects than they got queries for; those are sent in ONE
    batched LLM call that returns extra per-subject queries, merged in (deduped,
    capped at DETAILED_MAX_QUERIES). No-op when no shot is under-covered, so an
    ordinary single-subject video costs zero extra calls."""
    if not api_key or not shots:
        return shots

    cap = _detailed_max_queries()
    targets = []
    for s in shots:
        if s.get("priority") == "none":
            continue
        qs = [str(q).strip() for q in (s.get("search_queries") or []) if str(q).strip()]
        subjects = _count_distinct_subjects(s.get("text", ""))
        # Under-covered: more distinct subjects than current queries, and room to grow.
        if subjects >= 2 and subjects > len(qs) and len(qs) < cap:
            targets.append(s)

    if not targets:
        if progress_callback:
            progress_callback(1.0)
        return shots

    client = Groq(api_key=api_key)
    system_prompt = (
        "You are a B-roll search director. Each item below is ONE shot whose narration "
        "names MORE distinct physical subjects than it currently has search queries, so "
        "some subjects have NO footage. For each item, output one SHORT (2-5 word), "
        "concrete, visualizable stock-search query for EACH distinct physical subject "
        "that is NOT already covered by its existing queries. Lead with the subject noun, "
        "anchor every query to the overall video topic so ambiguous nouns resolve to the "
        "right domain, and do NOT reword or repeat existing queries — only ADD queries for "
        "uncovered subjects. If an item actually has only one real subject, return an empty "
        'add_queries for it. Return STRICT JSON only: '
        '{"shots": [{"slot_id": <int>, "add_queries": ["...", "..."]}]}'
    )
    if custom_instructions and custom_instructions.strip():
        system_prompt += f"\n\nUSER STYLE NOTES: {custom_instructions.strip()}"

    lines = []
    if video_topic and video_topic.strip():
        lines.append(f"OVERALL VIDEO TOPIC: {video_topic.strip()}\n")
    for s in targets:
        existing = " | ".join(str(q).strip() for q in (s.get("search_queries") or []) if str(q).strip())
        lines.append(
            f'slot_id {s.get("slot_id")}: "{s.get("text", "").strip()}"\n'
            f"  existing queries: {existing or '(none)'}"
        )
    user_msg = "\n".join(lines)

    try:
        data = _call_llm_json(client, system_prompt, user_msg, temperature=0.4, max_tokens=1200)
        by_id = {s.get("slot_id"): s for s in targets}
        for item in data.get("shots", []):
            tgt = by_id.get(item.get("slot_id"))
            if not tgt:
                continue
            existing = [str(q).strip() for q in (tgt.get("search_queries") or []) if str(q).strip()]
            seen = {q.lower() for q in existing}
            for q in item.get("add_queries", []):
                q = str(q).strip()
                if q and q.lower() not in seen:
                    existing.append(q)
                    seen.add(q.lower())
                    if len(existing) >= cap:
                        break
            tgt["search_queries"] = existing[:cap]
    except Exception as e:
        print(f"Error expanding multi-subject queries: {e}")

    if progress_callback:
        progress_callback(1.0)
    return shots


def generate_shot_list(script_text: str, wps: float, api_key: str, progress_callback=None,
                       custom_instructions: str = "", start_offset: float = 0.0,
                       video_topic: str = "") -> list:
    if not api_key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=api_key)
    system_prompt_template = load_director_prompt()

    system_prompt = _render_director_system_prompt(
        system_prompt_template, video_topic, custom_instructions
    )

    # Split text to bypass Groq 6k TPM limits. ~250 words per chunk, respecting sentences.
    from core.timing import split_script_into_smart_blocks
    blocks = split_script_into_smart_blocks(script_text, max_words=250)
    
    total_blocks = len(blocks)
    all_shots = []
    current_time = start_offset
    slot_id = 1
    prev_tail: list = []   # last 2 shots from the previous block (for cross-block continuity)

    for i, block in enumerate(blocks):
        try:
            user_msg = f"WPS: {wps:.2f}\nSCRIPT CHUNK:\n{block}"
            if prev_tail:
                ctx = "\n".join(
                    "  [{shot_type}] \"{text}\" → {intent} (query: {q})".format(
                        shot_type=s["shot_type"],
                        text=s["text"][:90].strip(),
                        intent=s["shot_intent"],
                        q=s["search_queries"][0] if s.get("search_queries") else "—",
                    )
                    for s in prev_tail
                )
                user_msg = (
                    "PREVIOUS SHOTS (for narrative continuity — avoid repeating shot_types "
                    "unless required, and let the context guide query direction):\n"
                    f"{ctx}\n\n{user_msg}"
                )

            data = _call_llm_json(client, system_prompt, user_msg, temperature=0.4, max_tokens=3000)
            shots = data.get("shots", [])
            
            for shot in shots:
                chunk_text = shot.get("script_chunk", "")
                
                # Calculate absolute time based on chunk word count
                chunk_words = len(chunk_text.split())
                if chunk_words == 0:
                    continue
                    
                duration = chunk_words / wps if wps > 0 else 1.0
                end_time = current_time + duration
                
                # Build the slot matching the new architecture. Timestamps are
                # kept as floats so the FCP XML lines up frame-accurately with
                # the voice; the *_str fields stay HH:MM:SS for display.
                all_shots.append({
                    "slot_id": slot_id,
                    "timestamp": round(float(current_time), 3),
                    "end_timestamp": round(float(end_time), 3),
                    "timestamp_start_str": format_time(int(current_time)),
                    "timestamp_end_str": format_time(int(end_time)),
                    "text": chunk_text,
                    "shot_intent": shot.get("shot_intent", "B-roll"),
                    "shot_type": shot.get("shot_type", "medium"),
                    "search_queries": shot.get("search_queries", []),
                    "duration_needed_sec": round(duration, 1),
                    "priority": shot.get("priority", "medium"),
                    "video_results": [] # Will be populated by Stage 2
                })
                
                current_time = end_time
                slot_id += 1

            prev_tail = all_shots[-2:]

        except Exception as e:
            print(f"Error processing block {i}: {e}")
            block_words = len(block.split())
            duration = block_words / wps if wps > 0 else 5.0
            end_time = current_time + duration
            all_shots.append({
                "slot_id": slot_id,
                "timestamp": round(float(current_time), 3),
                "end_timestamp": round(float(end_time), 3),
                "timestamp_start_str": format_time(int(current_time)),
                "timestamp_end_str": format_time(int(end_time)),
                "text": block,
                "shot_intent": "Error Fallback",
                "shot_type": "medium",
                "search_queries": [],
                "duration_needed_sec": round(duration, 1),
                "priority": "low",
                "video_results": []
            })
            current_time = end_time
            slot_id += 1
            
        if progress_callback:
            progress_callback(min(1.0, (i + 1) / total_blocks))

    if _detailed_queries_enabled():
        expand_multisubject_queries(all_shots, api_key, video_topic, custom_instructions)
    ensure_shot_queries(all_shots, video_topic)
    return all_shots

def generate_shot_list_from_transcription(segments: list, api_key: str, progress_callback=None,
                                          custom_instructions: str = "",
                                          video_topic: str = "",
                                          chunk_id: int = 0,
                                          segment_roadmap: dict = None) -> list:
    """
    Uses precise transcription segments to generate a shot list for the
    whole video in one pass (no chunking).

    When ``segment_roadmap`` (from :func:`segment_script_structure`) is provided,
    each block is annotated with the macro-subject(s) active over its time span,
    so the Director anchors localized queries to the overarching subject
    (Mode B/C). When ``None``, behavior is identical to the flat path (Mode A).
    """
    import json
    if not api_key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=api_key)
    system_prompt_template = load_director_prompt()

    context_aware = bool(segment_roadmap)
    system_prompt = _render_director_system_prompt(
        system_prompt_template, video_topic, custom_instructions,
        context_aware=context_aware,
    )
    system_prompt += "\n\nYou will receive transcription segments as '[start - end]: text'. Group them into logical cinematic shots. " \
                     "CRITICAL: For each shot, you MUST include 'start' and 'end' keys in the JSON (floats, in seconds) corresponding to the start of the first segment and end of the last segment in that shot. " \
                     "The 'script_chunk' must contain the verbatim text from those segments."

    # Block segments. Each block is ONE LLM call, processed sequentially — so on
    # a long transcript the call count (and the rate-limit pressure / wall time)
    # scales with the number of blocks. Raising DIRECTOR_BLOCK_SIZE packs more
    # segments per call, cutting the number of calls (e.g. 40 → 20 at size 40) at
    # the cost of larger per-call responses. Default 20.
    try:
        block_size = max(1, int(os.getenv("DIRECTOR_BLOCK_SIZE", "20")))
    except ValueError:
        block_size = 20
    all_shots = []
    slot_id = 1

    for i in range(0, len(segments), block_size):
        block = segments[i : i + block_size]
        user_msg = "\n".join([f"[{s['start']:.2f} - {s['end']:.2f}]: {s['text']}" for s in block])

        # Prepend the macro-subject(s) covering this block's time span so the
        # Director keeps the overarching topic in mind for each shot's queries.
        if context_aware and block:
            subjects = subjects_in_span(
                segment_roadmap,
                float(block[0].get("start", 0.0)),
                float(block[-1].get("end", 0.0)),
            )
            if subjects:
                user_msg = (
                    "STRUCTURAL CONTEXT (overarching subject at this moment): "
                    + " → ".join(subjects)
                    + "\n\n" + user_msg
                )

        try:
            data = _call_llm_json(client, system_prompt, user_msg, temperature=0.4, max_tokens=3000)
            shots = data.get("shots", [])
            if not shots:
                print(f"DEBUG: No shots found in LLM response for block. Response keys: {data.keys()}")
            
            for shot in shots:
                # We expect the AI to return the start/end in its JSON if we ask, 
                # but to be safe we can also try to parse it from the script_chunk it returns
                # or just trust the AI's logic if it groups them correctly.
                # Let's assume the AI provides 'start' and 'end' in the JSON for this mode.
                
                s_time = shot.get("start")
                e_time = shot.get("end")
                
                # If AI returns 0 or None, try to use current_time if we had one
                # but in transcription mode we really want the AI's values.
                if s_time is None: s_time = 0.0
                if e_time is None: e_time = s_time + 5.0
                
                all_shots.append({
                    "slot_id": slot_id,
                    "chunk_id": chunk_id,
                    "timestamp": round(float(s_time), 3),
                    "end_timestamp": round(float(e_time), 3),
                    "timestamp_start_str": format_time(int(float(s_time))),
                    "timestamp_end_str": format_time(int(float(e_time))),
                    "text": shot.get("script_chunk", ""),
                    "shot_intent": shot.get("shot_intent", "B-roll"),
                    "shot_type": shot.get("shot_type", "medium"),
                    "search_queries": shot.get("search_queries", []),
                    "duration_needed_sec": round(float(e_time) - float(s_time), 1),
                    "priority": shot.get("priority", "medium"),
                    "video_results": []
                })
                slot_id += 1
                
        except Exception as e:
            print(f"Error in director transcription block {i}: {e}")
            # Fallback: create one large shot from the entire block if AI fails
            if block:
                s_time = block[0]["start"]
                e_time = block[-1]["end"]
                all_shots.append({
                    "slot_id": slot_id,
                    "chunk_id": chunk_id,
                    "timestamp": round(float(s_time), 3),
                    "end_timestamp": round(float(e_time), 3),
                    "timestamp_start_str": format_time(int(float(s_time))),
                    "timestamp_end_str": format_time(int(float(e_time))),
                    "text": " ".join([s["text"] for s in block]),
                    "shot_intent": "Error Fallback",
                    "shot_type": "medium",
                    "search_queries": [],
                    "duration_needed_sec": round(float(e_time) - float(s_time), 1),
                    "priority": "low",
                    "video_results": []
                })
                slot_id += 1
            
        if progress_callback:
            progress_callback((i + len(block)) / len(segments))

    if _detailed_queries_enabled():
        expand_multisubject_queries(all_shots, api_key, video_topic, custom_instructions,
                                    progress_callback=progress_callback)
    ensure_shot_queries(all_shots, video_topic)
    return all_shots

def regenerate_shot_queries(
    shots: list,
    slot_ids: set,
    api_key: str,
    video_topic: str = "",
    custom_instructions: str = "",
    context_window: int = 2,
    progress_callback=None,
) -> list:
    """Re-generate search_queries for specific shots using full narrative context.

    Intended for shots that returned zero candidates on the first pass.
    Surrounding shots are included so the LLM can interpret ambiguous lines
    correctly and produce queries that diverge from the failed attempts.
    """
    if not api_key:
        raise ValueError("Groq API key is missing.")
    if not slot_ids:
        return shots

    client = Groq(api_key=api_key)
    system_prompt_template = load_director_prompt()
    system_prompt = _render_director_system_prompt(
        system_prompt_template, video_topic, custom_instructions
    )

    targets = [s for s in shots if s.get("slot_id") in slot_ids and s.get("priority") != "none"]
    total = max(len(targets), 1)

    for idx, target in enumerate(targets):
        pos = shots.index(target)
        before = shots[max(0, pos - context_window): pos]
        after  = shots[pos + 1: pos + 1 + context_window]

        ctx_lines = []
        for s in before:
            q0 = s["search_queries"][0] if s.get("search_queries") else "—"
            ctx_lines.append(
                "  [{shot_type}] \"{text}\" → {intent} (prev query: {q})".format(
                    shot_type=s["shot_type"],
                    text=s["text"][:100].strip(),
                    intent=s["shot_intent"],
                    q=q0,
                )
            )
        ctx_lines.append(f'  >>> TARGET: "{target["text"]}" <<<')
        for s in after:
            ctx_lines.append(f'  (next) [{s["shot_type"]}] "{s["text"][:80].strip()}"')

        old_q = " | ".join(target.get("search_queries", []))
        dur = target.get("duration_needed_sec", max(len(target["text"].split()) / 2.5, 1.0))
        wps_est = round(len(target["text"].split()) / max(dur, 0.1), 2)

        user_msg = (
            "NARRATIVE CONTEXT (surrounding shots):\n"
            + "\n".join(ctx_lines)
            + f"\n\nWPS: {wps_est:.2f}\nSCRIPT CHUNK:\n{target['text']}"
            + (f"\n\nPREVIOUSLY TRIED QUERIES (produced zero results — generate different ones): {old_q}" if old_q else "")
            + "\n\nReturn a JSON with a single-item 'shots' array. "
            "Produce NEW search_queries that differ from any previously tried queries."
        )

        try:
            data = _call_llm_json(client, system_prompt, user_msg, temperature=0.55, max_tokens=600)
            new_shots = data.get("shots", [])
            if new_shots:
                ns = new_shots[0]
                if ns.get("search_queries"):
                    target["search_queries"] = ns["search_queries"]
                if ns.get("shot_intent"):
                    target["shot_intent"] = ns["shot_intent"]
                if ns.get("shot_type"):
                    target["shot_type"] = ns["shot_type"]
        except Exception as e:
            print(f"Error regenerating slot {target.get('slot_id')}: {e}")

        if progress_callback:
            progress_callback((idx + 1) / total)

    # Safety net: if regeneration still produced no queries for a target, fill
    # deterministically so it never stays empty.
    ensure_shot_queries(targets, video_topic)
    return shots


def format_time(seconds) -> str:
    total = int(float(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

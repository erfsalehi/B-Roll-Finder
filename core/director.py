import os
import json
import time
from groq import Groq
from core.keywords import _call_llm_json

def load_director_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'director.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

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


def generate_shot_list(script_text: str, wps: float, api_key: str, progress_callback=None,
                       custom_instructions: str = "", start_offset: float = 0.0,
                       video_topic: str = "") -> list:
    if not api_key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=api_key)
    system_prompt_template = load_director_prompt()

    custom_block = _build_context_block(video_topic, custom_instructions)
    if custom_block:
        system_prompt = system_prompt_template.replace("{custom_instructions_block}", custom_block)
    else:
        system_prompt = system_prompt_template.replace("\n\n{custom_instructions_block}\n\n", "\n\n")

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
            
    return all_shots

def generate_shot_list_from_transcription(segments: list, api_key: str, progress_callback=None,
                                          custom_instructions: str = "",
                                          video_topic: str = "",
                                          chunk_id: int = 0) -> list:
    """
    Uses precise transcription segments to generate a shot list for the
    whole video in one pass (no chunking).
    """
    import json
    if not api_key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=api_key)
    system_prompt_template = load_director_prompt()

    custom_block = _build_context_block(video_topic, custom_instructions)
    if custom_block:
        system_prompt = system_prompt_template.replace("{custom_instructions_block}", custom_block)
    else:
        system_prompt = system_prompt_template.replace("\n\n{custom_instructions_block}\n\n", "\n\n")
    system_prompt += "\n\nYou will receive transcription segments as '[start - end]: text'. Group them into logical cinematic shots. " \
                     "CRITICAL: For each shot, you MUST include 'start' and 'end' keys in the JSON (floats, in seconds) corresponding to the start of the first segment and end of the last segment in that shot. " \
                     "The 'script_chunk' must contain the verbatim text from those segments."

    # Block segments
    block_size = 20
    all_shots = []
    slot_id = 1
    
    for i in range(0, len(segments), block_size):
        block = segments[i : i + block_size]
        user_msg = "\n".join([f"[{s['start']:.2f} - {s['end']:.2f}]: {s['text']}" for s in block])
        
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
    custom_block = _build_context_block(video_topic, custom_instructions)
    if custom_block:
        system_prompt = system_prompt_template.replace("{custom_instructions_block}", custom_block)
    else:
        system_prompt = system_prompt_template.replace("\n\n{custom_instructions_block}\n\n", "\n\n")

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

    return shots


def format_time(seconds) -> str:
    total = int(float(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

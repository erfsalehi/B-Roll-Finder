import os
import json
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
    system_prompt = system_prompt_template.replace("{custom_instructions_block}", custom_block)
    
    # Split text to bypass Groq 6k TPM limits. ~250 words per chunk, respecting sentences.
    from core.timing import split_script_into_smart_blocks
    blocks = split_script_into_smart_blocks(script_text, max_words=250)
    
    total_blocks = len(blocks)
    all_shots = []
    current_time = start_offset
    slot_id = 1
    
    for i, block in enumerate(blocks):
        try:
            # We explicitly tell the LLM the WPS so it can calculate duration_needed_sec internally if it wants, 
            # but we will definitively calculate timestamps here anyway.
            user_msg = f"WPS: {wps:.2f}\\nSCRIPT CHUNK:\\n{block}"
            
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
                
                # Build the slot matching the new architecture
                all_shots.append({
                    "slot_id": slot_id,
                    "timestamp": int(current_time),
                    "end_timestamp": int(end_time),
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
                
        except Exception as e:
            print(f"Error processing block {i}: {e}")
            block_words = len(block.split())
            duration = block_words / wps if wps > 0 else 5.0
            end_time = current_time + duration
            all_shots.append({
                "slot_id": slot_id,
                "timestamp": int(current_time),
                "end_timestamp": int(end_time),
                "timestamp_start_str": format_time(int(current_time)),
                "timestamp_end_str": format_time(int(end_time)),
                "text": block,
                "shot_intent": "Error Fallback",
                "shot_type": "medium",
                "search_queries": [f"Error: {e}"],
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
    system_prompt = system_prompt_template.replace("{custom_instructions_block}", custom_block)
    # The director prompt might need a small tweak to understand [start - end] segments, 
    # but the base prompt is usually smart enough if we explain the format.
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
                    "timestamp": int(float(s_time)),
                    "end_timestamp": int(float(e_time)),
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
            
        if progress_callback:
            progress_callback((i + len(block)) / len(segments))
            
    return all_shots

def format_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

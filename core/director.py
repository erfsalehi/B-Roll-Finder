import os
import json
from groq import Groq

def load_director_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'director.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_shot_list(script_text: str, wps: float, api_key: str, progress_callback=None, custom_instructions: str = "") -> list:
    if not api_key:
        raise ValueError("Groq API key is missing.")
        
    client = Groq(api_key=api_key)
    system_prompt_template = load_director_prompt()
    
    custom_block = ""
    if custom_instructions and custom_instructions.strip():
        custom_block = f"USER CUSTOM INSTRUCTIONS: {custom_instructions.strip()}\\nPlease ensure the shot list strictly adheres to these style guidelines."
    system_prompt = system_prompt_template.replace("{custom_instructions_block}", custom_block)
    
    # Split text to bypass Groq 6k TPM limits. ~250 words per chunk.
    words = script_text.split()
    block_size = 250
    blocks = [" ".join(words[i:i+block_size]) for i in range(0, len(words), block_size)]
    
    total_blocks = len(blocks)
    all_shots = []
    current_time = 0.0
    slot_id = 1
    
    for i, block in enumerate(blocks):
        try:
            # We explicitly tell the LLM the WPS so it can calculate duration_needed_sec internally if it wants, 
            # but we will definitively calculate timestamps here anyway.
            user_msg = f"WPS: {wps:.2f}\\nSCRIPT CHUNK:\\n{block}"
            
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.4, # Lower temp for strict JSON adherence and structure
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content
            data = json.loads(response_text)
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

def format_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

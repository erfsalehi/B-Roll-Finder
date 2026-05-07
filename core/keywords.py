import os
import re
import json
import requests
from groq import Groq, RateLimitError as GroqRateLimitError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, retry_if_not_exception_type

GROQ_MODEL       = "llama-3.3-70b-versatile"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_BASE  = "https://openrouter.ai/api/v1/chat/completions"

def load_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'visual_keywords.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception)
)
def call_groq_api(client: Groq, system_prompt: str, user_content: str) -> str:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1500,
    )
    return response.choices[0].message.content

def format_time(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"[{h:02d}:{m:02d}:{s:02d}]"

def generate_keywords_for_slots(slots: list, api_key: str, num_alternatives: int = 3, batch_size: int = 10, progress_callback=None, custom_instructions: str = "") -> list:
    if not api_key:
        raise ValueError("Groq API key is missing.")
        
    client = Groq(api_key=api_key)
    system_prompt_template = load_prompt()
    system_prompt = system_prompt_template.replace("{num_alternatives}", str(num_alternatives))
    
    if custom_instructions and custom_instructions.strip():
        system_prompt += f"\n\nUSER CUSTOM INSTRUCTIONS: {custom_instructions.strip()}\nPlease ensure the generated keywords strictly adhere to these specific style or content guidelines."
    
    total_slots = len(slots)
    
    for i in range(0, total_slots, batch_size):
        batch = slots[i:i+batch_size]
        
        user_content = ""
        for slot in batch:
            ts_str = format_time(slot['timestamp'])
            user_content += f"{ts_str}\nScript: \"{slot['text']}\"\n\n"
            
        try:
            response_text = call_groq_api(client, system_prompt, user_content)
            parse_groq_response(response_text, batch, num_alternatives)
            
        except Exception as e:
            # If batch fails after retries, assign error message to slots
            for slot in batch:
                if 'keywords' not in slot:
                    slot['keywords'] = [f"Error: {str(e)}"]
                    
        if progress_callback:
            # Progress from 0.0 to 1.0
            progress_callback(min(1.0, (i + len(batch)) / total_slots))
            
    return slots

def parse_groq_response(response_text: str, original_batch: list, num_alternatives: int):
    """
    Parses the response and mutates the dictionaries in original_batch to include 'keywords' list.
    """
    for slot in original_batch:
        if 'keywords' not in slot:
            slot['keywords'] = []
            
    lines = response_text.strip().split('\n')
    current_ts = None
    current_keywords = []
    
    ts_to_slot = {format_time(s['timestamp']): s for s in original_batch}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match timestamp exactly like [00:00:00]
        ts_match = re.match(r'^\[(\d{2}:\d{2}:\d{2})\]', line)
        if ts_match:
            if current_ts and current_ts in ts_to_slot:
                ts_to_slot[current_ts]['keywords'] = current_keywords[:num_alternatives]
            current_ts = ts_match.group(0)
            current_keywords = []
        else:
            if current_ts:
                # Remove common list prefixes AI might add (e.g. "1.", "- ", "* ")
                clean_line = re.sub(r'^[\d\.\-\*]+\s*', '', line).strip()
                # Remove quotes if AI quoted the keywords
                clean_line = clean_line.strip('"\'')
                if clean_line:
                    current_keywords.append(clean_line)
                    
    # Handle the final batch item
    if current_ts and current_ts in ts_to_slot:
        ts_to_slot[current_ts]['keywords'] = current_keywords[:num_alternatives]
        
    # Fallback if any slot is missing keywords (e.g. AI skipped it)
    for slot in original_batch:
        if not slot['keywords']:
            slot['keywords'] = [f"No keywords generated for {slot['text']}"]

def load_json_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'visual_keywords_json.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    # Don't retry rate-limit errors — fall back to OpenRouter instead
    retry=retry_if_not_exception_type(GroqRateLimitError)
)
def _call_groq_json(client: Groq, system_prompt: str, block: str) -> dict:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": block}
        ],
        model=GROQ_MODEL,
        temperature=0.7,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def _call_openrouter_json(system_prompt: str, user_content: str,
                          temperature: float = 0.7, max_tokens: int = 2000) -> dict:
    """Calls OpenRouter with the shared fallback model. Raises on failure."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OpenRouter API key is missing. Add it in Step 1 settings.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":           OPENROUTER_MODEL,
        "temperature":     temperature,
        "max_tokens":      max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    }
    resp = requests.post(OPENROUTER_BASE, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


def _call_llm_json(client: Groq, system_prompt: str, user_content: str,
                   temperature: float = 0.7, max_tokens: int = 2000) -> dict:
    """
    Try Groq first. On rate-limit, fall back to OpenRouter automatically.
    All other callers should use this instead of _call_groq_json directly.
    """
    try:
        return _call_groq_json(client, system_prompt, user_content)
    except GroqRateLimitError:
        print("Groq rate limit hit — falling back to OpenRouter.")
        return _call_openrouter_json(system_prompt, user_content, temperature, max_tokens)

def generate_keywords_with_ai_chunking(script_text: str, wps: float, api_key: str, num_alternatives: int = 3, progress_callback=None, custom_instructions: str = "", start_offset: float = 0.0) -> list:
    if not api_key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=api_key)
    system_prompt_template = load_json_prompt()
    system_prompt = system_prompt_template.replace("{num_alternatives}", str(num_alternatives))

    custom_block = ""
    if custom_instructions and custom_instructions.strip():
        custom_block = f"USER CUSTOM INSTRUCTIONS: {custom_instructions.strip()}\nPlease ensure the generated keywords strictly adhere to these specific style or content guidelines."
    system_prompt = system_prompt.replace("{custom_instructions_block}", custom_block)

    # Split text into blocks that respect sentence boundaries (~150 words per block)
    from core.timing import split_script_into_smart_blocks
    blocks = split_script_into_smart_blocks(script_text, max_words=150)

    total_blocks = len(blocks)
    all_slots = []
    current_time = start_offset

    for i, block in enumerate(blocks):
        try:
            data = _call_llm_json(client, system_prompt, block)
            chunks = data.get("chunks", [])
            
            for chunk in chunks:
                chunk_text = chunk.get("text", "")
                keywords = chunk.get("keywords", [])
                
                # Calculate duration based on words
                chunk_words = len(chunk_text.split())
                if chunk_words == 0:
                    continue
                    
                duration = chunk_words / wps if wps > 0 else 1.0
                end_time = current_time + duration
                
                all_slots.append({
                    "timestamp": int(current_time),
                    "end_timestamp": int(end_time),
                    "text": chunk_text,
                    "keywords": keywords[:num_alternatives]
                })
                
                current_time = end_time
                
        except Exception as e:
            # Fallback if a block fails
            print(f"Error processing block {i}: {e}")
            block_words = len(block.split())
            duration = block_words / wps if wps > 0 else 5.0
            end_time = current_time + duration
            all_slots.append({
                "timestamp": int(current_time),
                "end_timestamp": int(end_time),
                "text": block,
                "keywords": [f"Error generating keywords: {e}"]
            })
            current_time = end_time
            
        if progress_callback:
            progress_callback(min(1.0, (i + 1) / total_blocks))
            
    return all_slots

def generate_global_themes(script_text: str, api_key: str, num_themes: int = 5) -> list:
    import json
    if not api_key:
        raise ValueError("Groq API key is missing.")
        
    client = Groq(api_key=api_key)
    
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'global_themes.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read().replace("{num_themes}", str(num_themes))
        
    try:
        data = _call_llm_json(client, system_prompt, f"SCRIPT:\n{script_text}")
        themes = data.get("themes", [])
        for t in themes:
            t.setdefault('keywords', [])
            t.setdefault('video_results', [])
        return themes
    except Exception as e:
        print(f"Error generating global themes: {e}")
        return []

def generate_keywords_from_transcription(segments: list, api_key: str, num_alternatives: int = 3, progress_callback=None, custom_instructions: str = "") -> list:
    """
    Groups transcription segments into meaningful visual beats and generates keywords.
    """
    import json
    if not api_key:
        raise ValueError("Groq API key is missing.")
        
    client = Groq(api_key=api_key)
    
    # We use a specialized prompt for this that understands [start-end] segments
    system_prompt = f"""You are an expert video editor. You will receive a list of timestamped transcription segments.
Your task is to group these segments into "visual beats" (meaningful shots) and generate {num_alternatives} B-roll search keywords for each.

CRITICAL RULES:
1. Output MUST be valid JSON with a "chunks" array.
2. Each chunk must have:
   - "text": The combined text of the segments in this beat.
   - "start": The start timestamp of the first segment in the beat.
   - "end": The end timestamp of the last segment in the beat.
   - "keywords": Exactly {num_alternatives} search phrases.
3. NEVER break a sentence across two chunks. 
4. DO NOT change the text of the segments.
5. If a segment is very short, combine it with the next one to make a meaningful shot (typically 3-8 seconds).

{custom_instructions}
"""
    
    # Group segments into blocks of ~20 segments to avoid context limits
    block_size = 20
    all_slots = []
    
    for i in range(0, len(segments), block_size):
        block = segments[i : i + block_size]
        user_content = "\n".join([f"[{s['start']:.2f} - {s['end']:.2f}]: {s['text']}" for s in block])
        
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            chunks = data.get("chunks", [])
            
            for chunk in chunks:
                all_slots.append({
                    "timestamp": int(chunk.get("start", 0)),
                    "end_timestamp": int(chunk.get("end", 0)),
                    "text": chunk.get("text", ""),
                    "keywords": chunk.get("keywords", [])[:num_alternatives]
                })
        except Exception as e:
            print(f"Error processing transcription block {i}: {e}")
            
        if progress_callback:
            progress_callback((i + len(block)) / len(segments))
            
    return all_slots

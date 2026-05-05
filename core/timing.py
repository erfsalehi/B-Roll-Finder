import os
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.mp4 import MP4

def get_audio_duration(file_path: str) -> float:
    """
    Extracts the duration of an audio file in seconds using mutagen.
    Supports mp3, wav, and m4a/mp4.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.mp3':
            audio = MP3(file_path)
            return audio.info.length
        elif ext == '.wav':
            audio = WAVE(file_path)
            return audio.info.length
        elif ext in ['.m4a', '.mp4']:
            audio = MP4(file_path)
            return audio.info.length
        else:
            raise ValueError(f"Unsupported audio format: {ext}")
    except Exception as e:
        raise RuntimeError(f"Error reading audio duration for {file_path}: {e}")

def calculate_wps(script_text: str, duration_seconds: float) -> float:
    words = script_text.split()
    total_words = len(words)
    if total_words == 0 or duration_seconds <= 0:
        return 0.0
    return total_words / duration_seconds

def parse_script_to_slots(script_text: str, duration_seconds: float, intro_duration: float = 30.0, intro_interval: float = 1.0, body_interval: float = 2.0) -> list[dict]:
    """
    Parses a full script into timestamped slots based on average words per second.
    """
    words = script_text.split()
    total_words = len(words)
    if total_words == 0 or duration_seconds <= 0:
        return []
        
    wps = total_words / duration_seconds
    
    slots = []
    current_time = 0.0
    current_word_idx = 0
    
    while current_time < duration_seconds and current_word_idx < total_words:
        interval = intro_interval if current_time < intro_duration else body_interval
        
        # Determine how many words fall into this interval
        words_in_interval = int(wps * interval)
        if words_in_interval == 0 and current_word_idx < total_words:
            words_in_interval = 1
            
        end_idx = min(current_word_idx + words_in_interval, total_words)
        
        # Last interval should consume all remaining words
        if current_time + interval >= duration_seconds:
            end_idx = total_words
            
        chunk = " ".join(words[current_word_idx:end_idx])
        slots.append({
            "timestamp": int(current_time),
            "end_timestamp": int(min(current_time + interval, duration_seconds)),
            "text": chunk
        })
        
        current_word_idx = end_idx
        current_time += interval
        
    return slots

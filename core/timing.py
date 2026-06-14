import os
import re
from mutagen.mp3 import MP3
from mutagen.wave import WAVE
from mutagen.mp4 import MP4

def split_script_into_smart_blocks(script_text: str, max_words: int = 150) -> list[str]:
    """
    Splits a script into blocks that respect sentence boundaries.
    Attempts to keep blocks close to max_words but won't break a sentence.
    """
    # Split by sentences (period, exclamation, question mark followed by space)
    # We use lookbehind to keep the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', script_text.strip())
    
    blocks = []
    current_block = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If adding this sentence exceeds max_words and we already have some content,
        # finish the current block.
        if current_word_count + sentence_word_count > max_words and current_block:
            blocks.append(" ".join(current_block))
            current_block = []
            current_word_count = 0
            
        current_block.append(sentence)
        current_word_count += sentence_word_count
        
    # Add the final block if it exists
    if current_block:
        blocks.append(" ".join(current_block))
        
    return blocks

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

def slice_script_by_time(script_text: str, total_duration: float, start_time: float, end_time: float) -> str:
    """
    Extracts the portion of the script corresponding to the given time range.
    Uses linear WPS estimation.
    """
    words = script_text.split()
    total_words = len(words)
    if total_words == 0 or total_duration <= 0:
        return ""
        
    wps = total_words / total_duration
    
    start_idx = int(start_time * wps)
    end_idx = int(end_time * wps)
    
    # Ensure indices are within bounds
    start_idx = max(0, min(start_idx, total_words))
    end_idx = max(0, min(end_idx, total_words))
    
    if start_idx >= end_idx:
        # If the range is too small, at least return one word if possible
        if start_idx < total_words:
            return words[start_idx]
        return ""
        
    return " ".join(words[start_idx:end_idx])

_SENTENCE_END = (".", "!", "?", "。", "！", "？")


def chunk_segments_by_duration(segments: list, target_duration: float = 120.0,
                               max_duration: float = 180.0) -> list[dict]:
    """Group Whisper segments into manageable chunks at sentence boundaries.

    Each chunk targets ``target_duration`` seconds. We keep accumulating
    segments until the running duration ≥ target AND the latest segment
    ends a sentence (punctuated with ``.!?``); only then do we close the
    chunk. If we hit ``max_duration`` without finding a sentence boundary
    we force-cut to avoid one runaway chunk swallowing the whole script.

    Returns a list of chunk dicts:
        {
            "start":    float,         # seconds
            "end":      float,
            "text":     str,           # concatenated segment text
            "segments": list[dict],    # the original segments inside
        }
    """
    if not segments:
        return []

    chunks: list[dict] = []
    cur_segs: list[dict] = []
    cur_start: float | None = None

    def _close():
        if not cur_segs:
            return
        chunks.append({
            "start":    cur_start,
            "end":      cur_segs[-1]["end"],
            "text":     " ".join(s["text"].strip() for s in cur_segs).strip(),
            "segments": list(cur_segs),
        })

    for seg in segments:
        if cur_start is None:
            cur_start = seg["start"]
        cur_segs.append(seg)
        cur_dur = seg["end"] - cur_start

        text = seg["text"].strip()
        ends_sentence = bool(text) and text[-1] in _SENTENCE_END

        if (cur_dur >= target_duration and ends_sentence) or cur_dur >= max_duration:
            _close()
            cur_segs = []
            cur_start = None

    # Tail — anything left after the loop becomes a final chunk regardless
    # of whether it hit the target. Otherwise we'd silently drop content.
    _close()
    return chunks


def speech_onsets_from_segments(segments: list, min_silence: float = 0.35) -> list:
    """Absolute times (seconds) where the speaker resumes after a silence.

    A silence is a gap ≥ ``min_silence`` between one spoken word and the next;
    the *end* of that gap — the start of the next word — is a speech onset. These
    are the natural points to start a fresh b-roll clip so a visible cut lands
    exactly when the narration picks back up.

    Prefers per-word timings (``segment['words']`` from Whisper word
    granularity); falls back to segment boundaries when words are absent. Returns
    a sorted list of onset times, including the very first word/segment start.
    """
    if not segments:
        return []

    # Flatten words across segments, time-ordered.
    words = []
    for s in segments:
        for w in (s.get("words") or []):
            if w.get("start") is None:
                continue
            try:
                words.append((float(w["start"]), float(w.get("end", w["start"]))))
            except (TypeError, ValueError):
                continue
    words.sort()

    onsets = []
    if words:
        onsets.append(words[0][0])
        prev_end = words[0][1]
        for start, end in words[1:]:
            if start - prev_end >= min_silence:
                onsets.append(start)
            prev_end = max(prev_end, end)
    else:
        # No word timing — use segment starts that follow a real pause.
        prev_end = None
        for s in segments:
            try:
                st, en = float(s["start"]), float(s["end"])
            except (TypeError, ValueError, KeyError):
                continue
            if prev_end is None or st - prev_end >= min_silence:
                onsets.append(st)
            prev_end = en

    # De-dupe and sort (defensive against overlapping word lists).
    return sorted(set(round(o, 4) for o in onsets))


def attach_speech_onsets(shots: list, segments: list,
                         min_silence: float = 0.35) -> list:
    """Annotate each shot with the speech onsets that fall inside its voice range
    (``shot['speech_onsets']``), so :func:`core.output.generate_fcpxml` can begin
    a new sub-clip whenever the speaker resumes. Mutates and returns ``shots``."""
    onsets = speech_onsets_from_segments(segments, min_silence)
    for s in shots:
        try:
            st = float(s.get("timestamp", 0))
            en = float(s.get("end_timestamp", st))
        except (TypeError, ValueError):
            s["speech_onsets"] = []
            continue
        # Interior onsets only — the shot's own start/end are already boundaries.
        s["speech_onsets"] = [o for o in onsets if st < o < en]
    return shots


def parse_script_to_slots(script_text: str, duration_seconds: float, intro_duration: float = 30.0, intro_interval: float = 1.0, body_interval: float = 2.0, start_offset: float = 0.0) -> list[dict]:
    """
    Parses a full script into timestamped slots based on average words per second.
    """
    words = script_text.split()
    total_words = len(words)
    if total_words == 0 or duration_seconds <= 0:
        return []
        
    wps = total_words / duration_seconds
    
    slots = []
    current_time = start_offset
    current_word_idx = 0
    
    # Absolute end time for this portion
    max_time = start_offset + duration_seconds
    
    while current_time < max_time and current_word_idx < total_words:
        interval = intro_interval if current_time < (start_offset + intro_duration) else body_interval
        
        # Determine how many words fall into this interval
        words_in_interval = int(wps * interval)
        if words_in_interval == 0 and current_word_idx < total_words:
            words_in_interval = 1
            
        end_idx = min(current_word_idx + words_in_interval, total_words)
        
        # Last interval should consume all remaining words
        if current_time + interval >= max_time:
            end_idx = total_words
            
        chunk = " ".join(words[current_word_idx:end_idx])
        slots.append({
            "timestamp": round(float(current_time), 3),
            "end_timestamp": round(float(min(current_time + interval, max_time)), 3),
            "text": chunk
        })
        
        current_word_idx = end_idx
        current_time += interval
        
    return slots

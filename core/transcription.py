import os
from groq import Groq
from core.ffmpeg_utils import compress_audio_for_whisper

def transcribe_audio(file_path: str, api_key: str) -> list[dict]:
    """
    Transcribes audio using Groq's Whisper API.
    Returns a list of segments: [{'start': float, 'end': float, 'text': str}, ...]
    """
    if not api_key:
        raise ValueError("Groq API key is required for transcription.")
        
    client = Groq(api_key=api_key)
    
    # Check file size (Groq limit is 25MB)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    processed_path = file_path
    
    # If > 24MB (buffer), compress it
    if file_size_mb > 24.0:
        processed_path = compress_audio_for_whisper(file_path)
        
    try:
        with open(processed_path, "rb") as audio_file:
            # We use verbose_json to get segment-level timestamps
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(processed_path), audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",
            )
            
        # Clean up compressed file if created
        if processed_path != file_path and os.path.exists(processed_path):
            os.remove(processed_path)
            
        # Map to a clean list of segments
        segments = []
        # 'segments' is a list of dicts in the verbose_json response
        raw_segments = getattr(transcription, 'segments', [])
        for seg in raw_segments:
            segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip()
            })

        # Record audio length for per-job Whisper cost accounting (best-effort).
        try:
            from core import usage
            dur = getattr(transcription, 'duration', None)
            if dur is None and segments:
                dur = segments[-1]['end']
            usage.record_transcription("groq", "whisper-large-v3", dur or 0.0)
        except Exception:
            pass

        return segments
        
    except Exception as e:
        if processed_path != file_path and os.path.exists(processed_path):
            os.remove(processed_path)
        raise RuntimeError(f"Transcription failed: {e}")

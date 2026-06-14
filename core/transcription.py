import os
from groq import Groq
from core.ffmpeg_utils import compress_audio_for_whisper

def _attr(obj, key):
    """Read ``key`` from a dict or an SDK object uniformly."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def transcribe_audio(file_path: str, api_key: str) -> list[dict]:
    """
    Transcribes audio using Groq's Whisper API.

    Returns a list of segments, each:
        {'start': float, 'end': float, 'text': str, 'words': [...]}
    where ``words`` is the per-word timing within that segment
    ([{'word': str, 'start': float, 'end': float}, ...], possibly empty).

    Word-level timestamps are what let text overlays snap to the exact moment a
    phrase is spoken instead of guessing its position inside a multi-second
    sentence — see core.overlays_remotion._align_overlays_to_words.
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
            audio_bytes = audio_file.read()
        # verbose_json + word granularity gives both segment and per-word times.
        # Some Whisper deployments/SDKs reject the granularities kwarg (SDK
        # signature mismatch or an API-level 400) — fall back to a plain
        # verbose_json call (segment-only) so transcription never fails just
        # because word timing is unavailable. A second failure is the real error
        # and propagates.
        try:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(processed_path), audio_bytes),
                model="whisper-large-v3",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
        except Exception:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(processed_path), audio_bytes),
                model="whisper-large-v3",
                response_format="verbose_json",
            )

        # Clean up compressed file if created
        if processed_path != file_path and os.path.exists(processed_path):
            os.remove(processed_path)

        # Flatten the word list (top-level when word granularity is honored) so we
        # can attach each word to the segment it falls inside.
        flat_words = []
        for w in (getattr(transcription, 'words', None) or []):
            wt, ws, we = _attr(w, 'word'), _attr(w, 'start'), _attr(w, 'end')
            if wt is None or ws is None:
                continue
            try:
                flat_words.append({
                    'word': str(wt).strip(),
                    'start': float(ws),
                    'end': float(we if we is not None else ws),
                })
            except (TypeError, ValueError):
                continue
        flat_words.sort(key=lambda x: x['start'])

        # Map to a clean list of segments
        segments = []
        # 'segments' is a list of dicts in the verbose_json response
        raw_segments = getattr(transcription, 'segments', []) or []
        for seg in raw_segments:
            s_start, s_end = float(seg['start']), float(seg['end'])
            # Prefer words the API already nested in the segment; otherwise slice
            # them out of the flat list by their start time.
            seg_words = _attr(seg, 'words')
            if seg_words:
                words = []
                for w in seg_words:
                    wt, ws, we = _attr(w, 'word'), _attr(w, 'start'), _attr(w, 'end')
                    if wt is None or ws is None:
                        continue
                    words.append({'word': str(wt).strip(), 'start': float(ws),
                                  'end': float(we if we is not None else ws)})
            else:
                words = [w for w in flat_words
                         if s_start - 0.05 <= w['start'] < s_end + 0.05]
            segments.append({
                'start': s_start,
                'end': s_end,
                'text': seg['text'].strip(),
                'words': words,
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

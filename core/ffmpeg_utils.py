import subprocess
import os
import time
import threading

# Maximum time (seconds) to allow FFmpeg to normalize a single file.
# 10 minutes is generous even for large 4K sources on slow hardware.
_NORMALIZE_TIMEOUT = 600

# ── CPU guards for libx264 re-encoding ───────────────────────────────────────
# libx264 is itself multi-threaded and grabs every core by default. The
# DownloadManager can run up to 10 downloads at once (app.py), each able to
# trigger a normalize, so without limits N encodes × all-cores thrash the box.
# Cap threads per ffmpeg process and the number of simultaneous encodes so the
# product stays near the core count, leaving headroom for downloads.
def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return int(v) if v.isdigit() and int(v) > 0 else default

def _default_ffmpeg_threads() -> int:
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    return max(1, n // 4)

_FFMPEG_THREADS = _env_int("BROLL_FFMPEG_THREADS", _default_ffmpeg_threads())
_normalize_sem = threading.Semaphore(_env_int("BROLL_NORMALIZE_CONCURRENCY", 2))

def normalize_video(input_path: str, target_res: str = "1920x1080", task_state: dict = None) -> str:
    """
    Uses FFmpeg to conform a video to exactly target_res (width x height).
    Adds black bars (letterbox/pillarbox) to maintain aspect ratio.
    Supports cancellation via task_state and enforces a timeout so it never hangs.
    Returns the path to the normalized file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    width, height = map(int, target_res.split('x'))

    # ── Fast path: skip re-encoding if the file is already conformant ────────
    # Director Mode can queue 15-20 clips; re-encoding every one through
    # libx264 saturates the CPU. Most downloaded clips (Pexels/Pixabay 1080p,
    # and YouTube clips we already pulled as avc1) are *already* the target
    # resolution in H.264/yuv420p — nothing to do. We check codec + pixel
    # format too, not just dimensions, so a same-size-but-VP9/HEVC clip (which
    # Premiere chokes on) is still conformed.
    meta = get_video_metadata(input_path)
    already_conformant = (
        meta.get("width") == width
        and meta.get("height") == height
        and meta.get("vcodec") in ("h264", "avc1")
        and meta.get("pix_fmt") in ("yuv420p", "yuvj420p")
    )
    if already_conformant:
        return input_path

    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_normalized.mp4"

    cmd = [
        "ffmpeg", "-y",
        # Cap libx264's own thread pool so concurrent encodes don't each grab
        # every core (see _FFMPEG_THREADS).
        "-threads", str(_FFMPEG_THREADS),
        "-i", input_path,
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"format=yuv420p"
        ),
        "-c:v", "libx264",
        # 'veryfast' encodes ~3-4x quicker than 'fast' for a small size bump —
        # a good trade for B-roll that gets re-encoded again inside Premiere.
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]

    # Limit how many libx264 encodes run at once (see _normalize_sem). The
    # semaphore wraps only the encode, not the ffprobe fast-path above, so
    # already-conformant clips never wait on a slot.
    with _normalize_sem:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        start = time.monotonic()

        try:
            while proc.poll() is None:
                # Respect cancellation from the download manager
                if task_state and task_state.get('status') == 'cancelled':
                    proc.kill()
                    proc.wait()
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    return input_path  # leave original; caller will clean up

                # Hard timeout — never hang forever
                if time.monotonic() - start > _NORMALIZE_TIMEOUT:
                    proc.kill()
                    proc.wait()
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    raise TimeoutError(
                        f"FFmpeg timed out after {_NORMALIZE_TIMEOUT}s normalizing {os.path.basename(input_path)}"
                    )

                time.sleep(0.5)

            if proc.returncode != 0:
                stderr = proc.stderr.read().decode("utf-8", errors="replace")
                raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)

            # Swap normalized file over the original
            os.remove(input_path)
            os.rename(output_path, input_path)
            return input_path

        except Exception:
            # Clean up partial output on any failure
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise

def compress_audio_for_whisper(input_path: str) -> str:
    """
    Compresses an audio file to a low-bitrate mono MP3 (16kHz).
    This ensures the file is under the 25MB Groq/Whisper limit while remaining 
    perfectly legible for AI transcription.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_for_whisper.mp3"
    
    # 16kHz mono 32k bitrate is plenty for transcription
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "32k",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg compression error: {e.stderr}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e
def get_video_metadata(path: str) -> dict:
    """
    Uses ffprobe to extract duration, width, and height from a video file.
    Returns a dict with 'duration' (float), 'width' (int), and 'height' (int).
    Returns defaults if ffprobe fails or the file doesn't exist.
    """
    defaults = {"duration": 3600.0, "width": 1920, "height": 1080,
                "vcodec": "", "pix_fmt": ""}
    if not path or not os.path.exists(path):
        return defaults

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=width,height,codec_name,pix_fmt",
        "-of", "json", path
    ]
    try:
        import json
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        meta = {}
        # Try to get duration from format
        if "format" in data and "duration" in data["format"]:
            meta["duration"] = float(data["format"]["duration"])

        # Try to get geometry/codec from the first video stream
        if "streams" in data:
            for s in data["streams"]:
                if s.get("width") and s.get("height"):
                    meta["width"]   = int(s["width"])
                    meta["height"]  = int(s["height"])
                    meta["vcodec"]  = (s.get("codec_name") or "").lower()
                    meta["pix_fmt"] = (s.get("pix_fmt") or "").lower()
                    break

        return {**defaults, **meta}
    except Exception as e:
        print(f"[get_video_metadata] Error probing {path}: {e}")
        return defaults

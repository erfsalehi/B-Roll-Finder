import subprocess
import os
import time

# Maximum time (seconds) to allow FFmpeg to normalize a single file.
# 10 minutes is generous even for large 4K sources on slow hardware.
_NORMALIZE_TIMEOUT = 600

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
    base, _ = os.path.splitext(input_path)
    output_path = f"{base}_normalized.mp4"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"format=yuv420p"
        ),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]

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

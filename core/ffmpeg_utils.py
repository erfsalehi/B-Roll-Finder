import subprocess
import os

def normalize_video(input_path: str, target_res: str = "1920x1080") -> str:
    """
    Uses FFmpeg to conform a video to exactly target_res (width x height).
    Adds black bars (letterbox/pillarbox) to maintain aspect ratio.
    Returns the path to the normalized file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    width, height = map(int, target_res.split('x'))
    
    # Create a temporary output path
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_normalized.mp4"
    
    # FFmpeg command:
    # 1. Scale to fit within target width/height
    # 2. Pad to exactly target width/height
    # 3. Use yuv420p for maximum compatibility (Premiere Pro)
    # 4. H.264 codec
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Replace original with normalized
        os.remove(input_path)
        os.rename(output_path, input_path)
        return input_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e

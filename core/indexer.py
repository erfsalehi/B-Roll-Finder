import os
import subprocess
import json
import time
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from core.vector_db import VectorDB


class VideoIndexer:
    def __init__(self, download_dir="downloads/library", cache_dir=".cache/library"):
        self.download_dir = download_dir
        self.cache_dir = cache_dir
        # Use permanent=True so all indexed clips go to ~/.broll_director/
        self.db = VectorDB(permanent=True)
        self.model = None  # Lazy load

        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer('clip-ViT-B-32')

    def index_video(self, video_path, video_url=None, video_title=None, progress_cb=None):
        """Processes a single video: segments it, embeds frames, and adds to DB."""
        self._load_model()

        video_name = os.path.basename(video_path)
        safe_name = "".join([c if c.isalnum() else "_" for c in video_name])
        segment_dir = os.path.join(self.cache_dir, safe_name)
        os.makedirs(segment_dir, exist_ok=True)

        if progress_cb:
            progress_cb(0.1, "Analyzing scenes…")

        # Detect scenes and extract frames at each cut
        timestamps_file = os.path.join(segment_dir, "pts.txt")
        frame_pattern = os.path.join(segment_dir, "scene_%03d.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"select='gt(scene,0.4)',metadata=print:file={timestamps_file}",
            "-vsync", "vfr",
            frame_pattern,
        ]
        subprocess.run(cmd, capture_output=True)

        frames = sorted([f for f in os.listdir(segment_dir) if f.endswith(".jpg")])
        total = len(frames)

        if total == 0:
            # Fallback: take one frame at 2s
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:02", "-i", video_path,
                "-vframes", "1", os.path.join(segment_dir, "scene_001.jpg"),
            ], capture_output=True)
            frames = sorted([f for f in os.listdir(segment_dir) if f.endswith(".jpg")])
            total = len(frames)

        vectors = []
        metas = []

        for i, frame in enumerate(frames):
            frame_path = os.path.join(segment_dir, frame)
            if progress_cb:
                progress_cb(0.2 + (0.7 * (i / max(total, 1))), f"Analyzing scene {i+1}/{total}…")

            img = Image.open(frame_path)
            emb = self.model.encode(img)
            vectors.append(emb)
            metas.append({
                "video_path": video_path,
                "video_url": video_url,
                "video_title": video_title or video_name,
                "frame_path": frame_path,
                "scene_index": i,
            })

        if vectors:
            self.db.add_vectors(vectors, metas)

        if progress_cb:
            progress_cb(1.0, f"Indexed {total} scenes.")
        return total

    def download_and_index_youtube(self, url, progress_cb=None):
        """Downloads a YouTube video at 720p and indexes it into the Smart Library."""
        import yt_dlp
        from core.youtube import _get_cookie_opts

        if progress_cb:
            progress_cb(0.05, "Downloading YouTube video…")

        ydl_opts = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "outtmpl": os.path.join(self.download_dir, "%(title)s.%(ext)s"),
            "noplaylist": True,
            "merge_output_format": "mp4",
            "socket_timeout": 30,
            **_get_cookie_opts(),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            video_title = info.get("title", "YouTube Video")

        return self.index_video(video_path, video_url=url, video_title=video_title, progress_cb=progress_cb)

    # ── New: Section download ──────────────────────────────────────────────────

    def download_section(self, url, start_time, end_time, quality, output_path, progress_cb=None):
        """
        Downloads only the specified time range from a YouTube video at full quality.
        Uses yt-dlp's download_ranges to avoid fetching the entire file.
        """
        import yt_dlp
        from core.youtube import _get_cookie_opts

        if progress_cb:
            progress_cb(0.1, f"Downloading {int(start_time//60):02d}:{int(start_time%60):02d}"
                             f" → {int(end_time//60):02d}:{int(end_time%60):02d} at {quality}p…")

        fmt = f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]"

        ydl_opts = {
            "format": fmt,
            "outtmpl": output_path,
            "download_ranges": yt_dlp.utils.download_range_func(
                chapters=None,
                ranges=[(start_time, end_time)],
            ),
            "force_keyframes_at_cuts": True,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 30,
            **_get_cookie_opts(),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if progress_cb:
            progress_cb(1.0, "Download complete.")

    # ── New: Embed downloaded clip and add to permanent library ───────────────

    def embed_and_index(self, video_path, video_url=None, video_title=None,
                        timestamp_start=None, progress_cb=None):
        """
        Embeds a downloaded clip and adds it to the permanent Smart Library.
        Called automatically after a successful Step 6 section download.
        """
        self._load_model()

        try:
            video_name = os.path.basename(video_path)
            safe_name = "".join(c if c.isalnum() else "_" for c in video_name)
            frame_dir = os.path.join(self.cache_dir, safe_name + "_embed")
            os.makedirs(frame_dir, exist_ok=True)

            # Extract representative frames
            frame_path = os.path.join(frame_dir, "frame_001.jpg")
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:02", "-i", video_path,
                "-vframes", "1", frame_path,
            ], capture_output=True)

            if not os.path.exists(frame_path):
                return 0

            img = Image.open(frame_path)
            emb = self.model.encode(img)

            self.db.add_vectors([emb], [{
                "video_path": video_path,
                "video_url": video_url,
                "video_title": video_title or video_name,
                "frame_path": frame_path,
                "timestamp_start": timestamp_start,
                "scene_index": 0,
            }])

            if progress_cb:
                progress_cb(1.0, "Added to Smart Library.")
            return 1

        except Exception as e:
            print(f"[VideoIndexer] embed_and_index failed: {e}")
            return 0


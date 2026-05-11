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
        self.db = VectorDB()
        self.model = None # Lazy load
        
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

    def _load_model(self):
        if self.model is None:
            # Use multi-modal CLIP model from sentence-transformers
            self.model = SentenceTransformer('clip-ViT-B-32')

    def index_video(self, video_path, video_url=None, video_title=None, progress_cb=None):
        """
        Processes a single video: segments it, embeds frames, and adds to DB.
        """
        self._load_model()
        
        video_name = os.path.basename(video_path)
        safe_name = "".join([c if c.isalnum() else "_" for c in video_name])
        segment_dir = os.path.join(self.cache_dir, safe_name)
        os.makedirs(segment_dir, exist_ok=True)
        
        if progress_cb: progress_cb(0.1, "Analyzing scenes...")
        
        # 1. Detect scenes and extract frames at each cut
        # We use a scene change threshold of 0.4 as suggested.
        # This saves significant processing time and storage.
        frame_pattern = os.path.join(segment_dir, "scene_%03d.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "select='gt(scene,0.4)',metadata=print",
            "-vsync", "vfr",
            frame_pattern
        ]
        # We capture output to get the timestamps if needed, 
        # but for MVP we'll use the file order.
        subprocess.run(cmd, capture_output=True)
        
        frames = sorted([f for f in os.listdir(segment_dir) if f.endswith(".jpg")])
        total = len(frames)
        
        if total == 0:
            # Fallback: if no scene changes detected, just take one frame at 2s
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:02", "-i", video_path,
                "-vframes", "1", frame_pattern % 1
            ], capture_output=True)
            frames = sorted([f for f in os.listdir(segment_dir) if f.endswith(".jpg")])
            total = len(frames)

        vectors = []
        metas = []
        
        for i, frame in enumerate(frames):
            frame_path = os.path.join(segment_dir, frame)
            
            if progress_cb: 
                progress_cb(0.2 + (0.7 * (i / total)), f"Analyzing scene {i+1}/{total}...")
            
            # 3. Generate CLIP embedding
            img = Image.open(frame_path)
            emb = self.model.encode(img)
            
            vectors.append(emb)
            metas.append({
                "video_path": video_path,
                "video_url": video_url,
                "video_title": video_title or video_name,
                "frame_path": frame_path,
                "scene_index": i
            })
        
        # 4. Add to DB
        if vectors:
            self.db.add_vectors(vectors, metas)
        
        if progress_cb: progress_cb(1.0, f"Successfully indexed {total} segments.")
        return total

    def download_and_index_youtube(self, url, progress_cb=None):
        """Helper to download a YT video and index it immediately."""
        import yt_dlp
        
        if progress_cb: progress_cb(0.05, "Downloading YouTube video...")
        
        ydl_opts = {
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
            'outtmpl': os.path.join(self.download_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            video_title = info.get('title', 'YouTube Video')
            
        return self.index_video(video_path, video_url=url, video_title=video_title, progress_cb=progress_cb)

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
        
        if progress_cb: progress_cb(0.1, "Segmenting video...")
        
        # 1. Segment video into ~10s chunks
        # We use re-encoding to ensure we have frames at the start of each segment for embedding
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-f", "segment", "-segment_time", "10",
            "-reset_timestamps", "1",
            "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
            "-an", # No audio needed for visual indexing
            os.path.join(segment_dir, "seg_%03d.mp4")
        ]
        subprocess.run(cmd, capture_output=True)
        
        segments = sorted([f for f in os.listdir(segment_dir) if f.endswith(".mp4")])
        total = len(segments)
        
        vectors = []
        metas = []
        
        for i, seg in enumerate(segments):
            seg_path = os.path.join(segment_dir, seg)
            timestamp = i * 10
            
            if progress_cb: 
                progress_cb(0.1 + (0.8 * (i / total)), f"Embedding segment {i+1}/{total}...")
            
            # 2. Extract middle frame for embedding
            frame_path = os.path.join(segment_dir, f"frame_{i:03d}.jpg")
            # Get duration of segment to find middle
            # For simplicity, we just take 2s into the segment
            subprocess.run([
                "ffmpeg", "-y", "-ss", "00:00:02", "-i", seg_path,
                "-vframes", "1", frame_path
            ], capture_output=True)
            
            if os.path.exists(frame_path):
                # 3. Generate CLIP embedding
                img = Image.open(frame_path)
                emb = self.model.encode(img)
                
                vectors.append(emb)
                metas.append({
                    "video_path": video_path,
                    "video_url": video_url,
                    "video_title": video_title or video_name,
                    "segment_path": seg_path,
                    "timestamp": timestamp,
                    "duration": 10
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

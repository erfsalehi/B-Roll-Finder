"""
core/proxy_fetcher.py

Low-Res Proxy Fetch Engine for Smart Mode.

Workflow per shot:
  1. YouTube search (metadata only, no download)
  2. Download each candidate at lowest quality (144p/360p)
  3. Extract frames at scene cuts using ffmpeg scene detection
  4. Embed frames with CLIP, build in-memory FAISS index
  5. Search index with the shot's visual intent text
  6. Cut 6-second proxy clips around best-matching timestamps
  7. Return timestamped candidates ready for the gallery
"""

import os
import re
import subprocess
import shutil
import numpy as np

PROXY_CACHE_DIR = ".cache/proxies"


class ProxyFetcher:
    def __init__(self, proxy_cache_dir=PROXY_CACHE_DIR):
        self.proxy_cache_dir = proxy_cache_dir
        self.model = None
        os.makedirs(proxy_cache_dir, exist_ok=True)

    # ── CLIP model (lazy load) ────────────────────────────────────────────────

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('clip-ViT-B-32')

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_id(s):
        return "".join(c if c.isalnum() else "_" for c in s)[:60]

    @staticmethod
    def _fmt_ts(seconds):
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    # ── Step 1: YouTube search (metadata only) ────────────────────────────────

    def search_youtube(self, query, max_results=3):
        """Returns a list of {url, title, duration} dicts — no download."""
        import yt_dlp
        results = []
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "default_search": "ytsearch",
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
                for entry in info.get("entries", []) or []:
                    vid_id = entry.get("id", "")
                    url = entry.get("url") or (f"https://www.youtube.com/watch?v={vid_id}" if vid_id else None)
                    if url:
                        results.append({
                            "url": url,
                            "title": entry.get("title", "YouTube Video"),
                            "duration": entry.get("duration", 0),
                        })
        except Exception as e:
            print(f"[ProxyFetcher] YouTube search error: {e}")
        return results

    # ── Step 2: Download proxy at lowest resolution ───────────────────────────

    def download_proxy(self, url, video_id):
        """Download at the lowest available quality for fast analysis."""
        import yt_dlp

        output_path = os.path.join(self.proxy_cache_dir, f"{video_id}_proxy.mp4")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path  # Use cached proxy

        _cookie_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cookies.txt')

        ydl_opts = {
            "format": "worstvideo[ext=mp4]+worstaudio/worst[ext=mp4]/worst",
            "outtmpl": os.path.join(self.proxy_cache_dir, f"{video_id}_proxy.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mp4",
            "retries": 5,
            "extractor_retries": 3,
            'extractor_args': {
                'youtube': {
                    'player_client': ['web', 'tv_embedded', 'android'],
                }
            },
        }
        if os.path.exists(_cookie_file):
            ydl_opts["cookiefile"] = _cookie_file

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            actual = ydl.prepare_filename(info)

        # Normalise extension to .mp4 if needed
        if actual and os.path.exists(actual) and actual != output_path:
            shutil.move(actual, output_path)

        return output_path if os.path.exists(output_path) else None

    # ── Step 3: Extract frames with timestamps ────────────────────────────────

    def extract_frames_with_timestamps(self, video_path, threshold=0.3):
        """
        Uses ffmpeg scene detection to extract one frame per camera cut.
        Returns list of {frame_path, timestamp} dicts.
        """
        frames_dir = video_path + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        timestamps_file = os.path.join(frames_dir, "pts.txt")

        # Remove stale timestamps file
        if os.path.exists(timestamps_file):
            os.remove(timestamps_file)

        frame_pattern = os.path.join(frames_dir, "frame_%04d.jpg")

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"select='gt(scene,{threshold})',metadata=print:file={timestamps_file}",
            "-vsync", "vfr",
            frame_pattern,
        ]
        subprocess.run(cmd, capture_output=True)

        # Parse timestamps from metadata file
        frame_times = []
        if os.path.exists(timestamps_file):
            with open(timestamps_file, "r", errors="ignore") as f:
                content = f.read()
            for match in re.finditer(r"pts_time:([\d.]+)", content):
                frame_times.append(float(match.group(1)))

        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

        result = []
        for i, frame in enumerate(frames):
            t = frame_times[i] if i < len(frame_times) else float(i * 5)
            result.append({
                "frame_path": os.path.join(frames_dir, frame),
                "timestamp": t,
            })

        # Fallback: sample every 10 seconds if scene detection found nothing
        if not result:
            try:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                    capture_output=True, text=True
                )
                duration = float(probe.stdout.strip() or "60")
                for t in range(0, int(duration), 10):
                    fp = os.path.join(frames_dir, f"fallback_{t:06d}.jpg")
                    subprocess.run(
                        ["ffmpeg", "-y", "-ss", str(t), "-i", video_path, "-vframes", "1", fp],
                        capture_output=True
                    )
                    if os.path.exists(fp):
                        result.append({"frame_path": fp, "timestamp": float(t)})
            except Exception as e:
                print(f"[ProxyFetcher] Fallback frame extraction failed: {e}")

        return result

    # ── Step 4: CLIP embed ────────────────────────────────────────────────────

    def embed_frames(self, frame_list, progress_cb=None):
        """Returns a list of float32 numpy arrays (one per frame)."""
        self._load_model()
        from PIL import Image

        vectors = []
        for i, f in enumerate(frame_list):
            try:
                img = Image.open(f["frame_path"]).convert("RGB")
                emb = self.model.encode(img)
                vectors.append(emb.astype("float32"))
            except Exception:
                vectors.append(np.zeros(512, dtype="float32"))
            if progress_cb:
                progress_cb(i / max(len(frame_list), 1), f"Embedding frame {i+1}/{len(frame_list)}…")
        return vectors

    # ── Step 5: In-memory FAISS search ───────────────────────────────────────

    def search_frames(self, vectors, query_text, frame_list, top_k=3):
        """Searches an in-memory FAISS index with a text query via CLIP."""
        import faiss

        if not vectors:
            return []

        self._load_model()

        vecs = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vecs)

        index = faiss.IndexFlatIP(512)
        index.add(vecs)

        query_emb = self.model.encode(query_text).astype("float32")
        query_vec = np.array([query_emb])
        faiss.normalize_L2(query_vec)

        k = min(top_k, len(vectors))
        distances, indices = index.search(query_vec, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(frame_list):
                continue
            results.append({
                "timestamp": frame_list[idx]["timestamp"],
                "score": float(distances[0][i]),
                "frame_path": frame_list[idx]["frame_path"],
            })
        return results

    # ── Step 6: Cut proxy clip ────────────────────────────────────────────────

    def cut_proxy_clip(self, video_path, timestamp, clip_id, duration=6):
        """Cuts a short proxy clip around a timestamp for gallery preview."""
        start = max(0.0, timestamp - 2.0)
        clip_path = os.path.join(self.proxy_cache_dir, f"{clip_id}_{int(timestamp)}s.mp4")

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 500:
            return clip_path

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-crf", "30", "-preset", "ultrafast",
            "-vf", "scale=480:-2",
            "-an",
            clip_path,
        ]
        subprocess.run(cmd, capture_output=True)
        return clip_path if os.path.exists(clip_path) else None

    # ── Public API: Full pipeline for one shot ────────────────────────────────

    def fetch_for_shot(self, shot, max_videos=2, top_k_per_video=2, progress_cb=None):
        """
        Full Low-Res Proxy pipeline for a single shot.
        Returns a list of candidate dicts with proxy_clip_path and timestamps.
        """
        query = shot.get("shot_intent") or (shot.get("search_queries") or [""])[0]
        if not query:
            return []

        # ── YouTube search ────────────────────────────────────────────────────
        if progress_cb:
            progress_cb(0.02, f"🔍 Searching YouTube for: {query[:50]}…")

        videos = self.search_youtube(query, max_results=max_videos + 1)
        videos = videos[:max_videos]

        if not videos:
            return []

        candidates = []
        n_videos = len(videos)

        for vi, video in enumerate(videos):
            url = video["url"]
            video_id = self._safe_id(url.split("v=")[-1] if "v=" in url else url[-11:])
            base_p = vi / n_videos

            # Download proxy
            if progress_cb:
                progress_cb(base_p + 0.02, f"⬇ Downloading proxy [{vi+1}/{n_videos}]: {video['title'][:40]}…")
            try:
                proxy_path = self.download_proxy(url, video_id)
            except Exception as e:
                print(f"[ProxyFetcher] Proxy download failed for {url}: {e}")
                continue

            if not proxy_path or not os.path.exists(proxy_path):
                continue

            # Extract frames
            if progress_cb:
                progress_cb(base_p + 0.15, "🎬 Detecting scene cuts…")
            frames = self.extract_frames_with_timestamps(proxy_path)
            if not frames:
                continue

            # Embed
            def _emb_cb(p, msg):
                if progress_cb:
                    progress_cb(base_p + 0.15 + p * 0.45, f"🧠 {msg}")
            vectors = self.embed_frames(frames, progress_cb=_emb_cb)

            # Search
            if progress_cb:
                progress_cb(base_p + 0.65, "🔎 Finding best visual matches…")
            hits = self.search_frames(vectors, query, frames, top_k=top_k_per_video)

            # Cut proxy clips
            for hi, hit in enumerate(hits):
                ts = hit["timestamp"]
                clip_id = f"{video_id}_s{shot.get('slot_id', 'x')}"
                clip_path = self.cut_proxy_clip(proxy_path, ts, f"{clip_id}_{hi}")

                ts_start = max(0.0, ts - 2.0)
                ts_end = ts_start + 6.0

                candidates.append({
                    "title": video["title"],
                    "url": url,
                    "source": "smart_proxy",
                    "thumbnail": None,
                    "proxy_clip_path": clip_path,
                    "timestamp_start": ts_start,
                    "timestamp_end": ts_end,
                    "timestamp_start_fmt": self._fmt_ts(ts_start),
                    "timestamp_end_fmt": self._fmt_ts(ts_end),
                    "score": hit["score"],
                    "matched_query": query,
                    "duration": 6,
                })

        if progress_cb:
            progress_cb(1.0, f"✅ Found {len(candidates)} visual matches.")

        return candidates

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def clear_session_cache(self):
        """Removes all temporary proxy files. Call after Step 6 download."""
        if os.path.isdir(self.proxy_cache_dir):
            shutil.rmtree(self.proxy_cache_dir)
            os.makedirs(self.proxy_cache_dir, exist_ok=True)

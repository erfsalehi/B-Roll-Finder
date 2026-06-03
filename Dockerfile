# B-Roll Finder — server image. CPU-only, no GPU/torch (see deploy/README.md).
# Runs the always-on Telegram bot by default; the Streamlit UI is opt-in (see
# the CMD override at the bottom).
FROM python:3.12-slim

# ffmpeg + ffprobe: clip normalization, scene/frame extraction, audio downsample.
# ca-certificates: HTTPS to the stock / Groq / Telegram APIs and model downloads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer caches across code-only changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code (.dockerignore keeps secrets, caches, and downloads out).
COPY . .

# HOME governs where fastembed caches its ONNX model (~/.cache/fastembed). Point
# it at /app so the ~90 MB model lands on the mounted volume and is downloaded
# only once, surviving restarts.
ENV HOME=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Persist downloaded projects and caches (fastembed model, yt-dlp update marker,
# the SQLite clip library) across container restarts.
VOLUME ["/app/downloads", "/app/.cache"]

# Intentionally runs as root: the bot self-upgrades yt-dlp daily in-process
# (core/app_utils.update_yt_dlp does `pip install --upgrade yt-dlp`), which must
# be able to write to site-packages. YouTube breaks stale yt-dlp, so keeping it
# fresh without rebuilding the image matters for an always-on deployment.

# Default entrypoint: the Telegram bot. To run the Streamlit UI instead, override
# the command, e.g.:
#   docker run --env-file .env -p 8501:8501 broll-finder \
#     streamlit run app.py --server.port 8501 --server.address 0.0.0.0
CMD ["python", "-m", "bot.telegram_bot"]

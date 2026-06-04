# syntax=docker/dockerfile:1
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
# The BuildKit pip cache mount keeps successfully-downloaded wheels between
# build attempts, so a flaky link only re-fetches what's still missing instead
# of starting over. Generous timeout/retries for slow connections too.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout 120 --retries 10 -r requirements.txt

# nodejs-bin bundles the `node` binary inside its package dir but only registers
# a PATH entry point on Windows — on Linux it lands at
# .../site-packages/nodejs/bin/node, off PATH. yt-dlp's EJS JS-challenge solver
# needs `node` discoverable or it drops high-res YouTube formats, so symlink it
# (and npm/npx) into /usr/local/bin. `node --version` fails the build if broken.
RUN NODE_BIN="$(python -c 'import os,nodejs;print(os.path.join(os.path.dirname(nodejs.__file__),"bin"))')" \
    && ln -sf "$NODE_BIN/node" /usr/local/bin/node \
    && ln -sf "$NODE_BIN/npm"  /usr/local/bin/npm \
    && ln -sf "$NODE_BIN/npx"  /usr/local/bin/npx \
    && node --version

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

# Optional download-link server (BOT_FILE_SERVER=1). Map it with -p to hand out
# project-zip links; also used by the Streamlit UI override.
EXPOSE 8770 8501

# Intentionally runs as root: the bot self-upgrades yt-dlp daily in-process
# (core/app_utils.update_yt_dlp does `pip install --upgrade yt-dlp`), which must
# be able to write to site-packages. YouTube breaks stale yt-dlp, so keeping it
# fresh without rebuilding the image matters for an always-on deployment.

# Default entrypoint: the Telegram bot. To run the Streamlit UI instead, override
# the command, e.g.:
#   docker run --env-file .env -p 8501:8501 broll-finder \
#     streamlit run app.py --server.port 8501 --server.address 0.0.0.0
CMD ["python", "-m", "bot.telegram_bot"]

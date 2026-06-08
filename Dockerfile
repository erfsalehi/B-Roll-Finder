# syntax=docker/dockerfile:1
# B-Roll Finder — server image. CPU-only, no GPU/torch (see deploy/README.md).
# Runs the always-on Telegram bot by default; the Streamlit UI is opt-in (see
# the CMD override at the bottom).
FROM python:3.12-slim

# ffmpeg + ffprobe: clip normalization, scene/frame extraction, audio downsample.
# ca-certificates: HTTPS to the stock / Groq / Telegram APIs and model downloads.
# The lib*/fonts-* set are the system libraries Remotion's headless Chrome needs
# to render the animated text overlays.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg ca-certificates curl unzip \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 libxshmfence1 \
        fonts-liberation fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Deno: yt-dlp needs a JS runtime to solve YouTube's nsig / n-parameter
# challenge. Without it, current YouTube extraction returns only storyboard
# images (180p/90p/…) and every real format probe/download fails with
# "Requested format is not available". nodejs alone is no longer enough for the
# JS interpreter yt-dlp uses, so install Deno and put it on PATH (yt-dlp
# auto-detects it). `deno --version` fails the build if the download is broken.
RUN curl -fsSL -o /tmp/deno.zip \
        https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
    && unzip -o /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno \
    && deno --version

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
# and the local remotion CLI (node_modules/.bin/remotion, whose shebang is
# `env node`) both need `node` discoverable, so symlink just the real `node`
# binary into /usr/local/bin. npm/npx are deliberately NOT symlinked: in
# nodejs-bin they're JS CLI scripts whose target lacks the exec bit, so a symlink
# yields "npm: Permission denied" (exit 127). The build invokes npm via the
# cross-platform `python -m nodejs.npm` entrypoint instead (next step), which
# resolves node itself. `node --version` fails the build early if broken.
RUN NODE_BIN="$(python -c 'import os,nodejs;print(os.path.join(os.path.dirname(nodejs.__file__),"bin"))')" \
    && ln -sf "$NODE_BIN/node" /usr/local/bin/node \
    && node --version

# Remotion overlay renderer: install the Node project's deps and pre-fetch the
# headless Chrome it renders with. Done before `COPY . .` (and keyed on just
# package.json) so it caches across Python/code-only changes. Browser pre-fetch
# is best-effort — if it fails at build, Remotion fetches it on first render.
COPY remotion/package.json ./remotion/package.json
RUN cd remotion && python -m nodejs.npm install --no-audit --no-fund \
    && (./node_modules/.bin/remotion browser ensure || echo "[build] chrome pre-fetch deferred to first render")

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

# Health endpoint (bot/healthserver.py) so Docker/Coolify can verify liveness and
# do clean single-instance swaps (avoids the two-poller Telegram 409).
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('BOT_HEALTH_PORT','8000'), timeout=3)" || exit 1

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

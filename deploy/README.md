# Deploying B-Roll Finder on a server (Ubuntu)

Run the whole app on a clean-network server (e.g. your Hetzner CX43). The
server's datacenter connection makes the fetch / DNS / Telegram / DeepSeek
failures you hit locally disappear, and it's always-on for the Telegram bot.

The end product is identical to a local run: downloaded clips + a Premiere
FCPXML under `downloads/<project>/`. You pull the finished project to your
editing machine (scp or a single zip).

You can deploy either with **Docker** (simplest) or as a **venv + systemd
service** (sections 1–2). Pick one.

## Option A: Docker (recommended)

A `Dockerfile` at the repo root builds a CPU-only image (ffmpeg included, no
GPU/torch). Install Docker, then:

```bash
sudo apt update && sudo apt install -y docker.io git
git clone https://github.com/erfsalehi/B-Roll-Finder.git
cd B-Roll-Finder
# Create .env with your keys (see the list under section 1 below).
docker build -t broll-finder .
```

Run the always-on Telegram bot, persisting projects + caches on host volumes so
the fastembed model and downloads survive restarts:

```bash
docker run -d --name broll-bot --restart unless-stopped \
  --env-file .env \
  -v "$PWD/downloads:/app/downloads" \
  -v "$PWD/.cache:/app/.cache" \
  broll-finder
docker logs -f broll-bot          # live logs
```

To run the Streamlit UI instead (or alongside, with a different `--name`),
override the command:

```bash
docker run -d --name broll-ui --restart unless-stopped \
  --env-file .env -p 8501:8501 \
  -v "$PWD/downloads:/app/downloads" -v "$PWD/.cache:/app/.cache" \
  broll-finder \
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Reach the UI over an SSH tunnel (`ssh -L 8501:localhost:8501 user@SERVER`)
rather than exposing it publicly. Update with `git pull && docker build -t
broll-finder . && docker restart broll-bot`.

## Option B: venv + systemd

## 1. One-time setup

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip ffmpeg git
sudo useradd -m -d /opt/B-Roll-Finder broll || true
sudo -u broll -H bash -lc '
  cd /opt/B-Roll-Finder
  git clone https://github.com/erfsalehi/B-Roll-Finder.git . || git pull
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt
'
```

There is **no GPU/torch stack** — the Clip Library embeds with fastembed (ONNX
runtime, CPU). On first use it downloads the ~90 MB MiniLM model once into
`~/.cache/fastembed`.

### Tuning CPU usage (optional)

The app caps its own thread pools so it never saturates every vCPU (it shares
cores with concurrent downloads and ffmpeg). Defaults are sized for an 8-vCPU
box; override in `.env` if needed:

```bash
BROLL_TORCH_THREADS=4        # OMP/BLAS threads for numpy + the ONNX embedder (default: cores/2, max 4)
BROLL_FFMPEG_THREADS=2       # libx264 threads per normalize encode (default: cores/4)
BROLL_NORMALIZE_CONCURRENCY=2 # simultaneous libx264 encodes (default: 2)
```

Create `/opt/B-Roll-Finder/.env` with your keys (same ones you use locally:
`GROQ_API_KEY`, `PEXELS_API_KEY`, `YOUTUBE_API_KEY`, `DEEPSEEK_API_KEY`,
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USERS`, etc.). On the server you do
**not** need `APP_PROXY` / `BOT_PROXY` / `BROLL_BYPASS_HTTP_PROXY` — the network
is clean.

## 2. Run the Telegram bot as a service (always-on, auto-restart)

```bash
sudo cp /opt/B-Roll-Finder/deploy/broll-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now broll-bot
sudo systemctl status broll-bot          # should be "active (running)"
journalctl -u broll-bot -f               # live logs
```

Now send the bot a voice file from your phone; it runs the full pipeline and
writes the project to `/opt/B-Roll-Finder/downloads/<project>/`.

## 3. (Optional) Run the Streamlit UI for browser access

```bash
./venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Put it behind nginx + HTTPS + basic-auth, or just reach it over an SSH tunnel
from your laptop (safer, nothing public):

```bash
ssh -L 8501:localhost:8501 broll@YOUR_SERVER     # then open http://localhost:8501
```

## 4. Pull a finished project to your editing machine

Either copy the whole project folder:

```bash
scp -r broll@YOUR_SERVER:/opt/B-Roll-Finder/downloads/<project> .
```

…or make a single zip on the server first and copy that:

```bash
./venv/bin/python -c "from core.output import zip_project; print(zip_project('<project>'))"
scp broll@YOUR_SERVER:/opt/B-Roll-Finder/downloads/<project>.zip .
```

Unzip locally, then open the `.xml` in Premiere (File ▸ Import). If Premiere
prompts to locate media, point it at the project's `director/` folder once and
it relinks every clip.

## Updating

```bash
sudo -u broll -H bash -lc 'cd /opt/B-Roll-Finder && git pull && ./venv/bin/pip install -r requirements.txt'
sudo systemctl restart broll-bot
```

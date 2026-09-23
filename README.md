# B-Roll Finder

**From voiceover to Premiere-ready project, hands-free.**

Send a voiceover to a Telegram bot. B-Roll Finder transcribes it, cuts it into shots, searches YouTube, Pexels and your own clip library for footage, ranks and picks clips for every shot, renders animated text overlays, downloads everything, and delivers a zipped project with a Premiere Pro XML ready to import.

It runs as an **always-on Telegram bot** on a server (the main way to use it) or as a **Streamlit app** on your desktop when you want to pick clips by hand.

---

## Contents

- [What a run does](#what-a-run-does)
- [Telegram bot](#telegram-bot)
- [What you get](#what-you-get)
- [How it picks footage](#how-it-picks-footage)
- [Self-healing: no empty shots, no broken XML](#self-healing-no-empty-shots-no-broken-xml)
- [Text overlays](#text-overlays)
- [Extras and images](#extras-and-images)
- [Clip Library](#clip-library)
- [Deploying on a server](#deploying-on-a-server)
- [YouTube on a server: cookies and proxies](#youtube-on-a-server-cookies-and-proxies)
- [Streamlit desktop app](#streamlit-desktop-app)
- [API keys](#api-keys)
- [LLM providers and cost](#llm-providers-and-cost)
- [Configuration](#configuration)
- [Project layout](#project-layout)
- [Tests](#tests)

---

## What a run does

The headless pipeline (`core/pipeline.py`) runs these stages in order. It checks for cancellation between every stage.

```
 1   Transcribe            Whisper via Groq, word-level timestamps
 2   Topic                 one "smart" LLM pass names what the video is about
 3   Context pre-pass      optional: map the script into subject segments
 4   Shot list             AI director: timing, intent, 2–3 search queries per shot
 5   Fetch                 YouTube (yt-dlp) · Pexels · Clip Library, in parallel
 6   Filter                drop SD (YouTube Data API), Shorts, vertical, over-long clips
 7   Rank                  LLM-as-judge, batched, orders candidates per shot
 8   Auto-select           per-shot quota of Pexels + YouTube clips
 8b  Fill empty shots      multi-pass re-fetch, YouTube-first
 8c  Topic fallback        generic on-topic clip for anything still empty
 8d  Extra clips           brand / model / part / theme B-roll → Clip Library
 8e  Related images        Google stills → Clip Library
 8f  Per-shot images       3 Google images per shot, saved in the project
 9   QA review             AI "executive producer" reads the whole timeline
 9b  Auto-refine           re-pick the shots QA flagged, then review again
 9c  Boundary gate         every clip must pass quality/duration/orientation
 9e  YouTube coverage      every shot carries at least one YouTube clip
 9d  Text overlays         animated transparent overlays rendered with Remotion
       ── review gate: the bot pauses here for /download or /refine ──
10   Download              parallel, deduped, cached, with a repair loop
     Export                Premiere XML (validated + repaired) · shot SRT · links
```

Every API call is metered, so each job ends with a token and cost breakdown.

---

## Telegram bot

Start it on an always-on machine:

```bash
python -m bot.telegram_bot
```

Then **send the bot a voice message or audio file**. It posts progress for each stage, stops at the review gate with a summary and QA report, and after `/download` delivers the project. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USERS` in `.env`. The allowlist fails closed: if it's empty, nobody can use the bot.

On Windows, double-click **`start_bot.bat`**. It activates the venv, runs the bot in its own window, and restarts it if it crashes. To start it on boot, put a shortcut to it in `shell:startup`.

### The review gate

With **Pause to review** on (the default), a run stops after selection, QA and overlays. Then you choose:

- `/details`: a per-shot breakdown of what was picked.
- `/refine`: re-pick every shot QA flagged. `/refine 4 9` re-picks only shots 4 and 9.
- `/redo`: re-fetch shots that still have no clip.
- `/download`: download the clips and build the project.

Paused projects are saved to `.cache/`, so a crash or `/forcestop` doesn't lose them.

### Commands

| Command | What it does |
|---|---|
| *(send audio)* | Full run: footage, overlays, images, XML |
| `/settings` | Inline menu of per-chat options (see below) |
| `/status` | Checks the bot is up: API reachability, ffmpeg, keys, cookies, whether a job is running |
| `/test` | Preflight: live-tests the LLM, transcription, Pexels, yt-dlp search **and real downloads** in about 30 s. `/test quick` skips downloads |
| `/details` | Per-shot breakdown of the project waiting for review |
| `/download` | Download and build the reviewed project |
| `/refine [shots]` | Re-pick QA-flagged shots, or only the shots you name |
| `/redo` | Re-fetch empty shots, YouTube-first |
| `/cancel` | Stops at the next stage boundary, or discards a paused project |
| `/forcestop` | Hard stop plus bot restart, for when `/cancel` doesn't take |
| `/overlay` | Next audio file gets animated text overlays only, no footage |
| `/overlaytext <secs> <text>` | Renders one overlay clip for exact text and duration |
| `/extras` | Next audio file gets only the extra contextual clips |
| `/images` | Next audio file gets only related Google stills (into the Clip Library) |
| `/zip [name]` | Zips a finished project: attached if under ~50 MB, plus a download link |
| `/files [name]` | Lists every zip on the server, each with a fresh link |
| `/links [name]` | Per-shot source links, for re-downloading empty shots by hand |
| `/proxies` | Shows the validated YouTube proxy pool. `/proxies refresh` searches the list again |
| `/cleanup [name\|all\|overlays]` | Deletes projects or clears the overlay render cache. The Clip Library is kept |
| `/logs` | Sends this session's bot log as a file |
| `/cookies` | How to give the bot YouTube cookies. You can also just send `cookies.txt` |
| *(send `.xml`)* | Teach it your trims: send a Premiere XML export after editing |

Jobs run in a background thread, so `/status` and `/cancel` keep working while a job runs. A second audio file waits for the first job to finish.

### `/settings`

Each chat has its own settings, saved to `.cache/bot_settings.json`. The `.env` values are only the defaults a new chat starts with.

| Setting | Default | |
|---|---|---|
| Pexels / YouTube / Clip Library | on | Sources to search |
| Per-query counts | 3 / 4 / 5 | Candidates fetched per query (library: per shot) |
| Min height | 720p | Drop candidates below this |
| Download quality | 1080p | Download cap |
| QA review | on | Stage 9 |
| Auto-refine flagged | on | Stage 9b |
| Auto-fill empty shots | on | Stage 8b multi-pass fill |
| Pause to review | on | The review gate |
| Text overlays / Overlay style | on / Bold Yellow | Also: Clean White, Neon Glow, Boxed News |
| Extra clips / Related images | on | Library-only extras |
| Google images / shot, Images per shot | on, 3 | Needs `SERPER_API_KEY` |
| Detailed queries | off | Per-subject queries for shots that mention several subjects |
| Delete clips after zip | on | Keep only the zip on disk |

### Delivery

Telegram limits bot uploads to 50 MB, so the bot attaches small zips directly. For anything bigger it also sends a download link from its own file server. Turn it on with `BOT_FILE_SERVER=1`, `BOT_FILE_SERVER_PORT` (8770) and `BOT_PUBLIC_HOST`. The links are HMAC-signed, so they can't be guessed or pointed at another file. They **don't expire** unless you set `BOT_LINK_TTL`. `/files` gives you fresh links for every zip on disk.

Telegram also limits the files a bot can *receive* to 20 MB. For longer voiceovers, send a mono 64 kbps MP3 (a 30-minute voiceover is about 14 MB), or run a local Bot API server and set `TELEGRAM_API_BASE`.

---

## What you get

```
downloads/<project>/
├── <project>.xml            Premiere / FCP7 XML: B-roll on V1, overlays on V2, SFX on audio
├── <project>.shots.srt      one subtitle per shot number, to find shots on the timeline
├── download_links.txt       source URL of every clip
├── <clips>.mp4              named by shot number + query
├── overlays/                transparent ProRes 4444 .mov overlays (true alpha)
└── images/shots/shot_NN/    Google images per shot, plus sources.txt
```

Clips are placed on **speech onsets** rather than an even grid, so cuts land where the speaker starts a new beat. When you taught the bot a trim, or a reviewer marked the good part of a clip, the clip starts on that moment instead of at 0:00.

Before the XML is written it is **evaluated and repaired**: shot timings, SRT and XML structure are all checked. Gaps are filled by extending the previous clip (`FCPXML_FILL_GAPS`), so the B-roll track never has a hole, and the XML only references files that are actually on disk.

---

## How it picks footage

**Shot list.** An LLM director reads the transcript in blocks and writes one entry per shot: timing (taken from the transcript segments), visual intent, shot type and 2–3 search queries. Talking-head moments are marked `none` and skipped.

- **Context-aware keywords** (`ENABLE_CONTEXT_AWARE_KEYWORDS`): made for listicles and multi-product reviews. A pre-pass maps the script into subject segments (for example *"BMW M3 E90", 75–165 s*). Each shot's queries are then tied to the subject being discussed at that point, so "the transmission is jerky" searches for *that car's* gearbox.
- **Detailed queries** (`/settings`): when a shot names several subjects, each one gets its own query.

**Sources.** YouTube is searched with **yt-dlp + keywords**. The YouTube Data API is only used to find and drop SD clips (about 1 quota unit per 50 clips). Pexels rotates through several keys (`PEXELS_API_KEY_2`, …) when one hits its hourly limit. The Clip Library adds your own past footage without any API calls. Shorts, vertical clips and over-long uploads are filtered out. Downloads are capped at 1080p.

**Ranking.** An LLM judge ranks each shot's candidates against the narration and intent, several shots per call with jittered pacing, so large projects stay under rate limits.

**Selection.** Each shot gets a quota based on its length:

| Shot length | Pexels | YouTube |
|---|---|---|
| under 4 s | 1 | 1 |
| 4 s and longer | 2 | about one per 5 s (`ceil(dur/5)`) |

Every shot gets at least one YouTube clip. A look-back check keeps the same clip from appearing in consecutive shots.

### QA review and auto-refine

An AI "executive producer" reads the whole selected timeline in one pass. It flags thematic breaks, repeated visuals, pacing problems and clips that don't match the narration, each tied to a shot number with a suggested fix. With auto-refine on, the flagged shots get new keywords from the reviewer's suggestion and are fetched, ranked and selected again. Then QA runs once more, so the report you see reflects the fixes.

---

## Self-healing: no empty shots, no broken XML

A single failed search or a dead YouTube link shouldn't leave a black hole in the edit. After selection:

1. **Fill** (`AUTO_FILL`): any shot without a clip is re-fetched over several passes, YouTube first.
2. **Topic fallback** (`FILL_EMPTY_WITH_TOPIC`): a shot that is still empty gets a generic on-topic Pexels clip.
3. **Boundary gate**: every selected clip must pass the quality, duration and orientation rules. Shots that fail are re-picked for up to 3 rounds. If a clip still fails, it's dropped rather than breaking the XML.
4. **Download repair**: if a YouTube clip fails to download, another clip is picked for that shot and downloaded in its place. Clips that never made it to disk are removed before export.
5. **Gap fill**: when the XML is rendered, the previous clip is extended across any gap that remains.

---

## Text overlays

A reasoning LLM picks the headings, stats, money figures and emphasis words worth showing on screen. It works on transcript chunks of about 150 s in parallel, so long videos don't get cut short. Each overlay is rendered with **Remotion** (`remotion/`) as a **transparent ProRes 4444 `.mov`** with its sound effect baked in. Animations are chosen by overlay type (title card, stat pop, money count, lower third, emphasis pop). Overlay timing follows the word-level timestamps, and each overlay shows its title word for word.

- Four style presets: **Bold Yellow**, **Clean White**, **Neon Glow**, **Boxed News** (`/settings` or `OVERLAY_STYLE`).
- `/overlay` renders only the overlays for a voiceover. `/overlaytext 3.5 47% LESS WEAR` renders a single overlay.
- Rendering needs Node plus `npm ci` in `remotion/`. The Docker image includes both. If overlays fail, the main job still completes.

The Streamlit app also has its original PNG caption generator (fonts, outlines, emoji prefixes, Freesound SFX).

---

## Extras and images

These give the editor spare material. **None of it is placed on the timeline.**

- **Extra clips** (`ENABLE_EXTRA_CLIPS`, `/extras`): finds the brands, car models, parts and products the script names, plus the video's overall theme. It then downloads 2–3 HD landscape YouTube clips for each (brand → logo and factory, model → POV drive and review, part → "how it works", theme → iconic imagery). The clips go into the Clip Library as `Extra - <keyword>`.
- **Per-shot Google images** (`ENABLE_SHOT_IMAGES`): one short LLM call turns each shot into an image query. [Serper.dev](https://serper.dev) returns real Google Images results, and 3 per shot are saved to `images/shots/shot_NN/`. A `sources.txt` lists each image's page, since Google Images results **aren't licensed for reuse**. Costs about 1 Serper credit per shot.
- **Related images** (`ENABLE_RELATED_IMAGES`, `/images`): high-resolution stills of brands, models, products, parts and concepts. They are saved to the Clip Library, not the project. Uses Serper, or the Google Custom Search API as a fallback (Google shuts that API down on 2027-01-01).

---

## Clip Library

Every clip you download is stored in a local SQLite database (`.cache/clip_library.db`, or `CLIP_LIBRARY_DB`) with a 384-dim embedding of its shot description (`all-MiniLM-L6-v2` via **fastembed/ONNX**, so no torch or GPU is needed). Later projects search it before any external API. Results are ranked by similarity and weighted by how often a clip has been used. Over time it becomes the fastest and cheapest source for topics you cover often. `/cleanup` never touches it.

**Learned trims.** Send the bot, or import in the app, a Premiere/FCP7 XML after you've edited it. The app records how you cut each clip (in and out points) in `clip_preferred_trims`. When the same footage comes up again, your trim is used.

**Sharing between machines** (Streamlit sidebar → Library Health): **Export my library** writes a small JSON bundle (metadata, embeddings, trims). **Merge a teammate's export** combines it with yours, deduplicated by URL, and is safe to run twice. Only metadata is shared; the videos are downloaded again from their source URLs when reused.

<details>
<summary>What's stored per clip</summary>

| Field | What it is |
|---|---|
| `clip_url` | Source URL, used for deduplication and re-downloading |
| `shot_description` / `embedding` | The narration or intent text, and its vector |
| `clip_title`, `source`, `keywords`, `search_query` | Provenance |
| `project`, `slot_index` | Which project and shot it was used in |
| `duration`, `thumbnail_url`, `local_path` | Clip info and where the file is on this machine |
| `usage_count`, `last_used_at`, `created_at` | Reuse stats |

</details>

---

## Deploying on a server

The production setup is the bot in Docker on a CPU-only Linux server (about 8 vCPU / 16 GB). There's no torch or GPU, and transcription runs remotely on Groq. Full guide: **[deploy/README.md](deploy/README.md)** (Docker, Coolify, or venv + systemd).

```bash
git clone https://github.com/erfsalehi/B-Roll-Finder.git && cd B-Roll-Finder
docker build -t broll-finder .
docker run -d --name broll-bot --restart unless-stopped --env-file .env \
  -p 8770:8770 \
  -v "$PWD/downloads:/app/downloads" \
  -v "$PWD/.cache:/app/.cache" \
  -v "$PWD/cookies:/app/cookies:ro" \
  broll-finder
```

- The image includes ffmpeg, Deno and Node (for yt-dlp's JS challenge solver) and Remotion's headless Chrome.
- The health check is `GET /health` on port 8000. Run **one instance only**: two bots polling the same token cause Telegram `409 Conflict`. In Coolify, force-stop the old container before deploying.
- Keep `/app/.cache` and `/app/downloads` on persistent volumes. They hold cookies, caches, the Clip Library and finished projects.
- yt-dlp updates itself daily, because YouTube breaks old versions. `/test` shows the installed version.
- Optional thread caps: `BROLL_TORCH_THREADS`, `BROLL_FFMPEG_THREADS`, `BROLL_NORMALIZE_CONCURRENCY`.

---

## YouTube on a server: cookies and proxies

YouTube blocks datacenter IPs. If it isn't set up, you'll see "Sign in to confirm you're not a bot" or "This content isn't available", and runs quietly fall back to Pexels only.

1. **Cookies.** Export `cookies.txt` (Netscape format) from a browser that's logged into YouTube. Either drop it in `cookies/`, where any `*.txt` is picked up automatically, or just **send it to the bot**. Use a throwaway account. If a download fails with cookies, it's retried once without them.
2. **Proxy for downloads only.** `YT_DLP_PROXY` takes one proxy or a list of them, used in rotation with failover. A residential proxy is the reliable choice. Search runs direct, so only downloads go through the proxy.
3. **Free proxy lists.** Point `YT_DLP_PROXY_URL` at a list such as ProxyScrape. The bot tests proxies against YouTube, keeps a small pool of ones that work, and refills it as they die. See `/proxies`.

Run **`/test`** after any change. It performs real searches and downloads and tests the YouTube clients to show exactly what's blocked.

---

## Streamlit desktop app

For choosing clips by hand, with a gallery for each shot:

```bash
run.bat                          # Windows
chmod +x run.sh && ./run.sh      # macOS / Linux
```

It opens at `http://localhost:8501`, and the launcher creates the venv and installs dependencies. The app walks you through: **transcribe → shot list → fetch → rank (optional HD filter, auto-select) → review gallery → QA → PNG text overlays → export** (XML, `shot_list.json`, `.srt`). **🚀 Run everything automatically** runs the same pipeline with default settings and stops for review. Clip Library results show a purple border with a similarity % and usage count, and auto-picked clips get a 🤖 badge.

Manual install: `python -m venv venv`, activate it, `pip install -r requirements.txt`, `streamlit run app.py`. Needs Python 3.10+ and FFmpeg on PATH.

---

## API keys

Copy `.env.example` to `.env`. Every option is documented there.

| Key | Needed | Powers |
|---|---|---|
| `GROQ_API_KEY` (+ `_2`) | **yes** | Whisper transcription, and the free LLM tier |
| `DEEPSEEK_API_KEY` | recommended | **An OpenRouter key.** The paid DeepSeek tier, used first for every AI step |
| `PEXELS_API_KEY` (+ `_2`, `_3`…) | recommended | Pexels stock footage, rotated through on rate limits |
| `YOUTUBE_API_KEY` | optional | HD/SD check only (search uses yt-dlp) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USERS` | for the bot | The bot and its allowlist |
| `GEMINI_API_KEY` (+ `_2`) | optional | Backup vision model for describing library segments (OpenRouter GLM is tried first) |
| `SERPER_API_KEY` | optional | Per-shot and related Google images |
| `GOOGLE_CSE_API_KEY`, `GOOGLE_CSE_CX` | optional | Related images fallback |
| `OPENROUTER_API_KEY` | optional | Free LLM fallback |
| `PIXABAY_API_KEY` | optional | Pixabay (Streamlit app only) |
| `FREESOUND_API_KEY` | optional | SFX for the Streamlit PNG overlays |

---

## LLM providers and cost

Every text-LLM call goes through one dispatcher (`core/keywords.py`) that falls back automatically:

```
DeepSeek via OpenRouter (if DEEPSEEK_API_KEY)  →  Groq (key rotation)  →  OpenRouter free
```

The paid tier runs in two tiers with the same key:

| Tier | Model | Reasoning | Used for |
|---|---|---|---|
| **fast** | `~deepseek/deepseek-v4-flash-latest` | off | High-volume calls: shot slicing, ranking, keywords |
| **smart** | `xiaomi/mimo-v2.6-pro`, backup `~deepseek/deepseek-flash-latest` | on, 3,000-token budget | Once-per-video passes: topic, themes, segmenter, QA, overlay extraction |

You can override the models with `DEEPSEEK_MODEL_FAST`, `DEEPSEEK_MODEL_SMART` and `DEEPSEEK_MODEL_SMART_BACKUP`, and the thinking budget with `LLM_REASONING_BUDGET`. A reply cut off mid-thought is retried once with thinking off. `DEEPSEEK_NO_FALLBACK=true` retries DeepSeek with backoff instead of dropping to the free tiers. OpenRouter routing only sends calls to providers that support JSON mode and reasoning, and skips providers that return empty responses.

**Cost report.** Every LLM and Whisper call is metered (`core/usage.py`), and each job ends with a token and dollar breakdown. OpenRouter calls use the exact cost OpenRouter reports; the rest are estimates, so set `API_PRICING_JSON` / `WHISPER_USD_PER_HOUR` to match your actual rates (for example, 0 on a free Groq tier).

---

## Configuration

The most useful settings (all in `.env.example` with comments):

| Variable | Default | Effect |
|---|---|---|
| `AUTO_USE_PEXELS` / `AUTO_USE_YOUTUBE` / `AUTO_USE_LIBRARY` | `true` | Sources for automatic runs |
| `AUTO_PEXELS_NUM` / `AUTO_YOUTUBE_NUM` / `AUTO_LIBRARY_NUM` | `3` / `4` / `5` | Candidates per query or shot |
| `AUTO_MIN_HEIGHT` | `720` | Minimum candidate height |
| `ENABLE_QA_REVIEW` / `AUTO_REFINE` / `AUTO_FILL` | `true` / `false` / `true` | QA stages (the bot turns refine on per chat) |
| `FILL_EMPTY_WITH_TOPIC` / `FCPXML_FILL_GAPS` | `true` | Empty-shot fallbacks |
| `ENABLE_TEXT_OVERLAYS`, `OVERLAY_STYLE`, `OVERLAY_CHUNK_SEC` | `true`, `bold_yellow`, `150` | Overlays |
| `ENABLE_EXTRA_CLIPS`, `EXTRA_PER_KEYWORD`, `EXTRA_MAX_KEYWORDS` | `true`, `2`, `12` | Extras |
| `ENABLE_SHOT_IMAGES`, `SHOT_IMAGES_PER_SHOT` | `true`, `3` | Per-shot images |
| `ENABLE_RELATED_IMAGES` | `true` | Related stills |
| `ENABLE_CONTEXT_AWARE_KEYWORDS` / `ENABLE_DETAILED_QUERIES` | off | Query modes |
| `AUTO_SELECT_SHORT_SEC` / `AUTO_SELECT_YT_SECONDS` / `AUTO_SELECT_MIN_PEXELS` | `4` / `5` / `2` | Selection quota |
| `AUTO_SELECT_LOOKBACK` | `3` | Variety guard window |
| `DIRECTOR_BLOCK_SIZE` | `20` | Segments per shot-list call (raise it for long scripts) |
| `RANK_BATCH_SIZE` / `RANK_MAX_WORKERS` | `6` / `3` | Ranking throughput |
| `CLIP_LIBRARY_DB` | `.cache/clip_library.db` | Library location |
| `BOT_FILE_SERVER`, `BOT_PUBLIC_HOST`, `BOT_LINK_TTL` | off, auto, `0` (never expires) | Download links |
| `YT_DLP_PROXY`, `YT_DLP_PROXY_URL`, `YT_DOWNLOAD_NO_COOKIES` | none | YouTube access |
| `APP_PROXY` / `BOT_PROXY` | none | Route app / Telegram traffic through a local VPN proxy |

---

## Project layout

```
B-Roll Finder/
├── app.py                     Streamlit UI
├── bot/
│   ├── telegram_bot.py        commands, review gate, delivery
│   ├── settings.py            per-chat /settings menu
│   ├── pending_store.py       paused projects survive restarts
│   ├── fileserver.py          HMAC-signed download links
│   ├── healthserver.py        GET /health
│   └── logsetup.py            per-session logs for /logs
├── core/
│   ├── pipeline.py            headless end-to-end pipeline + self-healing loops
│   ├── transcription.py       Groq Whisper
│   ├── timing.py              audio duration, chunking, speech onsets
│   ├── director.py            topic, segmenter pre-pass, shot list
│   ├── director_search.py     parallel candidate fetch + query cache
│   ├── director_rank.py       LLM ranking, per-shot quota auto-select
│   ├── youtube.py             yt-dlp search/download, cookies, client fallbacks
│   ├── proxy_pool.py          validated YouTube proxy pool
│   ├── stock_apis.py          Pexels (key rotation) / Pixabay / YouTube Data API
│   ├── download_manager.py    parallel downloads, retries, dedup
│   ├── download_cache.py      cross-session URL → file registry
│   ├── extras.py              extra contextual clips
│   ├── shot_images.py         per-shot Google images (Serper)
│   ├── related_images.py      related stills → Clip Library
│   ├── overlays_remotion.py   overlay extraction + Remotion rendering
│   ├── captions.py            PNG overlays (Streamlit)
│   ├── clip_library.py        SQLite + fastembed library, export/merge
│   ├── xml_reimport.py        XML re-import → learned trims
│   ├── output.py              XML, SRT, zip; evaluate + repair
│   ├── selftest.py            /test preflight
│   ├── usage.py               per-job token/cost accounting
│   └── keywords.py            LLM dispatcher (DeepSeek → Groq → OpenRouter)
├── prompts/                   LLM system prompts
├── remotion/                  overlay renderer (Node / Remotion)
├── deploy/                    server guide + systemd unit
├── tests/                     pytest suite
├── Dockerfile · start_bot.bat · run.bat · run.sh
└── .env.example
```

---

## Tests

```bash
pytest tests/
```

Run `pytest tests/`, not bare `pytest`: the repo also has one-off scripts under `scratch/` that pytest would otherwise collect.

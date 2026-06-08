# B-Roll overlays (Remotion)

Animated, **transparent (alpha ProRes 4444)** text-overlay clips for the
pipeline. One parametrised composition (`Overlay`) is rendered once per highlight
by `core/overlays_remotion.py`, which passes the text/type/anim/sfx/duration as
input props.

## Setup
```bash
cd remotion
npm ci          # or: npm install
```
Remotion fetches a compatible headless Chrome on first render. On a server you
also need Chrome's system libs (see the repo Dockerfile).

## Sound effects
Put `swoosh.mp3`, `ding.mp3`, `thud.mp3` in `public/sfx/` (see that folder's
README). Missing files are skipped automatically (overlay renders silently).

## Preview in the browser
```bash
npm run studio
```

## Render one overlay manually (transparent ProRes)
```bash
npx remotion render src/index.ts Overlay out/test.mov \
  --props='{"text":"47% LESS WEAR","type":"stat","anim":"stat_pop","sfx":"ding","durationSec":3.5,"fps":30,"color":"#FFFFFF","accent":"#FFD400"}' \
  --codec=prores --prores-profile=4444 --pixel-format=yuva444p10le --image-format=png
```
The defaults in `remotion.config.ts` already select the alpha ProRes settings, so
the `--codec`/`--prores-profile`/`--pixel-format`/`--image-format` flags are
belt-and-suspenders.

Import `out/test.mov` onto a video track above your footage in Premiere — the
background is truly transparent (no chroma key needed).

## Animations (by `anim`)
- `title_card` — centered title + accent bar wipe (headings/titles)
- `stat_pop` / `money_count` — value pops on a dark pill (stats/numbers/money)
- `lower_third` — bar slides in bottom-left (secondary labels)
- `pop` — emphasis word scale-punch

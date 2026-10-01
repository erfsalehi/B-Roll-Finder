import React from 'react';
import {AbsoluteFill, Easing, Img, useCurrentFrame, useVideoConfig} from 'remotion';
import {loadFont} from '@remotion/google-fonts/Jost';

// "Image card": the split-screen reveal modelled on a reference edit (measured frame
// by frame, see core/motion_cards.py). A dark-green panel wipes in from the right
// while a photo card grows from the middle of the frame into the left half, and a
// caption fades in letter by letter underneath it. ONE eased progress value `p`
// drives the panel edge, the card scale and the card position — that is the whole
// trick. The right half then fills with whatever the planner chose: a big step
// number, a figure, or a second picture.
//
// Opaque, full frame (render to H.264, not alpha ProRes). Hard cut at the end.

// Loaded on first use inside a card render, not at import: Root.tsx imports this file
// for every composition, and a text-overlay render must not fetch Jost.
let fontStack: string | null = null;
const cardFont = (): string => {
  if (!fontStack) {
    const {fontFamily} = loadFont('normal', {weights: ['600', '700'], subsets: ['latin']});
    fontStack = `${fontFamily}, "Futura", "Century Gothic", "Arial Black", sans-serif`;
  }
  return fontStack;
};

export type CardRight =
  | {kind: 'number'; text: string; sub?: string}
  | {kind: 'stat'; text: string; sub?: string}
  | {kind: 'image'; image: string}
  | {kind: 'none'};

export type CardProps = {
  label: string;                 // caption under the card (also the headline when image is null)
  durationSec: number;
  fps: number;
  image: string | null;          // data URI / URL for the left card; null → typographic card
  right: CardRight;
  drift?: boolean | null;        // slow push-in while holding; null → only on long cards
};

export const DEFAULT_CARD_PROPS: CardProps = {
  label: 'CLEAN EXTERIOR',
  durationSec: 4,
  fps: 30,
  image: null,
  right: {kind: 'number', text: '1', sub: 'STEP'},
  drift: null,
};

// ── geometry, measured from the reference (1920×1080) ─────────────────────────
const W = 1920;
const H = 1080;
const SPLIT_X = 960;                 // where the green panel stops
const CARD_W = 663;
const CARD_H = 480;
const CARD_RADIUS = 22;              // scales with the card (it is a transform)
const CARD_END = {x: 518, y: 453.5}; // card centre once settled
const CARD_START = {x: 958, y: 520}; // …and while it is a thumbnail in the middle
const CARD_START_SCALE = 0.225;
const CAPTION_Y = 820;               // caption centre line
const CAPTION_SIZE = 45;
const TRANS_FRAMES = 34;             // the reference: 34 frames @30fps = 1.13 s

// Panel gradient: sampled from the reference (centre ≈ (1400, 440), falls off to
// near-black at the corners).
const PANEL_BG =
  'radial-gradient(circle 900px at 440px 440px, ' +
  'rgb(0,51,16) 0%, rgb(0,38,10) 25%, rgb(0,32,7) 35%, rgb(0,25,4) 50%, ' +
  'rgb(0,21,2) 58%, rgb(0,14,0) 75%, rgb(0,9,0) 90%, rgb(0,7,0) 100%)';

// Break a long label at the space nearest its middle.
function splitInTwo(text: string): string[] {
  const words = text.split(' ');
  if (words.length < 2) return [text];
  let best = 1;
  let bestDiff = Infinity;
  for (let i = 1; i < words.length; i++) {
    const diff = Math.abs(words.slice(0, i).join(' ').length - words.slice(i).join(' ').length);
    if (diff < bestDiff) {
      best = i;
      bestDiff = diff;
    }
  }
  return [words.slice(0, best).join(' '), words.slice(best).join(' ')];
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const easeInOut = Easing.inOut(Easing.cubic);
const easeOut = Easing.out(Easing.cubic);

// Each non-space letter fades in 1.67 frames after the previous one, over 2.5 frames
// (measured). Long captions are squeezed so the reveal never takes more than ~1 s.
function letterTimes(n: number, scale: number) {
  const stagger = Math.min(1.67 * scale, n > 1 ? (28 * scale) / (n - 1) : 1.67);
  return {stagger, fade: 2.5 * Math.max(0.6, scale)};
}

const Caption: React.FC<{text: string; frame: number; start: number; scale: number; cx: number;
                          y: number; size: number}> = ({text, frame, start, scale, cx, y, size}) => {
  const rows = text.split('\n');
  const letters = Array.from(text).filter((c) => c !== ' ' && c !== '\n').length;
  const {stagger, fade} = letterTimes(letters, scale);
  const lh = size * 1.24;
  let i = 0;
  return (
    <div
      style={{
        position: 'absolute',
        left: cx - 700,
        width: 1400,
        top: y - (rows.length * lh) / 2 + 1.5,
        textAlign: 'center',
        fontFamily: cardFont(),
        fontWeight: 700,
        fontSize: size,
        lineHeight: `${lh}px`,
        letterSpacing: '0.111em',
        color: '#fff',
        whiteSpace: 'pre',
      }}
    >
      {rows.map((row, r) => (
        <div key={r}>
          {Array.from(row).map((ch, k) => {
            if (ch === ' ') return <span key={k}>{' '}</span>;
            const t0 = start + i * stagger;
            i += 1;
            return (
              <span key={k} style={{opacity: clamp01((frame - t0) / fade)}}>{ch}</span>
            );
          })}
        </div>
      ))}
    </div>
  );
};

const PhotoCard: React.FC<{src: string; zoom: number}> = ({src, zoom}) => (
  <div
    style={{
      position: 'absolute', left: 0, top: 0, width: CARD_W, height: CARD_H,
      borderRadius: CARD_RADIUS, overflow: 'hidden', background: '#000',
    }}
  >
    <Img
      src={src}
      style={{width: '100%', height: '100%', objectFit: 'cover', transform: `scale(${zoom})`}}
    />
  </div>
);

// Big number / figure on the right panel.
const Numeral: React.FC<{text: string; sub?: string; frame: number; start: number}> = ({
  text, sub, frame, start,
}) => {
  const p = easeOut(clamp01((frame - start) / 16));
  const len = text.length;
  // Fit: one digit is huge, a long figure shrinks to stay inside the panel.
  const size = len <= 1 ? 640 : len === 2 ? 520 : len === 3 ? 400 : Math.max(150, Math.floor(760 / (len * 0.62)));
  return (
    <div
      style={{
        position: 'absolute', left: SPLIT_X, top: 0, width: W - SPLIT_X, height: H,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        opacity: p, transform: `translateY(${(1 - p) * 40}px) scale(${lerp(0.9, 1, p)})`,
      }}
    >
      {sub ? (
        <div
          style={{
            fontFamily: cardFont(), fontWeight: 600, fontSize: 44, letterSpacing: '0.32em',
            color: '#6fe3a0', opacity: 0.85, marginBottom: size > 300 ? -40 : 10,
            paddingLeft: '0.32em',
          }}
        >
          {sub}
        </div>
      ) : null}
      <div
        style={{
          fontFamily: cardFont(), fontWeight: 700, fontSize: size, lineHeight: 1, color: '#eafff1',
          textShadow: '0 0 90px rgba(60,255,140,0.28), 0 6px 18px rgba(0,0,0,0.35)',
          letterSpacing: len > 1 ? '0.01em' : 0,
        }}
      >
        {text}
      </div>
    </div>
  );
};

export const ImageCard: React.FC<CardProps> = (props) => {
  const frame = useCurrentFrame();
  const {durationInFrames} = useVideoConfig();
  const {label, image, right} = props;

  // The reveal needs ~34 frames; a short card compresses it instead of being cut off.
  const T = Math.min(TRANS_FRAMES, Math.max(10, Math.round(durationInFrames * 0.55)));
  const k = T / TRANS_FRAMES;
  const p = easeInOut(clamp01(frame / T));

  const long = props.drift == null ? props.durationSec > 4.5 : !!props.drift;
  const hold = clamp01((frame - T) / Math.max(1, durationInFrames - T));
  const zoom = long ? 1 + 0.045 * hold : 1;

  const panelLeft = lerp(W, SPLIT_X, p);
  const rightStart = Math.round(T * 0.55);
  const hasImage = !!image;

  // Left card: scale + position share the panel's clock.
  const s = lerp(CARD_START_SCALE, 1, p);
  const cx = lerp(CARD_START.x, CARD_END.x, p);
  const cy = lerp(CARD_START.y, CARD_END.y, p);
  // Typographic variant (no photo): the label itself is the left-hand headline, split
  // over two lines when long and sized so the widest line fits the left half.
  const headRows = label.length > 14 ? splitInTwo(label) : [label];
  const headSize = Math.max(40, Math.min(96, Math.floor(800 / (Math.max(...headRows.map((r) => r.length)) * 0.8))));

  return (
    <AbsoluteFill style={{background: '#000'}}>
      {/* green panel, wiping in from the right edge */}
      <div
        style={{
          position: 'absolute', left: panelLeft, top: 0, width: W - SPLIT_X, height: H,
          background: PANEL_BG,
        }}
      />

      {hasImage ? (
        <>
          <div
            style={{
              position: 'absolute', left: cx - CARD_W / 2, top: cy - CARD_H / 2,
              width: CARD_W, height: CARD_H, transform: `scale(${s})`, transformOrigin: '50% 50%',
            }}
          >
            <PhotoCard src={image as string} zoom={zoom} />
          </div>
          <Caption text={label} frame={frame} start={Math.round(T * 0.5)} scale={k}
                   cx={CARD_END.x} y={CAPTION_Y} size={CAPTION_SIZE} />
        </>
      ) : (
        <div style={{position: 'absolute', left: 0, top: 0, width: SPLIT_X, height: H, overflow: 'hidden'}}>
          <Caption text={headRows.join('\n')} frame={frame} start={Math.round(T * 0.3)} scale={k}
                   cx={CARD_END.x} y={H / 2 - 10} size={headSize} />
        </div>
      )}

      {right.kind === 'number' || right.kind === 'stat' ? (
        <Numeral text={right.text} sub={right.sub} frame={frame} start={rightStart} />
      ) : null}

      {right.kind === 'image' ? (() => {
        const q = easeOut(clamp01((frame - rightStart) / 18));
        return (
          <div
            style={{
              position: 'absolute', left: 1400 - CARD_W / 2 + (1 - q) * 70, top: CARD_END.y - CARD_H / 2,
              width: CARD_W, height: CARD_H, opacity: q,
            }}
          >
            <PhotoCard src={right.image} zoom={zoom} />
          </div>
        );
      })() : null}
    </AbsoluteFill>
  );
};

export function cardDurationFrames(props: CardProps): number {
  const fps = props.fps ?? 30;
  return Math.max(1, Math.round((props.durationSec ?? 4) * fps));
}

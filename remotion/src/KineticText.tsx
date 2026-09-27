import React from 'react';
import {AbsoluteFill, Easing, interpolate, random} from 'remotion';

// "Kinetic" look, modelled on the big-channel explainer style: huge flat-yellow
// heavy caps, a soft drop shadow, NO box — and a per-letter entrance. Four
// entrances (one per overlay, picked deterministically from the text so a video
// gets a mix but a re-render is identical):
//   pop      letters pop up from nothing in random order, dropping into place
//   fly      letters spin in from the lower right and assemble
//   converge letters start widely tracked + motion-blurred and snap together
//   mask     each line slides out from behind an invisible left edge
export type KineticEntrance = 'pop' | 'fly' | 'converge' | 'mask';
export const KINETIC_ENTRANCES: KineticEntrance[] = ['pop', 'fly', 'converge', 'mask'];

type Props = {
  text: string;
  frame: number;
  fps: number;
  durationInFrames: number;
  color: string;
  fontFamily: string;
  weight: number;
  upper: boolean;
  entrance: KineticEntrance;
};

// Canvas-relative sizing (1920×1080). The reference sets ~11-char lines at
// ~150-190px and lets short words go bigger, capped so a single word doesn't
// swallow the frame. A lone line runs wider than a stacked block.
const MAX_FONT = 200;
const MIN_FONT = 64;
const TARGET_LINE_W = 1250;   // px the longest line of a stack should fill
const TARGET_SINGLE_W = 1450; // px a single-line overlay should fill
const MAX_BLOCK_H = 760;      // px the whole text block may occupy
const LINE_HEIGHT = 1.2;
// Whole-entrance length; per-letter motions finish inside this window.
const ENTRANCE_SEC = 0.7;

const SHADOW = '0 6px 14px rgba(0,0,0,0.55), 0 2px 4px rgba(0,0,0,0.45)';

// Rough per-glyph advance for a black-weight geometric sans (Montserrat 900),
// in em. Only used to pick a font size that keeps the longest line in frame.
function glyphEm(ch: string): number {
  if (ch === ' ') return 0.3;
  if ('IJ1.,:;\'!|'.includes(ch)) return 0.38;
  if ('MW'.includes(ch)) return 1.05;
  if ('mw'.includes(ch)) return 0.95;
  if (ch >= 'a' && ch <= 'z') return 0.66;
  return 0.8;
}

function lineEm(line: string): number {
  let w = 0;
  for (const ch of line) w += glyphEm(ch);
  return w;
}

// Break into lines: a single word (or a tiny phrase) stays on one line; anything
// else is stacked into the fewest lines of ≤ ~14 chars (~22 for long titles), balanced so the block
// reads as a stack ("40 DEGREES / CELCIUS", "HVAC / SYSTEM").
export function wrapLines(text: string): string[] {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [''];
  const total = words.join(' ').length;
  if (total <= 7 || words.length === 1) return [words.join(' ')];
  // Full-sentence titles get wider lines so they read as 3-4 lines, not a
  // narrow tower of 6+.
  const perLine = total > 40 ? 22 : 14;
  const nLines = Math.max(2, Math.ceil(total / perLine));
  const target = total / nLines;
  const lines: string[] = [];
  let cur = '';
  for (const w of words) {
    const next = cur ? `${cur} ${w}` : w;
    const remainingLines = nLines - lines.length;
    // Break once this line reaches its share, unless it's the last line.
    if (cur && remainingLines > 1 && next.length > target + 2 &&
        Math.abs(cur.length - target) <= Math.abs(next.length - target)) {
      lines.push(cur);
      cur = w;
    } else {
      cur = next;
    }
  }
  if (cur) lines.push(cur);
  return lines;
}

function fitFont(lines: string[]): number {
  const widest = Math.max(...lines.map(lineEm), 1);
  const byWidth = (lines.length === 1 ? TARGET_SINGLE_W : TARGET_LINE_W) / widest;
  const byHeight = MAX_BLOCK_H / (lines.length * LINE_HEIGHT);
  return Math.round(Math.max(MIN_FONT, Math.min(MAX_FONT, byWidth, byHeight)));
}

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const easeOut = Easing.out(Easing.cubic);

export const KineticText: React.FC<Props> = ({
  text, frame, fps, durationInFrames, color, fontFamily, weight, upper, entrance,
}) => {
  const shown = upper ? text.toUpperCase() : text;
  const lines = wrapLines(shown);
  const fontSize = fitFont(lines);
  const entranceF = Math.max(8, Math.round(ENTRANCE_SEC * fps));
  // Clips can be short; never let the entrance eat more than ~45% of one.
  const E = Math.min(entranceF, Math.max(6, Math.floor(durationInFrames * 0.45)));
  const fadeOut = interpolate(
    frame, [durationInFrames - 8, durationInFrames - 1], [1, 0], clamp,
  );

  const lineStyle: React.CSSProperties = {
    display: 'block',
    whiteSpace: 'pre',
    lineHeight: LINE_HEIGHT,
  };

  let letterIdx = 0;
  const totalLetters = lines.join('').replace(/\s/g, '').length || 1;

  const renderLine = (line: string, li: number) => {
    if (entrance === 'mask') {
      // Whole line slides right out from behind a clip edge at its own left
      // side. Later lines trail slightly.
      const start = li * Math.round(E * 0.15);
      const p = interpolate(frame, [start, start + E], [0, 1], {
        ...clamp, easing: Easing.bezier(0.16, 1, 0.3, 1),
      });
      return (
        <span key={li} style={lineStyle}>
          {/* inline-block so the clip edge sits at the TEXT's left side, not
              the (wider, centred) block's. Clip only the left edge; leave
              room for the shadow elsewhere. */}
          <span style={{
            display: 'inline-block',
            clipPath: 'inset(-40% -40% -40% 0)',
          }}>
            <span style={{
              display: 'inline-block',
              transform: `translateX(${(p - 1) * 105}%)`,
            }}>
              {line}
            </span>
          </span>
        </span>
      );
    }

    const chars = Array.from(line);
    const mid = (chars.length - 1) / 2;
    return (
      <span key={li} style={lineStyle}>
        {chars.map((ch, ci) => {
          if (ch === ' ') return <span key={ci}>{' '}</span>;
          const i = letterIdx++;
          const seed = `${text}|${li}|${ci}`;
          const r1 = random(seed + 'a');
          const r2 = random(seed + 'b');
          const r3 = random(seed + 'c');
          let style: React.CSSProperties = {display: 'inline-block'};

          if (entrance === 'pop') {
            // Random order, each letter grows from a raised, tiny state and
            // drops onto the baseline with a small overshoot.
            const len = Math.max(5, Math.round(E * 0.45));
            const start = Math.round(r1 * (E - len));
            const p = interpolate(frame, [start, start + len], [0, 1], clamp);
            const s = interpolate(p, [0, 0.7, 1], [0.15, 1.08, 1]);
            const y = interpolate(p, [0, 1], [-0.45 - r2 * 0.25, 0], {
              easing: easeOut,
            });
            style = {
              ...style,
              opacity: p > 0 ? Math.min(1, p * 3) : 0,
              transform: `translateY(${y}em) scale(${s})`,
              transformOrigin: '50% 100%',
            };
          } else if (entrance === 'fly') {
            // Reading-order stagger; each letter travels from the lower right
            // along a spin into its slot.
            const len = Math.max(6, Math.round(E * 0.6));
            const start = Math.round((i / totalLetters) * (E - len) * 0.8 + r3 * 2);
            const p = interpolate(frame, [start, start + len], [0, 1], {
              ...clamp, easing: easeOut,
            });
            const dx = (1 - p) * (2.2 + r1 * 2.5);
            const dy = (1 - p) * (1.2 + r2 * 2.0);
            const rot = (1 - p) * (120 + r3 * 200) * (r1 > 0.5 ? 1 : -1);
            style = {
              ...style,
              opacity: interpolate(p, [0, 0.25], [0, 1], clamp),
              transform: `translate(${dx}em, ${dy}em) rotate(${rot}deg) scale(${0.6 + 0.4 * p})`,
            };
          } else {
            // converge: start spread far from the line's centre, stretched and
            // blurred horizontally, then snap together.
            const len = Math.max(6, Math.round(E * 0.75));
            const start = Math.round(r1 * (E - len));
            const p = interpolate(frame, [start, start + len], [0, 1], {
              ...clamp, easing: Easing.out(Easing.quad),
            });
            const spread = (ci - mid) * 0.9 + (r2 - 0.5) * 1.2;
            const dx = (1 - p) * spread;
            style = {
              ...style,
              opacity: interpolate(p, [0, 0.2], [0, 1], clamp),
              transform: `translateX(${dx}em) scaleX(${1 + (1 - p) * 0.8})`,
              filter: p < 1 ? `blur(${(1 - p) * 10}px)` : undefined,
            };
          }
          return <span key={ci} style={style}>{ch}</span>;
        })}
      </span>
    );
  };

  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      <div
        style={{
          opacity: fadeOut,
          textAlign: 'center',
          fontFamily,
          fontWeight: weight,
          fontSize,
          color,
          letterSpacing: 0,
          textShadow: SHADOW,
          maxWidth: '92%',
        }}
      >
        {lines.map(renderLine)}
      </div>
    </AbsoluteFill>
  );
};

// Stable pick of an entrance for a given overlay text.
export function pickEntrance(text: string, type?: string): KineticEntrance {
  // Figures read best as the letter pop (the "40 DEGREES" look).
  if (type === 'stat' || type === 'money' || type === 'number') return 'pop';
  const idx = Math.floor(random(`entrance|${text}`) * KINETIC_ENTRANCES.length);
  return KINETIC_ENTRANCES[Math.min(idx, KINETIC_ENTRANCES.length - 1)];
}

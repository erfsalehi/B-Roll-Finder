import React from 'react';
import {
  AbsoluteFill,
  Audio,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {loadFont} from '@remotion/google-fonts/Montserrat';

const {fontFamily: MONTSERRAT} = loadFont();

// Replace the first numeric token in `text` with `target * progress`, preserving
// currency symbols, %, commas, decimals, and surrounding words. Drives the
// count-up on stat / money / number overlays.
function countUp(text: string, progress: number): string {
  const m = text.match(/[\d,]+(?:\.\d+)?/);
  if (!m || m.index === undefined) return text;
  const numStr = m[0];
  const target = parseFloat(numStr.replace(/,/g, ''));
  if (!isFinite(target)) return text;
  const decimals = (numStr.split('.')[1] || '').length;
  const val = target * Math.max(0, Math.min(1, progress));
  const formatted = val.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return text.slice(0, m.index) + formatted + text.slice(m.index + numStr.length);
}

export type OverlayType =
  | 'title'
  | 'heading'
  | 'stat'
  | 'money'
  | 'number'
  | 'emphasis';

export type OverlayProps = {
  text: string;
  type: OverlayType;
  anim: 'title_card' | 'stat_pop' | 'money_count' | 'lower_third' | 'pop';
  sfx: 'swoosh' | 'ding' | 'thud' | 'none';
  durationSec: number;
  fps: number;
  color: string;   // primary text color
  accent: string;  // accent (bars, numbers)
};

export const DEFAULT_PROPS: OverlayProps = {
  text: 'SAMPLE TITLE',
  type: 'title',
  anim: 'title_card',
  sfx: 'swoosh',
  durationSec: 4,
  fps: 30,
  color: '#FFFFFF',
  accent: '#FFD400',
};

const FONT = `${MONTSERRAT}, "Arial Black", "Helvetica Neue", Arial, sans-serif`;

export const Overlay: React.FC<OverlayProps> = (props) => {
  const {text, anim, sfx, color, accent} = props;
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();

  // Entrance spring (0 → 1) and a symmetric fade-out near the end.
  const enter = spring({frame, fps, config: {damping: 200, mass: 0.6}});
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 10, durationInFrames - 1],
    [1, 0],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );
  const opacity = Math.min(enter, fadeOut);

  const Sfx =
    sfx && sfx !== 'none' ? (
      <Audio src={staticFile(`sfx/${sfx}.mp3`)} volume={0.8} />
    ) : null;

  // ── title_card: centered title with an accent bar that wipes in ──────────
  if (anim === 'title_card') {
    const slide = interpolate(enter, [0, 1], [60, 0]);
    const barW = interpolate(enter, [0.2, 1], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    return (
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        {Sfx}
        <div
          style={{
            opacity,
            transform: `translateY(${slide}px)`,
            textAlign: 'center',
            maxWidth: '80%',
          }}
        >
          <div
            style={{
              fontFamily: FONT,
              fontWeight: 900,
              fontSize: 110,
              lineHeight: 1.05,
              color,
              textShadow: '0 8px 30px rgba(0,0,0,0.55)',
              letterSpacing: -1,
            }}
          >
            {text}
          </div>
          <div
            style={{
              height: 14,
              marginTop: 26,
              background: accent,
              borderRadius: 8,
              width: `${barW * 100}%`,
              marginLeft: 'auto',
              marginRight: 'auto',
            }}
          />
        </div>
      </AbsoluteFill>
    );
  }

  // ── stat_pop / money_count / number: punchy value on a dark pill ─────────
  if (anim === 'stat_pop' || anim === 'money_count') {
    const pop = interpolate(enter, [0, 0.6, 1], [0.5, 1.12, 1], {
      extrapolateRight: 'clamp',
    });
    return (
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        {Sfx}
        <div
          style={{
            opacity,
            transform: `scale(${pop})`,
            padding: '34px 60px',
            background: 'rgba(10,10,12,0.82)',
            borderRadius: 28,
            border: `4px solid ${accent}`,
            boxShadow: '0 18px 60px rgba(0,0,0,0.5)',
          }}
        >
          <span
            style={{
              fontFamily: FONT,
              fontWeight: 900,
              fontSize: 150,
              color: accent,
              letterSpacing: -2,
            }}
          >
            {countUp(text, enter)}
          </span>
        </div>
      </AbsoluteFill>
    );
  }

  // ── lower_third: bar sliding in from bottom-left ─────────────────────────
  if (anim === 'lower_third') {
    const x = interpolate(enter, [0, 1], [-700, 80]);
    return (
      <AbsoluteFill>
        {Sfx}
        <div
          style={{
            position: 'absolute',
            bottom: 140,
            left: 0,
            opacity,
            transform: `translateX(${x}px)`,
            background: 'rgba(10,10,12,0.85)',
            borderLeft: `10px solid ${accent}`,
            padding: '24px 44px',
          }}
        >
          <span
            style={{fontFamily: FONT, fontWeight: 800, fontSize: 64, color}}
          >
            {text}
          </span>
        </div>
      </AbsoluteFill>
    );
  }

  // ── pop (emphasis): quick scale punch, centered ──────────────────────────
  const punch = interpolate(enter, [0, 0.5, 1], [0.6, 1.18, 1], {
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
      {Sfx}
      <span
        style={{
          opacity,
          transform: `scale(${punch}) rotate(-3deg)`,
          fontFamily: FONT,
          fontWeight: 900,
          fontSize: 170,
          color,
          textShadow: `0 0 40px ${accent}`,
          letterSpacing: -2,
        }}
      >
        {text}
      </span>
    </AbsoluteFill>
  );
};

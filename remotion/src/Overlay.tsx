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
  const hadGrouping = numStr.includes(',');
  const target = parseFloat(numStr.replace(/,/g, ''));
  if (!isFinite(target)) return text;
  const decimals = (numStr.split('.')[1] || '').length;
  // A year (bare 4-digit integer like 2026) is a label, not a quantity: never
  // add a thousands separator (2,026 is wrong) and never count up to it — render
  // the text exactly as written.
  const isYear =
    !hadGrouping && decimals === 0 && /^\d{4}$/.test(numStr) &&
    target >= 1000 && target <= 2999;
  if (isYear) return text;
  const val = target * Math.max(0, Math.min(1, progress));
  // Only group thousands when the source already did (10,000 / $4,999) — so we
  // never invent commas that weren't written.
  const formatted = val.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
    useGrouping: hadGrouping,
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

// Style preset tokens — a bundle applied to EVERY overlay in a video. Picked on
// the Python side (core/overlays_remotion.py OVERLAY_STYLES) and passed through
// props, so the same component renders bold-yellow, clean-white, neon, or boxed-
// news looks without changing the per-overlay `anim` layout.
export type StyleTokens = {
  weight: number;                       // font weight (400–900)
  upper: boolean;                       // ALL-CAPS the display text
  box: 'scrim' | 'none' | 'solid';      // background treatment behind text
  stroke: boolean;                      // bold dark outline on the glyphs
  glow: boolean;                        // accent-colored glow around the text
  radius: number;                       // corner radius of the box
  name: string;                         // preset id (for debugging)
};

export type OverlayProps = {
  text: string;
  type: OverlayType;
  anim: 'title_card' | 'stat_pop' | 'money_count' | 'lower_third' | 'pop';
  sfx: 'swoosh' | 'ding' | 'thud' | 'none';
  durationSec: number;
  fps: number;
  color: string;   // primary text color
  accent: string;  // accent (bars, numbers)
  style?: StyleTokens;
};

// The original "bold yellow" look — also the fallback when a clip was rendered
// before styles existed (props.style undefined).
const DEFAULT_STYLE: StyleTokens = {
  weight: 900, upper: true, box: 'scrim', stroke: true, glow: false,
  radius: 24, name: 'bold_yellow',
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
  style: DEFAULT_STYLE,
};

const FONT = `${MONTSERRAT}, "Arial Black", "Helvetica Neue", Arial, sans-serif`;

// Titles are now the FULL spoken title line (not a 1-5 word card), so they can
// run long. Scale the font down as the text grows so a whole-sentence title
// still fits the card instead of overflowing the frame.
function fitFontSize(text: string, big: number, small: number): number {
  const len = (text || '').trim().length;
  if (len <= 22) return big;
  if (len >= 90) return small;
  // Linear ramp between the short-title and long-title sizes.
  const t = (len - 22) / (90 - 22);
  return Math.round(big + (small - big) * t);
}

// Title/action-safe margins for the 1920×1080 canvas (~6%), so edge-anchored
// overlays (e.g. the lower third) never hug the frame border in a 1080p edit.
const SAFE_X = 120;
const SAFE_Y = 120;

// A dark scrim behind text so copy stays readable over ANY footage these alpha
// overlays get composited onto. The "solid" box is more opaque for the broadcast
// caption look.
const SCRIM = 'rgba(8,8,10,0.58)';
const SOLID_BG = 'rgba(12,12,14,0.96)';
// Layered drop shadow for text sitting directly on footage (no box behind it).
const TEXT_SHADOW = '0 4px 10px rgba(0,0,0,0.85), 0 2px 30px rgba(0,0,0,0.6)';
// Bold dark outline ("border") so vivid copy pops on any footage.
// paintOrder:'stroke' renders the outline BEHIND the fill so it doesn't eat the
// letterforms — a clean thick border rather than a thinned glyph.
const TEXT_STROKE = '3px rgba(0,0,0,0.9)';
const STROKED: React.CSSProperties = {
  WebkitTextStroke: TEXT_STROKE,
  paintOrder: 'stroke',
};

export const Overlay: React.FC<OverlayProps> = (props) => {
  const {text, anim, sfx, color, accent} = props;
  const st = props.style ?? DEFAULT_STYLE;
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();

  // ── style-derived helpers (applied across every anim branch) ──────────────
  const upper: React.CSSProperties = st.upper ? {textTransform: 'uppercase'} : {};
  const stroked: React.CSSProperties = st.stroke ? STROKED : {};
  const glow = st.glow ? `, 0 0 38px ${accent}, 0 0 14px ${accent}` : '';
  const textShadow = TEXT_SHADOW + glow;
  const boxBg =
    st.box === 'solid' ? SOLID_BG : st.box === 'scrim' ? SCRIM : 'transparent';
  // The broadcast "solid" box gets a thick accent edge; scrim/none don't.
  const solidEdge: React.CSSProperties =
    st.box === 'solid' ? {borderLeft: `12px solid ${accent}`} : {};
  const boxShadow =
    st.box === 'none' ? 'none' : '0 24px 70px rgba(0,0,0,0.45)';

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
    const padded = st.box !== 'none';
    return (
      <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center'}}>
        {Sfx}
        <div
          style={{
            opacity,
            transform: `translateY(${slide}px)`,
            textAlign: 'center',
            maxWidth: '78%',
            background: boxBg,
            padding: padded ? '36px 56px' : 0,
            borderRadius: st.radius,
            boxShadow,
            ...solidEdge,
          }}
        >
          <div
            style={{
              fontFamily: FONT,
              fontWeight: st.weight,
              fontSize: fitFontSize(text, 104, 52),
              lineHeight: 1.06,
              color,
              textShadow,
              letterSpacing: -1,
              ...stroked,
              ...upper,
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
            background: st.box === 'none' ? 'transparent' : SOLID_BG,
            borderRadius: Math.max(st.radius, 20),
            border: `4px solid ${accent}`,
            boxShadow: st.box === 'none' ? 'none' : '0 18px 60px rgba(0,0,0,0.5)',
          }}
        >
          <span
            style={{
              fontFamily: FONT,
              fontWeight: Math.max(st.weight, 800),
              fontSize: 150,
              color: accent,
              letterSpacing: -2,
              textShadow: st.glow ? glow.slice(2) : undefined,
              ...upper,
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
    // Slide in from off-screen left and rest at the safe-area margin (never at
    // the very edge of the frame).
    const x = interpolate(enter, [0, 1], [-700, 0]);
    return (
      <AbsoluteFill>
        {Sfx}
        <div
          style={{
            position: 'absolute',
            bottom: SAFE_Y,
            left: SAFE_X,
            maxWidth: 1920 - 2 * SAFE_X,
            opacity,
            transform: `translateX(${x}px)`,
            background: st.box === 'none' ? 'transparent' : SOLID_BG,
            borderLeft: `10px solid ${accent}`,
            borderRadius: Math.max(st.radius, 6),
            padding: '24px 44px',
            boxShadow: st.box === 'none' ? 'none' : '0 18px 50px rgba(0,0,0,0.45)',
          }}
        >
          <span
            style={{fontFamily: FONT, fontWeight: st.weight,
                    fontSize: fitFontSize(text, 64, 40), color,
                    textShadow, ...stroked, ...upper}}
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
          fontWeight: st.weight,
          fontSize: 170,
          color,
          // Dark drop shadow for legibility over bright footage + an accent glow
          // (always present on emphasis; intensified when the style asks for it).
          textShadow: `0 6px 22px rgba(0,0,0,0.8), 0 0 40px ${accent}${glow}`,
          letterSpacing: -2,
          ...stroked,
          ...upper,
        }}
      >
        {text}
      </span>
    </AbsoluteFill>
  );
};

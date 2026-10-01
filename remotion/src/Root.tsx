import React from 'react';
import {Composition} from 'remotion';
import {Overlay, DEFAULT_PROPS} from './Overlay';
import {ImageCard, DEFAULT_CARD_PROPS, cardDurationFrames} from './ImageCard';

// Two parametrised compositions. The Python wrapper renders them once per item,
// passing the props as JSON:
//   Overlay    — animated text over footage (transparent ProRes), core/overlays_remotion.py
//   ImageCard  — opaque split-screen photo card (H.264), core/motion_cards.py
// Durations are derived from the props via calculateMetadata.
export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="Overlay"
        component={Overlay}
        durationInFrames={120}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={DEFAULT_PROPS}
        calculateMetadata={({props}) => {
          const fps = props.fps ?? 30;
          const durationSec = props.durationSec ?? 4;
          return {
            fps,
            durationInFrames: Math.max(1, Math.round(durationSec * fps)),
          };
        }}
      />
      <Composition
        id="ImageCard"
        component={ImageCard}
        durationInFrames={120}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={DEFAULT_CARD_PROPS}
        calculateMetadata={({props}) => ({
          fps: props.fps ?? 30,
          durationInFrames: cardDurationFrames(props),
        })}
      />
    </>
  );
};

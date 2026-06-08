import React from 'react';
import {Composition} from 'remotion';
import {Overlay, DEFAULT_PROPS} from './Overlay';

// A single parametrised composition. The Python wrapper renders it once per
// highlight, passing the text/type/anim/sfx/duration as input props. Duration
// is derived from the highlight length via calculateMetadata.
export const RemotionRoot: React.FC = () => {
  return (
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
  );
};

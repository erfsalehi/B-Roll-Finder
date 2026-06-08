// Defaults so a bare `remotion render` produces a transparent ProRes 4444 .mov
// ready to composite on the overlay track in Premiere (true alpha — no green
// screen / chroma key needed). These can be overridden per-render via CLI flags.
import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('png');     // PNG frames carry the alpha channel
Config.setPixelFormat('yuva444p10le'); // 4:4:4 + alpha, 10-bit
Config.setCodec('prores');
Config.setProResProfile('4444');       // the alpha-capable ProRes profile
Config.setOverwriteOutput(true);

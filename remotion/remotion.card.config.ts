// Render settings for the opaque ImageCard clips (core/motion_cards.py passes this with
// --config). remotion.config.ts forces alpha ProRes for the text overlays, and Remotion
// refuses `--codec=h264` while a ProRes profile is configured, so cards get their own.
import {Config} from '@remotion/cli/config';

Config.setVideoImageFormat('jpeg');   // frames are opaque — JPEG is much faster than PNG
Config.setJpegQuality(95);
Config.setPixelFormat('yuv420p');
// Without this the JPEG frames come out tagged full-range BT.601 (yuvj420p), unlike the
// limited-range BT.709 footage next to them; Premiere can crush the near-black green.
Config.setColorSpace('bt709');
Config.setCodec('h264');
Config.setCrf(18);
Config.setOverwriteOutput(true);

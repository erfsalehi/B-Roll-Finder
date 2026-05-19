import os
import subprocess
import tempfile
import textwrap
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from groq import Groq
from core.keywords import _call_groq_json, _call_openrouter_json

_FONT_MAP = {
    "Arial Bold":           "C:\\Windows\\Fonts\\arialbd.ttf",
    "Arial":                "C:\\Windows\\Fonts\\arial.ttf",
    "Impact":               "C:\\Windows\\Fonts\\impact.ttf",
    "Segoe UI Bold":        "C:\\Windows\\Fonts\\segoeuib.ttf",
    "Times New Roman Bold": "C:\\Windows\\Fonts\\timesbd.ttf",
    "Trebuchet Bold":       "C:\\Windows\\Fonts\\trebucbd.ttf",
    "Verdana Bold":         "C:\\Windows\\Fonts\\verdanab.ttf",
    "DejaVu Bold (Linux)":  "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
}

# Render at this multiple of the final 1920x1080 canvas, then downsample with
# LANCZOS. Super-sampling is what gives the text crisp, photographic edges
# instead of the chunky aliasing PIL produces at native size — Premiere then
# rescales an already-sharp asset rather than amplifying jagged edges.
_SSAA = 2

# Final canvas size that ships to Premiere.
_W, _H = 1920, 1080


def get_available_fonts() -> dict:
    """Returns {display_name: path} for fonts present on this machine."""
    return {name: path for name, path in _FONT_MAP.items() if os.path.exists(path)}

def load_highlights_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'highlights.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_highlights(script_text: str, groq_key: str = None) -> list:
    system_prompt = load_highlights_prompt()
    user_content = f"Script for analysis:\n\n{script_text}"
    try:
        if groq_key:
            client = Groq(api_key=groq_key)
            res = _call_groq_json(client, system_prompt, user_content)
        else:
            res = _call_openrouter_json(system_prompt, user_content)
        return res.get("highlights", [])
    except Exception as e:
        print(f"Highlight extraction failed: {e}")
        return []

def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)

def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        if font_path and os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    for path in _FONT_MAP.values():
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_overlay_rgba(
    text: str,
    font_path: str = None,
    font_size: int = 120,
    color: str = "#FFFFFF",
    shadow_color: str = "#000000",
    y_position: int = 800,
    bg_color: str = None,
    bg_opacity: int = 160,
    outline: bool = False,
    auto_scale: bool = False,
    text_opacity: int = 255,
    x_offset: int = 0,
) -> Image.Image:
    """
    Core renderer that returns a 1920x1080 RGBA image with the caption drawn.
    Used by both create_text_overlay (which saves to PNG) and the live preview
    pipeline (which composites over a background frame).

    Super-samples at _SSAA x to give edges proper LANCZOS-quality antialiasing;
    PIL's native-resolution text rendering is what makes overlays look chunky
    when Premiere then rescales them on the timeline.
    """
    s = _SSAA
    W, H = _W * s, _H * s
    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    ss_size = max(8, int(font_size * s))
    font = _load_font(font_path, ss_size)

    if auto_scale:
        max_block_h = int(H * 0.60)
        for _ in range(20):
            max_chars = max(10, int((W * 0.94) / max(1, ss_size * 0.6)))
            lines = textwrap.wrap(text, width=max_chars) or [text]
            bboxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
            heights = [bb[3] - bb[1] for bb in bboxes]
            gap = max(4, ss_size // 8)
            total_h = sum(heights) + gap * (len(lines) - 1)
            if total_h <= max_block_h or ss_size <= 30 * s:
                break
            ss_size = max(30 * s, int(ss_size * 0.85))
            font = _load_font(font_path, ss_size)

    max_chars = max(10, int((W * 0.94) / max(1, ss_size * 0.6)))
    lines = textwrap.wrap(text, width=max_chars) or [text]

    line_bboxes  = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_widths  = [bb[2] - bb[0] for bb in line_bboxes]
    line_gap     = max(4, ss_size // 8)
    total_h      = sum(line_heights) + line_gap * (len(lines) - 1)
    max_w        = max(line_widths) if line_widths else 0
    y_start      = y_position * s - total_h // 2
    # X offset is specified in 1080p pixels; scale to the super-sampled canvas
    # so the visual shift matches what the user sees in the preview UI.
    x_shift      = int(x_offset) * s

    if bg_color:
        pad_x = ss_size // 3
        pad_y = ss_size // 5
        box_x1 = (W - max_w) // 2 - pad_x + x_shift
        box_y1 = y_start - pad_y
        box_x2 = (W + max_w) // 2 + pad_x + x_shift
        box_y2 = y_start + total_h + pad_y
        radius = max(8, ss_size // 6)
        box_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(box_layer).rounded_rectangle(
            [box_x1, box_y1, box_x2, box_y2],
            radius=radius,
            fill=_hex_to_rgba(bg_color, bg_opacity),
        )
        img = Image.alpha_composite(img, box_layer)
        draw = ImageDraw.Draw(img)

    text_rgba   = _hex_to_rgba(color, text_opacity)
    shadow_rgba = _hex_to_rgba(shadow_color, min(text_opacity, 220))

    if outline:
        # PIL's stroke_width gives a proper edge-antialiased outline. The old
        # 8-direction manual blit produced visibly jagged corners; this draws
        # the stroke once with FreeType's own AA pipeline.
        stroke_w = max(2, ss_size // 18)
        y = y_start
        for ln, bb, lh in zip(lines, line_bboxes, line_heights):
            text_w = bb[2] - bb[0]
            x = (W - text_w) / 2 - bb[0] + x_shift
            draw.text(
                (x, y), ln, font=font, fill=text_rgba,
                stroke_width=stroke_w, stroke_fill=shadow_rgba,
            )
            y += lh + line_gap
    else:
        # Soft drop shadow: render shadow text onto its own layer, blur, then
        # composite under the main text. Much more cinematic than the previous
        # single-pixel-offset hard shadow.
        shadow_layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        sh_draw = ImageDraw.Draw(shadow_layer)
        offset = max(2, ss_size // 22)
        y = y_start
        for ln, bb, lh in zip(lines, line_bboxes, line_heights):
            text_w = bb[2] - bb[0]
            x = (W - text_w) / 2 - bb[0] + x_shift
            sh_draw.text((x + offset, y + offset), ln, font=font, fill=shadow_rgba)
            y += lh + line_gap
        blur_radius = max(2, ss_size // 24)
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur_radius))
        img = Image.alpha_composite(img, shadow_layer)
        draw = ImageDraw.Draw(img)

        y = y_start
        for ln, bb, lh in zip(lines, line_bboxes, line_heights):
            text_w = bb[2] - bb[0]
            x = (W - text_w) / 2 - bb[0] + x_shift
            draw.text((x, y), ln, font=font, fill=text_rgba)
            y += lh + line_gap

    if s != 1:
        img = img.resize((_W, _H), Image.LANCZOS)
    return img


def create_text_overlay(
    text: str,
    filename: str,
    font_path: str = None,
    font_size: int = 120,
    color: str = "#FFFFFF",
    shadow_color: str = "#000000",
    y_position: int = 800,
    bg_color: str = None,
    bg_opacity: int = 160,
    outline: bool = False,
    auto_scale: bool = False,
    text_opacity: int = 255,
    x_offset: int = 0,
) -> str:
    """
    Generates a 1920x1080 transparent PNG with centered, wrapped text.

    ``x_offset`` shifts the text (and its background box) horizontally from
    centre by N 1080p pixels. y_position is the absolute Y centre; any
    vertical fine-tuning the caller wants should be folded into y_position
    before the call (the renderer doesn't need a separate y_offset
    parameter).
    """
    img = _render_overlay_rgba(
        text=text,
        font_path=font_path,
        font_size=font_size,
        color=color,
        shadow_color=shadow_color,
        y_position=y_position,
        bg_color=bg_color,
        bg_opacity=bg_opacity,
        outline=outline,
        auto_scale=auto_scale,
        text_opacity=text_opacity,
        x_offset=x_offset,
    )

    dir_part = os.path.dirname(filename)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    # optimize=True keeps the file small; compress_level=9 is the slowest/best
    # zlib setting — PNG is lossless so this never affects what Premiere sees.
    img.save(filename, format="PNG", optimize=True, compress_level=9)
    return filename


def extract_frame_from_video(video_path: str, timestamp_sec: float = 1.0) -> Image.Image:
    """
    Pull a single frame from a video file as a PIL Image (RGB, 1920x1080,
    letterboxed). Used as the background for the live overlay preview.

    Falls back to a neutral gray gradient if ffmpeg can't read the file.
    """
    if not video_path or not os.path.exists(video_path):
        return _fallback_background()

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_png = tf.name
        try:
            cmd = [
                "ffmpeg", "-y", "-ss", f"{timestamp_sec:.2f}", "-i", video_path,
                "-vframes", "1",
                "-vf", (
                    f"scale={_W}:{_H}:force_original_aspect_ratio=decrease,"
                    f"pad={_W}:{_H}:(ow-iw)/2:(oh-ih)/2"
                ),
                tmp_png,
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=15)
            if result.returncode != 0 or not os.path.exists(tmp_png) or os.path.getsize(tmp_png) == 0:
                return _fallback_background()
            return Image.open(tmp_png).convert("RGB")
        finally:
            try:
                os.remove(tmp_png)
            except OSError:
                pass
    except Exception:
        return _fallback_background()


def _fallback_background() -> Image.Image:
    """Neutral dark gradient — works when no clip is available for the preview."""
    bg = Image.new("RGB", (_W, _H), (40, 40, 48))
    top = (24, 24, 32)
    bot = (64, 64, 80)
    for y in range(_H):
        t = y / (_H - 1)
        r = int(top[0] * (1 - t) + bot[0] * t)
        g = int(top[1] * (1 - t) + bot[1] * t)
        b = int(top[2] * (1 - t) + bot[2] * t)
        bg.paste((r, g, b), (0, y, _W, y + 1))
    return bg


def render_overlay_preview(
    text: str,
    background: Image.Image = None,
    video_path: str = None,
    video_timestamp_sec: float = 1.0,
    **overlay_kwargs,
) -> Image.Image:
    """
    Render the caption on a 1920x1080 canvas composited over a real-looking
    background, so the user can judge size/legibility before generating all
    the PNGs.

    Pass either a PIL ``background`` image or a ``video_path`` (frame will be
    extracted at ``video_timestamp_sec``). Remaining kwargs are forwarded to
    the overlay renderer.
    """
    if background is None:
        background = extract_frame_from_video(video_path, video_timestamp_sec) if video_path else _fallback_background()

    if background.size != (_W, _H):
        background = background.resize((_W, _H), Image.LANCZOS)
    if background.mode != "RGBA":
        background = background.convert("RGBA")

    overlay = _render_overlay_rgba(text=text, **overlay_kwargs)
    return Image.alpha_composite(background, overlay).convert("RGB")

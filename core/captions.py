import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
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
    # Walk through _FONT_MAP fallbacks
    for path in _FONT_MAP.values():
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

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
) -> str:
    """
    Generates a 1920x1080 transparent PNG with centered, wrapped text.

    New params vs. original:
      bg_color    — hex colour for a rounded-rectangle background box (None = off)
      bg_opacity  — 0-255 alpha for the background box
      outline     — True: 8-direction stroke around text instead of single shadow
      auto_scale  — True: shrink font_size until wrapped block fits in 60% canvas height
      text_opacity — 0-255 alpha applied to the rendered text
    """
    img = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    font = _load_font(font_path, font_size)

    # Auto-scale: shrink until total block height ≤ 60% of canvas (648 px)
    if auto_scale:
        max_block_h = int(1080 * 0.60)
        for _ in range(20):
            max_chars = max(10, int(1800 / max(1, font_size * 0.6)))
            lines = textwrap.wrap(text, width=max_chars) or [text]
            bboxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
            heights = [bb[3] - bb[1] for bb in bboxes]
            gap = max(4, font_size // 8)
            total_h = sum(heights) + gap * (len(lines) - 1)
            if total_h <= max_block_h or font_size <= 30:
                break
            font_size = max(30, int(font_size * 0.85))
            font = _load_font(font_path, font_size)

    # Word-wrap
    max_chars = max(10, int(1800 / max(1, font_size * 0.6)))
    lines = textwrap.wrap(text, width=max_chars) or [text]

    # Measure block geometry
    line_bboxes  = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_widths  = [bb[2] - bb[0] for bb in line_bboxes]
    line_gap     = max(4, font_size // 8)
    total_h      = sum(line_heights) + line_gap * (len(lines) - 1)
    max_w        = max(line_widths) if line_widths else 0
    y_start      = y_position - total_h // 2
    offset       = max(2, font_size // 20)

    # Background box (rounded rectangle on a separate RGBA layer)
    if bg_color:
        pad_x = font_size // 3
        pad_y = font_size // 5
        box_x1 = (1920 - max_w) // 2 - pad_x
        box_y1 = y_start - pad_y
        box_x2 = (1920 + max_w) // 2 + pad_x
        box_y2 = y_start + total_h + pad_y
        radius = max(8, font_size // 6)
        box_layer = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
        ImageDraw.Draw(box_layer).rounded_rectangle(
            [box_x1, box_y1, box_x2, box_y2],
            radius=radius,
            fill=_hex_to_rgba(bg_color, bg_opacity),
        )
        img = Image.alpha_composite(img, box_layer)
        draw = ImageDraw.Draw(img)

    text_rgba   = _hex_to_rgba(color, text_opacity)
    shadow_rgba = _hex_to_rgba(shadow_color, min(text_opacity, 200))

    y = y_start
    for ln, bb, lh in zip(lines, line_bboxes, line_heights):
        text_w = bb[2] - bb[0]
        x = (1920 - text_w) / 2 - bb[0]

        if outline:
            # 8-direction stroke for crisp outline on any background
            for dx in (-offset, 0, offset):
                for dy in (-offset, 0, offset):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), ln, font=font, fill=shadow_rgba)
        elif shadow_color:
            draw.text((x + offset, y + offset), ln, font=font, fill=shadow_rgba)

        draw.text((x, y), ln, font=font, fill=text_rgba)
        y += lh + line_gap

    dir_part = os.path.dirname(filename)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    img.save(filename)
    return filename

import os
import json
from PIL import Image, ImageDraw, ImageFont
from groq import Groq
from core.keywords import _call_groq_json, _call_openrouter_json

def load_highlights_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'highlights.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_highlights(script_text: str, groq_key: str = None) -> list:
    """
    Calls LLM to extract high-impact text highlights from the script.
    """
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

def create_text_overlay(text: str, filename: str, font_path: str = None, 
                        font_size: int = 120, color: str = "#FFFFFF", 
                        shadow_color: str = "#000000", y_position: int = 800):
    """
    Generates a 1920x1080 transparent PNG with centered text.
    """
    # Create transparent RGBA image
    img = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Load Font
    try:
        if not font_path or not os.path.exists(font_path):
            # Windows defaults
            possible_fonts = [
                "C:\\Windows\\Fonts\\arialbd.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\segoeuib.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" # Linux fallback
            ]
            for f in possible_fonts:
                if os.path.exists(f):
                    font_path = f
                    break
        
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"Font loading failed, using default: {e}")
        font = ImageFont.load_default()

    # Word-wrap: split into lines so text never overflows 1920px wide canvas
    import textwrap
    max_chars = max(10, int(1800 / max(1, font_size * 0.6)))
    lines = textwrap.wrap(text, width=max_chars) or [text]

    # Measure total block height so we can centre vertically around y_position
    line_bboxes = [draw.textbbox((0, 0), ln, font=font) for ln in lines]
    line_heights = [bb[3] - bb[1] for bb in line_bboxes]
    line_gap = max(4, font_size // 8)
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)

    y = y_position - total_h // 2
    offset = max(2, font_size // 20)

    for ln, bb, lh in zip(lines, line_bboxes, line_heights):
        text_w = bb[2] - bb[0]
        x = (1920 - text_w) / 2 - bb[0]  # bb[0] corrects for left-side bearing

        # Draw Shadow (offset)
        if shadow_color:
            draw.text((x + offset, y + offset), ln, font=font, fill=shadow_color)

        # Draw Main Text
        draw.text((x, y), ln, font=font, fill=color)
        y += lh + line_gap
    
    # Save
    dir_part = os.path.dirname(filename)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)
    img.save(filename)
    return filename

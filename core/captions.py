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

    # Calculate text placement
    # textbbox is the modern Pillow method (v9.2.0+)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x = (1920 - text_w) / 2
    y = y_position
    
    # Draw Shadow (offset)
    if shadow_color:
        offset = max(2, font_size // 20)
        draw.text((x + offset, y + offset), text, font=font, fill=shadow_color)
    
    # Draw Main Text
    draw.text((x, y), text, font=font, fill=color)
    
    # Save
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    img.save(filename)
    return filename

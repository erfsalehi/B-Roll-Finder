import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def test_ranker():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    with open("prompts/director_rank.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().replace("{custom_instructions_block}", "OVERALL VIDEO TOPIC: Mechanical repairs\n")

    # Data from the user's screenshot
    shot_text = "and pull a slender steel blade from its sheath."
    shot_intent = "show the grandfather's tool"
    
    candidates = [
        "0. [PIXABAY] Cpu, Cpu Socket, Putting A Cpu Into A Socket, Hand, Technology | 3840x2160 (horizontal) | query: \"hand pulling out a screwdriver\"",
        "1. [PIXABAY] Smoke, Ascending, Haze, Blue, Step Out | 1920x1080 (horizontal) | query: \"hand pulling out a screwdriver\"",
        "2. [PIXABAY] Bud, Leaf, Sprout, Rise Up, Time Lapse, Leaf Bud, Eye, Set Out, Branch, Sprout | 3840x2160 (horizontal) | query: \"hand pulling out a screwdriver\"",
        "3. [PIXABAY] Cow, Ruminant, Pasture, Meadow, Nature, Animals On The Farm, Dairy Cows | 3840x2160 (horizontal) | query: \"hand pulling out a screwdriver\"",
        "4. [PEXELS] A Person Using A Screwdriver | 2160x4096 (vertical) | query: \"hand pulling out a screwdriver\""
    ]

    user_msg = (
        f"NARRATION: \"{shot_text}\"\n"
        f"SHOT INTENT: {shot_intent}\n\n"
        f"CANDIDATES:\n" + "\n".join(candidates)
    )

    print("--- Sending to LLM ---")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )
    
    print(response.choices[0].message.content)

test_ranker()

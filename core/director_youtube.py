from groq import Groq

from core.keywords import _call_llm_json


def seed_youtube_keywords(shots: list, max_keywords: int = 2) -> list:
    """Populate missing youtube_keywords from existing director queries."""
    for shot in shots:
        if shot.get("priority") == "none":
            shot["youtube_keywords"] = []
            continue
        existing = [str(kw).strip() for kw in shot.get("youtube_keywords", []) if str(kw).strip()]
        if existing:
            shot["youtube_keywords"] = existing[:max_keywords]
            continue
        queries = [str(q).strip() for q in shot.get("search_queries", []) if str(q).strip()]
        shot["youtube_keywords"] = queries[:max_keywords]
    return shots


def generate_youtube_keywords_for_shots(
    shots: list,
    api_key: str,
    video_topic: str = "",
    custom_instructions: str = "",
    max_keywords: int = 2,
    progress_callback=None,
) -> list:
    """Generate YouTube-oriented search phrases for Director shots."""
    if not api_key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=api_key)
    system_prompt = f"""
You generate concise YouTube search keywords for B-roll discovery.
Return valid JSON only: {{"shots": [{{"slot_id": 1, "youtube_keywords": ["...", "..."]}}]}}.
For each shot, write exactly {max_keywords} plain YouTube search phrases.
Prefer concrete subject + action + context. Avoid cinematic jargon, camera words, quotes, hashtags, and punctuation.
Use the overall topic to disambiguate vague script words.
""".strip()
    if video_topic.strip():
        system_prompt += f"\nOVERALL VIDEO TOPIC: {video_topic.strip()}"
    if custom_instructions.strip():
        system_prompt += f"\nUSER STYLE NOTES: {custom_instructions.strip()}"

    rankable = [s for s in shots if s.get("priority") != "none"]
    if not rankable:
        if progress_callback:
            progress_callback(1.0)
        return shots

    by_id = {s.get("slot_id"): s for s in shots}
    batch_size = 12
    for start in range(0, len(rankable), batch_size):
        batch = rankable[start:start + batch_size]
        user_lines = []
        for shot in batch:
            user_lines.append(
                f"slot_id: {shot.get('slot_id')}\n"
                f"narration: {shot.get('text', '')}\n"
                f"intent: {shot.get('shot_intent', '')}\n"
                f"stock_queries: {' | '.join(shot.get('search_queries', []))}"
            )
        data = _call_llm_json(
            client,
            system_prompt,
            "\n\n".join(user_lines),
            temperature=0.25,
            max_tokens=1600,
        )
        for row in data.get("shots", []):
            slot_id = row.get("slot_id")
            shot = by_id.get(slot_id)
            if not shot:
                continue
            keywords = row.get("youtube_keywords", [])
            clean = [str(kw).strip() for kw in keywords if str(kw).strip()]
            shot["youtube_keywords"] = clean[:max_keywords]
        if progress_callback:
            progress_callback(min(1.0, (start + len(batch)) / len(rankable)))

    return seed_youtube_keywords(shots, max_keywords=max_keywords)

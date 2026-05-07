import os
import json
from groq import Groq
from core.keywords import _call_groq_json


def _load_rank_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'director_rank.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def rank_shot_candidates(shots: list, api_key: str, custom_instructions: str = "",
                         progress_callback=None) -> list:
    """
    Stage 3: For each shot that has video_results, asks the LLM to rank candidates
    by relevance to the shot_intent. Reorders shot['video_results'] in place.
    Adds shot['rank_reason'] with the top pick's reasoning.
    """
    if not api_key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=api_key)

    system_prompt = _load_rank_prompt()
    custom_block = ""
    if custom_instructions and custom_instructions.strip():
        custom_block = f"USER STYLE NOTES: {custom_instructions.strip()}"
    system_prompt = system_prompt.replace("{custom_instructions_block}", custom_block)

    rankable = [s for s in shots if s.get('video_results') and s.get('priority') != 'none']
    total = len(rankable)
    done = 0

    for shot in rankable:
        candidates = shot['video_results']
        if len(candidates) <= 1:
            shot['rank_reason'] = ""
            done += 1
            if progress_callback:
                progress_callback(done / total)
            continue

        # Build candidate list for the LLM
        candidate_lines = []
        for i, c in enumerate(candidates):
            candidate_lines.append(
                f"{i}. [{c.get('source','?').upper()}] {c.get('title','?')} — {c.get('description','')}"
            )

        user_msg = (
            f"NARRATION: \"{shot.get('text', '')}\"\n"
            f"SHOT INTENT: {shot.get('shot_intent', '')}\n"
            f"SHOT TYPE: {shot.get('shot_type', '')}\n\n"
            f"CANDIDATES:\n" + "\n".join(candidate_lines)
        )

        try:
            data = _call_groq_json(client, system_prompt, user_msg)
            ranked = data.get('ranked', [])

            if ranked:
                # Reorder candidates according to LLM ranking
                index_order = [r['index'] for r in ranked if isinstance(r.get('index'), int)]
                # Include any missing indices at the end (safety net)
                all_indices = list(range(len(candidates)))
                for i in all_indices:
                    if i not in index_order:
                        index_order.append(i)

                shot['video_results'] = [candidates[i] for i in index_order if i < len(candidates)]
                shot['rank_reason'] = ranked[0].get('reason', '') if ranked else ''
        except Exception as e:
            print(f"Ranking failed for shot {shot.get('slot_id')}: {e}")
            shot['rank_reason'] = ''

        done += 1
        if progress_callback:
            progress_callback(done / total)

    return shots

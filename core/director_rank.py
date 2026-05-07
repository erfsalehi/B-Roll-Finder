import os
from groq import Groq
from core.keywords import _call_groq_json


def _load_rank_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'director_rank.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def rank_shot_candidates(shots: list, api_key: str, custom_instructions: str = "",
                         video_topic: str = "", progress_callback=None) -> list:
    """
    Stage 3: Ranks and filters each shot's video_results by relevance.
    - Reorders shot['video_results'] best → worst
    - Sets shot['rank_reason'] with the top pick's one-line reason
    - Marks irrelevant candidates with candidate['irrelevant'] = True
    """
    if not api_key:
        raise ValueError("Groq API key is missing.")

    client = Groq(api_key=api_key)

    system_prompt = _load_rank_prompt()
    custom_block = ""
    if video_topic and video_topic.strip():
        custom_block += f"OVERALL VIDEO TOPIC: {video_topic.strip()}\n"
    if custom_instructions and custom_instructions.strip():
        custom_block += f"USER STYLE NOTES: {custom_instructions.strip()}"
    system_prompt = system_prompt.replace("{custom_instructions_block}", custom_block)

    rankable = [s for s in shots if s.get('video_results') and s.get('priority') != 'none']
    total = len(rankable)
    done = 0

    for shot in rankable:
        candidates = shot['video_results']
        if len(candidates) <= 1:
            shot.setdefault('rank_reason', '')
            done += 1
            if progress_callback:
                progress_callback(done / total)
            continue

        candidate_lines = [
            f"{i}. [{c.get('source','?').upper()}] {c.get('title','?')} — {c.get('description','')}"
            for i, c in enumerate(candidates)
        ]

        user_msg = (
            f"NARRATION: \"{shot.get('text', '')}\"\n"
            f"SHOT INTENT: {shot.get('shot_intent', '')}\n\n"
            f"CANDIDATES:\n" + "\n".join(candidate_lines)
        )

        try:
            data = _call_groq_json(client, system_prompt, user_msg)
            ranked = data.get('ranked', [])

            if ranked:
                # Apply irrelevant flag to original candidates before reordering
                for r in ranked:
                    idx = r.get('index')
                    if isinstance(idx, int) and idx < len(candidates):
                        if r.get('irrelevant'):
                            candidates[idx]['irrelevant'] = True
                        else:
                            candidates[idx].pop('irrelevant', None)

                index_order = [r['index'] for r in ranked if isinstance(r.get('index'), int)]
                # Append any indices the LLM omitted (safety net)
                for i in range(len(candidates)):
                    if i not in index_order:
                        index_order.append(i)

                shot['video_results'] = [candidates[i] for i in index_order if i < len(candidates)]
                shot['rank_reason'] = ranked[0].get('reason', '')
        except Exception as e:
            print(f"Ranking failed for shot {shot.get('slot_id')}: {e}")
            shot.setdefault('rank_reason', '')

        done += 1
        if progress_callback:
            progress_callback(done / total)

    return shots

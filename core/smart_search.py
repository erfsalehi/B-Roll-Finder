from sentence_transformers import SentenceTransformer
from core.vector_db import VectorDB

class SmartSearch:
    def __init__(self):
        self.db = VectorDB()
        self.model = None

    def _load_model(self):
        """Loads the multi-modal CLIP model."""
        if self.model is None:
            # We use the same model as the indexer to ensure vector space compatibility
            self.model = SentenceTransformer('clip-ViT-B-32')

    def search(self, query_text, k=10, min_score=0.2):
        """
        Searches the library for segments matching the query text.
        Returns a list of matching segment metadata.
        """
        if self.db.get_total_count() == 0:
            return []
            
        self._load_model()
        
        # Encode text query into the CLIP vector space
        query_emb = self.model.encode(query_text)
        
        # Search the FAISS index
        raw_results = self.db.search(query_emb, k=k)
        
        # Filter by minimum similarity score
        filtered = [r for r in raw_results if r.get('score', 0) >= min_score]
        
        # Sort by score descending
        filtered.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return filtered

    def generate_match_reasons_batched(self, query_text, hits_metadata, api_key):
        """Sends all hits to Groq in a single request to avoid rate limits."""
        if not api_key or not hits_metadata:
            return ["Matches narration intent." for _ in hits_metadata]
            
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            
            hit_list = "\n".join([f"{i+1}. {h.get('video_title')}" for i, h in enumerate(hits_metadata)])
            
            prompt = f"""
            Narration: "{query_text}"
            
            Video Clips:
            {hit_list}
            
            Explain why each clip matches the narration visually.
            Return a JSON object with a key "reasons" containing an array of {len(hits_metadata)} strings.
            Format for each string: "Matches [topic] with [visual description]."
            """
            
            resp = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            import json
            data = json.loads(resp.choices[0].message.content)
            reasons = data.get("reasons", [])
            
            # Ensure we return the correct number of reasons even if LLM fails
            if len(reasons) < len(hits_metadata):
                reasons.extend(["High semantic relevance."] * (len(hits_metadata) - len(reasons)))
            return reasons[:len(hits_metadata)]
        except Exception:
            return ["High semantic relevance detected." for _ in hits_metadata]

    def get_library_stats(self):
        """Returns statistics about the indexed library."""
        total_segments = self.db.get_total_count()
        unique_videos = len(set(m.get('video_path') for m in self.db.metadata))
        
        return {
            "total_segments": total_segments,
            "unique_videos": unique_videos
        }

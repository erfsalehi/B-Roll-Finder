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

    def get_library_stats(self):
        """Returns statistics about the indexed library."""
        total_segments = self.db.get_total_count()
        unique_videos = len(set(m.get('video_path') for m in self.db.metadata))
        
        return {
            "total_segments": total_segments,
            "unique_videos": unique_videos
        }

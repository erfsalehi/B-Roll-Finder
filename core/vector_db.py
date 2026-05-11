import os
import json
import faiss
import numpy as np

class VectorDB:
    def __init__(self, index_path=".cache/index/faiss.index", meta_path=".cache/index/metadata.json"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dimension = 512 # Standard for CLIP ViT-B/32
        self.index = None
        self.metadata = []
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self.load()

    def load(self):
        """Loads the FAISS index and metadata from disk."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception:
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            # IndexFlatIP + normalize_L2 = Cosine Similarity
            self.index = faiss.IndexFlatIP(self.dimension)
            
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.metadata = json.load(f)

    def save(self):
        """Persists the index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_vectors(self, vectors, meta_list):
        """
        Adds vectors and their corresponding metadata to the database.
        vectors: np.array of shape (N, dimension)
        meta_list: list of dictionaries of length N
        """
        if len(vectors) == 0:
            return
            
        vectors = np.array(vectors).astype('float32')
        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)
        
        self.index.add(vectors)
        self.metadata.extend(meta_list)
        self.save()

    def search(self, query_vector, k=10):
        """
        Searches for the top K most similar vectors.
        Returns a list of metadata dictionaries with an added 'score' field.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query_vector = np.array([query_vector]).astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta['score'] = float(distances[0][i])
            results.append(meta)
            
        # Deduplicate results that point to the same file and similar timestamps if needed,
        # but for now we'll return raw hits.
        return results

    def clear(self):
        """Resets the entire database."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.save()

    def get_total_count(self):
        return len(self.metadata)

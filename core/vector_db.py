import os
import json
import faiss
import numpy as np

# Global Smart Library lives in the user's home directory so it persists
# across all projects and survives .cache/ wipes.
SMART_LIBRARY_DIR = os.path.join(os.path.expanduser("~"), ".broll_director")


class VectorDB:
    def __init__(self, permanent=True, index_path=None, meta_path=None):
        """
        permanent=True  → persistent Smart Library in ~/.broll_director/
        permanent=False → in-memory only (temporary proxy session index)
        """
        self.permanent = permanent
        self.dimension = 512  # CLIP ViT-B/32

        if permanent:
            os.makedirs(SMART_LIBRARY_DIR, exist_ok=True)
            self.index_path = index_path or os.path.join(SMART_LIBRARY_DIR, "smart_library.index")
            self.meta_path  = meta_path  or os.path.join(SMART_LIBRARY_DIR, "metadata.json")
        else:
            # No disk paths — everything stays in RAM
            self.index_path = None
            self.meta_path  = None

        self.index    = None
        self.metadata = []
        self.load()

    def load(self):
        """Loads the FAISS index and metadata from disk (permanent mode only)."""
        if self.index_path and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception:
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        if self.meta_path and os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)

    def save(self):
        """Persists the index and metadata to disk (permanent mode only)."""
        if not self.permanent:
            return  # Temporary index — never write to disk
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def add_vectors(self, vectors, meta_list):
        """
        Adds vectors and their corresponding metadata to the database.
        vectors: list of np.array of shape (dimension,)
        meta_list: list of dicts of the same length
        """
        if not vectors:
            return

        vecs = np.array(vectors, dtype="float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.metadata.extend(meta_list)
        self.save()

    def search(self, query_vector, k=10):
        """
        Searches for the top K most similar vectors.
        Returns metadata dicts with an added 'score' field.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(query_vector)

        actual_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, actual_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta["score"] = float(distances[0][i])
            results.append(meta)

        return results

    def clear(self):
        """Resets the entire database."""
        self.index    = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.save()

    def get_total_count(self):
        return len(self.metadata)


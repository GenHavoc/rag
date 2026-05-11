import faiss
import numpy as np
from typing import List, Dict

class FaissStore:
    """
    FAISS vector store for normalized embeddings (cosine similarity).
    """

    def __init__(self, dim: int):
        # Inner Product index = cosine similarity for normalized vectors
        self.index = faiss.IndexFlatIP(dim)

        self.texts: List[str] = []
        self.metadata: List[Dict] = []

    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: List[Dict]
    ):
        assert len(embeddings) == len(texts) == len(metadata), \
            "Embeddings, texts, and metadata must have same length"

        embeddings = np.asarray(embeddings).astype("float32")

        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray([query_embedding]).astype("float32")

        scores, idxs = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            results.append({
                "text": self.texts[idx],
                "meta": self.metadata[idx],
                "score": float(score)
            })

        return results

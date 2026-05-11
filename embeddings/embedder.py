from sentence_transformers import SentenceTransformer
from typing import List
from typing import Optional
import numpy as np

class Embedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model = SentenceTransformer(model_name, device=device)

        self.model = SentenceTransformer(
            model_name,
            device=device
        )

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Converts text chunks into normalized embeddings.

        Args:
            texts: List of text chunks
            batch_size: Batch size for encoding

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )

"""
embedder.py
Loads a SentenceTransformer model and runs embedding inference.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts)
        return np.asarray(vectors, dtype=float).tolist()

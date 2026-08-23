"""
brain.embedder — Sentence embedding model wrapper.

Uses all-MiniLM-L6-v2 (384-dim) for semantic similarity.
Fully local, no API calls.
"""

from typing import List, Optional


class TextEmbedder:
    """
    Local sentence embedding model.

    Usage:
        embedder = TextEmbedder()
        vec = embedder.embed("fix the VAD threshold")
        # → [0.23, -0.11, ...]  (384 floats)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            model_name: HuggingFace model name for sentence-transformers.
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self._model_name = model_name

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Input text.

        Returns:
            384-dimensional embedding as a list of floats.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of input strings.

        Returns:
            List of 384-dimensional embeddings.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings.tolist()

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Cosine similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score between -1.0 and 1.0 (higher = more similar).
        """
        emb_a = self.model.encode(text_a, normalize_embeddings=True)
        emb_b = self.model.encode(text_b, normalize_embeddings=True)
        return float(emb_a @ emb_b)

    def find_most_similar(
        self, query: str, candidates: List[str], top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar candidates to a query.

        Args:
            query: Query text.
            candidates: List of candidate texts.
            top_k: Number of results to return.

        Returns:
            List of (index, score, text) tuples, sorted by score descending.
        """
        query_emb = self.model.encode(query, normalize_embeddings=True)
        candidate_embs = self.model.encode(candidates, normalize_embeddings=True)
        scores = candidate_embs @ query_emb

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [(idx, float(score), candidates[idx]) for idx, score in ranked]

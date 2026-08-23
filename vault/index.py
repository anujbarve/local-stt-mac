"""
Vault Index
===========

FAISS-based vector index for semantic search.  Falls back to brute-force
cosine similarity when ``faiss`` / ``numpy`` are not installed.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from vault.config import INDEX_DIR, EMBEDDING_DIM

try:
    import faiss
    import numpy as np

    HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    HAS_FAISS = False


class VaultIndex:
    """FAISS-based vector index for semantic search."""

    def __init__(
        self,
        index_dir: str = INDEX_DIR,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.index_dir = index_dir
        self.embedding_dim = embedding_dim

        # FAISS index object (None when using brute-force fallback)
        self._index: Optional[Any] = None

        # Positional ID mapping: list of dicts
        # [{note_id, file_path, embedding (brute-force only), removed}, ...]
        self._id_map: List[Dict[str, Any]] = []

        # Brute-force storage when FAISS unavailable
        self._embeddings: List[List[float]] = []

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, reader: Any, embedder: Any = None) -> int:
        """Scan vault, embed all notes, build index.

        Returns number of notes indexed.
        """
        self.clear()
        notes = reader.read_all()
        count = 0
        for note in notes:
            embedding = self._embed_note(note, embedder)
            if embedding is not None:
                self.add(
                    note_id=note.get("id", ""),
                    file_path=note.get("file_path", ""),
                    embedding=embedding,
                )
                count += 1
        self.save()
        return count

    def rebuild(self, reader: Any, embedder: Any = None) -> int:
        """Clear and rebuild entire index."""
        self.clear()
        return self.build(reader, embedder)

    def add(self, note_id: str, file_path: str, embedding: List[float]) -> None:
        """Add a single note to the index."""
        vec = list(embedding)
        entry = {"note_id": note_id, "file_path": file_path, "removed": False}

        if HAS_FAISS and self._index is not None:
            self._index.add(np.array([vec], dtype=np.float32))
            self._id_map.append(entry)
        else:
            entry["embedding"] = vec
            self._id_map.append(entry)
            self._embeddings.append(vec)

    def add_batch(self, entries: List[Tuple[str, str, List[float]]]) -> int:
        """Add multiple entries at once (faster than individual adds).

        Args:
            entries: List of (note_id, file_path, embedding) tuples.

        Returns number of entries added.
        """
        if not entries:
            return 0

        if HAS_FAISS and self._index is not None:
            vectors = np.array([e[2] for e in entries], dtype=np.float32)
            self._index.add(vectors)
            for note_id, file_path, _ in entries:
                self._id_map.append({"note_id": note_id, "file_path": file_path, "removed": False})
        else:
            for note_id, file_path, embedding in entries:
                vec = list(embedding)
                self._id_map.append({"note_id": note_id, "file_path": file_path, "removed": False, "embedding": vec})
                self._embeddings.append(vec)

        return len(entries)

    def remove(self, note_id: str) -> bool:
        """Mark a note as removed.  Returns True if found."""
        for entry in self._id_map:
            if entry["note_id"] == note_id and not entry["removed"]:
                entry["removed"] = True
                return True
        return False

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find most similar notes.

        Returns ``[{note_id, file_path, score, rank}, ...]`` sorted by
        descending score.
        """
        if not self._id_map:
            return []

        if HAS_FAISS and self._index is not None and self._index.ntotal > 0:
            return self._search_faiss(query_embedding, top_k)
        return self._search_bruteforce(query_embedding, top_k)

    def search_by_text(self, text: str, embedder: Any, top_k: int = 10) -> List[Dict[str, Any]]:
        """Convenience: embed text then search."""
        embedding = self._embed_text(text, embedder)
        if embedding is None:
            return []
        return self.search(embedding, top_k=top_k)

    def save(self) -> None:
        """Save index + ID mapping to disk."""
        os.makedirs(self.index_dir, exist_ok=True)

        map_path = os.path.join(self.index_dir, "notes_map.json")
        # Don't persist embedding data in map when using FAISS
        serialisable = []
        for entry in self._id_map:
            e = {k: v for k, v in entry.items() if k != "embedding"}
            serialisable.append(e)
        with open(map_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)

        if HAS_FAISS and self._index is not None:
            idx_path = os.path.join(self.index_dir, "notes.index")
            faiss.write_index(self._index, idx_path)
        elif self._embeddings:
            emb_path = os.path.join(self.index_dir, "notes_embeddings.json")
            with open(emb_path, "w", encoding="utf-8") as fh:
                json.dump(self._embeddings, fh)

    def load(self) -> bool:
        """Load index from disk.  Returns True if successful."""
        return self._load()

    def clear(self) -> None:
        """Delete all data from index."""
        self._index = None
        self._id_map = []
        self._embeddings = []
        if HAS_FAISS:
            self._index = faiss.IndexFlatIP(self.embedding_dim)

    @property
    def size(self) -> int:
        """Number of active (non-removed) entries."""
        return sum(1 for e in self._id_map if not e.get("removed"))

    def stats(self) -> Dict[str, Any]:
        """Index statistics."""
        return {
            "total_entries": len(self._id_map),
            "active_entries": self.size,
            "removed_entries": len(self._id_map) - self.size,
            "embedding_dim": self.embedding_dim,
            "has_faiss": HAS_FAISS,
        }

    # ------------------------------------------------------------------
    # Internal – FAISS search
    # ------------------------------------------------------------------

    def _search_faiss(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        vec = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(vec, k)

        results: List[Dict[str, Any]] = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._id_map):
                continue
            entry = self._id_map[idx]
            if entry.get("removed"):
                continue
            results.append({
                "note_id": entry["note_id"],
                "file_path": entry["file_path"],
                "score": float(dist),
                "rank": rank,
            })
        return results

    # ------------------------------------------------------------------
    # Internal – brute-force fallback
    # ------------------------------------------------------------------

    def _search_bruteforce(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        scored: List[Tuple[int, float]] = []
        for i, entry in enumerate(self._id_map):
            if entry.get("removed"):
                continue
            emb = entry.get("embedding") or (self._embeddings[i] if i < len(self._embeddings) else None)
            if emb is None:
                continue
            score = _cosine_similarity(query_embedding, emb)
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(scored[:top_k], start=1):
            entry = self._id_map[idx]
            results.append({
                "note_id": entry["note_id"],
                "file_path": entry["file_path"],
                "score": score,
                "rank": rank,
            })
        return results

    # ------------------------------------------------------------------
    # Internal – embedding helpers
    # ------------------------------------------------------------------

    def _embed_note(self, note: Dict[str, Any], embedder: Any) -> Optional[List[float]]:
        """Embed a note's text content."""
        text = note.get("body", "") or note.get("summary", "") or note.get("text", "")
        if not text.strip():
            return None
        return self._embed_text(text, embedder)

    def _embed_text(self, text: str, embedder: Any) -> Optional[List[float]]:
        """Embed text using the provided embedder.

        The embedder is expected to have an ``encode(text)`` method that
        returns a list of floats, or an ``encode`` that returns a list of
        lists when given a list of strings.
        """
        if embedder is None:
            return None
        try:
            result = embedder.encode([text])
            if hasattr(result, "tolist"):
                return result.tolist()[0]
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], list):
                    return result[0]
                return result
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Internal – load / init
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        """Try to load existing index from disk."""
        map_path = os.path.join(self.index_dir, "notes_map.json")
        if not os.path.exists(map_path):
            self.clear()
            return False

        with open(map_path, "r", encoding="utf-8") as fh:
            self._id_map = json.load(fh)

        if HAS_FAISS:
            idx_path = os.path.join(self.index_dir, "notes.index")
            if os.path.exists(idx_path):
                self._index = faiss.read_index(idx_path)
                return True
            else:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                return True
        else:
            emb_path = os.path.join(self.index_dir, "notes_embeddings.json")
            if os.path.exists(emb_path):
                with open(emb_path, "r", encoding="utf-8") as fh:
                    self._embeddings = json.load(fh)
                return True
            self._embeddings = []
            return True


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors (pure Python fallback)."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

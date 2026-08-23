"""
brain.entity_linker — Entity resolution across notes.

Links extracted entities to existing entity pages in the vault
using embedding similarity. Uses a simple in-memory index
(FAISS is optional for large-scale use).
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from brain_contract import Entity, EntityType


class EntityLinker:
    """
    Resolve extracted entities to known vault entities.

    Maintains an in-memory index of known entities with their embeddings.
    Uses cosine similarity for matching.

    Usage:
        linker = EntityLinker()
        linker.add_entity("Sarah", EntityType.PERSON, embedding)
        match = linker.resolve(Entity(name="Sarah", entity_type=EntityType.PERSON))
        # → "sarah"
    """

    def __init__(self):
        # In-memory store: list of (name, type, embedding)
        self._entities: List[Dict[str, Any]] = []
        self._index_path: Optional[str] = None

    def resolve(
        self,
        entity: Entity,
        threshold: float = 0.85,
    ) -> Optional[str]:
        """
        Given an extracted entity, find the best match in the vault.

        Args:
            entity: The extracted entity to resolve.
            threshold: Minimum similarity to consider a match.

        Returns:
            Matching entity name or None if no match above threshold.
        """
        if not self._entities:
            return None

        # Filter by entity type first for efficiency
        candidates = [
            e for e in self._entities
            if e["type"] == entity.entity_type.value
        ]

        if not candidates:
            # Fall back to exact name match
            for e in self._entities:
                if e["name"].lower() == entity.name.lower():
                    return e["name"]
            return None

        # Exact match first
        for c in candidates:
            if c["name"].lower() == entity.name.lower():
                return c["name"]

        # If we have embeddings, use cosine similarity
        if entity.attributes.get("embedding") is not None:
            query_emb = entity.attributes["embedding"]
            best_match = None
            best_score = -1.0

            for c in candidates:
                if c["embedding"] is None:
                    continue
                score = self._cosine_similarity(query_emb, c["embedding"])
                if score > best_score:
                    best_score = score
                    best_match = c["name"]

            if best_score >= threshold:
                return best_match

        return None

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """
        Add a new entity to the index.

        Args:
            name: Entity name.
            entity_type: Type of entity.
            embedding: Optional 384-dim embedding vector.
        """
        self._entities.append({
            "name": name,
            "type": entity_type.value,
            "embedding": embedding,
        })

    def build_index(self, vault_path: str = "vault/") -> None:
        """
        Scan vault/ and build entity index from existing pages.

        This is a stub — in production, it would read vault entity pages.
        """
        if not os.path.exists(vault_path):
            return

        # Scan for entity pages (markdown files)
        for root, dirs, files in os.walk(vault_path):
            for fname in files:
                if fname.endswith(".md"):
                    # Extract entity name from filename
                    name = fname.replace(".md", "").replace("-", " ").title()
                    # Determine type from directory
                    rel_path = os.path.relpath(root, vault_path)
                    entity_type = self._dir_to_type(rel_path)
                    self.add_entity(name, entity_type)

    def _dir_to_type(self, rel_path: str) -> EntityType:
        """Map vault directory to entity type."""
        mapping = {
            "people": EntityType.PERSON,
            "projects": EntityType.PROJECT,
            "tools": EntityType.TOOL,
            "topics": EntityType.TOPIC,
        }
        top_dir = rel_path.split(os.sep)[0] if os.sep in rel_path else rel_path
        return mapping.get(top_dir, EntityType.CONCEPT)

    def _cosine_similarity(
        self, a: List[float], b: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def save(self, path: str) -> None:
        """Save entity index to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = []
        for e in self._entities:
            entry = {"name": e["name"], "type": e["type"]}
            if e["embedding"] is not None:
                entry["embedding"] = e["embedding"]
            data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._index_path = path

    def load(self, path: str) -> None:
        """Load entity index from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        self._entities = []
        for entry in data:
            self._entities.append({
                "name": entry["name"],
                "type": entry["type"],
                "embedding": entry.get("embedding"),
            })
        self._index_path = path

    @property
    def size(self) -> int:
        """Number of indexed entities."""
        return len(self._entities)

    def list_entities(
        self, entity_type: Optional[EntityType] = None
    ) -> List[str]:
        """List all indexed entity names, optionally filtered by type."""
        if entity_type is None:
            return [e["name"] for e in self._entities]
        return [
            e["name"] for e in self._entities
            if e["type"] == entity_type.value
        ]

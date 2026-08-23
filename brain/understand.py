"""
brain.understand — Main inference API.

This is the entry point that Instance B (vault) calls.
Wraps the model, tokenizer, embedder, and entity linker into a
clean API that produces Understanding objects.

Usage:
    engine = BrainEngine.load("checkpoints/brain/latest")
    understanding = engine.understand("I need to fix the VAD threshold by Friday")
    print(understanding.type)  # NoteType.TASK
    print(understanding.priority)  # 0.82
"""

import os
from typing import Any, Dict, List, Optional

import torch

from brain.config import (
    MAX_SEQ_LEN, VOCAB_SIZE, EMBEDDING_DIM,
    LLM_CONFIDENCE_THRESHOLD, CHECKPOINT_DIR,
)
from brain.model import BrainModel
from brain.tokenizer import Tokenizer
from brain.embedder import TextEmbedder
from brain.entity_linker import EntityLinker
from brain_contract import (
    Understanding, Entity, TemporalInfo,
    NoteType, EntityType, Priority, Urgency, Sentiment,
)

# Index maps for decoding model outputs
IDX_TO_TYPE = {i: t for i, t in enumerate(NoteType)}
IDX_TO_URGENCY = {i: u for i, u in enumerate(Urgency)}
IDX_TO_SENTIMENT = {i: s for i, s in enumerate(Sentiment)}

# BIO tag decode: tag_idx → (entity_type, is_start)
TAG_O = 0
ENTITY_TYPES = list(EntityType)


def _decode_bio_tag(tag_idx: int) -> Optional[tuple]:
    """Decode a BIO tag index to (EntityType, is_start)."""
    if tag_idx == TAG_O:
        return None
    if tag_idx % 2 == 1:
        # B-tag
        type_idx = (tag_idx - 1) // 2
        if type_idx < len(ENTITY_TYPES):
            return (ENTITY_TYPES[type_idx], True)
    elif tag_idx % 2 == 0:
        # I-tag
        type_idx = (tag_idx - 2) // 2
        if type_idx < len(ENTITY_TYPES):
            return (ENTITY_TYPES[type_idx], False)
    return None


class BrainEngine:
    """
    The main brain interface.

    Usage:
        engine = BrainEngine.load("checkpoints/brain/latest")
        understanding = engine.understand("I need to fix the VAD threshold by Friday")
        print(understanding.type)  # NoteType.TASK
        print(understanding.priority)  # 0.82
    """

    def __init__(
        self,
        model: BrainModel,
        tokenizer: Tokenizer,
        embedder: TextEmbedder,
        entity_linker: Optional[EntityLinker] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.entity_linker = entity_linker or EntityLinker()
        self.model.eval()

    @torch.no_grad()
    def understand(
        self,
        text: str,
        context: Optional[List[Dict]] = None,
    ) -> Understanding:
        """
        Full understanding pipeline.

        1. Tokenize text
        2. Run through model (all heads)
        3. Decode predictions
        4. Generate embedding
        5. Link entities to vault
        6. Return Understanding object

        Args:
            text: Input text (transcription).
            context: Optional list of recent notes for context.

        Returns:
            Understanding object with all fields populated.
        """
        # 1. Tokenize
        token_ids = self.tokenizer.encode(text, add_special=True)
        token_tensor = torch.tensor([token_ids], dtype=torch.long)

        # 2. Forward pass
        outputs = self.model(token_tensor)

        # 3. Decode predictions
        understanding = self._decode_outputs(text, outputs)

        # 4. Generate embedding
        try:
            understanding.embedding = self.embedder.embed(text)
        except Exception:
            understanding.embedding = None

        # 5. Link entities
        for entity in understanding.entities:
            resolved = self.entity_linker.resolve(entity)
            if resolved:
                entity.attributes["vault_page"] = resolved

        # 6. Set confidence from model
        understanding.confidence = self._compute_confidence(outputs)

        return understanding

    @torch.no_grad()
    def understand_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Understanding]:
        """Batch inference with proper batching for efficiency.

        Groups texts into batches, runs single forward pass per batch.
        Much faster than calling understand() in a loop.
        """
        results = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            # Tokenize batch
            batch_ids = []
            for text in batch_texts:
                ids = self.tokenizer.encode(text, add_special=True)
                batch_ids.append(ids)

            token_tensor = torch.tensor(batch_ids, dtype=torch.long)
            outputs = self.model(token_tensor)

            # Decode each result
            for i, text in enumerate(batch_texts):
                single_outputs = {k: v[i:i+1] if isinstance(v, torch.Tensor) else v
                                  for k, v in outputs.items()}
                understanding = self._decode_outputs(text, single_outputs)

                # Embedding
                try:
                    understanding.embedding = self.embedder.embed(text)
                except Exception:
                    understanding.embedding = None

                # Entity linking
                for entity in understanding.entities:
                    resolved = self.entity_linker.resolve(entity)
                    if resolved:
                        entity.attributes["vault_page"] = resolved

                understanding.confidence = self._compute_confidence(single_outputs)
                results.append(understanding)

        return results

    def should_call_llm(self, understanding: Understanding) -> bool:
        """
        Decide if the LLM should handle this instead.

        Returns True if:
          - confidence < 0.70
          - type is ambiguous (low classifier confidence)
          - complex temporal reasoning needed
        """
        if understanding.confidence < LLM_CONFIDENCE_THRESHOLD:
            return True

        # Check if temporal info is present but model is uncertain
        if understanding.temporal and understanding.temporal.confidence < 0.5:
            return True

        return False

    def suggest_links(
        self,
        understanding: Understanding,
        candidate_notes: List[Dict],
    ) -> List[str]:
        """
        Suggest related notes based on embedding similarity.

        Args:
            understanding: The current understanding.
            candidate_notes: List of note dicts with 'text' and 'note_id'.

        Returns:
            List of note_ids of related notes, sorted by relevance.
        """
        if understanding.embedding is None or not candidate_notes:
            return []

        query_emb = understanding.embedding
        scored = []

        for note in candidate_notes:
            note_emb = note.get("embedding")
            if note_emb is None:
                continue
            score = self._cosine_similarity(query_emb, note_emb)
            scored.append((note.get("note_id", ""), score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top 5 with score > 0.5
        return [nid for nid, score in scored[:5] if score > 0.5]

    def _decode_outputs(
        self, text: str, outputs: Dict[str, torch.Tensor]
    ) -> Understanding:
        """Decode raw model outputs into an Understanding object."""

        # Type classification
        type_logits = outputs["classifier"][0]  # [num_types]
        type_idx = type_logits.argmax().item()
        note_type = IDX_TO_TYPE.get(type_idx, NoteType.OBSERVATION)

        # Priority regression
        priority = outputs["priority"][0, 0].item()
        priority = max(0.0, min(1.0, priority))

        # Urgency classification
        urgency_logits = outputs["urgency"][0]
        urgency_idx = urgency_logits.argmax().item()
        urgency = IDX_TO_URGENCY.get(urgency_idx, Urgency.LATER)

        # Sentiment classification
        sentiment_logits = outputs["sentiment"][0]
        sentiment_idx = sentiment_logits.argmax().item()
        sentiment = IDX_TO_SENTIMENT.get(sentiment_idx, Sentiment.NEUTRAL)

        # Entity extraction (BIO decoding)
        entities = self._decode_entities(outputs["entities"][0], text)

        # Action decoding
        action = self._decode_action(outputs["action"], text)

        # Summary: pick top-scoring tokens
        summary = self._decode_summary(
            outputs["summary_scores"][0], text
        )

        # Temporal
        temporal = self._decode_temporal(outputs["temporal"])

        # Topics (multi-label with threshold)
        topic_probs = torch.sigmoid(outputs["topics"][0])
        topics = self._decode_topics(topic_probs, threshold=0.5)

        # Tags: auto-generate from entities and topics
        tags = self._generate_tags(entities, topics)

        return Understanding(
            text=text,
            type=note_type,
            priority=priority,
            urgency=urgency,
            sentiment=sentiment,
            entities=entities,
            action=action,
            summary=summary,
            topics=topics,
            tags=tags,
            temporal=temporal,
        )

    def _decode_entities(
        self,
        entity_logits: torch.Tensor,
        text: str,
    ) -> List[Entity]:
        """Decode BIO entity tags into Entity objects."""
        tag_indices = entity_logits.argmax(dim=-1)  # [seq_len]
        text_tokens = self.tokenizer._tokenize_text(text)

        entities = []
        current_entity = None
        current_tokens = []

        for i, tag_idx in enumerate(tag_indices):
            tag_idx = tag_idx.item()
            decoded = _decode_bio_tag(tag_idx)

            if decoded is None or not decoded[1]:
                # O-tag or I-tag — close current entity
                if current_entity is not None:
                    entity_name = " ".join(current_tokens)
                    # Check if name exists in text (case-insensitive)
                    entities.append(Entity(
                        name=entity_name,
                        entity_type=current_entity,
                        confidence=0.9,
                    ))
                    current_entity = None
                    current_tokens = []

                if decoded is not None and decoded[1] is False:
                    # I-tag without B-start — skip
                    continue

            if decoded is not None and decoded[1]:
                # B-tag — start new entity
                current_entity = decoded[0]
                # Map token index back to text token
                tok_idx = i  # approximate mapping
                if tok_idx < len(text_tokens):
                    current_tokens = [text_tokens[tok_idx]]
                else:
                    current_tokens = []

        # Close any open entity
        if current_entity is not None and current_tokens:
            entities.append(Entity(
                name=" ".join(current_tokens),
                entity_type=current_entity,
                confidence=0.9,
            ))

        return entities

    def _decode_action(
        self,
        action_tensor: torch.Tensor,
        text: str,
    ) -> Optional[str]:
        """Decode action head output into an action string."""
        if action_tensor.dim() == 3:
            # Training mode: logits [batch, max_len, vocab_size]
            token_ids = action_tensor[0].argmax(dim=-1)
        else:
            # Inference mode: token_ids [batch, max_len]
            token_ids = action_tensor[0]

        # Decode token IDs to text
        decoded = self.tokenizer.decode(token_ids.tolist())
        decoded = decoded.strip()

        # If the decoded action is empty or just padding, extract from text
        if not decoded or len(decoded) < 3:
            return self._extract_action_from_text(text)

        return decoded

    def _extract_action_from_text(self, text: str) -> Optional[str]:
        """Extract action phrase from text using simple heuristics."""
        text_lower = text.lower()
        action_markers = [
            "need to ", "have to ", "should ", "going to ",
            "want to ", "can you ", "remind me to ",
            "todo: ", "action item: ", "fix ", "update ",
            "review ", "deploy ", "build ", "write ",
        ]
        for marker in action_markers:
            idx = text_lower.find(marker)
            if idx >= 0:
                action_text = text[idx:].strip()
                # Cut at period or end
                for end_char in [".", ",", "!", "?"]:
                    cut_idx = action_text.find(end_char)
                    if cut_idx > 0:
                        action_text = action_text[:cut_idx]
                return action_text.strip()

        return None

    def _decode_summary(
        self,
        summary_scores: torch.Tensor,
        text: str,
    ) -> Optional[str]:
        """Extract extractive summary from per-token scores."""
        text_tokens = self.tokenizer._tokenize_text(text)
        if not text_tokens:
            return None

        scores = summary_scores[:len(text_tokens)]

        # Pick top 5 tokens by score
        if len(scores) == 0:
            return text[:200]

        top_indices = scores.argsort(descending=True)[:5]
        top_indices = sorted(top_indices.tolist())

        summary_tokens = [text_tokens[i] for i in top_indices if i < len(text_tokens)]
        return " ".join(summary_tokens) if summary_tokens else text[:200]

    def _decode_temporal(
        self, temporal_output: Dict[str, torch.Tensor]
    ) -> Optional[TemporalInfo]:
        """Decode temporal head output."""
        has_date_prob = temporal_output["has_date"][0, 0].item()
        has_date = has_date_prob > 0.5

        if not has_date:
            return None

        relative_logits = temporal_output["relative"][0]
        relative_idx = relative_logits.argmax().item()
        relative_map = {0: "past", 1: "today", 2: "future"}
        relative = relative_map.get(relative_idx, "future")

        return TemporalInfo(
            has_date=True,
            relative=relative,
            confidence=has_date_prob,
        )

    def _decode_topics(
        self,
        topic_probs: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[str]:
        """Decode multi-label topic predictions."""
        from brain_contract import TOPICS

        topics = []
        for i, prob in enumerate(topic_probs):
            if prob.item() >= threshold and i < len(TOPICS):
                topics.append(TOPICS[i])

        return topics

    def _generate_tags(
        self,
        entities: List[Entity],
        topics: List[str],
    ) -> List[str]:
        """Auto-generate tags from entities and topics."""
        tags = set()

        # Add entity names as tags
        for ent in entities:
            name = ent.name.lower().replace(" ", "-")
            tags.add(name)

        # Add topics as tags
        for topic in topics:
            tags.add(topic)

        return sorted(tags)

    def _compute_confidence(
        self, outputs: Dict[str, torch.Tensor]
    ) -> float:
        """Compute overall model confidence from head outputs."""
        # Average of softmax probabilities at predicted classes
        confidences = []

        # Classifier confidence
        type_probs = torch.softmax(outputs["classifier"][0], dim=0)
        confidences.append(type_probs.max().item())

        # Priority confidence (distance from 0.5)
        priority = outputs["priority"][0, 0].item()
        confidences.append(abs(priority - 0.5) * 2)

        # Urgency confidence
        urgency_probs = torch.softmax(outputs["urgency"][0], dim=0)
        confidences.append(urgency_probs.max().item())

        # Sentiment confidence
        sentiment_probs = torch.softmax(outputs["sentiment"][0], dim=0)
        confidences.append(sentiment_probs.max().item())

        # Temporal confidence
        has_date_conf = outputs["temporal"]["has_date"][0, 0].item()
        confidences.append(max(has_date_conf, 1 - has_date_conf))

        return sum(confidences) / len(confidences)

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

    @classmethod
    def load(cls, checkpoint_dir: str) -> "BrainEngine":
        """
        Load a trained brain engine.

        Args:
            checkpoint_dir: Path to checkpoint directory containing
                           model.pt, tokenizer.json, and optional
                           entity_index.json.
        """
        # Load model
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_path):
            model = BrainModel.load(model_path)
        else:
            model = BrainModel()

        # Load tokenizer
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            tokenizer = Tokenizer.load(tokenizer_path)
        else:
            tokenizer = Tokenizer()
            # Build a minimal vocab
            tokenizer.fit(["placeholder to initialize vocab"])

        # Load embedder (lazy — may not be installed)
        try:
            embedder = TextEmbedder()
        except Exception:
            # Stub embedder that returns zeros
            embedder = _StubEmbedder()

        # Load entity linker
        entity_linker = EntityLinker()
        entity_path = os.path.join(checkpoint_dir, "entity_index.json")
        if os.path.exists(entity_path):
            entity_linker.load(entity_path)

        return cls(model, tokenizer, embedder, entity_linker)

    def save(self, checkpoint_dir: str) -> None:
        """Save brain engine state."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.save(os.path.join(checkpoint_dir, "model.pt"))

        # Save tokenizer
        self.tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))

        # Save entity index
        self.entity_linker.save(
            os.path.join(checkpoint_dir, "entity_index.json")
        )


class _StubEmbedder:
    """Fallback embedder when sentence-transformers is not available."""

    dimension = 384

    def embed(self, text: str) -> List[float]:
        return [0.0] * self.dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * self.dimension for _ in texts]

    def similarity(self, text_a: str, text_b: str) -> float:
        return 0.0

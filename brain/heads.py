"""
brain.heads — All 10 task heads for the multi-task model.

Each head is a standalone nn.Module that takes backbone output
(context or tokens) and produces task-specific predictions.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.config import (
    EMBEDDING_DIM, CLASSIFIER_HIDDEN, ENTITY_HIDDEN,
    PRIORITY_HIDDEN, URGENCY_HIDDEN, SENTIMENT_HIDDEN,
    ACTION_DECODER_LAYERS, ACTION_DECODER_HEADS, ACTION_MAX_LEN,
    SUMMARY_HIDDEN, TEMPORAL_HIDDEN, TOPIC_HIDDEN, RELATION_HIDDEN,
    DROPOUT, VOCAB_SIZE,
)
from brain_contract import NoteType, EntityType, Urgency, Sentiment

# Count of BIO tags: O + 2 * number of entity types
NUM_BIO_TAGS = 1 + 2 * len(EntityType)  # = 45
NUM_URGENCY = len(Urgency)              # = 4
NUM_SENTIMENT = len(Sentiment)          # = 5
NUM_TYPES = len(NoteType)               # = 29
NUM_TOPICS = len([
    "technology", "health", "finance", "career", "relationships",
    "education", "entertainment", "food", "travel", "home",
    "fitness", "creativity", "spirituality", "science", "philosophy",
    "engineering", "design", "music", "gaming", "sports",
    "parenting", "cooking", "gardening", "reading", "writing",
    "politics", "environment", "productivity", "psychology",
])
NUM_RELATIONS = len([
    "works_on", "mentioned_in", "created", "uses", "depends_on",
    "related_to", "decided_by", "learned_from", "attended",
    "assigned_to", "lives_in", "prefers", "authored",
    "discussed_in", "follows_up_on", "blocks", "enables",
])


# =============================================================================
# Head 1: Content Classifier
# =============================================================================

class ClassifierHead(nn.Module):
    """Classify note type from context vector."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = CLASSIFIER_HIDDEN,
        num_classes: int = NUM_TYPES,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch, input_dim]
        Returns:
            [batch, num_classes] — logits
        """
        return self.net(context)


# =============================================================================
# Head 2: Entity Tagger (BIO)
# =============================================================================

class EntityHead(nn.Module):
    """Per-token BIO tagging for entity extraction."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = ENTITY_HIDDEN,
        num_tags: int = NUM_BIO_TAGS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len, input_dim]
        Returns:
            [batch, seq_len, num_tags] — per-token tag logits
        """
        return self.net(tokens)


# =============================================================================
# Head 3: Priority Predictor
# =============================================================================

class PriorityHead(nn.Module):
    """Regression: predict priority score 0.0 to 1.0."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = PRIORITY_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch, input_dim]
        Returns:
            [batch, 1] — priority score 0.0-1.0
        """
        return self.net(context)


# =============================================================================
# Head 4: Urgency Predictor
# =============================================================================

class UrgencyHead(nn.Module):
    """4-class classification: immediate, soon, later, someday."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = URGENCY_HIDDEN,
        num_classes: int = NUM_URGENCY,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch, input_dim]
        Returns:
            [batch, num_classes] — logits
        """
        return self.net(context)


# =============================================================================
# Head 5: Sentiment Analyzer
# =============================================================================

class SentimentHead(nn.Module):
    """5-class classification: positive, negative, neutral, mixed, uncertain."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = SENTIMENT_HIDDEN,
        num_classes: int = NUM_SENTIMENT,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch, input_dim]
        Returns:
            [batch, num_classes] — logits
        """
        return self.net(context)


# =============================================================================
# Head 6: Action Extractor (Seq2Seq)
# =============================================================================

class ActionHead(nn.Module):
    """
    Generate action text from context.

    During training: teacher forcing, returns logits.
    During inference: autoregressive generation.
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = EMBEDDING_DIM,
        n_heads: int = ACTION_DECODER_HEADS,
        n_layers: int = ACTION_DECODER_LAYERS,
        max_len: int = ACTION_MAX_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.max_len = max_len

        # Project context to decoder initial state
        self.input_proj = nn.Linear(input_dim, d_model)

        # Token embedding for decoder
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        context: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, input_dim] — from backbone
            target_ids: [batch, max_len] — teacher forcing IDs (training)

        Returns:
            Training: [batch, max_len, vocab_size] — logits
            Inference: [batch, max_len] — generated token IDs
        """
        batch_size = context.size(0)
        device = context.device

        # Project context to memory for decoder
        memory = self.input_proj(context).unsqueeze(1)  # [batch, 1, d_model]

        if target_ids is not None and self.training:
            # Teacher forcing
            tok_emb = self.token_emb(target_ids)
            positions = torch.arange(target_ids.size(1), device=device).unsqueeze(0)
            tok_emb = tok_emb + self.pos_emb(positions)

            # Causal mask
            seq_len = target_ids.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()

            decoded = self.decoder(tok_emb, memory, tgt_mask=causal_mask)
            return self.output_proj(decoded)
        else:
            # Autoregressive generation
            return self._generate(memory, device, batch_size)

    def _generate(
        self, memory: torch.Tensor, device: torch.device, batch_size: int
    ) -> torch.Tensor:
        """Autoregressively generate action tokens."""
        # Start with CLS token (id=2)
        generated = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)

        for _ in range(self.max_len - 1):
            tok_emb = self.token_emb(generated)
            positions = torch.arange(generated.size(1), device=device).unsqueeze(0)
            tok_emb = tok_emb + self.pos_emb(positions)

            seq_len = generated.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()

            decoded = self.decoder(tok_emb, memory, tgt_mask=causal_mask)
            logits = self.output_proj(decoded[:, -1:, :])
            next_token = logits.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)

        # Pad or truncate to max_len
        if generated.size(1) < self.max_len:
            pad = torch.zeros(
                batch_size, self.max_len - generated.size(1),
                dtype=torch.long, device=device
            )
            generated = torch.cat([generated, pad], dim=1)
        else:
            generated = generated[:, :self.max_len]

        return generated


# =============================================================================
# Head 7: Summary Extractor
# =============================================================================

class SummaryHead(nn.Module):
    """Per-token scoring for extractive summarization."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = SUMMARY_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len, input_dim]
        Returns:
            [batch, seq_len] — per-token importance scores
        """
        return self.net(tokens).squeeze(-1)


# =============================================================================
# Head 8: Temporal Resolver
# =============================================================================

class TemporalHead(nn.Module):
    """Predict temporal properties of text."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = TEMPORAL_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Sub-tasks
        self.has_date = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.relative = nn.Linear(hidden_dim, 3)   # past/today/future
        self.month = nn.Linear(hidden_dim, 12)      # month prediction

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            context: [batch, input_dim]
        Returns:
            Dict with:
              has_date: [batch, 1]
              relative: [batch, 3] — logits
              month: [batch, 12] — logits
        """
        h = self.shared(context)
        return {
            "has_date": self.has_date(h),
            "relative": self.relative(h),
            "month": self.month(h),
        }


# =============================================================================
# Head 9: Topic Classifier (multi-label)
# =============================================================================

class TopicHead(nn.Module):
    """Multi-label topic classification."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = TOPIC_HIDDEN,
        num_topics: int = NUM_TOPICS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_topics),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [batch, input_dim]
        Returns:
            [batch, num_topics] — per-topic probabilities (sigmoid)
        """
        return self.net(context)


# =============================================================================
# Head 10: Relationship Detector
# =============================================================================

class RelationHead(nn.Module):
    """Classify relationship between two entity embeddings."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM * 2,  # concatenated entity pair
        hidden_dim: int = RELATION_HIDDEN,
        num_relations: int = NUM_RELATIONS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_relations),
        )

    def forward(self, entity_pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entity_pair: [batch, input_dim*2] — concatenated entity embeddings
        Returns:
            [batch, num_relations] — logits
        """
        return self.net(entity_pair)

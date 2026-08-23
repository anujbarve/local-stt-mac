"""
brain.model — Full multi-task model.

Wires the backbone with all 10 heads into a single model.
One forward pass produces all outputs simultaneously.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from brain.backbone import Backbone
from brain.heads import (
    ClassifierHead, EntityHead, PriorityHead, UrgencyHead,
    SentimentHead, ActionHead, SummaryHead, TemporalHead,
    TopicHead, RelationHead, NUM_BIO_TAGS, NUM_TYPES,
)
from brain.config import VOCAB_SIZE, EMBEDDING_DIM


class BrainModel(nn.Module):
    """
    Multi-task neural network for text understanding.

    Architecture:
        Backbone → context + tokens
        context → classifier, priority, urgency, sentiment, action, temporal, topics
        tokens  → entity_tagger, summary
        entity_pair → relation
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = EMBEDDING_DIM,
    ):
        super().__init__()

        # Shared backbone
        self.backbone = Backbone(vocab_size=vocab_size, d_model=d_model)

        # Task heads
        self.classifier = ClassifierHead(input_dim=d_model)
        self.entity_tagger = EntityHead(input_dim=d_model)
        self.priority = PriorityHead(input_dim=d_model)
        self.urgency = UrgencyHead(input_dim=d_model)
        self.sentiment = SentimentHead(input_dim=d_model)
        self.action = ActionHead(input_dim=d_model, vocab_size=vocab_size)
        self.summary = SummaryHead(input_dim=d_model)
        self.temporal = TemporalHead(input_dim=d_model)
        self.topics = TopicHead(input_dim=d_model)
        self.relation = RelationHead(input_dim=d_model * 2)

    def forward(
        self,
        token_ids: torch.Tensor,
        target_action_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all heads.

        Args:
            token_ids: [batch, seq_len] — input token IDs
            target_action_ids: [batch, max_len] — teacher forcing for action head

        Returns:
            Dict with all head outputs:
              classifier:    [batch, num_types]
              entities:      [batch, seq_len, num_bio_tags]
              priority:      [batch, 1]
              urgency:       [batch, 4]
              sentiment:     [batch, 5]
              action:        [batch, max_len, vocab_size] or [batch, max_len]
              summary_scores:[batch, seq_len]
              temporal:      dict of tensors
              topics:        [batch, num_topics]
        """
        # Shared backbone
        context, tokens = self.backbone(token_ids)

        # All heads fire simultaneously
        outputs = {
            "classifier": self.classifier(context),
            "entities": self.entity_tagger(tokens),
            "priority": self.priority(context),
            "urgency": self.urgency(context),
            "sentiment": self.sentiment(context),
            "action": self.action(context, target_ids=target_action_ids),
            "summary_scores": self.summary(tokens),
            "temporal": self.temporal(context),
            "topics": self.topics(context),
        }

        return outputs

    def predict_relation(
        self, entity_emb_a: torch.Tensor, entity_emb_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict relationship between two entities.

        Args:
            entity_emb_a: [batch, d_model]
            entity_emb_b: [batch, d_model]
        Returns:
            [batch, num_relations] — logits
        """
        pair = torch.cat([entity_emb_a, entity_emb_b], dim=-1)
        return self.relation(pair)

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(
        self,
        path: str,
        version: str = "0.1.0",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save model with version metadata.

        Args:
            path: Output file path.
            version: Semantic version string (e.g. "1.2.0").
            metadata: Optional dict of training metrics, dataset info, etc.
        """
        import os
        from datetime import datetime, timezone

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "vocab_size": self.backbone.token_embedding.num_embeddings,
            "d_model": self.backbone.d_model,
            "version": version,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str) -> "BrainModel":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            d_model=checkpoint["d_model"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model

    @classmethod
    def load_versions(cls, checkpoint_dir: str) -> list:
        """List all model versions in a directory.

        Returns list of dicts with file, version, saved_at, metadata.
        """
        import os
        versions = []
        if not os.path.isdir(checkpoint_dir):
            return versions
        for fname in sorted(os.listdir(checkpoint_dir)):
            if fname.endswith(".pt"):
                fpath = os.path.join(checkpoint_dir, fname)
                try:
                    ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
                    versions.append({
                        "file": fname,
                        "version": ckpt.get("version", "unknown"),
                        "saved_at": ckpt.get("saved_at", "unknown"),
                        "metadata": ckpt.get("metadata", {}),
                    })
                except Exception:
                    continue
        return versions

    @classmethod
    def load_with_retry(
        cls,
        path: str,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> "BrainModel":
        """Load model with exponential backoff retry.

        Args:
            path: Path to checkpoint file.
            max_retries: Maximum number of retry attempts.
            backoff_factor: Multiplier for backoff delay.

        Returns:
            Loaded BrainModel.

        Raises:
            RuntimeError: If all retries fail.
        """
        import time as _time
        import logging

        logger = logging.getLogger(__name__)
        last_error = None

        for attempt in range(max_retries):
            try:
                return cls.load(path)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = backoff_factor * (2 ** attempt)
                    logger.warning(
                        "Model load attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1, max_retries, e, delay,
                    )
                    _time.sleep(delay)
                else:
                    logger.error(
                        "Model load failed after %d attempts: %s",
                        max_retries, e,
                    )

        raise RuntimeError(
            f"Failed to load model from {path} after {max_retries} attempts: {last_error}"
        )
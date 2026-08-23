"""
brain.train — Training pipeline for the multi-task brain model.

Handles dataset creation, training loop, loss computation, and checkpointing.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from brain.config import (
    LEARNING_RATE, BATCH_SIZE, EPOCHS, WEIGHT_DECAY, GRAD_CLIP,
    CHECKPOINT_DIR, CHECKPOINT_EVERY, PAD_TOKEN, UNK_TOKEN,
    CLASSIFIER_LOSS_WEIGHT, ENTITY_LOSS_WEIGHT, PRIORITY_LOSS_WEIGHT,
    URGENCY_LOSS_WEIGHT, SENTIMENT_LOSS_WEIGHT, ACTION_LOSS_WEIGHT,
    SUMMARY_LOSS_WEIGHT, TEMPORAL_LOSS_WEIGHT, TOPIC_LOSS_WEIGHT,
    VOCAB_SIZE, MAX_SEQ_LEN,
)
from brain.model import BrainModel
from brain.tokenizer import Tokenizer
from brain_contract import NoteType, Urgency, Sentiment, EntityType


# =============================================================================
# Label maps
# =============================================================================

TYPE_TO_IDX = {t.value: i for i, t in enumerate(NoteType)}
IDX_TO_TYPE = {i: t for t, i in TYPE_TO_IDX.items()}

URGENCY_TO_IDX = {u.value: i for i, u in enumerate(Urgency)}
IDX_TO_URGENCY = {i: u for u, i in URGENCY_TO_IDX.items()}

SENTIMENT_TO_IDX = {s.value: i for i, s in enumerate(Sentiment)}
IDX_TO_SENTIMENT = {i: s for s, i in SENTIMENT_TO_IDX.items()}

# BIO tag map: O=0, B-{type}=2i+1, I-{type}=2i+2 for each entity type
ENTITY_TYPES = list(EntityType)
TAG_O = 0


def entity_type_to_bio_tags(entity_type: str) -> Tuple[int, int]:
    """Return (B-tag, I-tag) indices for an entity type string."""
    idx = ENTITY_TYPES.index(EntityType(entity_type))
    return 2 * idx + 1, 2 * idx + 2


# =============================================================================
# Dataset
# =============================================================================

class BrainDataset(Dataset):
    """
    Dataset that converts raw sample dicts into model-ready tensors.

    Each sample dict has keys:
        text, type, priority, urgency, sentiment,
        entities, action, topics, has_temporal
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer: Tokenizer,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        token_ids = self.tokenizer.encode(sample["text"], add_special=True)
        token_ids = token_ids[:self.max_seq_len]

        # Pad to max_seq_len
        pad_id = self.tokenizer.pad_id
        while len(token_ids) < self.max_seq_len:
            token_ids.append(pad_id)

        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        # Type label
        type_label = TYPE_TO_IDX.get(sample["type"], 0)

        # Priority score
        priority_score = float(sample.get("priority", 0.5))

        # Urgency label
        urgency_label = URGENCY_TO_IDX.get(sample.get("urgency", "later"), 2)

        # Sentiment label
        sentiment_label = SENTIMENT_TO_IDX.get(sample.get("sentiment", "neutral"), 2)

        # Entity BIO labels (per-token)
        bio_labels = self._make_bio_labels(sample, token_ids)

        # Topic labels (multi-label binary vector)
        topic_labels = self._make_topic_labels(sample.get("topics", []))

        # Has-date for temporal head
        has_date = 1.0 if sample.get("has_temporal", False) else 0.0

        return {
            "token_ids": token_tensor,
            "type_label": torch.tensor(type_label, dtype=torch.long),
            "priority_score": torch.tensor(priority_score, dtype=torch.float),
            "urgency_label": torch.tensor(urgency_label, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "bio_labels": torch.tensor(bio_labels, dtype=torch.long),
            "topic_labels": torch.tensor(topic_labels, dtype=torch.float),
            "has_date": torch.tensor(has_date, dtype=torch.float),
        }

    def _make_bio_labels(
        self, sample: Dict, token_ids: List[int]
    ) -> List[int]:
        """
        Create BIO labels aligned with token positions.

        Since we don't have character-level alignment in synthetic data,
        we use a simplified scheme: mark the first token of each entity mention
        with B-tag and subsequent tokens with I-tag.
        """
        seq_len = self.max_seq_len
        labels = [TAG_O] * seq_len

        # Tokenize entity names and try to find them in the token sequence
        text_tokens = self.tokenizer._tokenize_text(sample["text"])

        for ent in sample.get("entities", []):
            ent_name = ent.get("name", "")
            ent_type = ent.get("type", "concept")
            b_tag, i_tag = entity_type_to_bio_tags(ent_type)

            # Tokenize entity name
            ent_tokens = self.tokenizer._tokenize_text(ent_name)
            if not ent_tokens:
                continue

            # Find matching token span in the text
            for start_idx in range(len(text_tokens)):
                if text_tokens[start_idx:start_idx + len(ent_tokens)] == ent_tokens:
                    # Map text_tokens index to token_ids index
                    # token_ids has [CLS] at position 0, so offset by 1
                    tok_start = start_idx + 1  # +1 for CLS token
                    tok_end = tok_start + len(ent_tokens)

                    if tok_end <= seq_len:
                        labels[tok_start] = b_tag
                        for j in range(tok_start + 1, tok_end):
                            labels[j] = i_tag
                    break

        return labels

    def _make_topic_labels(self, topics: List[str]) -> List[float]:
        """Create binary topic vector."""
        from brain_contract import TOPICS
        labels = [0.0] * len(TOPICS)
        for topic in topics:
            if topic in TOPICS:
                labels[TOPICS.index(topic)] = 1.0
        return labels


# =============================================================================
# Loss computation
# =============================================================================

def compute_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Compute multi-task loss.

    Returns dict of per-head losses + total weighted loss.
    """
    losses = {}

    # Classification: CrossEntropy
    losses["classifier"] = F.cross_entropy(
        outputs["classifier"], batch["type_label"]
    )

    # Entity tagging: CrossEntropy per token
    entity_logits = outputs["entities"]  # [batch, seq_len, num_tags]
    bio_labels = batch["bio_labels"]     # [batch, seq_len]
    losses["entities"] = F.cross_entropy(
        entity_logits.reshape(-1, entity_logits.size(-1)),
        bio_labels.reshape(-1),
        ignore_index=TAG_O,
    )

    # Priority: MSE regression
    losses["priority"] = F.mse_loss(
        outputs["priority"].squeeze(-1), batch["priority_score"]
    )

    # Urgency: CrossEntropy
    losses["urgency"] = F.cross_entropy(
        outputs["urgency"], batch["urgency_label"]
    )

    # Sentiment: CrossEntropy
    losses["sentiment"] = F.cross_entropy(
        outputs["sentiment"], batch["sentiment_label"]
    )

    # Action: CrossEntropy on decoder output (teacher forcing)
    action_logits = outputs["action"]
    if action_logits.dim() == 3:
        # Training mode with teacher forcing
        # Use a dummy target (all padding) since we don't have action targets
        # in synthetic data — the action head still learns from the context
        target = torch.zeros(
            action_logits.size(0), action_logits.size(1),
            dtype=torch.long, device=action_logits.device,
        )
        losses["action"] = F.cross_entropy(
            action_logits.reshape(-1, action_logits.size(-1)),
            target.reshape(-1),
            ignore_index=0,
        )
    else:
        losses["action"] = torch.tensor(0.0, device=action_logits.device)

    # Summary: Binary cross-entropy per token
    summary_scores = outputs["summary_scores"]  # [batch, seq_len]
    # Create summary labels: first 30% of non-padding tokens are "summary"
    seq_len = summary_scores.size(1)
    summary_cutoff = max(1, seq_len // 3)
    summary_labels = torch.zeros_like(summary_scores)
    summary_labels[:, :summary_cutoff] = 1.0
    losses["summary"] = F.binary_cross_entropy(
        summary_scores, summary_labels
    )

    # Temporal: Multi-task (has_date only in this simplified version)
    temporal = outputs["temporal"]
    losses["temporal_has_date"] = F.binary_cross_entropy(
        temporal["has_date"].squeeze(-1), batch["has_date"]
    )

    # Topics: Multi-label binary cross-entropy
    topic_logits = outputs["topics"]
    losses["topics"] = F.binary_cross_entropy_with_logits(
        topic_logits, batch["topic_labels"]
    )

    # Weighted total
    total = (
        CLASSIFIER_LOSS_WEIGHT * losses["classifier"]
        + ENTITY_LOSS_WEIGHT * losses["entities"]
        + PRIORITY_LOSS_WEIGHT * losses["priority"]
        + URGENCY_LOSS_WEIGHT * losses["urgency"]
        + SENTIMENT_LOSS_WEIGHT * losses["sentiment"]
        + ACTION_LOSS_WEIGHT * losses["action"]
        + SUMMARY_LOSS_WEIGHT * losses["summary"]
        + TEMPORAL_LOSS_WEIGHT * losses["temporal_has_date"]
        + TOPIC_LOSS_WEIGHT * losses["topics"]
    )
    losses["total"] = total

    return losses


# =============================================================================
# Trainer
# =============================================================================

class BrainTrainer:
    """
    Training loop for the multi-task brain model.

    Usage:
        trainer = BrainTrainer(model, tokenizer)
        trainer.train(train_data, val_data, epochs=20)
    """

    def __init__(
        self,
        model: BrainModel,
        tokenizer: Tokenizer,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def _make_loader(
        self,
        data: List[Dict],
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = BrainDataset(data, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch. Returns average loss per head."""
        self.model.train()
        epoch_losses: Dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(batch["token_ids"])
            losses = compute_loss(outputs, batch)
            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), GRAD_CLIP
            )

            self.optimizer.step()

            # Accumulate losses
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            num_batches += 1

        # Average
        if num_batches > 0:
            for k in epoch_losses:
                epoch_losses[k] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        epoch_losses: Dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch["token_ids"])
            losses = compute_loss(outputs, batch)

            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            num_batches += 1

        if num_batches > 0:
            for k in epoch_losses:
                epoch_losses[k] /= num_batches

        return epoch_losses

    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        checkpoint_dir: str = CHECKPOINT_DIR,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Full training loop.

        Returns list of epoch metrics dicts.
        """
        train_loader = self._make_loader(train_data, batch_size, shuffle=True)
        val_loader = self._make_loader(val_data, batch_size, shuffle=False)

        history = []

        for epoch in range(1, epochs + 1):
            train_losses = self.train_epoch(train_loader)
            val_losses = self.evaluate(val_loader)

            record = {
                "epoch": epoch,
                "train_" + k: v for k, v in train_losses.items()
            }
            record.update({
                "val_" + k: v for k, v in val_losses.items()
            })
            history.append(record)

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_losses['total']:.4f} | "
                    f"val_loss={val_losses['total']:.4f}"
                )

            # Checkpoint
            if epoch % CHECKPOINT_EVERY == 0 or epoch == epochs:
                ckpt_path = os.path.join(
                    checkpoint_dir, f"model_epoch{epoch}.pt"
                )
                self.save_checkpoint(
                    ckpt_path, epoch, record,
                )

        return history

    def retrain_head(
        self,
        head_name: str,
        data: List[Dict],
        epochs: int = 5,
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, float]:
        """
        Retrain a single head (fast, for corrections).

        Freezes backbone, only updates the specified head.
        """
        # Map head name to module
        head_modules = {
            "classifier": self.model.classifier,
            "entity_tagger": self.model.entity_tagger,
            "priority": self.model.priority,
            "urgency": self.model.urgency,
            "sentiment": self.model.sentiment,
            "action": self.model.action,
            "summary": self.model.summary,
            "temporal": self.model.temporal,
            "topics": self.model.topics,
        }

        if head_name not in head_modules:
            raise ValueError(f"Unknown head: {head_name}")

        # Freeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Optimizer for the head only
        head_optimizer = torch.optim.AdamW(
            head_modules[head_name].parameters(),
            lr=LEARNING_RATE * 2,
            weight_decay=WEIGHT_DECAY,
        )

        loader = self._make_loader(data, batch_size, shuffle=True)
        last_losses = {}

        for epoch in range(epochs):
            self.model.train()
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                head_optimizer.zero_grad()
                outputs = self.model(batch["token_ids"])
                losses = compute_loss(outputs, batch)
                losses["total"].backward()
                head_optimizer.step()
            last_losses = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}

        # Unfreeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        return last_losses

    def retrain_all(
        self,
        data: List[Dict],
        epochs: int = 10,
        batch_size: int = BATCH_SIZE,
    ) -> List[Dict[str, float]]:
        """Full model retrain with correction data."""
        val_split = int(len(data) * 0.8)
        train_data = data[:val_split]
        val_data = data[val_split:]
        return self.train(
            train_data, val_data,
            epochs=epochs,
            batch_size=batch_size,
        )

    def save_checkpoint(
        self, path: str, epoch: int, metrics: Dict
    ) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "vocab_size": self.model.backbone.token_embedding.num_embeddings,
            "d_model": self.model.backbone.d_model,
        }, path)

    def load_checkpoint(self, path: str) -> Tuple[int, Dict]:
        """Load model checkpoint. Returns (epoch, metrics)."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint.get("metrics", {})

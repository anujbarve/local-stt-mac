"""
brain.train — Training pipeline for the multi-task brain model.

Handles dataset creation, training loop, loss computation, metrics tracking,
learning rate scheduling, early stopping, data augmentation, and checkpointing.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
from brain_contract import NoteType, Urgency, Sentiment, EntityType, Priority


# =============================================================================
# Label maps
# =============================================================================

TYPE_TO_IDX = {t.value: i for i, t in enumerate(NoteType)}
IDX_TO_TYPE = {i: t for t, i in TYPE_TO_IDX.items()}

URGENCY_TO_IDX = {u.value: i for i, u in enumerate(Urgency)}
IDX_TO_URGENCY = {i: u for u, i in URGENCY_TO_IDX.items()}

SENTIMENT_TO_IDX = {s.value: i for i, s in enumerate(Sentiment)}
IDX_TO_SENTIMENT = {i: s for s, i in SENTIMENT_TO_IDX.items()}

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

        pad_id = self.tokenizer.pad_id
        while len(token_ids) < self.max_seq_len:
            token_ids.append(pad_id)

        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        type_label = TYPE_TO_IDX.get(sample["type"], 0)
        priority_score = float(sample.get("priority", 0.5))
        urgency_label = URGENCY_TO_IDX.get(sample.get("urgency", "later"), 2)
        sentiment_label = SENTIMENT_TO_IDX.get(sample.get("sentiment", "neutral"), 2)
        bio_labels = self._make_bio_labels(sample, token_ids)
        topic_labels = self._make_topic_labels(sample.get("topics", []))
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

    def _make_bio_labels(self, sample: Dict, token_ids: List[int]) -> List[int]:
        seq_len = self.max_seq_len
        labels = [TAG_O] * seq_len
        text_tokens = self.tokenizer._tokenize_text(sample["text"])

        for ent in sample.get("entities", []):
            ent_name = ent.get("name", "")
            ent_type = ent.get("type", "concept")
            try:
                b_tag, i_tag = entity_type_to_bio_tags(ent_type)
            except (ValueError, KeyError):
                continue

            ent_tokens = self.tokenizer._tokenize_text(ent_name)
            if not ent_tokens:
                continue

            for start_idx in range(len(text_tokens)):
                if text_tokens[start_idx:start_idx + len(ent_tokens)] == ent_tokens:
                    tok_start = start_idx + 1
                    tok_end = tok_start + len(ent_tokens)
                    if tok_end <= seq_len:
                        labels[tok_start] = b_tag
                        for j in range(tok_start + 1, tok_end):
                            labels[j] = i_tag
                    break
        return labels

    def _make_topic_labels(self, topics: List[str]) -> List[float]:
        from brain_contract import TOPICS
        labels = [0.0] * len(TOPICS)
        for topic in topics:
            if topic in TOPICS:
                labels[TOPICS.index(topic)] = 1.0
        return labels


# =============================================================================
# Metrics Tracker
# =============================================================================

class MetricsTracker:
    """Track per-head accuracy, F1, and loss across batches within an epoch."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Clear accumulated state for a new epoch."""
        self._type_correct = 0
        self._type_total = 0
        self._urgency_correct = 0
        self._urgency_total = 0
        self._sentiment_correct = 0
        self._sentiment_total = 0
        self._priority_errors: List[float] = []
        self._entity_tp = 0
        self._entity_fp = 0
        self._entity_fn = 0
        self._topic_tp = 0
        self._topic_fp = 0
        self._topic_fn = 0

    def update(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> None:
        """Accumulate predictions and labels from one batch."""
        with torch.no_grad():
            # Type accuracy
            type_pred = outputs["classifier"].argmax(dim=-1)
            self._type_correct += (type_pred == batch["type_label"]).sum().item()
            self._type_total += batch["type_label"].size(0)

            # Urgency accuracy
            urg_pred = outputs["urgency"].argmax(dim=-1)
            self._urgency_correct += (urg_pred == batch["urgency_label"]).sum().item()
            self._urgency_total += batch["urgency_label"].size(0)

            # Sentiment accuracy
            sent_pred = outputs["sentiment"].argmax(dim=-1)
            self._sentiment_correct += (sent_pred == batch["sentiment_label"]).sum().item()
            self._sentiment_total += batch["sentiment_label"].size(0)

            # Priority MAE
            pri_pred = outputs["priority"].squeeze(-1)
            errors = (pri_pred - batch["priority_score"]).abs()
            self._priority_errors.extend(errors.tolist())

            # Entity F1 (BIO tags)
            ent_pred = outputs["entities"].argmax(dim=-1)  # [batch, seq_len]
            bio_labels = batch["bio_labels"]
            mask = bio_labels != TAG_O
            self._entity_tp += ((ent_pred == bio_labels) & mask).sum().item()
            self._entity_fp += ((ent_pred != bio_labels) & (ent_pred != TAG_O) & mask).sum().item()
            self._entity_fn += ((ent_pred != bio_labels) & (bio_labels != TAG_O) & ~((ent_pred != bio_labels) & (ent_pred != TAG_O))).sum().item()

            # Topic F1
            topic_pred = (torch.sigmoid(outputs["topics"]) > 0.5).float()
            topic_labels = batch["topic_labels"]
            self._topic_tp += (topic_pred * topic_labels).sum().item()
            self._topic_fp += (topic_pred * (1 - topic_labels)).sum().item()
            self._topic_fn += ((1 - topic_pred) * topic_labels).sum().item()

    def compute(self) -> Dict[str, float]:
        """Compute epoch-level metrics."""
        type_acc = self._type_correct / self._type_total if self._type_total > 0 else 0.0
        urg_acc = self._urgency_correct / self._urgency_total if self._urgency_total > 0 else 0.0
        sent_acc = self._sentiment_correct / self._sentiment_total if self._sentiment_total > 0 else 0.0
        pri_mae = float(np.mean(self._priority_errors)) if self._priority_errors else 0.0

        ent_p = self._entity_tp / (self._entity_tp + self._entity_fp) if (self._entity_tp + self._entity_fp) > 0 else 0.0
        ent_r = self._entity_tp / (self._entity_tp + self._entity_fn) if (self._entity_tp + self._entity_fn) > 0 else 0.0
        ent_f1 = 2 * ent_p * ent_r / (ent_p + ent_r) if (ent_p + ent_r) > 0 else 0.0

        topic_p = self._topic_tp / (self._topic_tp + self._topic_fp) if (self._topic_tp + self._topic_fp) > 0 else 0.0
        topic_r = self._topic_tp / (self._topic_tp + self._topic_fn) if (self._topic_tp + self._topic_fn) > 0 else 0.0
        topic_f1 = 2 * topic_p * topic_r / (topic_p + topic_r) if (topic_p + topic_r) > 0 else 0.0

        return {
            "classifier_acc": type_acc,
            "urgency_acc": urg_acc,
            "sentiment_acc": sent_acc,
            "priority_mae": pri_mae,
            "entity_f1": ent_f1,
            "entity_precision": ent_p,
            "entity_recall": ent_r,
            "topic_f1": topic_f1,
            "topic_precision": topic_p,
            "topic_recall": topic_r,
        }


# =============================================================================
# Data Augmentation
# =============================================================================

SYNONYMS = {
    "fix": ["repair", "resolve", "patch"],
    "update": ["revise", "modify", "refresh"],
    "build": ["create", "construct", "develop"],
    "test": ["verify", "validate", "check"],
    "deploy": ["ship", "release", "launch"],
    "write": ["draft", "compose", "author"],
    "review": ["evaluate", "assess", "examine"],
    "debug": ["troubleshoot", "diagnose", "investigate"],
    "refactor": ["restructure", "reorganize", "rewrite"],
    "check": ["verify", "inspect", "confirm"],
    "install": ["set up", "configure", "setup"],
    "configure": ["set up", "adjust", "tune"],
    "optimize": ["improve", "enhance", "streamline"],
    "document": ["record", "note", "describe"],
    "clean": ["tidy", "organize", "sort"],
    "organize": ["arrange", "structure", "sort"],
    "schedule": ["plan", "arrange", "book"],
    "design": ["architect", "plan", "sketch"],
    "need": ["require", "want", "must"],
    "before": ["prior to", "ahead of", "by"],
    "meeting": ["call", "sync", "session"],
    "today": ["this morning", "right now", "currently"],
    "tomorrow": ["next day", "day after"],
    "quick": ["fast", "rapid", "brief"],
    "important": ["critical", "essential", "key"],
    "problem": ["issue", "bug", "challenge"],
    "working": ["functioning", "running", "operating"],
    "broken": ["not working", "failing", "down"],
    "happy": ["glad", "pleased", "satisfied"],
    "tired": ["exhausted", "worn out", "sleepy"],
}


class DataAugmenter:
    """Augment training data for better generalization."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def augment(self, data: List[Dict], factor: int = 2) -> List[Dict]:
        """Augment data by `factor`x."""
        augmented = list(data)
        for _ in range(factor - 1):
            for sample in data:
                new_sample = dict(sample)
                new_sample["text"] = self._apply_augmentation(sample["text"])
                augmented.append(new_sample)
        return augmented

    def _apply_augmentation(self, text: str) -> str:
        """Apply a random combination of augmentations."""
        methods = [self._synonym_replace, self._random_delete, self._random_swap]
        method = self.rng.choice(methods)
        return method(text)

    def _synonym_replace(self, text: str, n: int = 1) -> str:
        """Replace n random words with synonyms."""
        words = text.split()
        for _ in range(n):
            if not words:
                break
            idx = self.rng.randint(0, len(words) - 1)
            word = words[idx].lower()
            if word in SYNONYMS:
                words[idx] = self.rng.choice(SYNONYMS[word])
        return " ".join(words)

    def _random_delete(self, text: str, p: float = 0.1) -> str:
        """Randomly delete p fraction of words."""
        words = text.split()
        if len(words) <= 3:
            return text
        kept = [w for w in words if self.rng.random() > p]
        return " ".join(kept) if kept else text

    def _random_swap(self, text: str, n: int = 1) -> str:
        """Swap n pairs of adjacent words."""
        words = text.copy() if isinstance(text, list) else text.split()
        for _ in range(n):
            if len(words) < 2:
                break
            idx = self.rng.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)


# =============================================================================
# Loss computation
# =============================================================================

def compute_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute multi-task loss. Returns dict of per-head losses + total."""
    losses = {}

    losses["classifier"] = F.cross_entropy(outputs["classifier"], batch["type_label"])

    entity_logits = outputs["entities"]
    bio_labels = batch["bio_labels"]
    losses["entities"] = F.cross_entropy(
        entity_logits.reshape(-1, entity_logits.size(-1)),
        bio_labels.reshape(-1),
        ignore_index=TAG_O,
    )

    losses["priority"] = F.mse_loss(
        outputs["priority"].squeeze(-1), batch["priority_score"]
    )

    losses["urgency"] = F.cross_entropy(outputs["urgency"], batch["urgency_label"])

    losses["sentiment"] = F.cross_entropy(outputs["sentiment"], batch["sentiment_label"])

    action_logits = outputs["action"]
    if action_logits.dim() == 3:
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

    summary_scores = outputs["summary_scores"]
    seq_len = summary_scores.size(1)
    summary_cutoff = max(1, seq_len // 3)
    summary_labels = torch.zeros_like(summary_scores)
    summary_labels[:, :summary_cutoff] = 1.0
    losses["summary"] = F.binary_cross_entropy(summary_scores, summary_labels)

    temporal = outputs["temporal"]
    losses["temporal_has_date"] = F.binary_cross_entropy(
        temporal["has_date"].squeeze(-1), batch["has_date"]
    )

    topic_logits = outputs["topics"]
    losses["topics"] = F.binary_cross_entropy_with_logits(
        topic_logits, batch["topic_labels"]
    )

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

    Features:
        - Learning rate scheduling (cosine, plateau, step)
        - Early stopping on validation loss
        - Per-head metrics tracking
        - Best model checkpointing
    """

    def __init__(
        self,
        model: BrainModel,
        tokenizer: Tokenizer,
        lr: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        scheduler_type: str = "cosine",
        patience: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # LR scheduler
        self.scheduler_type = scheduler_type
        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=EPOCHS,
            )
        elif scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3,
            )
        elif scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5, gamma=0.5,
            )
        else:
            self.scheduler = None

        # Early stopping
        self.patience = patience
        self.best_val_loss = float("inf")
        self.wait = 0

        # Metrics
        self.metrics = MetricsTracker()

    def _make_loader(
        self,
        data: List[Dict],
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = BrainDataset(data, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch. Returns average loss per head."""
        self.model.train()
        self.metrics.reset()
        epoch_losses: Dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(batch["token_ids"])
            losses = compute_loss(outputs, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()

            self.metrics.update(outputs, batch)
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            num_batches += 1

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
        """Full training loop with early stopping and LR scheduling."""
        train_loader = self._make_loader(train_data, batch_size, shuffle=True)
        val_loader = self._make_loader(val_data, batch_size, shuffle=False)

        history = []
        self.best_val_loss = float("inf")
        self.wait = 0

        for epoch in range(1, epochs + 1):
            train_losses = self.train_epoch(train_loader)
            val_losses = self.evaluate(val_loader)

            record = {"epoch": epoch}
            record.update({"train_" + k: v for k, v in train_losses.items()})
            record.update({"val_" + k: v for k, v in val_losses.items()})

            # Add metrics
            train_metrics = self.metrics.compute()
            record.update({"train_" + k: v for k, v in train_metrics.items()})

            history.append(record)

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"train_loss={train_losses['total']:.4f} | "
                    f"val_loss={val_losses['total']:.4f} | "
                    f"acc={train_metrics['classifier_acc']:.3f}"
                )

            # LR scheduling
            if self.scheduler is not None:
                if self.scheduler_type == "plateau":
                    self.scheduler.step(val_losses["total"])
                else:
                    self.scheduler.step()

            # Early stopping
            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                self.wait = 0
                # Save best model
                best_path = os.path.join(checkpoint_dir, "model_best.pt")
                self.save_checkpoint(best_path, epoch, record)
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpoint
            if epoch % CHECKPOINT_EVERY == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"model_epoch{epoch}.pt")
                self.save_checkpoint(ckpt_path, epoch, record)

        # Save final
        final_path = os.path.join(checkpoint_dir, "model_final.pt")
        self.save_checkpoint(final_path, epoch if epochs > 0 else 0, history[-1] if history else {})

        return history

    def incremental_train(
        self,
        new_data: List[Dict],
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 1e-5,
    ) -> Dict[str, float]:
        """Fine-tune on new correction data (freezes backbone).

        Args:
            new_data: New correction samples.
            epochs: Training epochs (fewer than full train).
            batch_size: Batch size.
            lr: Learning rate (lower for fine-tuning).

        Returns:
            Training metrics.
        """
        if not new_data:
            return {"error": "no_data"}

        # Freeze backbone, only train heads
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Create optimizer for heads only
        head_params = [
            p for name, p in self.model.named_parameters()
            if "backbone" not in name and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=WEIGHT_DECAY)
        loader = self._make_loader(new_data, batch_size, shuffle=True)

        metrics = MetricsTracker()
        last_losses = {}

        for epoch in range(epochs):
            metrics.reset()
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(batch["token_ids"])
                losses = compute_loss(outputs, batch)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(head_params, GRAD_CLIP)
                optimizer.step()
                metrics.update(outputs, batch)
            last_losses = {k: v.item() if hasattr(v, 'item') else v for k, v in losses.items()}

        # Unfreeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        train_metrics = metrics.compute()
        result = {"epochs": epochs, "samples": len(new_data)}
        result.update({"train_" + k: v for k, v in train_metrics.items()})
        return result

    def retrain_head(
        self,
        head_name: str,
        data: List[Dict],
        epochs: int = 5,
        batch_size: int = BATCH_SIZE,
    ) -> Dict[str, float]:
        """Retrain a single head (freezes backbone)."""
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

        for param in self.model.backbone.parameters():
            param.requires_grad = False

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
        return self.train(data[:val_split], data[val_split:], epochs=epochs, batch_size=batch_size)

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict) -> None:
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


# =============================================================================
# Vault data loader
# =============================================================================

def load_vault_data(vault_root: str) -> List[Dict]:
    """Read notes from vault/ and convert to training format.

    Maps frontmatter fields to training labels.
    """
    import re

    data = []
    for dirpath, _dn, fnames in os.walk(vault_root):
        for fname in fnames:
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                sample = _parse_vault_note(fpath)
                if sample:
                    data.append(sample)
            except Exception:
                continue
    return data


def _parse_vault_note(fpath: str) -> Optional[Dict]:
    """Parse a single vault note into training format."""
    with open(fpath, "r", encoding="utf-8") as fh:
        raw = fh.read()

    if not raw.startswith("---"):
        return None

    end = raw.find("\n---", 3)
    if end == -1:
        return None

    fm_block = raw[3:end].strip()
    body = raw[end + 4:].lstrip("\n")

    # Parse frontmatter
    fm = {}
    for line in fm_block.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ": " in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip().strip("'\"")

    note_type = fm.get("type", "observation")

    # Parse priority float from label
    priority_label = fm.get("priority", "medium")
    priority_map = {"critical": 0.95, "high": 0.8, "medium": 0.5, "low": 0.3, "someday": 0.1}
    priority = priority_map.get(priority_label, 0.5)

    # Parse entities from lists
    entities = []
    for field, etype in [("people", "person"), ("projects", "project"), ("tools", "tool")]:
        val = fm.get(field, "[]")
        if val.startswith("[") and val.endswith("]"):
            items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
        else:
            items = [val] if val and val != "[]" else []
        for item in items:
            if item:
                entities.append({"name": item, "type": etype})

    # Parse tags
    tags_str = fm.get("tags", "[]")
    if tags_str.startswith("[") and tags_str.endswith("]"):
        topics = [t.strip().strip("'\"") for t in tags_str[1:-1].split(",") if t.strip()]
    else:
        topics = []

    return {
        "text": body.strip() if body.strip() else fm.get("id", "unknown"),
        "type": note_type,
        "priority": priority,
        "urgency": fm.get("urgency", "later"),
        "sentiment": fm.get("sentiment", "neutral"),
        "entities": entities,
        "topics": topics,
        "has_temporal": bool(fm.get("due_date")),
    }


# =============================================================================
# ONNX export
# =============================================================================

def export_onnx(
    model: BrainModel,
    tokenizer: Tokenizer,
    output_path: str = "checkpoints/brain/model.onnx",
    opset_version: int = 17,
) -> None:
    """Export trained model to ONNX format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.eval()
    dummy = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN))
    torch.onnx.export(
        model, dummy, output_path,
        opset_version=opset_version,
        input_names=["token_ids"],
        output_names=["classifier", "entities", "priority", "urgency",
                       "sentiment", "summary_scores", "topics"],
        dynamic_axes={"token_ids": {0: "batch"}},
    )


# =============================================================================
# History save/load
# =============================================================================

def save_history(history: List[Dict], path: str) -> None:
    """Save training history to JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def load_history(path: str) -> List[Dict]:
    """Load training history from JSON."""
    with open(path, "r") as f:
        return json.load(f)

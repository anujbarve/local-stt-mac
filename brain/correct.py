"""
brain.correct — Correction logging for continuous learning.

Manages a buffer of user corrections that accumulate until enough
are collected for retraining. Logs corrections to a JSONL file.
"""

import json
import os
from typing import Dict, List, Optional

from brain_contract import Correction, Understanding


# Minimum corrections before retraining is triggered
MIN_CORRECTIONS_FOR_RETRAIN = 20


class CorrectionLogger:
    """
    Manages the correction buffer for continuous learning.

    Usage:
        logger = CorrectionLogger()
        logger.log(correction)
        if logger.should_retrain():
            data = logger.get_training_data()
            trainer.retrain_all(data)
            logger.mark_applied(list(range(len(data))))
    """

    def __init__(
        self,
        buffer_path: str = "checkpoints/brain/corrections.jsonl",
    ):
        self.buffer_path = buffer_path
        self.buffer: List[Correction] = []
        self._load_buffer()

    def log(self, correction: Correction) -> None:
        """
        Add a correction to the buffer.

        Args:
            correction: A Correction object with original text,
                        predicted Understanding, and corrected Understanding.
        """
        self.buffer.append(correction)
        self._save_buffer()

    def get_training_data(
        self, min_corrections: int = MIN_CORRECTIONS_FOR_RETRAIN
    ) -> List[Dict]:
        """
        Get accumulated corrections as training data.

        Returns list of dicts suitable for BrainTrainer.
        """
        unapplied = [c for c in self.buffer if not c.applied]
        if len(unapplied) < min_corrections:
            return []

        training_data = []
        for correction in unapplied:
            # Use the corrected Understanding as the ground truth
            corrected = correction.corrected
            sample = {
                "text": correction.text,
                "type": corrected.type.value,
                "priority": corrected.priority,
                "urgency": corrected.urgency.value,
                "sentiment": corrected.sentiment.value,
                "entities": [
                    {"name": e.name, "type": e.entity_type.value}
                    for e in corrected.entities
                ],
                "action": corrected.action,
                "topics": corrected.topics,
                "has_temporal": corrected.temporal is not None
                    and corrected.temporal.has_date,
            }
            training_data.append(sample)

        return training_data

    def should_retrain(
        self,
        min_corrections: int = MIN_CORRECTIONS_FOR_RETRAIN,
    ) -> bool:
        """
        Check if enough corrections have accumulated.

        Returns True if there are >= min_corrections unapplied corrections.
        """
        unapplied = sum(1 for c in self.buffer if not c.applied)
        return unapplied >= min_corrections

    def mark_applied(self, indices: List[int]) -> None:
        """
        Mark corrections as used for retraining.

        Args:
            indices: List of buffer indices to mark as applied.
        """
        for idx in indices:
            if 0 <= idx < len(self.buffer):
                self.buffer[idx].applied = True
        self._save_buffer()

    def mark_all_applied(self) -> None:
        """Mark all corrections as applied."""
        for c in self.buffer:
            c.applied = True
        self._save_buffer()

    def stats(self) -> Dict[str, int]:
        """
        Get correction statistics.

        Returns dict with total, unapplied, applied counts.
        """
        total = len(self.buffer)
        applied = sum(1 for c in self.buffer if c.applied)
        return {
            "total": total,
            "applied": applied,
            "unapplied": total - applied,
        }

    def clear(self) -> None:
        """Clear the correction buffer."""
        self.buffer.clear()
        self._save_buffer()

    def _load_buffer(self) -> None:
        """Load corrections from JSONL file."""
        if not os.path.exists(self.buffer_path):
            return

        try:
            with open(self.buffer_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    predicted = Understanding.from_dict(data["predicted"])
                    corrected = Understanding.from_dict(data["corrected"])
                    correction = Correction(
                        text=data["text"],
                        predicted=predicted,
                        corrected=corrected,
                        timestamp=data.get("timestamp", ""),
                        applied=data.get("applied", False),
                    )
                    self.buffer.append(correction)
        except (json.JSONDecodeError, KeyError):
            # Corrupted file — start fresh
            self.buffer = []

    def _save_buffer(self) -> None:
        """Save corrections to JSONL file."""
        os.makedirs(os.path.dirname(self.buffer_path) or ".", exist_ok=True)
        with open(self.buffer_path, "w") as f:
            for c in self.buffer:
                data = c.to_dict()
                f.write(json.dumps(data) + "\n")

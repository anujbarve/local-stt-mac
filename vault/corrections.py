"""
Vault Corrections
=================

Store and manage user corrections in the vault.  Corrections are the
bridge between user feedback and model improvement — every correction
becomes training data for the brain.

Storage:
  - ``_corrections/corrections.jsonl`` — machine-readable buffer
  - ``_corrections/<month>.md`` — human-readable correction notes
  - ``_corrections/accuracy_log.json`` — accuracy trend over time
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from brain_contract import Understanding, NoteType, EntityType
from vault.config import VAULT_ROOT
from vault.schema import NoteSchema


class CorrectionVault:
    """Store and manage user corrections in the vault."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.corrections_dir = os.path.join(vault_root, "_corrections")
        self.buffer_path = os.path.join(self.corrections_dir, "corrections.jsonl")
        self.accuracy_log_path = os.path.join(self.corrections_dir, "accuracy_log.json")
        self._ensure_dir()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def log_correction(
        self,
        text: str,
        predicted: Understanding,
        corrected: Understanding,
    ) -> str:
        """Log a correction to the JSONL buffer.

        Returns the correction ID.
        """
        corr_id = self._generate_id()
        now = datetime.now(timezone.utc).isoformat()

        # Detect which fields changed
        changed = self._diff_fields(predicted, corrected)

        entry = {
            "id": corr_id,
            "timestamp": now,
            "text": text,
            "predicted": _understanding_to_dict(predicted),
            "corrected": _understanding_to_dict(corrected),
            "changed_fields": changed,
            "applied": False,
        }

        with open(self.buffer_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

        # Append to monthly markdown note
        self._append_correction_note(entry)

        return corr_id

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get all unapplied corrections."""
        return [c for c in self._read_all() if not c.get("applied")]

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all corrections."""
        return self._read_all()

    def get_stats(self) -> Dict[str, Any]:
        """Correction statistics.

        Returns dict with total, pending, applied, by_field breakdown.
        """
        all_corrs = self._read_all()
        pending = [c for c in all_corrs if not c.get("applied")]
        applied = [c for c in all_corrs if c.get("applied")]

        by_field: Dict[str, int] = {}
        for c in all_corrs:
            for field in c.get("changed_fields", []):
                by_field[field] = by_field.get(field, 0) + 1

        # Load accuracy trend
        accuracy_trend = self._load_accuracy_trend()

        return {
            "total": len(all_corrs),
            "pending": len(pending),
            "applied": len(applied),
            "by_field": by_field,
            "accuracy_trend": accuracy_trend,
            "buffer_path": self.buffer_path,
        }

    def export_training_data(self, output_path: str) -> int:
        """Export corrections as JSONL for brain retraining.

        Returns number of corrections exported.
        """
        pending = self.get_pending()
        if not pending:
            return 0

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for corr in pending:
                # Reformat for training: text → predicted vs corrected
                training_entry = {
                    "text": corr["text"],
                    "predicted": corr["predicted"],
                    "corrected": corr["corrected"],
                    "changed_fields": corr["changed_fields"],
                }
                fh.write(json.dumps(training_entry) + "\n")

        return len(pending)

    def mark_applied(self, correction_ids: List[str]) -> None:
        """Mark corrections as used for retraining."""
        ids_set = set(correction_ids)
        updated: List[Dict[str, Any]] = []

        for corr in self._read_all():
            if corr.get("id") in ids_set:
                corr["applied"] = True
            updated.append(corr)

        self._write_all(updated)

    def mark_all_applied(self) -> int:
        """Mark all pending corrections as applied. Returns count."""
        pending = self.get_pending()
        self.mark_applied([c["id"] for c in pending])
        return len(pending)

    def save_correction_note(self, correction: Dict[str, Any]) -> str:
        """Save a correction as a vault markdown note for human review.

        Returns file path of the created note.
        """
        note_dir = os.path.join(self.vault_root, "_corrections", "notes")
        os.makedirs(note_dir, exist_ok=True)

        corr_id = correction.get("id", "unknown")
        note_path = os.path.join(note_dir, f"{corr_id}.md")

        predicted = correction.get("predicted", {})
        corrected = correction.get("corrected", {})
        changed = correction.get("changed_fields", [])

        content = f"""---
id: {corr_id}
type: correction
created: {correction.get('timestamp', '')}
source: user_correction
---

# Correction: {correction.get('text', '')[:80]}

## Original Text
> {correction.get('text', '')}

## Brain Predicted
- **Type:** {predicted.get('type', '?')}
- **Priority:** {predicted.get('priority', '?')}
- **Action:** {predicted.get('action', 'none')}

## User Corrected
- **Type:** {corrected.get('type', '?')}
- **Priority:** {corrected.get('priority', '?')}
- **Action:** {corrected.get('action', 'none')}

## Changed Fields
{', '.join(changed) if changed else 'none'}

## Status
- **Applied:** {correction.get('applied', False)}
"""

        with open(note_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        return note_path

    def record_accuracy(self, accuracy: float) -> None:
        """Record a accuracy datapoint for trend tracking."""
        trend = self._load_accuracy_trend()
        trend.append(accuracy)
        # Keep last 100 datapoints
        trend = trend[-100:]
        with open(self.accuracy_log_path, "w", encoding="utf-8") as fh:
            json.dump(trend, fh)

    def clear(self) -> int:
        """Clear all corrections. Returns count removed."""
        count = len(self._read_all())
        if os.path.exists(self.buffer_path):
            os.unlink(self.buffer_path)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        """Create corrections directory if needed."""
        os.makedirs(self.corrections_dir, exist_ok=True)

    def _generate_id(self) -> str:
        """Generate unique correction ID."""
        now = datetime.now(timezone.utc)
        short = uuid.uuid4().hex[:6]
        return f"corr_{now.year}_{now.month:02d}_{now.day:02d}_{short}"

    def _read_all(self) -> List[Dict[str, Any]]:
        """Read all corrections from JSONL buffer."""
        if not os.path.exists(self.buffer_path):
            return []
        entries: List[Dict[str, Any]] = []
        with open(self.buffer_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def _write_all(self, entries: List[Dict[str, Any]]) -> None:
        """Rewrite the entire JSONL buffer."""
        with open(self.buffer_path, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")

    def _diff_fields(self, predicted: Understanding, corrected: Understanding) -> List[str]:
        """Detect which fields differ between predicted and corrected."""
        changed: List[str] = []

        if predicted.type != corrected.type:
            changed.append("type")
        if abs(predicted.priority - corrected.priority) > 0.05:
            changed.append("priority")
        if predicted.urgency != corrected.urgency:
            changed.append("urgency")
        if predicted.sentiment != corrected.sentiment:
            changed.append("sentiment")
        if (predicted.action or "") != (corrected.action or ""):
            changed.append("action")
        if (predicted.summary or "") != (corrected.summary or ""):
            changed.append("summary")

        # Compare entities
        pred_entities = {(e.name, e.entity_type.value) for e in predicted.entities}
        corr_entities = {(e.name, e.entity_type.value) for e in corrected.entities}
        if pred_entities != corr_entities:
            changed.append("entities")

        # Compare topics
        if set(predicted.topics) != set(corrected.topics):
            changed.append("topics")

        return changed

    def _append_correction_note(self, entry: Dict[str, Any]) -> None:
        """Append a correction to the monthly markdown note."""
        ts = entry.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            month_str = dt.strftime("%Y_%m")
        except (ValueError, AttributeError):
            month_str = datetime.now(timezone.utc).strftime("%Y_%m")

        note_path = os.path.join(self.corrections_dir, f"corrections_{month_str}.md")

        # Create header if file doesn't exist
        if not os.path.exists(note_path):
            header = f"""---
id: corrections_{month_str}
type: correction_log
created: {month_str}
---

# Corrections — {month_str.replace('_', '-')}

"""
            with open(note_path, "w", encoding="utf-8") as fh:
                fh.write(header)

        # Append entry
        predicted = entry.get("predicted", {})
        corrected = entry.get("corrected", {})
        changed = entry.get("changed_fields", [])

        block = f"""
## {entry.get('id', 'unknown')} — {entry.get('timestamp', '')}

**Text:** {entry.get('text', '')}

**Predicted:** type={predicted.get('type', '?')}, priority={predicted.get('priority', '?')}

**Corrected:** type={corrected.get('type', '?')}, priority={corrected.get('priority', '?')}

**Changed:** {', '.join(changed) if changed else 'none'}

---
"""
        with open(note_path, "a", encoding="utf-8") as fh:
            fh.write(block)

    def _load_accuracy_trend(self) -> List[float]:
        """Load accuracy trend from disk."""
        if not os.path.exists(self.accuracy_log_path):
            return []
        try:
            with open(self.accuracy_log_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _understanding_to_dict(u: Understanding) -> Dict[str, Any]:
    """Convert Understanding to a flat dict for JSON storage."""
    return {
        "text": u.text,
        "type": u.type.value if hasattr(u.type, "value") else str(u.type),
        "priority": u.priority,
        "urgency": u.urgency.value if hasattr(u.urgency, "value") else str(u.urgency),
        "sentiment": u.sentiment.value if hasattr(u.sentiment, "value") else str(u.sentiment),
        "confidence": u.confidence,
        "action": u.action,
        "summary": u.summary,
        "topics": u.topics,
        "tags": u.tags,
        "entities": [
            {"name": e.name, "entity_type": e.entity_type.value}
            for e in u.entities
        ],
    }

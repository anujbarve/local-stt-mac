"""
brain.interactive — Interactive correction loop.

After the brain produces an Understanding, this module shows it to the
user for review and correction. Corrections are logged for retraining.

Usage:
    from brain.interactive import InteractiveCorrector
    from brain.correct import CorrectionLogger

    corrector = InteractiveCorrector(CorrectionLogger())
    corrected, modified = corrector.present_and_correct(text, understanding)
"""

from typing import Optional, Tuple

from brain.correct import CorrectionLogger
from brain_contract import (
    Understanding,
    Correction,
    NoteType,
    Urgency,
    Sentiment,
)


class InteractiveCorrector:
    """Interactive correction loop for brain output.

    Shows the understanding to the user, lets them accept, skip,
    or edit individual fields. Corrections are logged for retraining.
    """

    def __init__(self, correction_logger: CorrectionLogger):
        self.logger = correction_logger

    def present_and_correct(
        self,
        text: str,
        understanding: Understanding,
    ) -> Tuple[Understanding, bool]:
        """Show the understanding to the user and collect corrections.

        Args:
            text: The original transcription text.
            understanding: The brain's understanding.

        Returns:
            (corrected_understanding, was_modified)
            If user skips, returns (original, False).
        """
        print(f"\n{'='*55}")
        print(f"  \U0001f4dd Transcription:")
        print(f"  \"{text}\"")
        print(f"{'='*55}")
        print(self._format_summary(understanding))
        print(f"{'='*55}")

        action = self._prompt_action()

        if action == "accept":
            return understanding, False

        if action == "skip":
            print("  \u23ed  Skipped.")
            return understanding, False

        if action == "edit":
            corrected = self._collect_corrections(understanding)
            if corrected is not None and corrected is not understanding:
                self._log_correction(text, understanding, corrected)
                return corrected, True
            return understanding, False

        # quit
        raise KeyboardInterrupt("User quit correction loop")

    def _format_summary(self, u: Understanding) -> str:
        """Format understanding for display."""
        lines = []

        # Type with confidence indicator
        conf = self._confidence_bar(u.confidence)
        lines.append(f"  \U0001f9e0 Type:        {u.type.value:<15} {conf}")

        # Priority
        pri_label = self._priority_label(u.priority)
        lines.append(f"     Priority:    {u.priority:.2f} ({pri_label})")

        # Urgency
        lines.append(f"     Urgency:     {u.urgency.value}")

        # Sentiment
        lines.append(f"     Sentiment:   {u.sentiment.value}")

        # Action
        if u.action:
            lines.append(f"     Action:      {u.action}")

        # Entities
        if u.entities:
            ent_strs = [f"{e.name} ({e.entity_type.value})" for e in u.entities]
            lines.append(f"     Entities:    {', '.join(ent_strs)}")

        # Topics
        if u.topics:
            lines.append(f"     Topics:      {', '.join(u.topics)}")

        # Temporal
        if u.temporal and u.temporal.has_date:
            rel = u.temporal.relative or "unknown"
            lines.append(f"     Temporal:    {rel}")

        return "\n".join(lines)

    def _confidence_bar(self, confidence: float) -> str:
        """Format confidence as a visual bar."""
        filled = int(confidence * 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"{bar} {confidence:.0%}"

    def _priority_label(self, priority: float) -> str:
        """Map priority score to a human-readable label."""
        if priority >= 0.9:
            return "critical"
        elif priority >= 0.7:
            return "high"
        elif priority >= 0.4:
            return "medium"
        elif priority >= 0.2:
            return "low"
        else:
            return "someday"

    def _prompt_action(self) -> str:
        """Prompt user for action: accept, edit, skip, or quit."""
        print()
        print("  [\u2713] Accept   [e] Edit   [s] Skip   [q] Quit")
        response = input("  > ").strip().lower()

        if response in ("", "y", "yes", "a", "\u2713"):
            return "accept"
        elif response in ("e", "edit", "c", "change"):
            return "edit"
        elif response in ("s", "skip", "n", "no"):
            return "skip"
        elif response in ("q", "quit", "x"):
            return "quit"
        else:
            # Default to accept
            return "accept"

    def _collect_corrections(
        self,
        original: Understanding,
    ) -> Optional[Understanding]:
        """Prompt user for field-by-field corrections.

        Returns corrected Understanding or None if no changes made.
        """
        corrected = Understanding(
            text=original.text,
            timestamp=original.timestamp,
            type=original.type,
            priority=original.priority,
            urgency=original.urgency,
            sentiment=original.sentiment,
            entities=list(original.entities),
            action=original.action,
            summary=original.summary,
            topics=list(original.topics),
            tags=list(original.tags),
            embedding=original.embedding,
            temporal=original.temporal,
        )

        changes_made = False

        # Edit type
        new_type = self._prompt_field(
            "Type", original.type.value,
            options=[t.value for t in NoteType],
        )
        if new_type:
            try:
                corrected.type = NoteType(new_type)
                changes_made = True
            except ValueError:
                print(f"    \u26a0 Invalid type '{new_type}', keeping original.")

        # Edit priority
        new_pri = self._prompt_field(
            "Priority", f"{original.priority:.2f}",
            hint="float 0.0-1.0",
        )
        if new_pri:
            try:
                corrected.priority = float(new_pri)
                changes_made = True
            except ValueError:
                print("    \u26a0 Invalid number, keeping original.")

        # Edit urgency
        new_urg = self._prompt_field(
            "Urgency", original.urgency.value,
            options=[u.value for u in Urgency],
        )
        if new_urg:
            try:
                corrected.urgency = Urgency(new_urg)
                changes_made = True
            except ValueError:
                print(f"    \u26a0 Invalid urgency '{new_urg}', keeping original.")

        # Edit sentiment
        new_sent = self._prompt_field(
            "Sentiment", original.sentiment.value,
            options=[s.value for s in Sentiment],
        )
        if new_sent:
            try:
                corrected.sentiment = Sentiment(new_sent)
                changes_made = True
            except ValueError:
                print(f"    \u26a0 Invalid sentiment '{new_sent}', keeping original.")

        # Edit action
        new_action = self._prompt_field(
            "Action", original.action or "(none)",
        )
        if new_action and new_action != "(none)":
            corrected.action = new_action
            changes_made = True

        if not changes_made:
            print("  No changes made.")
            return None

        return corrected

    def _prompt_field(
        self,
        name: str,
        current: str,
        options: Optional[list] = None,
        hint: Optional[str] = None,
    ) -> Optional[str]:
        """Prompt for a single field change.

        Returns new value string or None if unchanged.
        """
        prompt = f"    {name} [{current}]"
        if options:
            prompt += f"  ({', '.join(options[:5])}{'...' if len(options) > 5 else ''})"
        if hint:
            prompt += f"  # {hint}"

        new_val = input(prompt + ": ").strip()

        if not new_val:
            return None

        return new_val

    def _log_correction(
        self,
        text: str,
        predicted: Understanding,
        corrected: Understanding,
    ) -> None:
        """Log the correction to the buffer."""
        correction = Correction(
            text=text,
            predicted=predicted,
            corrected=corrected,
        )
        self.logger.log(correction)
        stats = self.logger.stats()
        print(f"  \U0001f4be Correction logged ({stats['unapplied']} pending)")

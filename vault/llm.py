"""
Vault LLM
=========

Local LLM wrapper for Gemma 3 1B via MLX.  Falls back gracefully when
MLX / the model is not installed.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from brain_contract import (
    Understanding,
    NoteType,
    EntityType,
    Urgency,
    Sentiment,
    Entity,
)
from vault.config import LLM_MODEL, LLM_MLX_MODEL_PATH, LLM_MAX_TOKENS, LLM_TEMPERATURE

try:
    import mlx_lm

    HAS_LLM = True
except ImportError:
    HAS_LLM = False


class LocalLLM:
    """Local LLM for deep reasoning."""

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        model_path: Optional[str] = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.model_name = model_name
        self.model_path = model_path or LLM_MLX_MODEL_PATH
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        """Load model on first use.  Returns True if successful."""
        if not HAS_LLM:
            return False
        if self._model is not None:
            return True
        try:
            self._model, self._tokenizer = mlx_lm.load(self.model_path)
            return True
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        """Check if LLM is loaded and available."""
        return self._ensure_loaded()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt."""
        if not self._ensure_loaded():
            return ""
        try:
            response = mlx_lm.generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens or self.max_tokens,
                temp=temperature or self.temperature,
            )
            return response
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # High-level methods
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        context: Optional[List[Dict[str, Any]]] = None,
    ) -> Understanding:
        """Full analysis of transcription text.

        Returns a filled ``Understanding`` object.
        """
        prompt = self._build_analysis_prompt(text, context)
        response = self.generate(prompt)
        return self._parse_analysis_response(response, text)

    def answer(
        self,
        query: str,
        vault_context: List[Dict[str, Any]],
    ) -> str:
        """Answer a question using vault context."""
        prompt = self._build_answer_prompt(query, vault_context)
        return self.generate(prompt)

    def find_connections(
        self,
        new_note: Dict[str, Any],
        related_notes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Discover non-obvious connections between notes.

        Returns ``[{note_id, connection_description}, ...]``.
        """
        if not related_notes:
            return []
        prompt = self._build_connection_prompt(new_note, related_notes)
        response = self.generate(prompt)
        return self._parse_connections(response, related_notes)

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_analysis_prompt(
        self, text: str, context: Optional[List[Dict[str, Any]]]
    ) -> str:
        ctx_block = ""
        if context:
            ctx_block = "\nContext from recent notes:\n"
            for c in context[:5]:
                ctx_block += f"- {c.get('body', '')[:120]}\n"

        return f"""Analyze this text and extract structured information.

Text: "{text}"
{ctx_block}
Extract the following as JSON:
{{
  "type": "task|idea|observation|learning|decision|meeting|reminder|person_info|preference|journal|other",
  "priority": 0.0-1.0,
  "urgency": "immediate|soon|later|someday",
  "sentiment": "positive|negative|neutral|mixed",
  "confidence": 0.0-1.0,
  "action": "extracted action phrase or null",
  "summary": "one-line summary",
  "topics": ["topic1", "topic2"],
  "entities": [{{"name": "...", "type": "person|project|tool|date|other"}}],
  "tags": ["tag1", "tag2"]
}}

Respond with ONLY the JSON, no other text."""

    def _build_answer_prompt(
        self, query: str, vault_context: List[Dict[str, Any]]
    ) -> str:
        notes_block = ""
        for i, note in enumerate(vault_context[:8], 1):
            body = note.get("body", "")[:200]
            notes_block += f"\n--- Note {i} (type: {note.get('type', '?')}) ---\n{body}\n"

        return f"""Based on the following notes from my knowledge base:
{notes_block}

Answer this question: {query}

Be specific. Cite which note the information comes from."""

    def _build_connection_prompt(
        self,
        new_note: Dict[str, Any],
        related_notes: List[Dict[str, Any]],
    ) -> str:
        new_body = new_note.get("body", "")[:200]
        candidates = ""
        for i, n in enumerate(related_notes[:5], 1):
            candidates += f"\n--- Candidate {i} ---\n{n.get('body', '')[:150]}\n"

        return f"""A new note says:
"{new_body}"

These existing notes might be related:
{candidates}

For each candidate that has a non-obvious connection, respond with JSON:
[{{"note_index": 1, "connection": "brief description"}}]
If none, respond with []"""

    # ------------------------------------------------------------------
    # Response parsers
    # ------------------------------------------------------------------

    def _parse_analysis_response(
        self, response: str, original_text: str
    ) -> Understanding:
        """Parse LLM JSON response into Understanding."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}
        except (json.JSONDecodeError, AttributeError):
            data = {}

        # Map entity types
        entities: List[Entity] = []
        for e in data.get("entities", []):
            try:
                etype = EntityType(e.get("type", "concept"))
            except (ValueError, KeyError):
                etype = EntityType.CONCEPT
            entities.append(Entity(
                name=e.get("name", ""),
                entity_type=etype,
            ))

        # Map note type
        try:
            note_type = NoteType(data.get("type", "observation"))
        except ValueError:
            note_type = NoteType.OBSERVATION

        # Map urgency
        try:
            urgency = Urgency(data.get("urgency", "later"))
        except ValueError:
            urgency = Urgency.LATER

        # Map sentiment
        try:
            sentiment = Sentiment(data.get("sentiment", "neutral"))
        except ValueError:
            sentiment = Sentiment.NEUTRAL

        return Understanding(
            text=original_text,
            type=note_type,
            priority=float(data.get("priority", 0.5)),
            urgency=urgency,
            sentiment=sentiment,
            confidence=float(data.get("confidence", 0.5)),
            entities=entities,
            action=data.get("action"),
            summary=data.get("summary"),
            topics=data.get("topics", []),
            tags=data.get("tags", []),
        )

    def _parse_connections(
        self, response: str, related_notes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse connection discovery response."""
        try:
            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                connections = json.loads(json_match.group())
                result: List[Dict[str, Any]] = []
                for conn in connections:
                    idx = conn.get("note_index", 1) - 1
                    if 0 <= idx < len(related_notes):
                        result.append({
                            "note_id": related_notes[idx].get("id", ""),
                            "connection": conn.get("connection", ""),
                        })
                return result
        except (json.JSONDecodeError, AttributeError):
            pass
        return []

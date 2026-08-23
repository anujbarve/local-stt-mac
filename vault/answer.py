"""
Vault Answerer
==============

End-to-end question answering: parse query → search vault → generate answer
with vault context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vault.config import SEARCH_TOP_K


class VaultAnswerer:
    """Answer questions using vault search + LLM."""

    def __init__(
        self,
        search: Any,
        reader: Any,
        llm: Any = None,
    ):
        self.search = search
        self.reader = reader
        self.llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, query: str, embedder: Any = None) -> Dict[str, Any]:
        """Full question answering pipeline.

        Returns:
            {
                "query": "original query",
                "answer": "text answer",
                "sources": [{"note_id": ..., "title": ..., "relevance": ...}],
                "strategy": "hybrid",
                "confidence": 0.82,
            }
        """
        # 1. Search vault
        query_result = self.search.search(
            query, strategy="hybrid", top_k=SEARCH_TOP_K, embedder=embedder
        )

        # 2. Format sources
        sources: List[Dict[str, Any]] = []
        for note, score in zip(query_result.notes, query_result.scores):
            title = note.get("body", "").split("\n")[0].lstrip("# ").strip()[:80]
            sources.append({
                "note_id": note.get("id", ""),
                "title": title or note.get("id", ""),
                "relevance": round(score, 3),
            })

        # 3. Generate answer
        answer_text = ""
        confidence = 0.5

        if self.llm and self.llm.is_available:
            answer_text = self.llm.answer(query, query_result.notes)
            confidence = 0.8
        elif query_result.notes:
            # Fallback: synthesize from top result
            top = query_result.notes[0]
            answer_text = self._format_fallback_answer(query, top)
            confidence = 0.6
        else:
            answer_text = "I couldn't find any relevant notes in the vault."
            confidence = 0.1

        return {
            "query": query,
            "answer": answer_text,
            "sources": sources,
            "strategy": query_result.strategy,
            "confidence": confidence,
        }

    def summarize(self, topic: str, embedder: Any = None) -> str:
        """Aggregate notes about a topic into a summary."""
        # Search for related notes
        result = self.search.search(
            topic, strategy="hybrid", top_k=20, embedder=embedder
        )

        if not result.notes:
            return f"No notes found about '{topic}'."

        if self.llm and self.llm.is_available:
            return self.llm.answer(f"Summarize what I know about {topic}", result.notes)

        # Fallback: concatenate key info
        lines = [f"## Summary: {topic}\n"]
        lines.append(f"Found {len(result.notes)} related notes.\n")

        for i, note in enumerate(result.notes[:5], 1):
            body_preview = note.get("body", "").split("\n")[0].lstrip("# ").strip()
            ntype = note.get("type", "note")
            lines.append(f"{i}. **[{ntype}]** {body_preview}")

        return "\n".join(lines)

    def explain(self, note_id: str) -> str:
        """Explain a decision or lesson with full context."""
        note = self.reader.read_by_id(note_id)
        if note is None:
            return f"Note '{note_id}' not found."

        body = note.get("body", "")
        ntype = note.get("type", "observation")

        if self.llm and self.llm.is_available:
            # Get related notes for context
            related = self.search.search(
                body[:100], strategy="hybrid", top_k=5
            )
            return self.llm.answer(
                f"Explain this {ntype} in detail:\n{body}", related.notes
            )

        # Fallback: return the note body with context
        backlinks = note.get("backlinks", [])
        explanation = f"### {ntype.title()}\n\n{body}\n"
        if backlinks:
            explanation += f"\n**Referenced by:** {', '.join(backlinks)}\n"
        return explanation

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _format_fallback_answer(self, query: str, note: Dict[str, Any]) -> str:
        """Format a simple answer from a single note."""
        ntype = note.get("type", "note")
        body = note.get("body", "").strip()
        # Take first meaningful paragraph
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        if paragraphs:
            main = paragraphs[0]
        else:
            main = body[:300]

        return f"Based on your {ntype} notes:\n\n{main}"

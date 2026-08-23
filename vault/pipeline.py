"""
Vault Pipeline
==============

End-to-end processor: ``Understanding`` → vault write + index + link.
This is the main entry point that ``transcribe.py`` will call at merge time.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from brain_contract import Understanding, NoteType
from vault.config import VAULT_ROOT, INDEX_DIR
from vault.reader import VaultReader
from vault.writer import VaultWriter
from vault.index import VaultIndex
from vault.graph import VaultGraph
from vault.linker import VaultLinker
from vault.search import VaultSearch
from vault.query import QueryParser
from vault.moc import VaultMOC
from vault.llm import LocalLLM
from vault.answer import VaultAnswerer


class VaultPipeline:
    """End-to-end: Understanding → vault write + index + link."""

    def __init__(
        self,
        vault_root: str = VAULT_ROOT,
        index_dir: str = INDEX_DIR,
    ):
        self.vault_root = vault_root
        self.index_dir = index_dir

        # Core components
        self.reader = VaultReader(vault_root)
        self.writer = VaultWriter(vault_root)
        self.index = VaultIndex(index_dir)
        self.graph = VaultGraph(vault_root)

        # Derived components
        self.linker = VaultLinker(self.reader, self.graph)
        self.search = VaultSearch(self.reader, self.index, self.graph)
        self.query_parser = QueryParser()
        self.moc = VaultMOC(vault_root, self.reader, self.graph)
        self.llm = LocalLLM()
        self.answerer = VaultAnswerer(self.search, self.reader, self.llm)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, understanding: Understanding, embedder: Any = None) -> str:
        """Full pipeline: Understanding → vault.

        Steps:
            1. Write note to vault (writer.py)
            2. Add to search index (index.py)
            3. Update graph (graph.py)
            4. Auto-link to related notes (linker.py)

        Returns file_path of created note.
        """
        # 1. Write note
        note_path = self.writer.write(understanding)

        # 2. Update index with embedding
        if understanding.embedding:
            note_id = understanding.note_id or ""
            self.index.add(note_id, note_path, understanding.embedding)

        # 3. Rebuild graph from vault (lightweight)
        self.graph.build()

        # 4. Auto-link
        if understanding.note_id:
            self.linker.link(understanding)

        return note_path

    def process_batch(
        self,
        understandings: List[Understanding],
        embedder: Any = None,
    ) -> List[str]:
        """Process multiple understandings efficiently."""
        paths: List[str] = []
        for u in understandings:
            path = self.process(u, embedder)
            paths.append(path)
        # Rebuild index once after batch
        if any(u.embedding for u in understandings):
            self.index.rebuild(self.reader, embedder)
        # Rebuild graph once
        self.graph.build()
        return paths

    def query(self, text: str, embedder: Any = None) -> Dict[str, Any]:
        """Search and answer a question about the vault."""
        return self.answerer.answer(text, embedder)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def rebuild_index(self, embedder: Any = None) -> int:
        """Rebuild the entire search index from vault files."""
        return self.index.rebuild(self.reader, embedder)

    def rebuild_graph(self) -> int:
        """Rebuild the entire graph from vault files."""
        return self.graph.build()

    def rebuild_mocs(self) -> int:
        """Regenerate all Maps of Content."""
        return self.moc.auto_generate_all()

    def stats(self) -> Dict[str, Any]:
        """Vault statistics."""
        return {
            "note_count": self.reader.count(),
            "index_size": self.index.size,
            "graph_nodes": self.graph.num_nodes,
            "graph_edges": self.graph.num_edges,
            "moc_count": len(self.moc.list_mocs()),
            "vault_root": self.vault_root,
        }

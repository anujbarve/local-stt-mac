"""
Vault Pipeline
==============

End-to-end processor: ``Understanding`` → vault write + index + link.
This is the main entry point that ``transcribe.py`` will call at merge time.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from brain_contract import Understanding, NoteType
from vault.config import VAULT_ROOT, INDEX_DIR, DRY_RUN

logger = logging.getLogger(__name__)
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
from vault.corrections import CorrectionVault


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
        self.correction_vault = CorrectionVault(vault_root)

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
    # Correction support
    # ------------------------------------------------------------------

    def process_with_correction(
        self,
        understanding: Understanding,
        corrected: Optional[Understanding] = None,
    ) -> tuple:
        """Process with optional pre-corrected understanding.

        Args:
            understanding: Original brain output
            corrected: Pre-corrected version (if user already corrected)

        Returns:
            (file_path, was_corrected)
        """
        if corrected is not None:
            # Log the correction
            self.correction_vault.log_correction(
                understanding.text, understanding, corrected
            )
            understanding = corrected

        path = self.process_safe(understanding)
        return path, corrected is not None

    def get_correction_stats(self) -> Dict[str, Any]:
        """Get correction statistics."""
        return self.correction_vault.get_stats()

    def export_corrections(self, output_path: str) -> int:
        """Export corrections for brain retraining."""
        return self.correction_vault.export_training_data(output_path)

    # ------------------------------------------------------------------
    # Safe methods (never raise)
    # ------------------------------------------------------------------

    def process_safe(self, understanding: Understanding) -> Optional[str]:
        """Process with full error handling. Never raises.

        Returns file_path on success, None on failure.
        """
        try:
            if DRY_RUN:
                plan = self.dry_run(understanding)
                return plan.get("file_path")
            return self.process(understanding)
        except Exception as e:
            logger.error("Pipeline error: %s", e)
            try:
                return self._fallback_write(understanding)
            except Exception as e2:
                logger.error("Fallback write also failed: %s", e2)
                return None

    def process_batch_safe(
        self,
        understandings: List[Understanding],
    ) -> List[Optional[str]]:
        """Process multiple understandings with error handling.

        Returns list of file_paths (None for failures).
        """
        paths: List[Optional[str]] = []
        for u in understandings:
            paths.append(self.process_safe(u))
        return paths

    def dry_run(self, understanding: Understanding) -> Dict[str, Any]:
        """Simulate processing without writing anything.

        Returns dict describing what WOULD happen.
        """
        from vault.schema import NoteSchema
        from brain_contract import NOTE_TYPE_DIRS

        subdir = NOTE_TYPE_DIRS.get(understanding.type, "daily")
        if understanding.action:
            slug = NoteSchema.slugify(understanding.action)
        elif understanding.topics:
            slug = NoteSchema.slugify(understanding.topics[0])
        else:
            slug = NoteSchema.slugify(understanding.text[:60])

        file_path = os.path.join(self.vault_root, subdir, f"{slug}.md")

        # Count entities that would be linked
        linked = []
        from brain_contract import ENTITY_PAGE_TYPES
        for e in understanding.entities:
            if e.entity_type in ENTITY_PAGE_TYPES:
                linked.append(NoteSchema.slugify(e.name))

        return {
            "file_path": file_path,
            "note_type": understanding.type.value,
            "entities_linked": linked,
            "index_added": understanding.embedding is not None,
            "would_write": not DRY_RUN,
        }

    def health_check(self) -> Dict[str, Any]:
        """Check vault health and return diagnostics."""
        import shutil as _shutil

        note_count = self.reader.count()
        index_size = self.index.size
        graph_edges = self.graph.num_edges
        orphans = len(self.graph.orphans())

        # Disk usage
        disk_mb = 0.0
        if os.path.isdir(self.vault_root):
            total = 0
            for dirpath, _dn, fnames in os.walk(self.vault_root):
                for f in fnames:
                    fp = os.path.join(dirpath, f)
                    try:
                        total += os.path.getsize(fp)
                    except OSError:
                        pass
            disk_mb = round(total / (1024 * 1024), 2)

        return {
            "vault_exists": os.path.isdir(self.vault_root),
            "note_count": note_count,
            "index_size": index_size,
            "graph_edges": graph_edges,
            "orphan_count": orphans,
            "broken_links": 0,  # requires schema.VaultValidator
            "disk_usage_mb": disk_mb,
        }

    def backup(self, note_path: str) -> Optional[str]:
        """Create a backup before overwriting a note.

        Returns backup path or None if backup disabled or file missing.
        """
        if not os.path.exists(note_path):
            return None
        return self.writer._create_backup(note_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback_write(self, understanding: Understanding) -> Optional[str]:
        """Emergency fallback: write raw text to daily journal."""
        try:
            return self.writer.write_daily_entry(understanding)
        except Exception as e:
            logger.error("Fallback write failed: %s", e)
            return None

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

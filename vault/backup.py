"""
Vault Backup & Restore
======================

Full and incremental backup system for the vault.  Creates tarball
archives with integrity verification and manifest tracking.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tarfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class VaultBackup:
    """Full and incremental backup for the vault."""

    def __init__(self, vault_root: str):
        self.vault_root = vault_root
        self.backup_dir = os.path.join(vault_root, "_backups")
        self.manifest_path = os.path.join(self.backup_dir, "manifest.json")
        self._ensure_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def full_backup(self, label: Optional[str] = None) -> str:
        """Create a full tarball backup of the vault.

        Returns the backup file path.
        """
        ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
        name = f"full_{ts}"
        if label:
            name += f"_{label}"
        backup_path = os.path.join(self.backup_dir, f"{name}.tar.gz")

        self._create_tarball(backup_path, self.vault_root)

        # Update manifest
        entry = self._make_manifest_entry(backup_path, "full")
        self._update_manifest(entry)

        return backup_path

    def incremental_backup(self) -> str:
        """Backup only files changed since last full backup.

        Returns the backup file path.
        """
        last_full = self._find_last_full_backup()
        since = 0
        if last_full:
            since = os.path.getmtime(last_full)

        ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"incr_{ts}.tar.gz")

        changed_files = self._find_changed_files(since)
        if not changed_files:
            # Nothing changed — create empty marker
            with open(backup_path, "w") as fh:
                fh.write("")
            entry = self._make_manifest_entry(backup_path, "incremental")
            entry["file_count"] = 0
            self._update_manifest(entry)
            return backup_path

        self._create_tarball_from_files(backup_path, changed_files)

        entry = self._make_manifest_entry(backup_path, "incremental")
        entry["file_count"] = len(changed_files)
        self._update_manifest(entry)

        return backup_path

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata."""
        manifest = self._load_manifest()
        return manifest.get("backups", [])

    def restore(self, backup_path: str, target: Optional[str] = None) -> bool:
        """Restore vault from backup.

        Args:
            backup_path: Path to backup tarball.
            target: Optional target directory (defaults to vault_root).

        Returns True on success.
        """
        if not os.path.exists(backup_path):
            return False

        target = target or self.vault_root

        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(path=target)
            return True
        except (tarfile.TarError, OSError):
            return False

    def verify(self, backup_path: str) -> Dict[str, Any]:
        """Verify backup integrity.

        Returns ``{"valid": True, "file_count": 42, "size_mb": 1.2}``.
        """
        if not os.path.exists(backup_path):
            return {"valid": False, "error": "file not found"}

        try:
            size_bytes = os.path.getsize(backup_path)
            with tarfile.open(backup_path, "r:gz") as tar:
                members = tar.getmembers()
                file_count = len([m for m in members if m.isfile()])

            return {
                "valid": True,
                "file_count": file_count,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            }
        except (tarfile.TarError, OSError) as e:
            return {"valid": False, "error": str(e)}

    def delete_backup(self, backup_path: str) -> bool:
        """Delete a specific backup."""
        if os.path.exists(backup_path):
            os.unlink(backup_path)
            # Remove from manifest
            manifest = self._load_manifest()
            manifest["backups"] = [
                b for b in manifest.get("backups", [])
                if b.get("path") != backup_path
            ]
            self._save_manifest(manifest)
            return True
        return False

    def cleanup(self, keep_last: int = 5) -> int:
        """Delete old backups, keeping the last N.

        Returns number of backups deleted.
        """
        backups = self.list_backups()
        if len(backups) <= keep_last:
            return 0

        # Sort by timestamp, delete oldest
        sorted_backups = sorted(backups, key=lambda b: b.get("timestamp", ""))
        to_delete = sorted_backups[: len(sorted_backups) - keep_last]
        count = 0
        for b in to_delete:
            path = b.get("path", "")
            if path and os.path.exists(path):
                os.unlink(path)
                count += 1

        # Rebuild manifest
        manifest = self._load_manifest()
        manifest["backups"] = [
            b for b in manifest.get("backups", [])
            if b.get("path") not in {d.get("path") for d in to_delete}
        ]
        self._save_manifest(manifest)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        os.makedirs(self.backup_dir, exist_ok=True)

    def _create_tarball(self, output_path: str, source_dir: str) -> None:
        """Create a tar.gz from an entire directory."""
        with tarfile.open(output_path, "w:gz") as tar:
            for dirpath, _dn, fnames in os.walk(source_dir):
                # Skip the _backups directory itself
                if "_backups" in dirpath:
                    continue
                for fname in fnames:
                    full_path = os.path.join(dirpath, fname)
                    arcname = os.path.relpath(full_path, os.path.dirname(source_dir))
                    tar.add(full_path, arcname=arcname)

    def _create_tarball_from_files(self, output_path: str, files: List[str]) -> None:
        """Create a tar.gz from a list of specific files."""
        with tarfile.open(output_path, "w:gz") as tar:
            for fpath in files:
                if os.path.exists(fpath):
                    arcname = os.path.relpath(fpath, os.path.dirname(self.vault_root))
                    tar.add(fpath, arcname=arcname)

    def _find_changed_files(self, since: float) -> List[str]:
        """Find files modified after *since* timestamp."""
        changed: List[str] = []
        for dirpath, _dn, fnames in os.walk(self.vault_root):
            if "_backups" in dirpath:
                continue
            for fname in fnames:
                fpath = os.path.join(dirpath, fname)
                try:
                    if os.path.getmtime(fpath) > since:
                        changed.append(fpath)
                except OSError:
                    pass
        return changed

    def _find_last_full_backup(self) -> Optional[str]:
        """Find the most recent full backup path."""
        backups = self.list_backups()
        full_backups = [b for b in backups if b.get("backup_type") == "full"]
        if not full_backups:
            return None
        full_backups.sort(key=lambda b: b.get("timestamp", ""), reverse=True)
        return full_backups[0].get("path")

    def _make_manifest_entry(self, path: str, backup_type: str) -> Dict[str, Any]:
        """Create a manifest entry for a new backup."""
        size_bytes = os.path.getsize(path) if os.path.exists(path) else 0
        return {
            "path": path,
            "backup_type": backup_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "size_bytes": size_bytes,
            "file_count": 0,
        }

    def _update_manifest(self, entry: Dict[str, Any]) -> None:
        """Append an entry to the manifest."""
        manifest = self._load_manifest()
        manifest.setdefault("backups", []).append(entry)
        self._save_manifest(manifest)

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        return {"backups": []}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

"""
Vault Configuration
===================

Centralised settings for the vault layer. All paths, thresholds, and
tuning knobs live here so every other module can ``from vault.config import …``.

Rules:
  - Instance B owns this file.
  - Never hard-code paths elsewhere; import from here.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VAULT_ROOT = os.getenv("VAULT_PATH", os.path.join(os.getcwd(), "vault"))
INDEX_DIR = os.getenv("INDEX_PATH", os.path.join(os.getcwd(), "vault_index"))
CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints", "vault")

# ---------------------------------------------------------------------------
# FAISS / embedding
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 384               # must match brain/embedder.py
FAISS_INDEX_TYPE = "flat"         # "flat" or "ivf" for larger vaults

# ---------------------------------------------------------------------------
# LLM (Gemma 3 1 B via MLX)
# ---------------------------------------------------------------------------
LLM_MODEL = "google/gemma-3-1b-it"
LLM_MLX_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "gemma-3-1b")
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
SEARCH_TOP_K = 10                 # max results to return
SEARCH_MIN_SCORE = 0.3            # minimum similarity score
SEARCH_EXPAND_THRESHOLD = 5       # if fewer results, expand search

# ---------------------------------------------------------------------------
# Auto-linking
# ---------------------------------------------------------------------------
LINK_CONFIDENCE_THRESHOLD = 0.75  # minimum entity match confidence
AUTO_LINK_MAX_DISTANCE = 3        # max graph hops for link suggestions

# ---------------------------------------------------------------------------
# Daily journal
# ---------------------------------------------------------------------------
DAILY_DIR = os.path.join(VAULT_ROOT, "daily")

# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
BRAIN_CHECKPOINT_DIR = os.getenv(
    "BRAIN_CHECKPOINT",
    os.path.join(os.getcwd(), "checkpoints", "brain"),
)

# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------
DRY_RUN = os.getenv("VAULT_DRY_RUN", "false").lower() == "true"
BACKUP_ENABLED = True
BACKUP_DIR = os.path.join(VAULT_ROOT, "_backups")
ATOMIC_WRITES = True

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
VAULT_VERSION = "0.2.0"

# ---------------------------------------------------------------------------
# Back-compat alias (some modules may reference this directly)
# ---------------------------------------------------------------------------
INDEX_PATH = INDEX_DIR

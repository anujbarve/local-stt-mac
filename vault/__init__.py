"""
vault – Storage, retrieval, and reasoning layer for the second brain.

Instance B owns this package.  It consumes ``Understanding`` objects from
``brain_contract`` and persists them as a linked markdown vault.

Sub-modules (foundation first, then advanced):

  config   – paths and settings
  schema   – frontmatter validation & ID generation
  templates – markdown renderers for every note type
  writer   – Understanding → vault file
  reader   – vault file → structured dict
  index    – FAISS vector index
  search   – hybrid retrieval (structural + semantic + graph)
  graph    – wikilink / backlink graph
  linker   – auto-linking via entity resolution
  moc      – Map of Content auto-generation
  query    – NL → structured query parser
  llm      – Gemma 3 1B local inference
  answer   – query answering with vault context
  pipeline – end-to-end: Understanding → vault write
"""

from vault.config import VAULT_ROOT, INDEX_DIR, EMBEDDING_DIM, VAULT_VERSION
from vault.pipeline import VaultPipeline
from vault.reader import VaultReader
from vault.search import VaultSearch
from vault.corrections import CorrectionVault

__version__ = VAULT_VERSION
__all__ = [
    "VAULT_ROOT", "INDEX_DIR", "EMBEDDING_DIM", "VAULT_VERSION",
    "VaultPipeline", "VaultReader", "VaultSearch", "CorrectionVault",
]

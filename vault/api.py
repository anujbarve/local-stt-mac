"""
Vault REST API
==============

FastAPI-based REST interface for external tools and dashboards.
Gracefully degrades when FastAPI/uvicorn are not installed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from vault.config import VAULT_ROOT

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


def create_app(vault_root: Optional[str] = None):
    """Create FastAPI app instance.

    Returns None if FastAPI is not installed.
    """
    if not HAS_FASTAPI:
        return None

    from vault.pipeline import VaultPipeline

    app = FastAPI(
        title="Second Brain API",
        version="0.3.0",
        description="REST API for the second brain knowledge system",
    )

    pipeline = VaultPipeline(vault_root=vault_root or VAULT_ROOT)

    class QueryRequest(BaseModel):
        text: str
        max_results: int = 10

    class NoteRequest(BaseModel):
        text: str
        type: str = "observation"
        priority: float = 0.5

    class CorrectionRequest(BaseModel):
        text: str
        predicted_type: str
        corrected_type: str
        predicted_priority: float = 0.5
        corrected_priority: float = 0.5

    @app.get("/health")
    def health():
        """Vault health check."""
        return pipeline.health_check()

    @app.get("/stats")
    def stats():
        """Vault statistics."""
        return pipeline.stats()

    @app.get("/corrections/stats")
    def correction_stats():
        """Correction statistics."""
        return pipeline.get_correction_stats()

    @app.post("/search")
    def search(req: QueryRequest):
        """Search the vault."""
        return pipeline.query(req.text)

    @app.post("/notes")
    def create_note(req: NoteRequest):
        """Create a new note."""
        from brain_contract import Understanding, NoteType
        try:
            note_type = NoteType(req.type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid type: {req.type}")

        u = Understanding(text=req.text, type=note_type, priority=req.priority)
        path = pipeline.process_safe(u)
        if path:
            return {"file_path": path, "status": "created"}
        raise HTTPException(status_code=500, detail="Failed to create note")

    @app.post("/corrections")
    def log_correction(req: CorrectionRequest):
        """Log a user correction."""
        from brain_contract import Understanding, NoteType
        try:
            predicted = Understanding(
                text=req.text,
                type=NoteType(req.predicted_type),
                priority=req.predicted_priority,
            )
            corrected = Understanding(
                text=req.text,
                type=NoteType(req.corrected_type),
                priority=req.corrected_priority,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        corr_id = pipeline.correction_vault.log_correction(
            req.text, predicted, corrected
        )
        return {"correction_id": corr_id, "status": "logged"}

    @app.get("/notes/{note_id}")
    def get_note(note_id: str):
        """Get a note by ID."""
        note = pipeline.reader.read_by_id(note_id)
        if note:
            return note
        raise HTTPException(status_code=404, detail="Note not found")

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    vault_root: Optional[str] = None,
) -> None:
    """Run the API server."""
    app = create_app(vault_root=vault_root)
    if app is None:
        print("Error: FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
        return

    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("Error: uvicorn not installed. Run: pip install uvicorn")

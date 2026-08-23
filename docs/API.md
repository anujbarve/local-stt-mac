# REST API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Set `VAULT_API_KEY` environment variable. Pass key via `X-API-Key` header:

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/health
```

If no key is configured, all endpoints are open.

---

## Endpoints

### `GET /health`

Health check.

**Response:**
```json
{
  "vault_exists": true,
  "note_count": 42,
  "index_size": 42,
  "graph_edges": 128,
  "orphan_count": 3,
  "broken_links": 0,
  "disk_usage_mb": 1.2
}
```

---

### `GET /stats`

Vault statistics.

**Response:**
```json
{
  "note_count": 42,
  "index_size": 42,
  "graph_nodes": 35,
  "graph_edges": 128,
  "moc_count": 5,
  "vault_root": "/app/vault_data"
}
```

---

### `POST /search`

Search the vault.

**Request:**
```json
{
  "text": "what projects am I working on?",
  "max_results": 10
}
```

**Response:**
```json
{
  "answer": "Based on your project notes...",
  "sources": [
    {"note_id": "note_2026_08_23_001", "title": "Voice Assistant", "relevance": 0.87}
  ],
  "strategy": "hybrid",
  "confidence": 0.8
}
```

---

### `POST /notes`

Create a new note.

**Request:**
```json
{
  "text": "Fix the VAD threshold by Friday",
  "type": "task",
  "priority": 0.8
}
```

**Response:**
```json
{
  "file_path": "/app/vault_data/tasks/fix-vad-threshold.md",
  "status": "created"
}
```

---

### `GET /notes/{note_id}`

Get a note by ID.

**Response:**
```json
{
  "id": "note_2026_08_23_001",
  "type": "task",
  "body": "# Fix VAD Threshold\n\n...",
  "tags": ["bug", "vad"]
}
```

---

### `POST /corrections`

Log a user correction.

**Request:**
```json
{
  "text": "Sarah told me her birthday",
  "predicted_type": "observation",
  "corrected_type": "person_info",
  "predicted_priority": 0.3,
  "corrected_priority": 0.5
}
```

**Response:**
```json
{
  "correction_id": "corr_2026_08_23_001",
  "status": "logged"
}
```

---

### `GET /corrections/stats`

Correction statistics.

**Response:**
```json
{
  "total": 45,
  "pending": 28,
  "applied": 17,
  "by_field": {"type": 12, "priority": 8}
}
```

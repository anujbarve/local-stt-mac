# Deployment Guide

## Quick Start (Docker)

```bash
# Build and run
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f vault
```

## Manual Setup

```bash
# 1. Clone the repository
git clone https://github.com/anujbarve/atlasai.git
cd atlasai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize the vault
python -c "from vault.pipeline import VaultPipeline; VaultPipeline()"

# 4. Start the API server
python -m vault.api serve

# 5. Or use the CLI
python scripts/unified_cli.py vault stats
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VAULT_PATH` | `./vault` | Vault root directory |
| `INDEX_PATH` | `./vault_index` | FAISS index directory |
| `VAULT_API_KEY` | _(none)_ | API authentication key |
| `VAULT_DRY_RUN` | `false` | Dry-run mode |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `BRAIN_CHECKPOINT` | `./checkpoints/brain` | Brain model path |

## Docker Configuration

```yaml
# docker-compose.yml override
services:
  vault:
    environment:
      - VAULT_API_KEY=your-secret-key
      - LOG_LEVEL=DEBUG
    volumes:
      - /host/vault:/app/vault_data
```

## Production Checklist

- [ ] Set `VAULT_API_KEY` for authentication
- [ ] Set `LOG_LEVEL=WARNING` (not DEBUG)
- [ ] Configure backup schedule (`vault/backup.py`)
- [ ] Set up monitoring (health endpoint at `/health`)
- [ ] Run `vault/migration.py validate_all()` before first use
- [ ] Configure notifications for deadline alerts

## Backup Strategy

```bash
# Full backup
python -c "from vault.backup import VaultBackup; VaultBackup('vault').full_backup('pre-deploy')"

# List backups
python -c "from vault.backup import VaultBackup; print(VaultBackup('vault').list_backups())"

# Restore
python -c "from vault.backup import VaultBackup; VaultBackup('vault').restore('path/to/backup.tar.gz')"
```

## Scaling Notes

- For vaults with 10K+ notes, use FAISS IVF index (`FAISS_INDEX_TYPE=ivf`)
- Enable batch indexing for bulk imports (`vault/index.py add_batch()`)
- Use `vault/resilience.py` circuit breakers for external service calls
- Monitor with `GET /health` and `GET /stats` endpoints

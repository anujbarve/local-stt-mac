# Usage Guide

## Voice Mode

```bash
# Start with brain understanding + vault writing
python transcribe.py --brain --vault

# With interactive corrections
python transcribe.py --brain --vault --interactive

# Verbose mode (shows brain reasoning)
python transcribe.py --brain --vault --verbose
```

## CLI Mode

```bash
# Quick inference
python scripts/brain_cli.py understand "Meeting with Sarah about the project"

# JSON output
python scripts/brain_cli.py understand "TODO: fix the VAD threshold" --json

# Interactive correction mode
python scripts/brain_cli.py interactive

# Batch processing
echo '["text1", "text2"]' > input.json
python scripts/brain_cli.py batch input.json --output results.json
```

## Training

```bash
# Generate synthetic data and train
python scripts/train.py --synthetic 15000 --epochs 20

# Quick test run
python scripts/train.py --quick

# Train from vault data
python scripts/train.py --vault vault/

# Export to ONNX
python scripts/train.py --export-onnx
```

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py

# With confusion matrix
python scripts/evaluate.py --confusion --per-type

# Save report
python scripts/evaluate.py --output reports/eval.json
```

## Search

```python
from vault.pipeline import VaultPipeline

pipeline = VaultPipeline()

# Ask a question
results = pipeline.query("What projects am I working on?")
print(results["answer"])

# Health check
health = pipeline.health_check()
print(f"Vault has {health['note_count']} notes, {health['graph_edges']} graph edges")

# Dry run (see what would happen without writing)
from brain_contract import Understanding, NoteType
u = Understanding(text="Fix the bug by Friday", type=NoteType.TASK)
plan = pipeline.dry_run(u)
print(f"Would write to: {plan['file_path']}")
```

#!/usr/bin/env python3
"""
Unified Brain + Vault CLI
==========================

Single entry point for all brain and vault operations.

Brain commands delegate to ``scripts/brain_cli.py`` via subprocess.
Vault commands use vault modules directly.

Usage::

    # Brain operations
    python scripts/unified_cli.py brain understand "Fix the VAD threshold"
    python scripts/unified_cli.py brain interactive
    python scripts/unified_cli.py brain evaluate --test-data data/test.json
    python scripts/unified_cli.py brain retrain --epochs 5

    # Vault operations
    python scripts/unified_cli.py vault stats
    python scripts/unified_cli.py vault search "what projects am I working on?"
    python scripts/unified_cli.py vault health
    python scripts/unified_cli.py vault process "I need to fix the VAD"

    # Corrections
    python scripts/unified_cli.py corrections stats
    python scripts/unified_cli.py corrections export --output corrections.jsonl
    python scripts/unified_cli.py corrections history
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# Brain CLI delegation
# ---------------------------------------------------------------------------

BRAIN_CLI = os.path.join(os.path.dirname(__file__), "brain_cli.py")


def _run_brain_cli(*args: str) -> int:
    """Run brain_cli.py as a subprocess."""
    cmd = [sys.executable, BRAIN_CLI] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0 and result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def cmd_brain_understand(args: argparse.Namespace) -> None:
    """Brain inference — delegates to brain_cli.py."""
    brain_args = ["understand", args.text]
    if args.json:
        brain_args.append("--json")
    if args.checkpoint:
        brain_args.extend(["--checkpoint", args.checkpoint])
    sys.exit(_run_brain_cli(*brain_args))


def cmd_brain_interactive(args: argparse.Namespace) -> None:
    """Interactive brain mode — delegates to brain_cli.py."""
    brain_args = ["interactive"]
    if args.checkpoint:
        brain_args.extend(["--checkpoint", args.checkpoint])
    sys.exit(_run_brain_cli(*brain_args))


def cmd_brain_evaluate(args: argparse.Namespace) -> None:
    """Brain evaluation — delegates to brain_cli.py."""
    brain_args = ["evaluate", "--test-data", args.test_data]
    if args.output:
        brain_args.extend(["--output", args.output])
    sys.exit(_run_brain_cli(*brain_args))


def cmd_brain_retrain(args: argparse.Namespace) -> None:
    """Brain retrain — delegates to brain_cli.py."""
    brain_args = ["retrain", "--epochs", str(args.epochs)]
    sys.exit(_run_brain_cli(*brain_args))


# ---------------------------------------------------------------------------
# Vault operations (direct)
# ---------------------------------------------------------------------------

def cmd_vault_stats(args: argparse.Namespace) -> None:
    """Vault statistics."""
    from vault.pipeline import VaultPipeline
    pipeline = VaultPipeline(vault_root=args.vault_path)
    stats = pipeline.stats()
    print(json.dumps(stats, indent=2))


def cmd_vault_search(args: argparse.Namespace) -> None:
    """Search the vault."""
    from vault.pipeline import VaultPipeline
    pipeline = VaultPipeline(vault_root=args.vault_path)
    result = pipeline.query(args.query)
    print(f"Answer: {result.get('answer', 'No answer')}")
    sources = result.get("sources", [])
    if sources:
        print(f"Sources ({len(sources)}):")
        for s in sources[:5]:
            print(f"  - [{s.get('relevance', 0):.2f}] {s.get('title', s.get('note_id', '?'))}")


def cmd_vault_health(args: argparse.Namespace) -> None:
    """Vault health check."""
    from vault.pipeline import VaultPipeline
    pipeline = VaultPipeline(vault_root=args.vault_path)
    health = pipeline.health_check()
    print(json.dumps(health, indent=2))


def cmd_vault_process(args: argparse.Namespace) -> None:
    """Process text through vault pipeline."""
    from brain_contract import Understanding, NoteType
    from vault.pipeline import VaultPipeline

    pipeline = VaultPipeline(vault_root=args.vault_path)
    u = Understanding(text=args.text, type=NoteType.OBSERVATION)
    path = pipeline.process_safe(u)
    if path:
        print(f"[Vault] Saved: {path}")
    else:
        print("[Vault] Failed to save")


# ---------------------------------------------------------------------------
# Corrections (direct)
# ---------------------------------------------------------------------------

def cmd_corrections_stats(args: argparse.Namespace) -> None:
    """Correction statistics."""
    from vault.corrections import CorrectionVault
    vault = CorrectionVault(vault_root=args.vault_path)
    stats = vault.get_stats()
    print(json.dumps(stats, indent=2))


def cmd_corrections_export(args: argparse.Namespace) -> None:
    """Export corrections for retraining."""
    from vault.corrections import CorrectionVault
    vault = CorrectionVault(vault_root=args.vault_path)
    count = vault.export_training_data(args.output)
    print(f"Exported {count} corrections → {args.output}")


def cmd_corrections_history(args: argparse.Namespace) -> None:
    """Show recent corrections."""
    from vault.corrections import CorrectionVault
    vault = CorrectionVault(vault_root=args.vault_path)
    all_corrs = vault.get_all()
    recent = all_corrs[-10:] if all_corrs else []

    if not recent:
        print("No corrections logged yet.")
        return

    print(f"Last {len(recent)} corrections (of {len(all_corrs)} total):\n")
    for c in recent:
        status = "✓ applied" if c.get("applied") else "⏳ pending"
        changed = ", ".join(c.get("changed_fields", []))
        text_preview = c.get("text", "")[:60]
        print(f"  [{status}] {c.get('id', '?')}")
        print(f"    \"{text_preview}\"")
        print(f"    changed: {changed or 'none'}")
        print()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def cmd_pipeline(args: argparse.Namespace) -> None:
    """End-to-end pipeline — write text to vault."""
    from brain_contract import Understanding, NoteType
    from vault.pipeline import VaultPipeline

    pipeline = VaultPipeline(vault_root=args.vault_path)
    u = Understanding(text=args.text, type=NoteType.OBSERVATION)
    path = pipeline.process_safe(u)
    if path:
        print(f"[Vault] Saved: {path}")
    else:
        print("[Vault] Failed to save")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unified-cli",
        description="Unified Brain + Vault CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--vault-path",
        default=os.getenv("VAULT_PATH", "vault"),
        help="Vault root directory",
    )

    sub = parser.add_subparsers(dest="command")

    # ── Brain ──
    brain_sub = sub.add_parser("brain", help="Brain operations (delegates to brain_cli.py)")
    brain_cmds = brain_sub.add_subparsers(dest="brain_command")

    p = brain_cmds.add_parser("understand", help="Classify text")
    p.add_argument("text")
    p.add_argument("--json", action="store_true")
    p.add_argument("--checkpoint", default="checkpoints/brain")

    brain_cmds.add_parser("interactive", help="Interactive correction mode")

    p = brain_cmds.add_parser("evaluate", help="Evaluate model")
    p.add_argument("--test-data", required=True)
    p.add_argument("--output")

    p = brain_cmds.add_parser("retrain", help="Retrain from corrections")
    p.add_argument("--epochs", type=int, default=5)

    # ── Vault ──
    vault_sub = sub.add_parser("vault", help="Vault operations")
    vault_cmds = vault_sub.add_subparsers(dest="vault_command")

    vault_cmds.add_parser("stats", help="Vault statistics")
    vault_cmds.add_parser("health", help="Health check")

    p = vault_cmds.add_parser("search", help="Search the vault")
    p.add_argument("query")

    p = vault_cmds.add_parser("process", help="Write text to vault")
    p.add_argument("text")

    # ── Corrections ──
    corr_sub = sub.add_parser("corrections", help="Correction management")
    corr_cmds = corr_sub.add_subparsers(dest="corr_command")

    corr_cmds.add_parser("stats", help="Correction statistics")

    p = corr_cmds.add_parser("export", help="Export for retraining")
    p.add_argument("--output", default="corrections.jsonl")

    corr_cmds.add_parser("history", help="Show recent corrections")

    # ── Pipeline ──
    pipe_sub = sub.add_parser("pipeline", help="End-to-end pipeline")
    pipe_sub.add_argument("text")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    dispatch = {
        ("brain", "understand"): cmd_brain_understand,
        ("brain", "interactive"): cmd_brain_interactive,
        ("brain", "evaluate"): cmd_brain_evaluate,
        ("brain", "retrain"): cmd_brain_retrain,
        ("vault", "stats"): cmd_vault_stats,
        ("vault", "search"): cmd_vault_search,
        ("vault", "health"): cmd_vault_health,
        ("vault", "process"): cmd_vault_process,
        ("corrections", "stats"): cmd_corrections_stats,
        ("corrections", "export"): cmd_corrections_export,
        ("corrections", "history"): cmd_corrections_history,
        ("pipeline", None): cmd_pipeline,
    }

    sub_cmd = (
        getattr(args, "brain_command", None)
        or getattr(args, "vault_command", None)
        or getattr(args, "corr_command", None)
    )
    key = (args.command, sub_cmd)
    handler = dispatch.get(key)

    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

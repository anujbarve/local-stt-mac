"""
scripts/train.py — CLI training pipeline.

Usage:
    python scripts/train.py [OPTIONS]

Options:
    --synthetic N       Number of synthetic samples (default: 15000)
    --data PATH         Load data from JSON file instead of generating
    --vault PATH        Load real data from vault directory
    --epochs N          Training epochs (default: 20)
    --batch-size N      Batch size (default: 32)
    --lr FLOAT          Learning rate (default: 1e-4)
    --checkpoint-dir    Checkpoint directory (default: checkpoints/brain)
    --resume PATH       Resume training from checkpoint
    --scheduler TYPE    LR scheduler: cosine, plateau, step, none
    --patience N        Early stopping patience (default: 5)
    --augment           Enable data augmentation
    --quick             Quick test mode (1000 samples, 5 epochs)
    --export-onnx       Export to ONNX after training
    --verbose           Print detailed metrics per epoch
"""

import argparse
import json
import os
import random
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    args = parse_args()

    print("=" * 60)
    print("Brain Training Pipeline")
    print("=" * 60)

    # 1. Generate or load data
    print("\n[1/5] Preparing data...")
    if args.data:
        data = load_json_data(args.data)
        print(f"  Loaded {len(data)} samples from {args.data}")
    elif args.vault:
        data = load_vault_data(args.vault)
        print(f"  Loaded {len(data)} notes from {args.vault}")
    else:
        n = 1000 if args.quick else args.synthetic
        data = generate_synthetic_data(n)
        print(f"  Generated {len(data)} synthetic samples")

    if not data:
        print("  ERROR: No training data available")
        sys.exit(1)

    # 2. Split train/val
    print(f"\n[2/5] Splitting data ({len(data)} samples)...")
    train_data, val_data = split_data(data, val_ratio=0.15)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # 3. Train tokenizer
    print("\n[3/5] Training tokenizer...")
    tokenizer = train_tokenizer(train_data)
    print(f"  Vocab size: {tokenizer.vocab_size_actual()}")

    # 4. Train model
    epochs = 5 if args.quick else args.epochs
    print(f"\n[4/5] Training model ({epochs} epochs)...")

    from brain.model import BrainModel
    from brain.train import BrainTrainer, save_history, export_onnx

    model = BrainModel(vocab_size=tokenizer.vocab_size_actual())
    trainer = BrainTrainer(
        model, tokenizer,
        lr=args.lr,
        scheduler_type=args.scheduler,
        patience=args.patience,
    )

    if args.resume:
        epoch, metrics = trainer.load_checkpoint(args.resume)
        print(f"  Resumed from epoch {epoch}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start = time.time()
    history = trainer.train(
        train_data, val_data,
        epochs=epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose,
    )
    elapsed = time.time() - start

    # 5. Save final model and tokenizer
    print("\n[5/5] Saving...")
    final_dir = args.checkpoint_dir
    model.save(os.path.join(final_dir, "model.pt"))
    tokenizer.save(os.path.join(final_dir, "tokenizer.json"))
    save_history(history, os.path.join(final_dir, "history.json"))

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Epochs: {len(history)}")
    if history:
        print(f"  Final train loss: {history[-1].get('train_total', 0):.4f}")
        print(f"  Final val loss: {history[-1].get('val_total', 0):.4f}")
        print(f"  Best val accuracy: {history[-1].get('train_classifier_acc', 0):.1%}")
    print(f"  Checkpoints: {final_dir}/")

    # Export ONNX if requested
    if args.export_onnx:
        print("\n  Exporting to ONNX...")
        export_onnx(model, tokenizer, os.path.join(final_dir, "model.onnx"))
        print(f"  Done: {final_dir}/model.onnx")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the brain model")
    parser.add_argument("--synthetic", type=int, default=15000)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--vault", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/brain")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default="cosine",
                       choices=["cosine", "plateau", "step", "none"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def generate_synthetic_data(n: int) -> list:
    from brain.data import SyntheticDataGenerator
    return SyntheticDataGenerator().generate(n)


def load_json_data(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def load_vault_data(vault_root: str) -> list:
    from brain.train import load_vault_data as _load
    return _load(vault_root)


def train_tokenizer(data: list):
    from brain.tokenizer import Tokenizer
    from brain.config import VOCAB_SIZE
    tok = Tokenizer(vocab_size=VOCAB_SIZE)
    tok.fit([s["text"] for s in data])
    return tok


def split_data(data: list, val_ratio: float = 0.15):
    shuffled = data.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_ratio))
    return shuffled[:split], shuffled[split:]


if __name__ == "__main__":
    main()

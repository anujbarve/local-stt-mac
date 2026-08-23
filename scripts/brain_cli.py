"""
scripts/brain_cli.py — Unified brain CLI.

Commands:
    understand TEXT      Quick inference on text
    interactive         Interactive correction mode
    batch INPUT OUTPUT  Batch process text file
    train [OPTIONS]     Train the model
    evaluate [OPTIONS]  Evaluate the model
    stats               Show model stats and correction buffer info
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Brain CLI")
    subparsers = parser.add_subparsers(dest="command")

    # understand
    p_understand = subparsers.add_parser("understand")
    p_understand.add_argument("text")
    p_understand.add_argument("--explain", action="store_true")
    p_understand.add_argument("--json", action="store_true")
    p_understand.add_argument("--checkpoint", default="checkpoints/brain")

    # interactive
    p_interactive = subparsers.add_parser("interactive")
    p_interactive.add_argument("--checkpoint", default="checkpoints/brain")

    # batch
    p_batch = subparsers.add_parser("batch")
    p_batch.add_argument("input")
    p_batch.add_argument("--output", default="output.json")
    p_batch.add_argument("--checkpoint", default="checkpoints/brain")

    # train
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--synthetic", type=int, default=15000)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--quick", action="store_true")

    # evaluate
    p_evaluate = subparsers.add_parser("evaluate")
    p_evaluate.add_argument("--confusion", action="store_true")
    p_evaluate.add_argument("--per-type", action="store_true")
    p_evaluate.add_argument("--quick", action="store_true")

    # stats
    subparsers.add_parser("stats")

    args = parser.parse_args()

    if args.command == "understand":
        cmd_understand(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "stats":
        cmd_stats()
    else:
        parser.print_help()


def cmd_understand(args):
    from brain.understand import BrainEngine
    engine = BrainEngine.load(args.checkpoint)
    understanding = engine.understand(args.text)

    if args.json:
        print(json.dumps(understanding.to_dict(), indent=2))
    elif args.explain:
        print(engine.explain(understanding))
    else:
        print(f"Type:       {understanding.type.value}")
        print(f"Priority:   {understanding.priority:.2f}")
        print(f"Urgency:    {understanding.urgency.value}")
        print(f"Sentiment:  {understanding.sentiment.value}")
        print(f"Confidence: {understanding.confidence:.2f}")
        if understanding.entities:
            print(f"Entities:   {[(e.name, e.entity_type.value) for e in understanding.entities]}")
        if understanding.action:
            print(f"Action:     {understanding.action}")
        if understanding.topics:
            print(f"Topics:     {understanding.topics}")


def cmd_interactive(args):
    from brain.understand import BrainEngine
    from brain.correct import CorrectionLogger

    engine = BrainEngine.load(args.checkpoint)
    logger = CorrectionLogger()

    print("Brain Interactive Mode (Ctrl+C to exit)")
    print("=" * 50)

    try:
        while True:
            text = input("\nYou: ").strip()
            if not text:
                continue

            understanding = engine.understand(text)
            print(f"\nBrain: {engine.explain(understanding)}")

            response = input("\nCorrect? [Y/n/edit]: ").strip().lower()
            if response == "n":
                print("Skipped.")
                continue
            elif response == "edit":
                corrected = logger.interactive_correct(text, understanding)
                if corrected:
                    print(f"Corrected: {corrected.type.value}, priority={corrected.priority:.2f}")
            else:
                print("Accepted.")

    except KeyboardInterrupt:
        print("\nExiting.")
        print(f"Buffer: {logger.stats()}")


def cmd_batch(args):
    from brain.understand import BrainEngine

    engine = BrainEngine.load(args.checkpoint)

    with open(args.input) as f:
        texts = json.load(f)

    results = []
    for text in texts:
        u = engine.understand(text)
        results.append(u.to_dict())

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} texts → {args.output}")


def cmd_train(args):
    import subprocess
    cmd = [sys.executable, "scripts/train.py"]
    if args.synthetic:
        cmd.extend(["--synthetic", str(args.synthetic)])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.quick:
        cmd.append("--quick")
    subprocess.run(cmd)


def cmd_evaluate(args):
    import subprocess
    cmd = [sys.executable, "scripts/evaluate.py"]
    if args.confusion:
        cmd.append("--confusion")
    if args.per_type:
        cmd.append("--per-type")
    if args.quick:
        cmd.append("--quick")
    subprocess.run(cmd)


def cmd_stats():
    from brain.correct import CorrectionLogger
    logger = CorrectionLogger()
    stats = logger.stats()

    print("Brain Stats")
    print("=" * 40)
    print(f"Corrections:  {stats['total']} total, {stats['unapplied']} unapplied")
    print(f"Model:        checkpoints/brain/model.pt")
    print(f"Tokenizer:    checkpoints/brain/tokenizer.json")


if __name__ == "__main__":
    main()

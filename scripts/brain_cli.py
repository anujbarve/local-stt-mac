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
    p_evaluate.add_argument("--test-data")
    p_evaluate.add_argument("--confusion", action="store_true")
    p_evaluate.add_argument("--per-type", action="store_true")
    p_evaluate.add_argument("--output")
    p_evaluate.add_argument("--checkpoint", default="checkpoints/brain")

    # export
    p_export = subparsers.add_parser("export")
    p_export.add_argument("--onnx", action="store_true", help="Export to ONNX format")
    p_export.add_argument("--all", action="store_true", help="Export ONNX + TorchScript + model card")
    p_export.add_argument("--output", help="Output path")
    p_export.add_argument("--versions", action="store_true", help="List model versions")
    p_export.add_argument("--checkpoint", default="checkpoints/brain")

    # corrections
    p_corrections = subparsers.add_parser("corrections")
    p_corrections.add_argument("--stats", action="store_true")
    p_corrections.add_argument("--export")
    p_corrections.add_argument("--import-file", dest="import_file")

    # retrain
    p_retrain = subparsers.add_parser("retrain")
    p_retrain.add_argument("--epochs", type=int, default=5)
    p_retrain.add_argument("--checkpoint", default="checkpoints/brain")

    # stats
    subparsers.add_parser("stats")

    args = parser.parse_args()

    handlers = {
        "understand": cmd_understand,
        "interactive": cmd_interactive,
        "batch": cmd_batch,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "corrections": cmd_corrections,
        "retrain": cmd_retrain,
        "stats": cmd_stats,
    }
    handler = handlers.get(args.command)
    if handler:
        handler(args)
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
    """Evaluate model on test data."""
    from brain.understand import BrainEngine
    from brain.evaluate import BrainEvaluator

    engine = BrainEngine.load(args.checkpoint)
    evaluator = BrainEvaluator(engine.model, engine.tokenizer)

    if args.test_data:
        with open(args.test_data) as f:
            test_data = json.load(f)

        report = evaluator.evaluate(test_data)
        print(json.dumps(report, indent=2, default=str))

        if args.output:
            evaluator.generate_report(test_data, args.output)
            print(f"Report saved to {args.output}")
    else:
        # No test data — show basic model info
        print("Brain Evaluator")
        print("=" * 40)
        print(f"Model loaded: {args.checkpoint}")
        print(f"Usage: python scripts/brain_cli.py evaluate --test-data data/test.json")


def cmd_export(args):
    """Export model to ONNX, TorchScript, or list versions."""
    if args.versions:
        from brain.model import BrainModel
        versions = BrainModel.load_versions(args.checkpoint)
        if not versions:
            print(f"No model versions found in {args.checkpoint}")
        else:
            print(json.dumps(versions, indent=2))
        return

    from brain.understand import BrainEngine

    engine = BrainEngine.load(args.checkpoint)

    if args.all:
        # Full export: ONNX + TorchScript + model card
        from brain.export import ModelExporter
        exporter = ModelExporter(engine)
        output_dir = args.output or os.path.join(args.checkpoint, "exports")
        artifacts = exporter.export_all(output_dir)
        print("Exported artifacts:")
        for art_type, path in artifacts.items():
            print(f"  {art_type}: {path}")
    elif args.onnx:
        output = args.output or os.path.join(args.checkpoint, "model.onnx")
        from brain.export import ModelExporter
        exporter = ModelExporter(engine)
        exporter.export_onnx(output)
        print(f"ONNX exported to {output}")
    else:
        # Default: ONNX only
        output = args.output or os.path.join(args.checkpoint, "model.onnx")
        from brain.export import ModelExporter
        exporter = ModelExporter(engine)
        exporter.export_onnx(output)
        print(f"Model exported to {output}")


def cmd_corrections(args):
    """Manage corrections."""
    from brain.correct import CorrectionLogger
    logger = CorrectionLogger()

    if args.stats:
        stats = logger.get_stats()
        print(json.dumps(stats, indent=2))
    elif args.export:
        count = logger.export_training_data(args.export)
        print(f"Exported {count} corrections to {args.export}")
    elif args.import_file:
        count = logger.import_training_data(args.import_file)
        print(f"Imported {count} corrections from {args.import_file}")
    else:
        # Default: show stats
        stats = logger.stats()
        print(json.dumps(stats, indent=2))


def cmd_retrain(args):
    """Retrain from corrections."""
    from brain.correct import CorrectionLogger
    from brain.distill import DistillationPipeline
    from brain.understand import BrainEngine

    engine = BrainEngine.load(args.checkpoint)
    logger = CorrectionLogger()
    distiller = DistillationPipeline(engine, llm=None, correction_logger=logger)

    if distiller.should_retrain():
        metrics = distiller.retrain(epochs=args.epochs)
        print(json.dumps(metrics, indent=2))
        engine.save(args.checkpoint)
        print("Model retrained and saved.")
    else:
        stats = logger.stats()
        print(f"Not enough corrections for retraining ({stats['unapplied']} pending, need {20})")


def cmd_stats():
    from brain.correct import CorrectionLogger
    logger = CorrectionLogger()
    stats = logger.get_stats()

    print("Brain Stats")
    print("=" * 40)
    print(f"Corrections:  {stats['total_corrections']} total, {stats['pending']} pending")
    if stats['by_field']:
        print(f"By field:     {stats['by_field']}")
    print(f"Model:        checkpoints/brain/model.pt")
    print(f"Tokenizer:    checkpoints/brain/tokenizer.json")


if __name__ == "__main__":
    main()

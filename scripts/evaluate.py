"""
scripts/evaluate.py — CLI evaluation script.

Usage:
    python scripts/evaluate.py [OPTIONS]

Options:
    --checkpoint PATH    Model checkpoint (default: checkpoints/brain/model.pt)
    --data PATH          Test data JSON file
    --vault PATH         Evaluate against vault notes
    --output PATH        Save metrics report to JSON
    --confusion          Print confusion matrices
    --per-type           Print per-type accuracy breakdown
    --quick              Quick smoke test (100 samples)
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    args = parse_args()

    print("=" * 60)
    print("Brain Evaluation")
    print("=" * 60)

    # 1. Load model
    print("\n[1/3] Loading model...")
    from brain.understand import BrainEngine
    ckpt_dir = os.path.dirname(args.checkpoint)
    if not ckpt_dir:
        ckpt_dir = "checkpoints/brain"
    engine = BrainEngine.load(ckpt_dir)
    print(f"  Model loaded from {ckpt_dir}")

    # 2. Load test data
    print("\n[2/3] Loading test data...")
    if args.data:
        data = load_json_data(args.data)
    elif args.vault:
        from brain.train import load_vault_data
        data = load_vault_data(args.vault)
    else:
        from brain.data import SyntheticDataGenerator
        data = SyntheticDataGenerator(seed=99).generate(1000)

    if args.quick:
        data = data[:100]

    print(f"  {len(data)} test samples")

    # 3. Evaluate
    print("\n[3/3] Evaluating...")
    metrics = evaluate(engine, data)

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Type accuracy:      {metrics['type_acc']:.1%}")
    print(f"  Priority MAE:       {metrics['priority_mae']:.3f}")
    print(f"  Urgency accuracy:   {metrics['urgency_acc']:.1%}")
    print(f"  Sentiment accuracy: {metrics['sentiment_acc']:.1%}")
    print(f"  Entity F1:          {metrics['entity_f1']:.1%}")
    print(f"  Topic F1:           {metrics['topic_f1']:.1%}")

    if args.confusion:
        print_confusion_matrix(metrics["confusion"])

    if args.per_type:
        print_per_type_accuracy(metrics["per_type"])

    if args.output:
        save_report(metrics, args.output)
        print(f"\n  Report saved to {args.output}")


def evaluate(engine, data):
    """Run full evaluation and return metrics dict."""
    from collections import defaultdict

    type_correct = 0
    urgency_correct = 0
    sentiment_correct = 0
    priority_errors = []
    entity_tp = entity_fp = entity_fn = 0
    topic_tp = topic_fp = topic_fn = 0
    confusion = defaultdict(lambda: defaultdict(int))
    per_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for sample in data:
        text = sample["text"]
        expected_type = sample.get("type", "observation")

        understanding = engine.understand(text)
        predicted_type = understanding.type.value

        # Type
        if predicted_type == expected_type:
            type_correct += 1
            per_type[expected_type]["correct"] += 1
        per_type[expected_type]["total"] += 1
        confusion[expected_type][predicted_type] += 1

        # Priority
        expected_priority = float(sample.get("priority", 0.5))
        priority_errors.append(abs(understanding.priority - expected_priority))

        # Urgency
        if understanding.urgency.value == sample.get("urgency", "later"):
            urgency_correct += 1

        # Sentiment
        if understanding.sentiment.value == sample.get("sentiment", "neutral"):
            sentiment_correct += 1

        # Entity F1
        expected_ents = {(e["name"].lower(), e["type"]) for e in sample.get("entities", [])}
        predicted_ents = {(e.name.lower(), e.entity_type.value) for e in understanding.entities}
        entity_tp += len(expected_ents & predicted_ents)
        entity_fp += len(predicted_ents - expected_ents)
        entity_fn += len(expected_ents - predicted_ents)

        # Topic F1
        expected_topics = set(sample.get("topics", []))
        predicted_topics = set(understanding.topics)
        topic_tp += len(expected_topics & predicted_topics)
        topic_fp += len(predicted_topics - expected_topics)
        topic_fn += len(expected_topics - predicted_topics)

    n = len(data)
    ep = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0
    er = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0
    ef1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0
    tp = topic_tp / (topic_tp + topic_fp) if (topic_tp + topic_fp) > 0 else 0
    tr = topic_tp / (topic_tp + topic_fn) if (topic_tp + topic_fn) > 0 else 0
    tf1 = 2 * tp * tr / (tp + tr) if (tp + tr) > 0 else 0

    return {
        "type_acc": type_correct / n if n > 0 else 0,
        "priority_mae": sum(priority_errors) / len(priority_errors) if priority_errors else 0,
        "urgency_acc": urgency_correct / n if n > 0 else 0,
        "sentiment_acc": sentiment_correct / n if n > 0 else 0,
        "entity_f1": ef1,
        "entity_precision": ep,
        "entity_recall": er,
        "topic_f1": tf1,
        "topic_precision": tp,
        "topic_recall": tr,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "per_type": {k: dict(v) for k, v in per_type.items()},
        "num_samples": n,
    }


def print_confusion_matrix(confusion):
    types = sorted(confusion.keys())
    print("\nConfusion Matrix (rows=expected, cols=predicted):")
    print(f"{'':>15}", end="")
    for t in types[:8]:
        print(f"{t[:8]:>9}", end="")
    print()
    for row in types[:8]:
        print(f"{row[:14]:>15}", end="")
        for col in types[:8]:
            print(f"{confusion[row].get(col, 0):>9}", end="")
        print()


def print_per_type_accuracy(per_type):
    print("\nPer-Type Accuracy:")
    for type_name, counts in sorted(per_type.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {type_name:20s} {acc:.1%} ({counts['correct']}/{counts['total']})")


def save_report(metrics, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def load_json_data(path):
    with open(path) as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the brain model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/brain/model.pt")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--vault", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--confusion", action="store_true")
    parser.add_argument("--per-type", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

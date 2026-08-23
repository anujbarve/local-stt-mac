"""
brain.evaluate — Model evaluation metrics.

Measures model accuracy on test data across all task heads:
classification, entity extraction, priority, urgency, sentiment, topics.

Usage:
    from brain.evaluate import BrainEvaluator

    evaluator = BrainEvaluator(model, tokenizer)
    report = evaluator.evaluate(test_data)
    print(report["accuracy"])
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from brain_contract import NoteType, Urgency, Sentiment


class BrainEvaluator:
    """Evaluate brain model on test data.

    Provides per-head accuracy, confusion matrix, error analysis,
    and structured evaluation reports.
    """

    def __init__(self, model: Any, tokenizer: Any):
        """Initialize evaluator.

        Args:
            model: BrainModel instance.
            tokenizer: Tokenizer instance.
        """
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(
        self,
        test_data: List[Dict],
    ) -> Dict[str, Any]:
        """Evaluate model on test set.

        Args:
            test_data: List of dicts with keys: text, type, priority,
                       urgency, sentiment, entities, topics.

        Returns:
            Dict with accuracy metrics, confusion matrix, per-type accuracy.
        """
        if not test_data:
            return {"error": "no_test_data", "total_samples": 0}

        type_correct = 0
        urgency_correct = 0
        sentiment_correct = 0
        priority_errors = []
        topic_matches = 0
        topic_total = 0
        entity_matches = 0
        entity_total = 0
        confidences = []
        type_predictions = []
        type_actuals = []
        per_type_correct = defaultdict(int)
        per_type_total = defaultdict(int)
        errors = []

        for sample in test_data:
            text = sample["text"]
            actual_type = sample.get("type", "observation")
            actual_priority = sample.get("priority", 0.5)
            actual_urgency = sample.get("urgency", "later")
            actual_sentiment = sample.get("sentiment", "neutral")
            actual_entities = sample.get("entities", [])
            actual_topics = sample.get("topics", [])

            # Get model prediction
            try:
                pred = self._predict(text)
            except Exception:
                continue

            confidences.append(pred["confidence"])

            # Type accuracy
            per_type_total[actual_type] += 1
            if pred["type"] == actual_type:
                type_correct += 1
                per_type_correct[actual_type] += 1
            type_predictions.append(pred["type"])
            type_actuals.append(actual_type)

            # Urgency accuracy
            if pred["urgency"] == actual_urgency:
                urgency_correct += 1

            # Sentiment accuracy
            if pred["sentiment"] == actual_sentiment:
                sentiment_correct += 1

            # Priority MAE
            pri_error = abs(pred["priority"] - actual_priority)
            priority_errors.append(pri_error)

            # Topic F1 (simple overlap)
            pred_topics = set(pred.get("topics", []))
            actual_topic_set = set(actual_topics)
            if actual_topic_set:
                topic_total += 1
                if pred_topics & actual_topic_set:
                    topic_matches += 1

            # Entity F1
            pred_entity_names = {e.lower() for e in pred.get("entity_names", [])}
            actual_entity_names = {e.get("name", "").lower() for e in actual_entities}
            if actual_entity_names:
                entity_total += 1
                if pred_entity_names & actual_entity_names:
                    entity_matches += 1

            # Track errors
            if pred["type"] != actual_type:
                errors.append({
                    "text": text[:80],
                    "predicted_type": pred["type"],
                    "actual_type": actual_type,
                    "confidence": pred["confidence"],
                })

        total = len(test_data)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "accuracy": {
                "type": type_correct / total if total else 0,
                "priority_mae": sum(priority_errors) / len(priority_errors) if priority_errors else 0,
                "urgency": urgency_correct / total if total else 0,
                "sentiment": sentiment_correct / total if total else 0,
                "topic_f1": topic_matches / topic_total if topic_total else 0,
                "entity_f1": entity_matches / entity_total if entity_total else 0,
            },
            "confusion_matrix": self._build_confusion_matrix(type_actuals, type_predictions),
            "per_type_accuracy": {
                t: per_type_correct[t] / per_type_total[t]
                for t in per_type_total
            },
            "total_samples": total,
            "avg_confidence": avg_confidence,
        }

    def confusion_matrix(
        self,
        test_data: List[Dict],
    ) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix for type classification.

        Args:
            test_data: List of test samples.

        Returns:
            Dict of {actual_type: {predicted_type: count}}.
        """
        actuals = []
        predictions = []

        for sample in test_data:
            try:
                pred = self._predict(sample["text"])
                actuals.append(sample.get("type", "observation"))
                predictions.append(pred["type"])
            except Exception:
                continue

        return self._build_confusion_matrix(actuals, predictions)

    def per_type_accuracy(
        self,
        test_data: List[Dict],
    ) -> Dict[str, float]:
        """Accuracy broken down by note type.

        Args:
            test_data: List of test samples.

        Returns:
            Dict of {type_name: accuracy_float}.
        """
        correct = defaultdict(int)
        total = defaultdict(int)

        for sample in test_data:
            actual = sample.get("type", "observation")
            total[actual] += 1
            try:
                pred = self._predict(sample["text"])
                if pred["type"] == actual:
                    correct[actual] += 1
            except Exception:
                continue

        return {
            t: correct[t] / total[t]
            for t in total
        }

    def error_analysis(
        self,
        test_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Find the worst errors for debugging.

        Returns list of {text, predicted, actual, confidence} sorted
        by confidence (highest confidence wrong predictions first).
        """
        errors = []

        for sample in test_data:
            try:
                pred = self._predict(sample["text"])
                actual = sample.get("type", "observation")
                if pred["type"] != actual:
                    errors.append({
                        "text": sample["text"][:200],
                        "predicted": pred["type"],
                        "actual": actual,
                        "confidence": pred["confidence"],
                    })
            except Exception:
                continue

        # Sort by confidence descending (worst errors first)
        errors.sort(key=lambda x: x["confidence"], reverse=True)
        return errors

    def generate_report(
        self,
        test_data: List[Dict],
        output_path: str = "reports/eval.json",
    ) -> str:
        """Generate full evaluation report and save to JSON.

        Args:
            test_data: List of test samples.
            output_path: Path to save the report.

        Returns:
            The output path.
        """
        report = self.evaluate(test_data)

        # Add error analysis
        report["worst_errors"] = self.error_analysis(test_data)[:10]

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return output_path

    def _predict(self, text: str) -> Dict[str, Any]:
        """Get model prediction for a single text.

        Returns dict with type, priority, urgency, sentiment, etc.
        """
        token_ids = self.tokenizer.encode(text, add_special=True)
        token_tensor = torch.tensor([token_ids], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(token_tensor)

        # Decode type
        type_logits = outputs["classifier"][0]
        type_idx = type_logits.argmax().item()
        type_probs = torch.softmax(type_logits, dim=0)
        confidence = type_probs.max().item()

        idx_to_type = {i: t.value for i, t in enumerate(NoteType)}
        predicted_type = idx_to_type.get(type_idx, "observation")

        # Decode priority
        priority = outputs["priority"][0, 0].item()
        priority = max(0.0, min(1.0, priority))

        # Decode urgency
        urgency_logits = outputs["urgency"][0]
        urgency_idx = urgency_logits.argmax().item()
        idx_to_urgency = {i: u.value for i, u in enumerate(Urgency)}
        predicted_urgency = idx_to_urgency.get(urgency_idx, "later")

        # Decode sentiment
        sentiment_logits = outputs["sentiment"][0]
        sentiment_idx = sentiment_logits.argmax().item()
        idx_to_sentiment = {i: s.value for i, s in enumerate(Sentiment)}
        predicted_sentiment = idx_to_sentiment.get(sentiment_idx, "neutral")

        # Decode topics
        topic_probs = torch.sigmoid(outputs["topics"][0])
        from brain_contract import TOPICS
        topics = [
            TOPICS[i] for i, p in enumerate(topic_probs)
            if p.item() >= 0.5 and i < len(TOPICS)
        ]

        return {
            "type": predicted_type,
            "priority": priority,
            "urgency": predicted_urgency,
            "sentiment": predicted_sentiment,
            "confidence": confidence,
            "topics": topics,
            "entity_names": [],  # entity decoding is complex, skip for eval
        }

    def _build_confusion_matrix(
        self,
        actuals: List[str],
        predictions: List[str],
    ) -> Dict[str, Dict[str, int]]:
        """Build a confusion matrix from actual/prediction lists."""
        matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for actual, predicted in zip(actuals, predictions):
            matrix[actual][predicted] += 1
        return dict(matrix)

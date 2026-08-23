"""Tests for brain/evaluate.py — BrainEvaluator."""

import json
import os

import pytest

from brain.evaluate import BrainEvaluator
from brain.model import BrainModel
from brain.tokenizer import Tokenizer


@pytest.fixture
def evaluator():
    model = BrainModel(vocab_size=1000)
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.fit(["placeholder text for vocab"])
    return BrainEvaluator(model, tokenizer)


def _make_sample(text="test text", note_type="observation", priority=0.5, urgency="later", sentiment="neutral", topics=None, entities=None):
    return {
        "text": text,
        "type": note_type,
        "priority": priority,
        "urgency": urgency,
        "sentiment": sentiment,
        "topics": topics or [],
        "entities": entities or [],
    }


class TestBrainEvaluator:
    def test_evaluate_returns_metrics(self, evaluator):
        """evaluate returns expected keys."""
        data = [_make_sample() for _ in range(10)]
        result = evaluator.evaluate(data)

        assert "accuracy" in result
        assert "total_samples" in result
        assert "avg_confidence" in result
        assert "confusion_matrix" in result
        assert result["total_samples"] == 10

    def test_evaluate_accuracy_keys(self, evaluator):
        """accuracy dict has expected fields."""
        data = [_make_sample() for _ in range(5)]
        result = evaluator.evaluate(data)
        acc = result["accuracy"]

        assert "type" in acc
        assert "priority_mae" in acc
        assert "urgency" in acc
        assert "sentiment" in acc
        assert "topic_f1" in acc
        assert "entity_f1" in acc

    def test_evaluate_empty_data(self, evaluator):
        """evaluate with empty data returns error."""
        result = evaluator.evaluate([])
        assert "error" in result
        assert result["total_samples"] == 0

    def test_confusion_matrix(self, evaluator):
        """confusion_matrix returns nested dict."""
        data = [_make_sample(note_type="task"), _make_sample(note_type="idea")]
        result = evaluator.confusion_matrix(data)
        assert isinstance(result, dict)

    def test_per_type_accuracy(self, evaluator):
        """per_type_accuracy returns dict of floats."""
        data = [
            _make_sample(note_type="task"),
            _make_sample(note_type="idea"),
            _make_sample(note_type="task"),
        ]
        result = evaluator.per_type_accuracy(data)
        assert isinstance(result, dict)
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_error_analysis(self, evaluator):
        """error_analysis returns list of error dicts."""
        data = [_make_sample(text=f"sample {i}") for i in range(5)]
        result = evaluator.error_analysis(data)
        assert isinstance(result, list)
        for err in result:
            assert "text" in err
            assert "predicted" in err
            assert "actual" in err
            assert "confidence" in err

    def test_generate_report(self, evaluator, tmp_path):
        """generate_report creates a JSON file."""
        data = [_make_sample() for _ in range(5)]
        output_path = str(tmp_path / "eval.json")
        result_path = evaluator.generate_report(data, output_path)

        assert os.path.exists(result_path)
        with open(result_path) as f:
            report = json.load(f)
        assert "accuracy" in report
        assert "worst_errors" in report

    def test_predict_returns_dict(self, evaluator):
        """_predict returns expected keys."""
        result = evaluator._predict("hello world")
        assert "type" in result
        assert "priority" in result
        assert "urgency" in result
        assert "sentiment" in result
        assert "confidence" in result
        assert "topics" in result

    def test_predict_priority_in_range(self, evaluator):
        """_predict priority is between 0 and 1."""
        result = evaluator._predict("some text")
        assert 0.0 <= result["priority"] <= 1.0

    def test_predict_confidence_in_range(self, evaluator):
        """_predict confidence is between 0 and 1."""
        result = evaluator._predict("some text")
        assert 0.0 <= result["confidence"] <= 1.0

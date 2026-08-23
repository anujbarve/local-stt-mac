"""Tests for vault/corrections.py"""

import json
import os

import pytest

from brain_contract import Understanding, NoteType, Entity, EntityType, Urgency
from vault.corrections import CorrectionVault


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def corrections(tmp_path):
    vault_root = str(tmp_path / "vault")
    os.makedirs(vault_root, exist_ok=True)
    return CorrectionVault(vault_root=vault_root)


def _make_observation():
    return Understanding(
        text="The sky is blue today",
        type=NoteType.OBSERVATION,
        priority=0.2,
        topics=["nature"],
    )


def _make_task():
    return Understanding(
        text="Fix the VAD threshold by Friday",
        type=NoteType.TASK,
        priority=0.82,
        urgency=Urgency.SOON,
        action="fix VAD threshold",
        topics=["technology"],
    )


def _make_person():
    return Understanding(
        text="Sarah told me her birthday is March 15th",
        type=NoteType.PERSON_INFO,
        priority=0.5,
        entities=[
            Entity(name="Sarah", entity_type=EntityType.PERSON),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCorrectionVault:
    def test_log_correction(self, corrections):
        predicted = _make_observation()
        corrected = _make_person()
        corr_id = corrections.log_correction(
            "Sarah told me her birthday", predicted, corrected
        )
        assert corr_id.startswith("corr_")
        assert os.path.exists(corrections.buffer_path)

    def test_log_correction_content(self, corrections):
        predicted = _make_observation()
        corrected = _make_person()
        corrections.log_correction("test text", predicted, corrected)

        with open(corrections.buffer_path) as f:
            line = f.readline()
        entry = json.loads(line)
        assert entry["text"] == "test text"
        assert entry["predicted"]["type"] == "observation"
        assert entry["corrected"]["type"] == "person_info"
        assert "type" in entry["changed_fields"]

    def test_get_pending(self, corrections):
        predicted = _make_observation()
        corrected = _make_task()
        corrections.log_correction("text1", predicted, corrected)
        corrections.log_correction("text2", predicted, corrected)

        pending = corrections.get_pending()
        assert len(pending) == 2

    def test_get_all(self, corrections):
        predicted = _make_observation()
        corrected = _make_task()
        corrections.log_correction("text1", predicted, corrected)
        corrections.log_correction("text2", predicted, corrected)

        all_corrs = corrections.get_all()
        assert len(all_corrs) == 2

    def test_get_stats_empty(self, corrections):
        stats = corrections.get_stats()
        assert stats["total"] == 0
        assert stats["pending"] == 0
        assert stats["applied"] == 0

    def test_get_stats_with_corrections(self, corrections):
        predicted = _make_observation()
        corrected = _make_task()
        corrections.log_correction("text1", predicted, corrected)

        stats = corrections.get_stats()
        assert stats["total"] == 1
        assert stats["pending"] == 1
        assert stats["applied"] == 0
        assert "type" in stats["by_field"]

    def test_export_training_data(self, corrections, tmp_path):
        predicted = _make_observation()
        corrected = _make_task()
        corrections.log_correction("text1", predicted, corrected)

        output = str(tmp_path / "training.jsonl")
        count = corrections.export_training_data(output)
        assert count == 1
        assert os.path.exists(output)

        with open(output) as f:
            line = f.readline()
        entry = json.loads(line)
        assert entry["text"] == "text1"

    def test_export_empty(self, corrections, tmp_path):
        output = str(tmp_path / "training.jsonl")
        count = corrections.export_training_data(output)
        assert count == 0

    def test_mark_applied(self, corrections):
        predicted = _make_observation()
        corrected = _make_task()
        corr_id = corrections.log_correction("text1", predicted, corrected)

        corrections.mark_applied([corr_id])
        pending = corrections.get_pending()
        assert len(pending) == 0

    def test_mark_all_applied(self, corrections):
        predicted = _make_observation()
        corrected = _make_task()
        corrections.log_correction("text1", predicted, corrected)
        corrections.log_correction("text2", predicted, corrected)

        count = corrections.mark_all_applied()
        assert count == 2
        assert len(corrections.get_pending()) == 0

    def test_save_correction_note(self, corrections):
        predicted = _make_observation()
        corrected = _make_person()
        corr_id = corrections.log_correction(
            "Sarah birthday", predicted, corrected
        )

        all_corrs = corrections.get_all()
        note_path = corrections.save_correction_note(all_corrs[0])
        assert os.path.exists(note_path)
        assert note_path.endswith(".md")

    def test_record_accuracy(self, corrections):
        corrections.record_accuracy(0.72)
        corrections.record_accuracy(0.78)
        trend = corrections._load_accuracy_trend()
        assert len(trend) == 2
        assert trend[0] == 0.72

    def test_clear(self, corrections):
        predicted = _make_observation()
        corrected = _make_task()
        corrections.log_correction("text1", predicted, corrected)
        corrections.log_correction("text2", predicted, corrected)

        count = corrections.clear()
        assert count == 2
        assert len(corrections.get_all()) == 0

    def test_diff_fields_type_change(self, corrections):
        a = _make_observation()
        b = _make_task()
        changed = corrections._diff_fields(a, b)
        assert "type" in changed

    def test_diff_fields_priority_change(self, corrections):
        a = Understanding(text="test", type=NoteType.TASK, priority=0.2)
        b = Understanding(text="test", type=NoteType.TASK, priority=0.9)
        changed = corrections._diff_fields(a, b)
        assert "priority" in changed

    def test_diff_fields_same(self, corrections):
        a = _make_task()
        b = Understanding(
            text=a.text, type=a.type, priority=a.priority,
            urgency=a.urgency, sentiment=a.sentiment,
            action=a.action, summary=a.summary,
            topics=list(a.topics), tags=list(a.tags),
        )
        changed = corrections._diff_fields(a, b)
        assert changed == []

    def test_monthly_note_created(self, corrections):
        predicted = _make_observation()
        corrected = _make_task()
        corrections.log_correction("text1", predicted, corrected)

        # Monthly note should exist
        notes = [f for f in os.listdir(corrections.corrections_dir) if f.startswith("corrections_")]
        assert len(notes) >= 1

    def test_multiple_corrections_append(self, corrections):
        predicted = _make_observation()
        corrected = _make_task()
        corrections.log_correction("text1", predicted, corrected)
        corrections.log_correction("text2", predicted, corrected)

        all_corrs = corrections.get_all()
        assert len(all_corrs) == 2

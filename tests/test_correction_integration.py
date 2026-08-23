"""Integration tests for correction system."""

import json
import os

import pytest

from brain_contract import Understanding, NoteType, Entity, EntityType, Urgency
from vault.pipeline import VaultPipeline
from vault.corrections import CorrectionVault


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(tmp_path):
    vault_root = str(tmp_path / "vault")
    index_dir = str(tmp_path / "index")
    return VaultPipeline(vault_root=vault_root, index_dir=index_dir)


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

class TestCorrectionRoundtrip:
    def test_log_and_read_back(self, pipeline):
        """Log a correction, read it back."""
        predicted = _make_observation()
        corrected = _make_task()

        corr_id = pipeline.correction_vault.log_correction(
            predicted.text, predicted, corrected
        )
        assert corr_id.startswith("corr_")

        all_corrs = pipeline.correction_vault.get_all()
        assert len(all_corrs) == 1
        assert all_corrs[0]["id"] == corr_id

    def test_correction_preserves_text(self, pipeline):
        """Correction stores original text."""
        predicted = _make_observation()
        corrected = _make_task()

        pipeline.correction_vault.log_correction(
            "original text here", predicted, corrected
        )

        all_corrs = pipeline.correction_vault.get_all()
        assert all_corrs[0]["text"] == "original text here"


class TestPipelineWithCorrection:
    def test_process_with_correction(self, pipeline):
        """process_with_correction logs correction and writes corrected note."""
        predicted = _make_observation()
        corrected = _make_task()

        path, was_corrected = pipeline.process_with_correction(
            predicted, corrected=corrected
        )

        assert path is not None
        assert was_corrected is True
        assert os.path.exists(path)

        # Correction should be logged
        stats = pipeline.get_correction_stats()
        assert stats["total"] == 1

    def test_process_without_correction(self, pipeline):
        """process_with_correction without correction just processes normally."""
        u = _make_task()

        path, was_corrected = pipeline.process_with_correction(u)

        assert path is not None
        assert was_corrected is False

        # No correction logged
        stats = pipeline.get_correction_stats()
        assert stats["total"] == 0

    def test_correction_stats_after_pipeline(self, pipeline):
        """Multiple corrections accumulate stats."""
        p1 = _make_observation()
        c1 = _make_task()
        pipeline.process_with_correction(p1, corrected=c1)

        p2 = _make_observation()
        c2 = _make_person()
        pipeline.process_with_correction(p2, corrected=c2)

        stats = pipeline.get_correction_stats()
        assert stats["total"] == 2
        assert stats["pending"] == 2


class TestExportImportCycle:
    def test_export_after_correction(self, pipeline, tmp_path):
        """Export corrections after logging."""
        predicted = _make_observation()
        corrected = _make_task()
        pipeline.process_with_correction(predicted, corrected=corrected)

        output = str(tmp_path / "export.jsonl")
        count = pipeline.export_corrections(output)
        assert count == 1
        assert os.path.exists(output)

    def test_mark_applied_prevents_re_export(self, pipeline, tmp_path):
        """After marking applied, export returns 0."""
        predicted = _make_observation()
        corrected = _make_task()
        pipeline.process_with_correction(predicted, corrected=corrected)

        # Mark applied
        all_corrs = pipeline.correction_vault.get_all()
        pipeline.correction_vault.mark_applied([all_corrs[0]["id"]])

        output = str(tmp_path / "export2.jsonl")
        count = pipeline.export_corrections(output)
        assert count == 0


class TestCorrectionIntegration:
    def test_full_cycle(self, pipeline, tmp_path):
        """Full cycle: correct → log → export → mark applied."""
        # 1. Process with correction
        p1 = _make_observation()
        c1 = _make_task()
        path1, _ = pipeline.process_with_correction(p1, corrected=c1)
        assert path1 is not None

        p2 = _make_observation()
        c2 = _make_person()
        path2, _ = pipeline.process_with_correction(p2, corrected=c2)
        assert path2 is not None

        # 2. Check stats
        stats = pipeline.get_correction_stats()
        assert stats["total"] == 2
        assert stats["pending"] == 2

        # 3. Export
        output = str(tmp_path / "training.jsonl")
        count = pipeline.export_corrections(output)
        assert count == 2

        # 4. Mark applied
        all_corrs = pipeline.correction_vault.get_all()
        ids = [c["id"] for c in all_corrs]
        pipeline.correction_vault.mark_applied(ids)

        # 5. Verify no more pending
        stats2 = pipeline.get_correction_stats()
        assert stats2["pending"] == 0
        assert stats2["applied"] == 2

    def test_health_check_still_works(self, pipeline):
        """Health check works alongside corrections."""
        u = _make_task()
        pipeline.process(u)

        health = pipeline.health_check()
        assert health["note_count"] >= 1

        stats = pipeline.get_correction_stats()
        assert stats["total"] == 0

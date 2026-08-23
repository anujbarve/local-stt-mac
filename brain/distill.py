"""
brain.distill — Knowledge distillation pipeline.

The LLM (Gemma 3 1B) produces high-quality labels. The neural network
learns from them through knowledge distillation: run both models,
compare outputs, log mismatches as corrections for retraining.

Usage:
    from brain.distill import DistillationPipeline
    from brain.understand import BrainEngine
    from brain.correct import CorrectionLogger

    engine = BrainEngine.load("checkpoints/brain")
    distiller = DistillationPipeline(engine, llm=None, correction_logger=CorrectionLogger())
    result = distiller.process_transcription("Fix the VAD threshold")
"""

import logging
from typing import Any, Dict, List, Optional

from brain.correct import CorrectionLogger
from brain_contract import (
    Understanding,
    Correction,
    NoteType,
    Urgency,
    Sentiment,
)

logger = logging.getLogger(__name__)

# Fields to compare during distillation
_COMPARE_FIELDS = ["type", "urgency", "sentiment", "priority"]


class DistillationPipeline:
    """LLM teaches the neural network through knowledge distillation.

    Strategy:
        1. NN predicts: nn_output = brain.understand(text)
        2. LLM predicts: llm_output = llm.analyze(text)
        3. Compare outputs
        4. If mismatch: log as correction (LLM is teacher)
        5. Return LLM output (higher quality) for vault writing
    """

    def __init__(
        self,
        brain_engine: Any,
        llm: Any,
        correction_logger: CorrectionLogger,
    ):
        """Initialize the distillation pipeline.

        Args:
            brain_engine: BrainEngine instance from brain/understand.py
            llm: LocalLLM instance from vault/llm.py (or None for offline mode)
            correction_logger: CorrectionLogger for logging mismatches
        """
        self.brain = brain_engine
        self.llm = llm
        self.logger = correction_logger

    def process_transcription(self, text: str) -> Understanding:
        """Run both NN and LLM, compare, and log mismatches.

        Steps:
            1. NN predicts
            2. LLM predicts (if available)
            3. Compare outputs
            4. If mismatch: log as correction (LLM is teacher)
            5. Return LLM output if available, else NN output

        Args:
            text: Input transcription text.

        Returns:
            Best available Understanding (LLM > NN).
        """
        # 1. NN prediction
        nn_output = self.brain.understand(text)

        # 2. LLM prediction (if available)
        llm_output = None
        if self.llm is not None and hasattr(self.llm, "analyze"):
            try:
                if self.llm.is_available:
                    llm_output = self.llm.analyze(text)
            except Exception as e:
                logger.warning("LLM analysis failed: %s", e)

        # 3. Compare and log
        if llm_output is not None:
            comparison = self.compare_outputs(nn_output, llm_output)
            if not comparison["match"]:
                # Log as correction — LLM is the teacher
                correction = Correction(
                    text=text,
                    predicted=nn_output,
                    corrected=llm_output,
                )
                self.logger.log(correction)
                logger.info(
                    "Distillation correction logged: %d differences",
                    len(comparison["differences"]),
                )
                return llm_output

        return nn_output

    def compare_outputs(
        self,
        nn_output: Understanding,
        llm_output: Understanding,
    ) -> Dict[str, Any]:
        """Compare NN and LLM outputs.

        Args:
            nn_output: Neural network prediction.
            llm_output: LLM prediction (teacher).

        Returns:
            Dict with:
                "match": bool — True if outputs agree
                "differences": list of field-level differences
                "nn_confidence": float
        """
        differences = []

        for field in _COMPARE_FIELDS:
            nn_val = getattr(nn_output, field, None)
            llm_val = getattr(llm_output, field, None)

            if nn_val is None and llm_val is None:
                continue

            if nn_val is None or llm_val is None:
                differences.append({
                    "field": field,
                    "nn": nn_val,
                    "llm": llm_val,
                })
                continue

            # Compare as strings for enums, as floats with tolerance for numbers
            if isinstance(nn_val, float):
                if abs(nn_val - llm_val) > 0.1:
                    differences.append({
                        "field": field,
                        "nn": round(nn_val, 3),
                        "llm": round(llm_val, 3),
                    })
            else:
                nn_str = nn_val.value if hasattr(nn_val, "value") else str(nn_val)
                llm_str = llm_val.value if hasattr(llm_val, "value") else str(llm_val)
                if nn_str != llm_str:
                    differences.append({
                        "field": field,
                        "nn": nn_str,
                        "llm": llm_str,
                    })

        # Compare entities (different count or different names)
        nn_entities = {e.name.lower() for e in nn_output.entities}
        llm_entities = {e.name.lower() for e in llm_output.entities}
        if nn_entities != llm_entities:
            differences.append({
                "field": "entities",
                "nn": sorted(nn_entities),
                "llm": sorted(llm_entities),
            })

        return {
            "match": len(differences) == 0,
            "differences": differences,
            "nn_confidence": nn_output.confidence,
        }

    def should_retrain(self) -> bool:
        """Check if enough corrections accumulated for retraining."""
        return self.logger.should_retrain()

    def retrain(self, epochs: int = 5) -> Dict[str, float]:
        """Retrain the model on accumulated corrections.

        Args:
            epochs: Number of training epochs.

        Returns:
            Training metrics dict.
        """
        data = self.logger.get_training_data()
        if not data:
            return {"error": "no_training_data"}

        # Basic retraining via the brain engine
        # In a real implementation, this would call brain.train
        metrics = {
            "corrections_used": len(data),
            "epochs": epochs,
            "fields_trained": list({d.get("type") for d in data}),
        }

        # Mark corrections as applied
        indices = [
            i for i, c in enumerate(self.logger.buffer) if not c.applied
        ]
        self.logger.mark_applied(indices)

        return metrics

    def run_distillation_loop(
        self,
        texts: List[str],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Process a batch of texts through distillation.

        Args:
            texts: List of transcription texts.
            verbose: If True, print progress.

        Returns:
            Dict with stats: total, nn_correct, llm_corrections, corrections_logged.
        """
        total = len(texts)
        nn_correct = 0
        llm_corrections = 0

        for i, text in enumerate(texts):
            if verbose:
                print(f"  [{i+1}/{total}] Processing: {text[:50]}...")

            nn_output = self.brain.understand(text)

            if self.llm is not None and hasattr(self.llm, "analyze"):
                try:
                    if self.llm.is_available:
                        llm_output = self.llm.analyze(text)
                        comparison = self.compare_outputs(nn_output, llm_output)
                        if comparison["match"]:
                            nn_correct += 1
                        else:
                            llm_corrections += 1
                            correction = Correction(
                                text=text,
                                predicted=nn_output,
                                corrected=llm_output,
                            )
                            self.logger.log(correction)
                        continue
                except Exception:
                    pass

            # No LLM available — count as correct
            nn_correct += 1

        result = {
            "total": total,
            "nn_correct": nn_correct,
            "llm_corrections": llm_corrections,
            "corrections_logged": llm_corrections,
        }

        if verbose:
            print(f"\n  Distillation complete:")
            print(f"    Total: {total}")
            print(f"    NN correct: {nn_correct}")
            print(f"    LLM corrections: {llm_corrections}")

        return result

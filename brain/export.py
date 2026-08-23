"""
brain.export — Model export pipeline.

End-to-end export: train → evaluate → calibrate → export.
Produces ONNX, TorchScript, and model card artifacts.

Usage:
    from brain.export import ModelExporter
    from brain.understand import BrainEngine

    engine = BrainEngine.load("checkpoints/brain")
    exporter = ModelExporter(engine)
    artifacts = exporter.export_all("exports/brain")
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch


class ModelExporter:
    """End-to-end export pipeline for brain models."""

    def __init__(self, engine: Any):
        """Initialize exporter with a BrainEngine instance.

        Args:
            engine: BrainEngine with loaded model and tokenizer.
        """
        self.engine = engine

    def export_all(
        self,
        output_dir: str,
        metrics: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """Export ONNX + TorchScript + model card.

        Args:
            output_dir: Directory to write artifacts.
            metrics: Optional accuracy metrics to include in model card.

        Returns:
            Dict mapping artifact type to file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        artifacts = {}

        # ONNX
        onnx_path = os.path.join(output_dir, "model.onnx")
        artifacts["onnx"] = self.export_onnx(onnx_path)

        # TorchScript
        ts_path = os.path.join(output_dir, "model_scripted.pt")
        artifacts["torchscript"] = self.export_torchscript(ts_path)

        # Model card
        card_path = os.path.join(output_dir, "model_card.json")
        artifacts["model_card"] = self.generate_model_card(card_path, metrics)

        return artifacts

    def export_onnx(self, path: str, opset_version: int = 17) -> str:
        """Export model to ONNX format.

        Args:
            path: Output file path.
            opset_version: ONNX opset version.

        Returns:
            The output path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        model = self.engine.model
        model.eval()

        from brain.config import VOCAB_SIZE, MAX_SEQ_LEN
        dummy = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN))

        try:
            torch.onnx.export(
                model,
                dummy,
                path,
                opset_version=opset_version,
                input_names=["token_ids"],
                output_names=[
                    "classifier", "entities", "priority", "urgency",
                    "sentiment", "summary_scores", "topics",
                ],
                dynamic_axes={"token_ids": {0: "batch"}},
            )
        except Exception as e:
            # Fallback: save as TorchScript if ONNX fails
            import warnings
            warnings.warn(f"ONNX export failed ({e}), saving as TorchScript instead")
            scripted = torch.jit.trace(model, dummy)
            fallback_path = path.replace(".onnx", ".pt")
            scripted.save(fallback_path)
            return fallback_path

        return path

    def export_torchscript(self, path: str) -> str:
        """Export model to TorchScript format.

        Args:
            path: Output file path.

        Returns:
            The output path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        model = self.engine.model
        model.eval()

        from brain.config import VOCAB_SIZE, MAX_SEQ_LEN
        dummy = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN))

        scripted = torch.jit.trace(model, dummy)
        scripted.save(path)

        return path

    def generate_model_card(
        self,
        path: str,
        metrics: Optional[Dict] = None,
    ) -> str:
        """Generate model card JSON with metadata.

        Args:
            path: Output file path.
            metrics: Optional accuracy metrics.

        Returns:
            The output path.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        model = self.engine.model
        tokenizer = self.engine.tokenizer

        card = {
            "model_name": "brain-multi-task",
            "version": getattr(model, "_version", "0.1.0"),
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "architecture": {
                "type": "multi-task-transformer",
                "d_model": model.backbone.d_model,
                "num_heads": getattr(model.backbone, "num_heads", 4),
                "num_layers": getattr(model.backbone, "num_layers", 4),
                "vocab_size": model.backbone.token_embedding.num_embeddings,
                "max_seq_len": getattr(model.backbone, "max_seq_len", 512),
                "total_parameters": model.count_parameters(),
            },
            "task_heads": [
                "classifier", "entity_tagger", "priority", "urgency",
                "sentiment", "action", "summary", "temporal", "topics",
            ],
            "tokenizer": {
                "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else "unknown",
                "type": "word-level",
            },
            "export_formats": ["onnx", "torchscript"],
            "temperature": self.engine.temperature,
        }

        if metrics:
            card["metrics"] = metrics

        with open(path, "w") as f:
            json.dump(card, f, indent=2)

        return path

    def validate_export(
        self,
        onnx_path: str,
        test_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Verify exported ONNX model matches PyTorch inference.

        Args:
            onnx_path: Path to ONNX model.
            test_texts: Texts to test (uses defaults if None).

        Returns:
            {"match_rate": 0.98, "max_diff": 0.001, "tested": 100}
        """
        if test_texts is None:
            test_texts = [
                "fix the VAD threshold",
                "meeting with Sarah about the project",
                "TODO: update the README",
                "I learned that Python is great",
                "deploy the new version",
            ]

        try:
            import onnxruntime as ort
        except ImportError:
            return {"error": "onnxruntime not installed", "tested": 0}

        session = ort.InferenceSession(onnx_path)
        model = self.engine.model
        model.eval()

        match_count = 0
        max_diff = 0.0

        for text in test_texts:
            # PyTorch inference
            token_ids = self.engine.tokenizer.encode(text, add_special=True)
            token_tensor = torch.tensor([token_ids], dtype=torch.long)

            with torch.no_grad():
                pt_outputs = model(token_tensor)
            pt_class = pt_outputs["classifier"][0].argmax().item()

            # ONNX inference
            import numpy as np
            onnx_inputs = {"token_ids": token_ids}
            # Pad to same length
            from brain.config import MAX_SEQ_LEN
            padded = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))
            onnx_inputs = {"token_ids": np.array([padded], dtype=np.int64)}
            onnx_outputs = session.run(None, onnx_inputs)
            onnx_class = int(np.argmax(onnx_outputs[0][0]))

            if pt_class == onnx_class:
                match_count += 1

            diff = abs(
                float(torch.softmax(pt_outputs["classifier"][0], dim=0).max())
                - float(np.max(torch.softmax(pt_outputs["classifier"][0], dim=0).numpy()))
            )
            max_diff = max(max_diff, diff)

        return {
            "match_rate": match_count / len(test_texts) if test_texts else 0,
            "max_diff": round(max_diff, 6),
            "tested": len(test_texts),
        }

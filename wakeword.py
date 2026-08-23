"""Local Wake Word Detection using openWakeWord and ONNX Runtime."""

import os
import pathlib
import shutil
from typing import Dict, List, Optional, Tuple
from huggingface_hub import hf_hub_download
import numpy as np
import openwakeword
from openwakeword.model import Model


# Standard feature models required by openWakeWord
FEATURE_REPO = "harvestsu/openwakeword-onnx"
FEATURE_FILES = ["embedding_model.onnx", "melspectrogram.onnx"]

# Community and pre-trained wake-word models available out of the box
BUILTIN_MODELS: Dict[str, Tuple[str, str]] = {
    "hey_jarvis": (FEATURE_REPO, "hey_jarvis_v0.1.onnx"),
    "hey_handy": (FEATURE_REPO, "hey_jarvis_v0.1.onnx"),  # Default alias
    "hey_glados": ("johnthenerd/openwakeword-hey-glados", "hey_glados.onnx"),
    "hey_fable": ("12384-0ashntoi/hey-fable-openwakeword", "hey_fable.onnx"),
    "mr_president": ("12384-0ashntoi/mr-president-openwakeword", "mister_president.onnx"),
    "mister_president": ("12384-0ashntoi/mr-president-openwakeword", "mister_president.onnx"),
    "hey_beemo": ("amrmantawi/bmo-openwakeword", "hey_beemo.onnx"),
    "bmo": ("amrmantawi/bmo-openwakeword", "hey_beemo.onnx"),
}


def get_models_directory() -> str:
    """Return directory where openWakeWord models are stored."""
    target_dir = os.path.join(
        pathlib.Path(openwakeword.__file__).parent.resolve(),
        "resources",
        "models"
    )
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def get_local_project_models_dir() -> str:
    """Return local workspace models directory (./models)."""
    local_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir


def list_available_wakewords() -> List[str]:
    """Return list of all built-in and locally discovered wake word models."""
    names = set(BUILTIN_MODELS.keys())
    # Check local workspace ./models folder
    local_dir = os.path.join(os.getcwd(), "models")
    if os.path.exists(local_dir):
        for f in os.listdir(local_dir):
            if f.endswith(".onnx") and not f.startswith("silero_vad") and not f.startswith("embedding_") and not f.startswith("melspectrogram"):
                names.add(os.path.splitext(f)[0])
    return sorted(list(names))


def ensure_feature_models(models_dir: Optional[str] = None) -> None:
    """Ensure base feature extraction ONNX models exist in target directory."""
    if models_dir is None:
        models_dir = get_models_directory()

    for fname in FEATURE_FILES:
        dest_path = os.path.join(models_dir, fname)
        if not os.path.exists(dest_path):
            try:
                cached_path = hf_hub_download(repo_id=FEATURE_REPO, filename=fname)
                shutil.copyfile(cached_path, dest_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download required openWakeWord feature model '{fname}': {e}\n"
                    "Please verify internet connection for initial model caching."
                ) from e


def resolve_model_path(model_identifier: str, models_dir: Optional[str] = None) -> str:
    """Resolve a model name or file path to a valid ONNX model file path.
    
    Args:
        model_identifier: File path (e.g. 'models/hey_handy.onnx') or alias (e.g. 'hey_jarvis', 'hey_glados').
        models_dir: Directory for storing built-in models.
        
    Returns:
        Absolute path to the ONNX model file.
    """
    if models_dir is None:
        models_dir = get_models_directory()

    # 1. Direct file path check
    if os.path.isfile(model_identifier):
        return os.path.abspath(model_identifier)

    # Check relative to local ./models directory
    local_models_dir = os.path.join(os.getcwd(), "models")
    local_candidate = os.path.join(local_models_dir, model_identifier)
    if os.path.isfile(local_candidate):
        return os.path.abspath(local_candidate)
    if not model_identifier.endswith(".onnx") and os.path.isfile(f"{local_candidate}.onnx"):
        return os.path.abspath(f"{local_candidate}.onnx")

    # Check relative to openwakeword package models_dir
    in_models_dir = os.path.join(models_dir, model_identifier)
    if os.path.isfile(in_models_dir):
        return in_models_dir
    if not model_identifier.endswith(".onnx") and os.path.isfile(f"{in_models_dir}.onnx"):
        return f"{in_models_dir}.onnx"

    # 2. Check built-in online mappings
    norm_key = model_identifier.lower().replace("-", "_").replace(" ", "_")
    if norm_key in BUILTIN_MODELS:
        repo_id, filename = BUILTIN_MODELS[norm_key]
        dest_path = os.path.join(models_dir, filename)
        if not os.path.exists(dest_path):
            cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copyfile(cached_path, dest_path)
        return dest_path

    # Check if a built-in openWakeWord model exists
    if norm_key in openwakeword.MODELS:
        default_path = openwakeword.MODELS[norm_key]["model_path"].replace(".tflite", ".onnx")
        if os.path.exists(default_path):
            return default_path

    available = ", ".join(list_available_wakewords())
    raise FileNotFoundError(
        f"Wake word model '{model_identifier}' could not be found.\n"
        f"Provide a path to a custom '.onnx' model file or one of the available names:\n"
        f"  {available}"
    )


class WakeWordDetector:
    """Continuous wake-word detection using openWakeWord and ONNX Runtime."""

    def __init__(
        self,
        wake_word: str = "hey_jarvis",
        threshold: float = 0.5,
        inference_framework: str = "onnx",
    ):
        self.wake_word_identifier = wake_word
        self.threshold = threshold
        self.inference_framework = inference_framework
        self.models_dir = get_models_directory()

        # Ensure base feature models exist
        ensure_feature_models(self.models_dir)

        # Resolve model path
        self.model_path = resolve_model_path(wake_word, self.models_dir)
        self.model_name = os.path.splitext(os.path.basename(self.model_path))[0]

        # Initialize openWakeWord Model instance
        self._model = Model(
            wakeword_models=[self.model_path],
            inference_framework=self.inference_framework,
        )

    def process_chunk(self, chunk: np.ndarray) -> Tuple[bool, str, float]:
        """Process an audio chunk (16-bit PCM, 16kHz, 1280 samples).
        
        Returns:
            Tuple of (detected: bool, model_name: str, confidence_score: float)
        """
        if chunk.dtype != np.int16:
            chunk = (chunk * 32767).astype(np.int16) if chunk.dtype == np.float32 else chunk.astype(np.int16)
        if chunk.ndim > 1:
            chunk = chunk.flatten()

        predictions = self._model.predict(chunk)
        
        best_model = self.model_name
        best_score = 0.0
        for name, score in predictions.items():
            if score > best_score:
                best_score = float(score)
                best_model = name

        detected = best_score >= self.threshold
        return detected, best_model, best_score

    def reset(self) -> None:
        """Reset internal frame prediction buffer."""
        self._model.reset()

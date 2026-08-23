"""Unit tests for wake word model resolution and detection."""

import os
import numpy as np
import pytest
from wakeword import (
    BUILTIN_MODELS,
    WakeWordDetector,
    ensure_feature_models,
    get_models_directory,
    list_available_wakewords,
    resolve_model_path,
)


def test_models_directory_exists():
    mdir = get_models_directory()
    assert os.path.isdir(mdir)


def test_ensure_feature_models():
    ensure_feature_models()
    mdir = get_models_directory()
    assert os.path.exists(os.path.join(mdir, "embedding_model.onnx"))
    assert os.path.exists(os.path.join(mdir, "melspectrogram.onnx"))


def test_list_available_wakewords():
    models = list_available_wakewords()
    assert isinstance(models, list)
    assert "hey_jarvis" in models
    assert "hey_glados" in models
    assert "hey_beemo" in models


def test_resolve_builtin_model_path():
    path = resolve_model_path("hey_jarvis")
    assert os.path.exists(path)
    assert path.endswith(".onnx")


def test_resolve_invalid_model_path():
    with pytest.raises(FileNotFoundError):
        resolve_model_path("non_existent_model_12345")


def test_wakeword_detector_initialization_and_predict():
    detector = WakeWordDetector(wake_word="hey_jarvis", threshold=0.5)
    assert detector.model_name is not None

    # Process a silent chunk (1280 samples)
    silent_chunk = np.zeros(1280, dtype=np.int16)
    detected, model_name, score = detector.process_chunk(silent_chunk)
    assert isinstance(detected, bool)
    assert isinstance(score, float)
    assert detected is False  # Silence should not trigger wake word

    # Reset
    detector.reset()

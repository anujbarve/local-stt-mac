"""Unit tests for speech recognition module."""

from unittest.mock import patch
import numpy as np
import pytest
from speech import MODEL_ALIASES, SpeechRecognizer, resolve_model_identifier


def test_resolve_model_identifier():
    assert resolve_model_identifier("tiny") == "mlx-community/whisper-tiny"
    assert resolve_model_identifier("base") == "mlx-community/whisper-base"
    assert resolve_model_identifier("small") == "mlx-community/whisper-small"
    assert resolve_model_identifier("custom/repo") == "custom/repo"


def test_speech_recognizer_empty_audio():
    recognizer = SpeechRecognizer(model_name="tiny")
    empty_audio = np.array([], dtype=np.float32)
    assert recognizer.transcribe(empty_audio) == ""


def test_speech_recognizer_transcribe_mock():
    recognizer = SpeechRecognizer(model_name="tiny")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    with patch("mlx_whisper.transcribe", return_value={"text": " Hello world  "}):
        result = recognizer.transcribe(dummy_audio)
        assert result == "Hello world"

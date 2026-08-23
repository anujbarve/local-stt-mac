"""Integration tests for the end-to-end processing pipeline without hardware microphone."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from audio import CircularBuffer
from vad import SegmenterState, SpeechSegmenter, VoiceActivityDetector
from wakeword import WakeWordDetector
from speech import SpeechRecognizer


def test_mocked_end_to_end_pipeline():
    """Test full cycle: wake word -> speech recording -> silence detection -> STT."""
    # 1. Setup mock components
    mock_wakeword = MagicMock(spec=WakeWordDetector)
    mock_vad = MagicMock(spec=VoiceActivityDetector)
    mock_vad.threshold = 0.5
    mock_stt = MagicMock(spec=SpeechRecognizer)
    mock_stt.transcribe.return_value = "Explain how Docker networking works."

    preroll = CircularBuffer(8000)  # 500ms
    segmenter = SpeechSegmenter(
        vad=mock_vad,
        silence_duration=0.5,
        min_speech_duration=0.2,
        initial_timeout=2.0,
        sample_rate=16000,
    )

    # 2. Simulate wake word trigger
    mock_wakeword.process_chunk.return_value = (True, "hey_jarvis", 0.85)
    chunk = np.zeros(1280, dtype=np.int16)
    preroll.append(chunk)

    detected, model_name, score = mock_wakeword.process_chunk(chunk)
    assert detected is True

    # 3. Simulate speech recording
    segmenter.start(preroll_audio=preroll.get_all())
    mock_vad.process_chunk.return_value = [0.8, 0.8]  # Speech detected

    for _ in range(4):  # 4 * 1280 samples = 0.32s speech
        state = segmenter.process_chunk(np.ones(1280, dtype=np.int16))
    assert state == SegmenterState.RECORDING_SPEECH

    # 4. Simulate silence until segmenter completes
    mock_vad.process_chunk.return_value = [0.01, 0.01]  # Silence detected
    for _ in range(7):  # 7 * 1280 samples = 0.56s silence >= 0.5s silence_duration
        state = segmenter.process_chunk(np.zeros(1280, dtype=np.int16))
    assert state == SegmenterState.COMPLETED

    # 5. Simulate transcription
    audio = segmenter.get_audio()
    assert len(audio) > 0
    text = mock_stt.transcribe(audio)
    assert text == "Explain how Docker networking works."

"""Unit tests for Voice Activity Detection and speech segmenting."""

from unittest.mock import MagicMock
import numpy as np
import pytest
from vad import (
    get_vad_model_path,
    SegmenterState,
    SpeechSegmenter,
    VAD_FRAME_SIZE,
    VoiceActivityDetector,
)


def test_vad_model_path_exists():
    path = get_vad_model_path()
    assert path.endswith(".onnx")


def test_vad_silence_prediction():
    vad = VoiceActivityDetector()
    silence = np.zeros(VAD_FRAME_SIZE, dtype=np.int16)
    prob = vad.predict_frame(silence)
    assert 0.0 <= prob <= 1.0
    assert prob < 0.1  # Silence probability should be near 0


def test_vad_chunk_processing():
    vad = VoiceActivityDetector()
    # 1280 samples = 2 full 512 frames + 256 leftover
    chunk = np.zeros(1280, dtype=np.int16)
    probs = vad.process_chunk(chunk)
    assert len(probs) == 2

    # Second chunk of 1280 will combine with 256 leftover -> 1536 samples -> 3 full frames
    probs2 = vad.process_chunk(chunk)
    assert len(probs2) == 3


def test_segmenter_transitions_with_mock_vad():
    mock_vad = MagicMock(spec=VoiceActivityDetector)
    mock_vad.threshold = 0.5

    segmenter = SpeechSegmenter(
        vad=mock_vad,
        silence_duration=0.5,
        min_speech_duration=0.2,
        initial_timeout=2.0,
        sample_rate=16000,
    )
    segmenter.start()
    assert segmenter.state == SegmenterState.WAITING_FOR_SPEECH

    # 1. Feed silence -> stays WAITING_FOR_SPEECH
    mock_vad.process_chunk.return_value = [0.05, 0.05]
    state = segmenter.process_chunk(np.zeros(1280, dtype=np.int16))
    assert state == SegmenterState.WAITING_FOR_SPEECH

    # 2. Feed speech -> transitions to RECORDING_SPEECH
    mock_vad.process_chunk.return_value = [0.8, 0.9]
    for _ in range(4):  # 4 * 1280 = 5120 samples = 0.32s > min_speech_duration
        state = segmenter.process_chunk(np.ones(1280, dtype=np.int16))
    assert state == SegmenterState.RECORDING_SPEECH

    # 3. Feed silence -> silence accumulates
    mock_vad.process_chunk.return_value = [0.02, 0.02]
    # Need 0.5s of silence = 8000 samples = 7 chunks of 1280
    for _ in range(7):
        state = segmenter.process_chunk(np.zeros(1280, dtype=np.int16))

    assert state == SegmenterState.COMPLETED
    audio = segmenter.get_audio()
    assert len(audio) > 0
    assert audio.dtype == np.float32


def test_segmenter_initial_timeout_with_mock_vad():
    mock_vad = MagicMock(spec=VoiceActivityDetector)
    mock_vad.threshold = 0.5
    mock_vad.process_chunk.return_value = [0.01]

    segmenter = SpeechSegmenter(
        vad=mock_vad,
        initial_timeout=0.3,
        sample_rate=16000,
    )
    segmenter.start()

    # Feed 0.4s of silence without speech (0.4 * 16000 = 6400 samples)
    chunk = np.zeros(6400, dtype=np.int16)
    state = segmenter.process_chunk(chunk)
    assert state == SegmenterState.TIMEOUT

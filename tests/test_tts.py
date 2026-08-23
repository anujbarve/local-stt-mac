"""Unit tests for KittenTTS engine."""

import numpy as np
import pytest
from tts import KittenTTSEngine, DEFAULT_VOICE, SAMPLE_RATE


def test_tts_engine_initialization():
    engine = KittenTTSEngine(voice="rosie")
    assert engine.voice_name == "rosie"
    assert engine.speed == 1.0


def test_tts_generate_empty_string():
    engine = KittenTTSEngine(voice="rosie")
    audio = engine.generate("")
    assert len(audio) == 0


def test_tts_generate_audio():
    engine = KittenTTSEngine(voice="rosie")
    audio = engine.generate("Hello Atlas")
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
    assert audio.dtype == np.float32

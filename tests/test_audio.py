"""Unit tests for audio capture and circular buffering."""

import numpy as np
import pytest
from audio import CircularBuffer, format_device_list, list_input_devices, MicrophonePermissionError, AudioDeviceError


def test_circular_buffer_basic():
    buf = CircularBuffer(capacity_samples=100)
    assert len(buf.get_all()) == 0

    # Add 40 samples
    samples1 = np.arange(40, dtype=np.int16)
    buf.append(samples1)
    out = buf.get_all()
    assert len(out) == 40
    np.testing.assert_array_equal(out, samples1)

    # Add 40 more samples (total 80 <= 100)
    samples2 = np.arange(40, 80, dtype=np.int16)
    buf.append(samples2)
    out = buf.get_all()
    assert len(out) == 80
    np.testing.assert_array_equal(out, np.arange(80, dtype=np.int16))

    # Add 40 more samples (overflow capacity of 100, should keep newest 100 samples)
    samples3 = np.arange(80, 120, dtype=np.int16)
    buf.append(samples3)
    out = buf.get_all()
    assert len(out) == 100
    np.testing.assert_array_equal(out, np.arange(20, 120, dtype=np.int16))


def test_circular_buffer_large_chunk():
    buf = CircularBuffer(capacity_samples=50)
    samples = np.arange(200, dtype=np.int16)
    buf.append(samples)
    out = buf.get_all()
    assert len(out) == 50
    np.testing.assert_array_equal(out, np.arange(150, 200, dtype=np.int16))


def test_circular_buffer_clear():
    buf = CircularBuffer(capacity_samples=50)
    buf.append(np.arange(30, dtype=np.int16))
    assert len(buf.get_all()) == 30
    buf.clear()
    assert len(buf.get_all()) == 0


def test_list_input_devices():
    devices = list_input_devices()
    assert isinstance(devices, list)
    text = format_device_list()
    assert isinstance(text, str)
    assert len(text) > 0


def test_custom_exceptions():
    perm_err = MicrophonePermissionError("Permission denied")
    assert "Permission denied" in str(perm_err)

    dev_err = AudioDeviceError("Device not found")
    assert "Device not found" in str(dev_err)

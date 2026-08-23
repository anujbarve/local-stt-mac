"""Audio capture module for continuous microphone streaming."""

import queue
import sys
from typing import Generator, List, Optional, Tuple, Union
import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000  # 16 kHz standard for STT & Wake Word
CHUNK_SAMPLES = 1280  # 80ms chunk at 16kHz (compatible with openWakeWord)


class AudioDeviceError(Exception):
    """Raised when there is an issue accessing audio hardware."""
    pass


class MicrophonePermissionError(Exception):
    """Raised when macOS microphone permission is denied."""
    pass


def list_input_devices() -> List[Tuple[int, str, int, float]]:
    """Return a list of available input audio devices.
    
    Returns:
        List of tuples: (device_index, name, max_input_channels, default_samplerate)
    """
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append((
                idx,
                dev["name"],
                dev["max_input_channels"],
                dev.get("default_samplerate", 44100.0),
            ))
    return devices


def format_device_list() -> str:
    """Format available audio devices for display."""
    devices = list_input_devices()
    if not devices:
        return "No audio input devices found."
    lines = ["Available Audio Input Devices:"]
    default_dev = sd.default.device[0]
    for idx, name, channels, sr in devices:
        is_def = " (Default)" if idx == default_dev else ""
        lines.append(f"  [{idx}] {name}{is_def} - {channels} ch, {int(sr)} Hz")
    return "\n".join(lines)


class CircularBuffer:
    """A fixed-capacity circular buffer for storing recent audio chunks (pre-roll)."""

    def __init__(self, capacity_samples: int):
        self.capacity = capacity_samples
        self._buffer = np.zeros(capacity_samples, dtype=np.int16)
        self._write_pos = 0
        self._size = 0

    def append(self, samples: np.ndarray) -> None:
        """Append samples to the circular buffer."""
        n = len(samples)
        if n == 0:
            return
        if n >= self.capacity:
            self._buffer[:] = samples[-self.capacity:]
            self._write_pos = 0
            self._size = self.capacity
            return

        end_pos = self._write_pos + n
        if end_pos <= self.capacity:
            self._buffer[self._write_pos:end_pos] = samples
        else:
            first_part = self.capacity - self._write_pos
            self._buffer[self._write_pos:] = samples[:first_part]
            self._buffer[:n - first_part] = samples[first_part:]
        
        self._write_pos = (self._write_pos + n) % self.capacity
        self._size = min(self.capacity, self._size + n)

    def get_all(self) -> np.ndarray:
        """Retrieve all currently buffered samples in chronological order."""
        if self._size == 0:
            return np.array([], dtype=np.int16)
        if self._size < self.capacity:
            return self._buffer[:self._size].copy()
        # Full buffer: unwrapping
        return np.concatenate((self._buffer[self._write_pos:], self._buffer[:self._write_pos]))

    def clear(self) -> None:
        """Clear the buffer."""
        self._write_pos = 0
        self._size = 0


class AudioStream:
    """Manages continuous microphone audio capture via sounddevice."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        device: Optional[Union[int, str]] = None,
        preroll_ms: int = 500,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.device = device
        self.queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        preroll_samples = int((preroll_ms / 1000.0) * sample_rate)
        self.preroll_buffer = CircularBuffer(preroll_samples)

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal callback invoked by sounddevice for each new audio chunk."""
        if status:
            pass  # Audio buffer underflow/overflow handling
        # indata is shape (frames, 1), convert to 1D int16 array
        chunk = indata[:, 0].copy()
        self.preroll_buffer.append(chunk)
        self.queue.put_nowait(chunk)

    def start(self) -> None:
        """Start the microphone audio stream."""
        if self._stream is not None and self._stream.active:
            return

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_samples,
                device=self.device,
                callback=self._audio_callback,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            err_msg = str(e).lower()
            if "permission" in err_msg or "not authorized" in err_msg or "access" in err_msg:
                raise MicrophonePermissionError(
                    "Error: Microphone access denied.\n"
                    "Enable microphone access for Terminal in:\n"
                    "System Settings → Privacy & Security → Microphone"
                ) from e
            elif "invalid device" in err_msg or "device unavailable" in err_msg:
                raise AudioDeviceError(
                    f"Selected audio device ({self.device}) is unavailable.\n"
                    f"{format_device_list()}"
                ) from e
            else:
                raise AudioDeviceError(f"Failed to open audio input stream: {e}") from e
        except Exception as e:
            raise AudioDeviceError(f"Unexpected audio stream error: {e}") from e

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            finally:
                self._stream = None

    def read_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read the next audio chunk from the queue."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_preroll(self) -> np.ndarray:
        """Get the audio captured immediately prior to the current moment."""
        return self.preroll_buffer.get_all()

    def clear_queue(self) -> None:
        """Discard any accumulated chunks in queue."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

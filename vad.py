"""Voice Activity Detection (VAD) and speech segmenting using Silero VAD ONNX."""

from enum import Enum
import os
import pathlib
import shutil
from typing import List, Optional, Tuple
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort


SILERO_VAD_REPO = "onnx-community/silero-vad"
SILERO_VAD_FILE = "onnx/model.onnx"
VAD_FRAME_SIZE = 512  # Silero VAD accepts 512 samples at 16kHz (32ms)
SAMPLE_RATE = 16000


def get_vad_model_path() -> str:
    """Download and return path to Silero VAD ONNX model."""
    local_dir = os.path.join(
        pathlib.Path(__file__).parent.resolve(),
        "models"
    )
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, "silero_vad.onnx")
    if not os.path.exists(local_path):
        cached_path = hf_hub_download(repo_id=SILERO_VAD_REPO, filename=SILERO_VAD_FILE)
        shutil.copyfile(cached_path, local_path)
    return local_path


class VoiceActivityDetector:
    """Lightweight Silero VAD v5 ONNX engine."""

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        if model_path is None:
            model_path = get_vad_model_path()
        self.model_path = model_path
        self.threshold = threshold

        # Initialize ONNX inference session with minimal threads for low idle overhead
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(self.model_path, sess_options=opts)

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self._leftover = np.array([], dtype=np.int16)

    def reset(self) -> None:
        """Reset internal RNN state."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._leftover = np.array([], dtype=np.int16)

    def predict_frame(self, frame_512: np.ndarray) -> float:
        """Calculate speech probability for a single 512-sample frame."""
        if len(frame_512) != VAD_FRAME_SIZE:
            raise ValueError(f"Expected frame of size {VAD_FRAME_SIZE}, got {len(frame_512)}")

        # Normalize to [-1.0, 1.0] float32
        if frame_512.dtype == np.int16:
            frame_float = frame_512.astype(np.float32) / 32768.0
        else:
            frame_float = frame_512.astype(np.float32)

        inp = np.expand_dims(frame_float, axis=0)  # Shape (1, 512)
        out, new_state = self._session.run(
            None,
            {"input": inp, "state": self._state, "sr": self._sr}
        )
        self._state = new_state
        return float(out[0][0])

    def process_chunk(self, chunk: np.ndarray) -> List[float]:
        """Process an arbitrary audio chunk, returning speech probabilities for each 512-sample frame."""
        if chunk.ndim > 1:
            chunk = chunk.flatten()

        samples = np.concatenate((self._leftover, chunk)) if len(self._leftover) > 0 else chunk
        probs = []
        num_frames = len(samples) // VAD_FRAME_SIZE
        for i in range(num_frames):
            frame = samples[i * VAD_FRAME_SIZE:(i + 1) * VAD_FRAME_SIZE]
            prob = self.predict_frame(frame)
            probs.append(prob)

        rem_idx = num_frames * VAD_FRAME_SIZE
        self._leftover = samples[rem_idx:].copy()
        return probs


class SegmenterState(Enum):
    WAITING_FOR_SPEECH = 1
    RECORDING_SPEECH = 2
    COMPLETED = 3
    TIMEOUT = 4


class SpeechSegmenter:
    """Manages audio recording state from wake word detection to end-of-speech silence."""

    def __init__(
        self,
        vad: VoiceActivityDetector,
        silence_duration: float = 1.5,
        min_speech_duration: float = 0.3,
        initial_timeout: float = 5.0,
        max_recording_duration: float = 45.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.vad = vad
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.initial_timeout = initial_timeout
        self.max_recording_duration = max_recording_duration
        self.sample_rate = sample_rate

        self.state = SegmenterState.WAITING_FOR_SPEECH
        self.recorded_chunks: List[np.ndarray] = []
        self.speech_samples = 0
        self.silence_samples = 0
        self.total_samples = 0

    def start(self, preroll_audio: Optional[np.ndarray] = None) -> None:
        """Start segmenter recording with optional pre-roll audio buffer."""
        self.vad.reset()
        self.state = SegmenterState.WAITING_FOR_SPEECH
        self.recorded_chunks = []
        self.speech_samples = 0
        self.silence_samples = 0
        self.total_samples = 0

        if preroll_audio is not None and len(preroll_audio) > 0:
            self.recorded_chunks.append(preroll_audio)
            self.total_samples += len(preroll_audio)

    def process_chunk(self, chunk: np.ndarray) -> SegmenterState:
        """Process incoming audio chunk and update speech state.
        
        Returns:
            Current SegmenterState (e.g. RECORDING_SPEECH, COMPLETED, TIMEOUT)
        """
        if self.state in (SegmenterState.COMPLETED, SegmenterState.TIMEOUT):
            return self.state

        self.recorded_chunks.append(chunk)
        chunk_len = len(chunk)
        self.total_samples += chunk_len

        # Check total duration limit
        if (self.total_samples / self.sample_rate) >= self.max_recording_duration:
            self.state = SegmenterState.COMPLETED
            return self.state

        probs = self.vad.process_chunk(chunk)
        # If any frame in chunk has speech prob >= threshold, chunk has speech
        is_speech = any(p >= self.vad.threshold for p in probs) if probs else False

        if self.state == SegmenterState.WAITING_FOR_SPEECH:
            if is_speech:
                self.state = SegmenterState.RECORDING_SPEECH
                self.speech_samples += chunk_len
                self.silence_samples = 0
            else:
                # Check initial silence timeout (no speech started after wake word)
                if (self.total_samples / self.sample_rate) >= self.initial_timeout:
                    self.state = SegmenterState.TIMEOUT
                    return self.state

        elif self.state == SegmenterState.RECORDING_SPEECH:
            if is_speech:
                self.speech_samples += chunk_len
                self.silence_samples = 0
            else:
                self.silence_samples += chunk_len
                silence_sec = self.silence_samples / self.sample_rate
                speech_sec = self.speech_samples / self.sample_rate
                if silence_sec >= self.silence_duration and speech_sec >= self.min_speech_duration:
                    self.state = SegmenterState.COMPLETED

        return self.state

    def get_audio(self) -> np.ndarray:
        """Return all recorded audio as float32 array in range [-1.0, 1.0]."""
        if not self.recorded_chunks:
            return np.array([], dtype=np.float32)
        raw = np.concatenate(self.recorded_chunks)
        if raw.dtype == np.int16:
            return (raw.astype(np.float32) / 32768.0)
        return raw.astype(np.float32)

"""Speech-to-text transcription engine optimized for Apple Silicon using MLX."""

import contextlib
import io
import os
import sys
from typing import Dict, Optional, Union
import mlx_whisper
import numpy as np


# Standard model aliases mapped to high-performance MLX Community HuggingFace models
MODEL_ALIASES: Dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny",
    "tiny.en": "mlx-community/whisper-tiny.en",
    "base": "mlx-community/whisper-base",
    "base.en": "mlx-community/whisper-base.en",
    "small": "mlx-community/whisper-small",
    "small.en": "mlx-community/whisper-small.en",
    "medium": "mlx-community/whisper-medium",
    "medium.en": "mlx-community/whisper-medium.en",
    "large": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

DEFAULT_MODEL = "base"


def resolve_model_identifier(model_name_or_path: str) -> str:
    """Resolve model alias or return direct path/repo id."""
    key = model_name_or_path.lower().strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    return model_name_or_path


class SpeechRecognizer:
    """High-performance local speech recognizer using MLX Whisper."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        language: Optional[str] = "en",
        temperature: float = 0.0,
        verbose: bool = False,
    ):
        self.model_identifier = resolve_model_identifier(model_name)
        self.language = language
        self.temperature = temperature
        self.verbose = verbose
        self._is_warmed_up = False

    def preload(self) -> None:
        """Warm up the MLX model and cache weights in unified memory."""
        if self._is_warmed_up:
            return
        dummy_audio = np.zeros(1600, dtype=np.float32)
        # Suppress stdout/stderr during preload to prevent progress bar output
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                mlx_whisper.transcribe(
                    dummy_audio,
                    path_or_hf_repo=self.model_identifier,
                    verbose=False,
                )
        self._is_warmed_up = True

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe an audio buffer (float32, 16kHz mono).
        
        Args:
            audio: 1D numpy float32 array normalized to [-1.0, 1.0].
            sample_rate: Audio sample rate (default 16000).
            
        Returns:
            Clean transcribed text.
        """
        if len(audio) == 0:
            return ""

        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            else:
                audio = audio.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.flatten()

        decode_options = {
            "temperature": self.temperature,
            "condition_on_previous_text": False,
        }
        if self.language:
            decode_options["language"] = self.language

        if not self.verbose:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    result = mlx_whisper.transcribe(
                        audio,
                        path_or_hf_repo=self.model_identifier,
                        verbose=False,
                        **decode_options,
                    )
        else:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_identifier,
                verbose=True,
                **decode_options,
            )

        text = result.get("text", "").strip()
        return text

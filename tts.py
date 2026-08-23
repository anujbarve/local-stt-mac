"""Lightweight, local Text-to-Speech (TTS) using KittenTTS ONNX with Rosie voice."""

import contextlib
import os
import re
import sys
from typing import Optional
from huggingface_hub import hf_hub_download
import numpy as np
import onnxruntime as ort
import phonemizer
import sounddevice as sd
import h5py


KITTEN_REPO = "alexisStacksCode/Kitten_TTS_Nano-0.8-ONNX"
MODEL_FILE = "kitten_tts_nano-0.8-fp32.onnx"
DEFAULT_VOICE = "rosie"
SAMPLE_RATE = 24000

# Voice file mappings in repository
VOICE_FILES = {
    "rosie": "voices/f_rosie.h5",
    "f_rosie": "voices/f_rosie.h5",
    "bella": "voices/f_bella.h5",
    "kiki": "voices/f_kiki.h5",
    "luna": "voices/f_luna.h5",
    "bruno": "voices/m_bruno.h5",
    "hugo": "voices/m_hugo.h5",
    "jasper": "voices/m_jasper.h5",
    "leo": "voices/m_leo.h5",
}


class KittenTTSEngine:
    """Fast local TTS engine using KittenTTS ONNX."""

    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = 1.0, verbose: bool = False):
        self.voice_name = voice.lower()
        self.speed = speed
        self.verbose = verbose

        # 1. Download/load ONNX model
        model_path = hf_hub_download(KITTEN_REPO, MODEL_FILE)
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(model_path, sess_options=opts)

        # 2. Download/load voice style embedding
        voice_file = VOICE_FILES.get(self.voice_name, "voices/f_rosie.h5")
        voice_path = hf_hub_download(KITTEN_REPO, voice_file)
        with h5py.File(voice_path, "r") as f:
            # Use style vector (shape 1, 256)
            self._style = np.array(f["_"])[0:1, :].astype(np.float32)

        # 3. Phonemizer backend & dictionary
        self._phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True
        )
        symbols = list('$;:,.!?¡¿—…"«»"" ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘\'̩\'ᵻ')
        self._word_index_dict = {symbol: i for i, symbol in enumerate(symbols)}

    def generate(self, text: str, speed: Optional[float] = None) -> np.ndarray:
        """Synthesize text to 24kHz audio waveform (float32 array)."""
        if not text or not text.strip():
            return np.array([], dtype=np.float32)

        sp = speed if speed is not None else self.speed
        clean_text = text.strip()

        # Phonemize
        phonemes = self._phonemizer.phonemize([clean_text])[0]
        words = " ".join(re.findall(r"\w+|[^\w\s]", phonemes))
        token_ids = [0] + [self._word_index_dict[c] for c in words if c in self._word_index_dict] + [0]
        input_ids = np.array([token_ids], dtype=np.int64)

        out = self._session.run(None, {
            "input_ids": input_ids,
            "style": self._style,
            "speed": np.array([sp], dtype=np.float32),
        })
        return out[0].astype(np.float32)

    def speak(self, text: str, speed: Optional[float] = None, wait: bool = True) -> None:
        """Generate audio and play it through sounddevice."""
        if not text or not text.strip():
            return

        audio = self.generate(text, speed=speed)
        if len(audio) == 0:
            return

        sd.play(audio, samplerate=SAMPLE_RATE)
        if wait:
            sd.wait()

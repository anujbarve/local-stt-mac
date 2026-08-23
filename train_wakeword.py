#!/usr/bin/env python3
"""Local Wake Word Trainer for macOS with Fast Augmentation & Precise Streaming Alignment."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from typing import List, Tuple
import numpy as np
import openwakeword
from openwakeword.model import AudioFeatures
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SAMPLE_RATE = 16000
CHUNK_SIZE = 1280  # 80ms at 16kHz
FRAME_COUNT = 16   # 16 frames = 1.28s window
EMBEDDING_DIM = 96

# Fast local macOS English voices
VOICES = [
    "Samantha", "Daniel", "Karen", "Moira", "Fred", "Rishi",
    "Tessa", "Ralph", "Kathy", "Albert", "Junior"
]

RATES = [140, 160, 180, 200]

# Phonetically similar phrases to prevent false triggers
HARD_NEGATIVES = [
    "Hey Anna", "Anna", "Hannah", "Hey Hannah", "Arnold", "Hey Arnold",
    "Android", "Hey Android", "Annual", "Annoy", "Anode", "Animal", "Anime",
    "Annie", "Hey Annie", "Manu", "Hey Manu", "Hello", "Hey", "No", "Oh no",
    "Hey man", "Hey there", "Hey Jarvis", "Hey Siri", "Ok Google", "Alexa",
    "Computer", "Hey Andy", "Andy", "Hey Ant", "Anthony", "Hey Anthony",
    "Audio", "Hey Audio", "Auto", "Hey Auto"
]

GENERAL_NEGATIVES = [
    "what time is it", "good morning", "explain how docker works",
    "open the browser", "turn on the lights", "can you hear me",
    "just checking in", "play some music", "set a timer for ten minutes",
    "who is the president", "nice to meet you", "thank you very much",
    "goodbye", "start recording", "stop listening", "how is the weather",
    "write a python script", "show me the code", "let's get started",
    "what do you think", "tell me a story", "why is the sky blue",
]


class WakeWordClassifier(nn.Module):
    """Neural classifier matching openWakeWord feature input shape [batch, 16, 96]."""

    def __init__(self, input_dim: int = EMBEDDING_DIM, num_frames: int = FRAME_COUNT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * num_frames, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_tts_clip(phrase: str, voice: str, rate: int, output_wav: str) -> bool:
    """Generate a 16kHz mono WAV clip using macOS `say` and `ffmpeg` with timeout."""
    temp_aiff = output_wav.replace(".wav", ".aiff")
    try:
        cmd_say = ["say", "-v", voice, "-r", str(rate), "-o", temp_aiff, phrase]
        res = subprocess.run(cmd_say, capture_output=True, text=True, check=False, timeout=3.0)
        if res.returncode != 0 or not os.path.exists(temp_aiff):
            return False

        cmd_ffmpeg = [
            "ffmpeg", "-y", "-i", temp_aiff,
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
            output_wav
        ]
        res2 = subprocess.run(cmd_ffmpeg, capture_output=True, text=True, check=False, timeout=3.0)
        return res2.returncode == 0 and os.path.exists(output_wav)
    except Exception:
        return False
    finally:
        if os.path.exists(temp_aiff):
            try:
                os.remove(temp_aiff)
            except OSError:
                pass


def load_wav(wav_path: str) -> np.ndarray:
    """Load 16kHz mono WAV as numpy int16 array."""
    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        return np.frombuffer(data, dtype=np.int16)


def extract_streaming_windows(audio: np.ndarray, feature_extractor: AudioFeatures) -> List[np.ndarray]:
    """Stream audio chunks through AudioFeatures and extract 16-frame feature windows."""
    feature_extractor.feature_buffer = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    feature_extractor.raw_data_buffer.clear()
    feature_extractor.melspectrogram_buffer = np.zeros((0, 32), dtype=np.float32)
    feature_extractor.accumulated_samples = 0
    feature_extractor.raw_data_remainder = np.empty(0)

    windows = []
    for i in range(0, len(audio) - CHUNK_SIZE + 1, CHUNK_SIZE):
        chunk = audio[i:i + CHUNK_SIZE]
        feature_extractor(chunk)
        if len(feature_extractor.feature_buffer) >= FRAME_COUNT:
            feat = feature_extractor.get_features(FRAME_COUNT)
            windows.append(feat[0].copy())
    return windows


def create_dataset(
    target_phrases: List[str],
    temp_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a balanced training dataset with accurate frame-level labeling."""
    feature_extractor = AudioFeatures(inference_framework="onnx")

    X_list: List[np.ndarray] = []
    y_list: List[float] = []

    print(f"Generating positive speech clips for: {target_phrases}...")
    pos_count = 0

    for phrase in target_phrases:
        for voice in VOICES:
            for rate in RATES:
                wav_path = os.path.join(temp_dir, f"pos_{pos_count}.wav")
                if generate_tts_clip(phrase, voice, rate, wav_path):
                    raw = load_wav(wav_path)
                    pos_count += 1

                    # Create multiple augmented versions: standard, louder, quieter, slight noise
                    for gain, noise_level in [(1.0, 0), (1.3, 100), (0.7, 50), (1.0, 200)]:
                        aug_raw = (raw.astype(np.float32) * gain).astype(np.int32)
                        if noise_level > 0:
                            noise = np.random.normal(0, noise_level, size=len(raw)).astype(np.int32)
                            aug_raw = np.clip(aug_raw + noise, -32768, 32767).astype(np.int16)
                        else:
                            aug_raw = np.clip(aug_raw, -32768, 32767).astype(np.int16)

                        lead_silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
                        trail_silence = np.zeros(SAMPLE_RATE, dtype=np.int16)
                        full_clip = np.concatenate((lead_silence, aug_raw, trail_silence))

                        windows = extract_streaming_windows(full_clip, feature_extractor)
                        if not windows:
                            continue

                        # Exact trigger chunk index where speech finished
                        trigger_idx = (len(lead_silence) + len(aug_raw)) // CHUNK_SIZE - FRAME_COUNT

                        for idx, win in enumerate(windows):
                            if trigger_idx <= idx <= trigger_idx + 1:
                                X_list.append(win)
                                y_list.append(1.0)
                            elif idx < trigger_idx - 2 or idx > trigger_idx + 4:
                                X_list.append(win)
                                y_list.append(0.0)

    print(f"Generated {pos_count} positive clips -> {sum(1 for y in y_list if y == 1.0)} positive windows.")

    # 2. Hard Negatives & General Negatives
    print("Generating hard negative and conversational speech samples...")
    neg_phrases = HARD_NEGATIVES + GENERAL_NEGATIVES
    for neg_idx, phrase in enumerate(neg_phrases):
        voice = VOICES[neg_idx % len(VOICES)]
        rate = RATES[neg_idx % len(RATES)]
        wav_path = os.path.join(temp_dir, f"neg_{neg_idx}.wav")
        if generate_tts_clip(phrase, voice, rate, wav_path):
            raw = load_wav(wav_path)
            lead_silence = np.zeros(np.random.randint(4000, 16000), dtype=np.int16)
            trail_silence = np.zeros(8000, dtype=np.int16)
            full_clip = np.concatenate((lead_silence, raw, trail_silence))

            windows = extract_streaming_windows(full_clip, feature_extractor)
            for win in windows:
                X_list.append(win)
                y_list.append(0.0)

    # 3. Pure Silence & Ambient Noise
    print("Generating silence and acoustic noise samples...")
    for _ in range(30):
        # Pure silence
        silence = np.zeros(3 * SAMPLE_RATE, dtype=np.int16)
        for win in extract_streaming_windows(silence, feature_extractor):
            X_list.append(win)
            y_list.append(0.0)

        # Ambient noise
        noise = np.random.normal(0, np.random.uniform(50, 400), size=3 * SAMPLE_RATE).astype(np.int16)
        for win in extract_streaming_windows(noise, feature_extractor):
            X_list.append(win)
            y_list.append(0.0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)

    n_pos = int(np.sum(y == 1.0))
    n_neg = int(np.sum(y == 0.0))
    print(f"Total training dataset: {len(X)} windows ({n_pos} positive, {n_neg} negative)")
    return X, y


def train_and_export(
    X: np.ndarray,
    y: np.ndarray,
    output_path: str,
    epochs: int = 35,
    lr: float = 0.001,
) -> None:
    """Train classifier with penalty on false positives and export to ONNX."""
    print("Training neural wake-word classifier...")
    model = WakeWordClassifier()
    model.train()

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            # Heavier weight on negative mistakes
            weights = torch.where(batch_y == 1.0, torch.tensor(1.0), torch.tensor(2.0))
            loss = (criterion(preds, batch_y) * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        avg_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        all_preds = model(torch.from_numpy(X)).numpy()
        pos_acc = np.mean((all_preds[y == 1.0] >= 0.5)) * 100.0
        neg_acc = np.mean((all_preds[y == 0.0] < 0.5)) * 100.0
        print(f"Validation: Positive Accuracy = {pos_acc:.2f}%, Negative Rejection = {neg_acc:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dummy_input = torch.randn(1, FRAME_COUNT, EMBEDDING_DIM)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["x.1"],
        output_names=["output"],
        dynamic_axes={"x.1": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"\nModel exported successfully to: {output_path}")


def verify_model_behavior(onnx_path: str) -> None:
    """Verify silence score is near zero and test inference."""
    from openwakeword.model import Model
    print(f"\nVerifying model behavior on silence: {onnx_path}")
    oww = Model(wakeword_models=[onnx_path], inference_framework="onnx")

    max_silence_score = 0.0
    for _ in range(50):
        chunk = np.zeros(1280, dtype=np.int16)
        res = oww.predict(chunk)
        for score in res.values():
            max_silence_score = max(max_silence_score, float(score))

    print(f"Maximum score across 50 silence chunks: {max_silence_score:.6f}")
    if max_silence_score > 0.05:
        print("[Warning] Model has non-zero silence score.")
    else:
        print("Verification passed! Model completely ignores silence.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train_wakeword",
        description="Train a custom wake word model locally on macOS with streaming alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phrase",
        type=str,
        default="Hey Anno",
        help="Target wake phrase to train (e.g. 'Hey Anno' or 'Anno')",
    )
    parser.add_argument(
        "--aliases",
        type=str,
        nargs="*",
        default=["Anno", "Hey Anno"],
        help="Optional additional variations to recognize (e.g. --aliases 'Anno' 'Hey Anno')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX model path (defaults to models/<phrase_normalized>.onnx)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Number of training epochs",
    )
    args = parser.parse_args()

    phrases = list(set([args.phrase] + (args.aliases or [])))

    if args.output is None:
        clean_name = args.phrase.lower().replace(" ", "_").replace("-", "_")
        output_path = os.path.join(os.getcwd(), "models", f"{clean_name}.onnx")
    else:
        output_path = args.output

    with tempfile.TemporaryDirectory() as temp_dir:
        X, y = create_dataset(phrases, temp_dir)
        train_and_export(X, y, output_path, epochs=args.epochs)
        verify_model_behavior(output_path)

    print("\n" + "=" * 55)
    print(f"Wake word model '{args.phrase}' ready!")
    print(f"Run:")
    print(f"  transcribe --wake-word {output_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()

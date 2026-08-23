#!/usr/bin/env python3
"""Minimal, completely local Speech-to-Text CLI application for macOS with Hotkey & KittenTTS voice reply."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

from audio import (
    AudioDeviceError,
    AudioStream,
    CHUNK_SAMPLES,
    MicrophonePermissionError,
    SAMPLE_RATE,
    format_device_list,
)
from hotkey import HotkeyListener
from speech import DEFAULT_MODEL, MODEL_ALIASES, SpeechRecognizer
from tts import KittenTTSEngine
from vad import SpeechSegmenter, SegmenterState, VoiceActivityDetector
from wakeword import (
    BUILTIN_MODELS,
    WakeWordDetector,
    list_available_wakewords,
)


CONFIG_FILE = "config.json"


def load_config() -> Dict[str, Any]:
    """Load configuration from config.json if it exists."""
    config: Dict[str, Any] = {}
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to parse {CONFIG_FILE}: {e}", file=sys.stderr)
    return config


def parse_arguments(args=None) -> argparse.Namespace:
    """Parse command line arguments with config file & env var fallbacks."""
    config = load_config()

    default_model = os.getenv("VOICEASSIST_MODEL", config.get("model", DEFAULT_MODEL))
    default_wakeword = os.getenv(
        "VOICEASSIST_WAKE_WORD",
        os.getenv("WAKE_WORD", config.get("wake_word", None))
    )
    default_threshold = float(
        os.getenv("VOICEASSIST_THRESHOLD", config.get("threshold", 0.45))
    )
    default_silence = float(
        os.getenv("VOICEASSIST_SILENCE", config.get("silence", 1.5))
    )
    default_device = os.getenv("VOICEASSIST_DEVICE", config.get("device", None))
    default_hotkey = os.getenv("VOICEASSIST_HOTKEY", config.get("hotkey", "alt_r"))
    default_tts = config.get("tts", True)
    default_tts_voice = os.getenv("VOICEASSIST_TTS_VOICE", config.get("tts_voice", "rosie"))

    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Minimal, fully local speech-to-text CLI with hotkey & optional wake word for macOS Apple Silicon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Speech recognition model. Choices: {', '.join(sorted(MODEL_ALIASES.keys()))} or custom path",
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        default=default_wakeword,
        help="Enable wake word detection with specified model name or .onnx path (disabled by default)",
    )
    parser.add_argument(
        "--enable-wake-word",
        action="store_true",
        help="Enable wake word detection using default model (hey_atlas)",
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        default=default_hotkey,
        help="Global hotkey to trigger recording (default: alt_r / Right Option)",
    )
    parser.add_argument(
        "--no-hotkey",
        action="store_true",
        help="Disable hotkey listening",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        default=default_tts,
        help="Enable KittenTTS voice reply (Rosie voice) for transcribed speech",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS audio playback",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default=default_tts_voice,
        help="KittenTTS voice to use (default: rosie)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=default_threshold,
        help="Confidence threshold for wake word detection (0.0 - 1.0)",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=default_silence,
        help="Duration of silence (seconds) to detect end of speech",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        help="Audio input device index or substring name",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--list-wake-words",
        action="store_true",
        help="List available built-in and local wake word models and exit",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Transcribe a single utterance after trigger and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose / debug output",
    )
    parser.add_argument(
        "--brain",
        action="store_true",
        help="Enable brain understanding (classify, extract entities, determine priority)",
    )
    parser.add_argument(
        "--vault",
        action="store_true",
        help="Enable vault writing (save understanding to vault)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive correction after understanding",
    )
    parser.add_argument(
        "--brain-checkpoint",
        type=str,
        default="checkpoints/brain",
        help="Brain model checkpoint directory",
    )
    parser.add_argument(
        "--vault-path",
        type=str,
        default="vault/",
        help="Vault root path",
    )
    parsed = parser.parse_args(args)

    if parsed.enable_wake_word and not parsed.wake_word:
        parsed.wake_word = "hey_atlas"

    return parsed


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the main voice assistant pipeline."""
    if args.list_devices:
        print(format_device_list())
        return 0

    if args.list_wake_words:
        available = list_available_wakewords()
        print("Available Wake Word Models:")
        for name in available:
            print(f"  • {name}")
        print("\nYou can also pass any custom model path: transcribe --wake-word ./models/my_model.onnx")
        return 0

    device_arg = None
    if args.device is not None:
        try:
            device_arg = int(args.device)
        except ValueError:
            device_arg = args.device

    enable_tts = args.tts and not args.no_tts

    if args.verbose:
        print("[Debug] Initializing components...")

    # 1. Initialize Speech Recognizer
    try:
        if args.verbose:
            print(f"[Debug] Loading speech recognition model: {args.model}")
        recognizer = SpeechRecognizer(
            model_name=args.model,
            verbose=args.verbose,
        )
        recognizer.preload()
    except Exception as e:
        print(f"Error loading speech recognition model: {e}", file=sys.stderr)
        return 1

    # 2. Initialize VAD
    try:
        if args.verbose:
            print("[Debug] Initializing Silero VAD...")
        vad = VoiceActivityDetector()
        segmenter = SpeechSegmenter(vad=vad, silence_duration=args.silence)
    except Exception as e:
        print(f"Error initializing Voice Activity Detector: {e}", file=sys.stderr)
        return 1

    # 3. Initialize Wake Word Detector (only if wake-word parameter specified)
    wakeword = None
    if args.wake_word:
        try:
            if args.verbose:
                print(f"[Debug] Initializing Wake Word Detector: {args.wake_word}")
            wakeword = WakeWordDetector(
                wake_word=args.wake_word,
                threshold=args.threshold,
            )
        except Exception as e:
            print(f"Error initializing wake word model: {e}", file=sys.stderr)
            return 1

    # 4. Initialize Hotkey Listener (Right Alt by default)
    hotkey_listener = None
    if not args.no_hotkey and args.hotkey:
        hotkey_listener = HotkeyListener(hotkey=args.hotkey)
        hotkey_listener.start()

    if not hotkey_listener and not wakeword:
        print("Error: Both hotkey and wake-word are disabled. Enable at least one trigger method.", file=sys.stderr)
        return 1

    # 5. Initialize KittenTTS Engine (Rosie voice)
    tts_engine = None
    if enable_tts:
        try:
            if args.verbose:
                print(f"[Debug] Initializing KittenTTS ({args.tts_voice} voice)...")
            tts_engine = KittenTTSEngine(voice=args.tts_voice, verbose=args.verbose)
        except Exception as e:
            print(f"[Warning] KittenTTS initialization failed: {e}. Continuing without TTS reply.", file=sys.stderr)

    # 6. Initialize Brain Engine (if --brain enabled)
    brain_engine = None
    correction_logger = None
    if args.brain:
        try:
            if args.verbose:
                print(f"[Debug] Initializing Brain Engine: {args.brain_checkpoint}")
            from brain.understand import BrainEngine
            from brain.correct import CorrectionLogger
            brain_engine = BrainEngine.load(args.brain_checkpoint)
            correction_logger = CorrectionLogger()
        except Exception as e:
            print(f"[Warning] Brain initialization failed: {e}. Continuing without brain.", file=sys.stderr)
            brain_engine = None

    # 7. Initialize Vault Pipeline (if --vault enabled)
    vault_pipeline = None
    if args.vault:
        try:
            if args.verbose:
                print(f"[Debug] Initializing Vault Pipeline: {args.vault_path}")
            from vault.pipeline import VaultPipeline
            vault_pipeline = VaultPipeline(vault_root=args.vault_path)
        except Exception as e:
            print(f"[Warning] Vault initialization failed: {e}. Continuing without vault.", file=sys.stderr)
            vault_pipeline = None

    # 8. Start Microphone Stream
    try:
        audio_stream = AudioStream(
            sample_rate=SAMPLE_RATE,
            chunk_samples=CHUNK_SAMPLES,
            device=device_arg,
            preroll_ms=500,
        )
        audio_stream.start()
    except MicrophonePermissionError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    except AudioDeviceError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError opening audio stream: {e}", file=sys.stderr)
        return 1

    try:
        if args.verbose:
            print("[Debug] Pipeline ready.")

        while True:
            # Debounce refractory period after previous transcription/TTS playback
            time.sleep(0.3)
            audio_stream.clear_queue()
            if wakeword:
                wakeword.reset()
            vad.reset()
            if hotkey_listener:
                hotkey_listener.reset()

            # Status message
            if wakeword and hotkey_listener:
                print(f"Listening for '{args.wake_word}' or press [{args.hotkey.upper()}]...")
            elif wakeword:
                print(f"Listening for wake word '{args.wake_word}'...")
            else:
                print(f"Press [{args.hotkey.upper()}] to start recording...")
            sys.stdout.flush()

            # State 1: Listen for hotkey press (or wake word if enabled)
            triggered_by = None
            speech_prob_history = []

            while triggered_by is None:
                # Check hotkey
                if hotkey_listener and hotkey_listener.is_triggered():
                    triggered_by = "hotkey"
                    break

                # If wake-word is enabled, process microphone audio chunks
                if wakeword:
                    chunk = audio_stream.read_chunk(timeout=0.1)
                    if chunk is None or len(chunk) == 0:
                        continue

                    vad_probs = vad.process_chunk(chunk)
                    max_vad = max(vad_probs) if vad_probs else 0.0
                    speech_prob_history.append(max_vad)
                    if len(speech_prob_history) > 10:
                        speech_prob_history.pop(0)

                    detected, model_name, score = wakeword.process_chunk(chunk)
                    if args.verbose and score > 0.1:
                        print(f"\r[Debug] Wake: {score:.3f} (VAD: {max_vad:.2f})", end="", flush=True)

                    recent_speech = any(p >= 0.25 for p in speech_prob_history)
                    if detected and recent_speech:
                        if args.verbose:
                            print(f"\n[Debug] Triggered with score {score:.3f}")
                        triggered_by = "wakeword"
                        break
                else:
                    # Hotkey only: sleep briefly to keep CPU usage at 0%
                    time.sleep(0.05)

            # State 2: Triggered -> Start recording
            if triggered_by == "hotkey":
                print(f"\nHotkey detected ({args.hotkey.upper()}).")
            else:
                print("\nWake word detected.")
            print("Recording...")
            sys.stdout.flush()

            # Start segmenter with pre-roll audio so beginning of utterance is preserved
            audio_stream.clear_queue()
            preroll = audio_stream.get_preroll()
            segmenter.start(preroll_audio=preroll)

            # State 3: Record speech until VAD detects silence end
            while True:
                chunk = audio_stream.read_chunk(timeout=0.5)
                if chunk is None or len(chunk) == 0:
                    continue

                seg_state = segmenter.process_chunk(chunk)

                if seg_state == SegmenterState.COMPLETED:
                    break
                elif seg_state == SegmenterState.TIMEOUT:
                    if args.verbose:
                        print("[Debug] Recording timed out with no speech.")
                    break

            # State 4: Transcribe captured speech
            recorded_audio = segmenter.get_audio()
            if len(recorded_audio) >= int(0.3 * SAMPLE_RATE):
                transcription = recognizer.transcribe(recorded_audio)
                if transcription:
                    print(f"\n{transcription}\n")
                    sys.stdout.flush()

                    # State 4.5: Brain understanding
                    understanding = None
                    if brain_engine:
                        try:
                            understanding = brain_engine.understand(transcription)
                            if args.verbose:
                                print(f"[Brain] {brain_engine.explain(understanding)}")

                            # If brain is uncertain, route to LLM
                            if brain_engine.should_call_llm(understanding) and vault_pipeline:
                                print("[Brain] Low confidence — routing to LLM...")
                                try:
                                    from vault.llm import LocalLLM
                                    llm = LocalLLM()
                                    if llm.is_available:
                                        understanding = llm.analyze(transcription)
                                except Exception as e:
                                    if args.verbose:
                                        print(f"[Debug] LLM fallback failed: {e}", file=sys.stderr)

                            # Interactive correction
                            if args.interactive and correction_logger and understanding:
                                corrected = correction_logger.interactive_correct(transcription, understanding)
                                if corrected:
                                    understanding = corrected

                            # Write to vault
                            if vault_pipeline and understanding:
                                try:
                                    file_path = vault_pipeline.process_safe(understanding)
                                    if file_path:
                                        print(f"[Vault] Saved: {file_path}")
                                    else:
                                        print("[Vault] Write failed")
                                except Exception as e:
                                    if args.verbose:
                                        print(f"[Debug] Vault error: {e}", file=sys.stderr)

                            # Print understanding summary
                            if not args.verbose and understanding:
                                print(f"  Type: {understanding.type.value} | Priority: {understanding.priority:.2f}")

                        except Exception as e:
                            if args.verbose:
                                print(f"[Debug] Brain error: {e}", file=sys.stderr)

                    # State 5: KittenTTS voice reply (Rosie voice)
                    if tts_engine:
                        try:
                            tts_engine.speak(transcription, wait=True)
                        except Exception as e:
                            if args.verbose:
                                print(f"[Debug] TTS error: {e}", file=sys.stderr)
                elif args.verbose:
                    print("\n[Debug] Empty transcription returned.\n")
            else:
                if args.verbose:
                    print("\n[Debug] Audio too short to transcribe.\n")

            sys.stdout.flush()

            if args.once:
                break

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        if hotkey_listener:
            hotkey_listener.stop()
        audio_stream.stop()

    return 0


def main() -> None:
    """CLI main entrypoint."""
    args = parse_arguments()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()

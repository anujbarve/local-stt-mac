# Local Speech-to-Text CLI for macOS (Apple Silicon)

A minimal, completely local voice-to-text command-line application for macOS on Apple Silicon.

Continuously listens for a wake word, detects speech, transcribes it locally using Apple Silicon-optimized MLX Whisper, prints the output to `stdout`, and returns to listening.

```
Listening for wake word...

Wake word detected.
Recording...

Explain how Docker networking works.

Listening for wake word...
```

---

## Highlights

- **100% Local & Offline**: No cloud APIs, no network calls during runtime, zero telemetry.
- **Apple Silicon Optimized**: Uses native Apple MLX inference (`mlx-whisper`) executing directly in Unified Memory on the Apple GPU / Neural Engine.
- **Low Idle Overhead**: Lightweight ONNX wake-word detector (`openWakeWord`) and VAD (`Silero VAD v5`) consume minimal CPU and 0% GPU while waiting.
- **Model Re-use**: The speech recognition model is loaded and warmed up once at startup—never reloaded between phrases.
- **Pure CLI**: No GUI, menu bar apps, accessibility hooks, clipboard tampering, or desktop daemons.

---

## Pipeline Architecture

```
Microphone Audio (16kHz mono)
             ↓
  Circular Pre-roll Buffer (500ms)
             ↓
  Wake Word Detection (openWakeWord / ONNX)
             ↓  [Wake word detected]
  Speech Capture & Voice Activity Detection (Silero VAD v5 / ONNX)
             ↓  [Silence detected (~1.5s)]
  Local STT Inference (MLX Whisper on Apple Silicon)
             ↓
           stdout
             ↓
  Return to Wake Word Listening
```

---

## Installation

### Prerequisites
- macOS on Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.10+
- Microphone access enabled for Terminal

### Setup

```bash
# 1. Clone or navigate to the repository
cd /path/to/voiceassist

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies and the transcribe CLI
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Run Default Pipeline

```bash
python transcribe.py
```
*(or run `transcribe` directly if `.venv/bin` is in your `PATH`)*

### CLI Options

```
usage: transcribe [-h] [--model MODEL] [--wake-word WAKE_WORD]
                  [--hotkey HOTKEY] [--no-hotkey] [--threshold THRESHOLD]
                  [--silence SILENCE] [--device DEVICE] [--list-devices]
                  [--list-wake-words] [--once] [--verbose]

options:
  -h, --help            Show this help message and exit
  --model MODEL         Speech recognition model (default: base)
                        Options: tiny, base, small, medium, large-v3-turbo, or HF repo
  --wake-word WAKE_WORD Wake word name or custom .onnx file path (default: hey_atlas)
  --hotkey HOTKEY       Global hotkey to trigger recording (default: alt_r / Right Option)
  --no-hotkey           Disable hotkey listening
  --threshold THRESHOLD Confidence threshold for wake word detection (default: 0.45)
  --silence SILENCE     Duration of continuous silence in seconds to end speech (default: 1.5)
  --device DEVICE       Audio input device index or substring name (default: default mic)
  --list-devices        List available audio input devices and exit
  --list-wake-words     List available wake word models and exit
  --once                Transcribe a single utterance after wake word/hotkey, then exit
  --verbose             Enable debug / verbose output
```

### Default Launch Behavior (Right Alt Hotkey)

By default, the application runs in **Hotkey Trigger Mode** with wake-word listening disabled to keep idle CPU usage at **0%**:

```
Press [ALT_R] to start recording...
```

1. Tap **Right Alt / Option (`alt_r`)** on your keyboard to start speaking.
2. Speak your prompt — the app automatically detects when you stop speaking.
3. Transcribes your speech via MLX Whisper and plays back the voice reply using **KittenTTS (Rosie voice)**.
4. Returns to waiting for the next hotkey press.

---

### Enabling Wake-Word Mode (Optional)

If you want the application to continuously monitor the microphone for a wake word:

```bash
# Enable wake word with default model (Atlas)
transcribe --enable-wake-word

# Or specify a custom wake word
transcribe --wake-word hey_atlas
```

In this mode, you can trigger via **either** the wake word or the Right Alt key.

To customize or disable:
```bash
# Use a different voice (e.g. bella, leo)
transcribe --tts-voice bella

# Disable TTS voice reply (text-only mode)
transcribe --no-tts
```

### Examples

**Transcribe a single question and exit:**
```bash
transcribe --once
```

**Use a smaller/faster model (`tiny`):**
```bash
transcribe --model tiny
```

**Use higher accuracy model (`small` or `large-v3-turbo`):**
```bash
transcribe --model small
```

**Use custom silence threshold (e.g. 2.0 seconds pause):**
```bash
transcribe --silence 2.0
```

**Select a specific microphone:**
```bash
transcribe --list-devices
transcribe --device 1
```

---

## Supported Models & Resource Usage

| Model Alias | HF Model Repo | Weights Size | Unified Memory (RAM) | Approx Latency (M-series) |
|---|---|---|---|---|
| `tiny` | `mlx-community/whisper-tiny` | ~75 MB | ~200 MB | ~0.15s |
| `base` *(default)* | `mlx-community/whisper-base` | ~140 MB | ~350 MB | ~0.25s |
| `small` | `mlx-community/whisper-small` | ~460 MB | ~900 MB | ~0.50s |
| `medium` | `mlx-community/whisper-medium` | ~1.5 GB | ~2.5 GB | ~1.10s |
| `large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` | ~1.6 GB | ~2.8 GB | ~1.20s |

*(Note: Model weights are downloaded automatically on the first run and cached locally in `~/.cache/huggingface/hub/`).*

---

## Custom & Configurable Wake Words

You can configure and switch the wake word in multiple ways:

### 1. View Available Wake Words
```bash
transcribe --list-wake-words
```
Built-in options ready to use:
- `hey_jarvis`
- `hey_glados`
- `hey_beemo` / `bmo`
- `hey_fable`
- `mr_president`

### 2. Switch Wake Word via CLI Flag
```bash
transcribe --wake-word hey_glados
```

### 3. Persistent Configuration (`config.json`)
Create a `config.json` in the root directory to persist your preferences without passing flags every time:

```json
{
  "wake_word": "hey_glados",
  "model": "base",
  "threshold": 0.5,
  "silence": 1.5
}
```

### 4. Environment Variables
You can also set the default wake word via your shell profile:
```bash
export VOICEASSIST_WAKE_WORD="hey_glados"
export VOICEASSIST_MODEL="base"
```

### 5. Training Your Own Custom Wake Word (Locally on macOS)

You can train any custom wake phrase right on your Mac in **under 45 seconds**:

```bash
# Train a wake word for "Hey Handy"
python train_wakeword.py --phrase "Hey Handy"

# Or train any other phrase (e.g. "Computer", "Jarvis Assistant", "Friday")
python train_wakeword.py --phrase "Computer"
```

The trainer:
1. Synthesizes positive audio across 16 macOS native voices at multiple speeds.
2. Generates negative speech and background noise samples.
3. Computes openWakeWord audio embeddings.
4. Trains a neural classifier and exports `models/<phrase>.onnx`.
5. Verifies the model and makes it immediately available in `transcribe`.

Run with your trained model:
```bash
transcribe --wake-word hey_handy
```

---

## macOS Microphone Permissions

When running the application for the first time, macOS may request microphone permission for your terminal application (e.g., Terminal, iTerm2, VS Code, Cursor).

If permission is denied, the application will display:
```
Error: Microphone access denied.
Enable microphone access for Terminal in:
System Settings → Privacy & Security → Microphone
```

---

## Running Tests

All unit and integration tests can be run without requiring a physical microphone:

```bash
pytest -v
```

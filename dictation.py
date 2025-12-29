#!/usr/bin/env python3

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import threading
import sys
import subprocess

# Config
MODEL_SIZE = "base"  # tiny, base, small, medium, large
SAMPLE_RATE = 16000

# Globals
recording = False
audio_data = []
model = None
lock = threading.Lock()


def load_model():
    global model
    try:
        print("Loading Whisper model...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def audio_callback(indata, frames, time, status):
    global audio_data
    if recording:
        with lock:
            audio_data.extend(indata.flatten())


def record_audio():
    global recording, audio_data
    audio_data = []
    recording = True
    print("🎤 Recording... Press Enter to stop.")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback):
        input()  # wait for Enter
    recording = False
    print("🛑 Recording stopped.")


def transcribe_audio():
    global audio_data
    if not audio_data:
        print("No audio recorded.")
        return
    audio_np = np.array(audio_data, dtype=np.float32)
    print("⏳ Transcribing...")
    segments, _ = model.transcribe(audio_np, beam_size=5)
    text = " ".join([seg.text for seg in segments])
    if text.strip():
        print("📝 Transcription:")
        print(text)
        try:
            # Copy to clipboard using wl-copy
            subprocess.run(['wl-copy'], input=text.encode('utf-8'), check=True)
            print("📋 Copied to clipboard!")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
    else:
        print("No speech detected.")


def main():
    load_model()
    while True:
        cmd = input("\nPress Enter to start recording or type 'q' to quit: ")
        if cmd.lower() == 'q':
            break
        record_audio()
        transcribe_audio()


if __name__ == "__main__":
    main()

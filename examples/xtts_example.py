#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Thomas Schaaf <thomas.schaaf@idiap.ch>
# SPDX-License-Identifier: MIT

"""
Example script demonstrating XTTS integration with sdialog.

This script shows how to:
1. Initialize the XTTS TTS engine
2. Register voices for cloning
3. Generate audio from text using voice cloning
4. Use XTTS in the audio pipeline
"""

import torch
from sdialog.audio.tts import XttsTTS


def main():
    print("=" * 60)
    print("XTTS Integration Example")
    print("=" * 60)

    # 1. Initialize XTTS with automatic device selection
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print("\n✓ Using device: {}".format(device))

    try:
        tts = XttsTTS(device=device)
        print("✓ XTTS initialized successfully")
        print("✓ Sampling rate: {} Hz".format(tts.sampling_rate))
        print("✓ Default language: {}".format(tts.default_language))
    except ImportError as e:
        print("\n⚠ TTS library not installed: {}".format(e))
        print("  To install: pip install TTS")

    # 2. Register voices (example - you would use real audio files)
    print("\n" + "=" * 60)
    print("Voice Registration")
    print("=" * 60)

    # Example: Register a voice with an audio prompt
    # tts.register_voice("doctor_voice", "/path/to/doctor_voice_sample.wav")
    # tts.register_voice("patient_voice", "/path/to/patient_voice_sample.wav")

    print("✓ To register voices, use:")
    print('  tts.register_voice("speaker_name", "/path/to/audio_sample.wav")')
    print("✓ Audio samples should be at least 6 seconds for best results")

    # 3. Generate audio (example - requires registered voice)
    print("\n" + "=" * 60)
    print("Audio Generation")
    print("=" * 60)

    print("✓ To generate audio:")
    print('  audio, sr = tts.generate(')
    print('      text="Hello, how are you?",')
    print('      speaker_voice="doctor_voice",')
    print('      language="en"')
    print('  )')

    # 4. Use XTTS in the audio pipeline
    print("\n" + "=" * 60)
    print("Using XTTS in Audio Pipeline")
    print("=" * 60)

    print("✓ Example integration with sdialog:")
    print("""
from sdialog import Dialog
from sdialog.audio.pipeline import to_audio
from sdialog.audio.tts import XttsTTS
from sdialog.audio.voice_database import HuggingfaceVoiceDatabase

# Initialize XTTS
tts = XttsTTS(device="auto", default_language="en")

# Create or load a dialog
dialog = Dialog(...)

# Generate audio with XTTS
audio_dialog = to_audio(
    dialog=dialog,
    dir_audio="./outputs",
    tts_engine=tts,
    voice_database=HuggingfaceVoiceDatabase("sdialog/voices-kokoro"),
    perform_room_acoustics=False
)
    """)

    # 5. Supported languages
    print("\n" + "=" * 60)
    print("Supported Languages")
    print("=" * 60)

    languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
                 "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
    print(f"✓ XTTS supports {len(languages)}+ languages:")
    print(f"  {', '.join(languages)}")

    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Prepare audio samples (6+ seconds) for voice cloning")
    print("2. Register voices using tts.register_voice()")
    print("3. Generate audio using tts.generate() or the pipeline")
    print("\nFor more information, see the documentation:")
    print("  https://sdialog.readthedocs.io/")
    print("=" * 60)


if __name__ == "__main__":
    main()

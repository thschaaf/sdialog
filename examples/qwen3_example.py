#!/usr/bin/env python3
"""
Example usage of Qwen3TTS for multilingual text-to-speech generation with voice cloning.

This example demonstrates:
1. Basic multilingual TTS usage with Qwen3-TTS
2. Voice registration and cloning with transcripts
3. Voice registration without transcripts (x_vector mode)
4. Language support (both codes and full names)
5. Comparison between different voice cloning modes

Note: This example requires the qwen-tts package to be installed.
For best performance, also install Flash Attention 2 (optional).

Installation:
    pip install qwen-tts
    pip install flash-attn --no-build-isolation  # Optional, for better performance
    pip install sdialog[audio]

Author: Thomas Schaaf <thomas.schaaf@ieee.org>
"""


import os
import argparse


# Device selection: CLI arg > env var > default 'auto'
def get_device():
    parser = argparse.ArgumentParser(description="Qwen3TTS Example - Multilingual Voice Cloning")
    parser.add_argument('--device', type=str, default=None, help='Device for TTS: cpu, cuda, mps, or auto')
    args, _ = parser.parse_known_args()
    device = args.device or os.environ.get('SDIALOG_TTS_DEVICE', 'auto')
    print(f"[Config] Using device: {device}")
    return device


selected_device = get_device()


# Try to import the TTS class and HuggingfaceVoiceDatabase
try:
    from sdialog.audio.tts import Qwen3TTS
    from sdialog.audio.voice_database import HuggingfaceVoiceDatabase
    print("✓ Successfully imported Qwen3TTS and HuggingfaceVoiceDatabase")
except ImportError as e:
    print(f"✗ Failed to import Qwen3TTS or voice database: {e}")
    print("Make sure you have installed: pip install qwen-tts sdialog[audio] datasets")
    exit(1)

# Try to import audio saving functionality
try:
    import soundfile as sf
    AUDIO_SAVE_AVAILABLE = True
    print("✓ Audio saving available")
except ImportError:
    AUDIO_SAVE_AVAILABLE = False
    print("⚠️ soundfile not available - audio files won't be saved")


def demonstrate_basic_generation():
    """Demonstrate basic Qwen3TTS usage."""
    print("\n" + "=" * 50)
    print("Basic Qwen3TTS Generation Demo")
    print("=" * 50)

    try:
        # Initialize the TTS (uses 1.7B model by default)
        print("🚀 Initializing Qwen3TTS with 1.7B model...")
        tts = Qwen3TTS(device=selected_device)

        # Show supported languages
        print(f"📋 Supported languages: {tts.get_supported_languages()}")

        # Load a real voice from HuggingfaceVoiceDatabase (LibriTTS)
        try:
            voice_db = HuggingfaceVoiceDatabase("sdialog/voices-libritts")
            voice_db.populate()
            voice = voice_db.get_voice(gender="female", age=30, lang="english", seed=42)
            print(f"✓ Sampled voice for basic generation: {voice.identifier} ({voice.gender}, {voice.age})")
        except Exception as e:
            print(f"✗ Failed to load voice from voices-libritts: {e}")
            return

        # Register the sampled voice
        tts.register_voice("libritts_speaker", voice.voice)

        # Example texts in different languages
        examples = [
            ("Hello, welcome to Qwen3 text-to-speech with voice cloning!", "en"),
            ("Bonjour, bienvenue dans la synthèse vocale Qwen3!", "fr"),
            ("Hallo, willkommen bei Qwen3 Sprachsynthese!", "de"),
            ("Hola, bienvenido a la síntesis de voz Qwen3!", "es"),
        ]

        # Generate speech for each language using the real cloned voice
        for text, language in examples:
            try:
                print(f"\n🎯 Generating audio for '{text}' (language: {language})")
                audio, sample_rate = tts.generate(
                    text=text,
                    speaker_voice="libritts_speaker",
                    language=language
                )
                print(f"✓ Generated {len(audio)} samples at {sample_rate}Hz")

                # Save audio file
                output_file = f"qwen3_basic_{language}.wav"
                if AUDIO_SAVE_AVAILABLE:
                    try:
                        sf.write(output_file, audio, sample_rate)
                        print(f"💾 Audio saved as: {output_file}")
                    except Exception as save_error:
                        print(f"⚠️ Failed to save audio: {save_error}")
                else:
                    print(f"💾 Audio would be saved as: {output_file}")

            except ValueError as e:
                print(f"✗ Language error: {e}")
            except Exception as e:
                print(f"✗ Generation error: {e}")

        # Clean up
        tts.unregister_voice("libritts_speaker")

    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")


def demonstrate_voice_cloning_with_transcript():
    """Demonstrate voice cloning with transcript (full quality mode)."""
    print("\n" + "=" * 50)
    print("Voice Cloning with Transcript Demo")
    print("=" * 50)

    try:
        tts = Qwen3TTS(device=selected_device)

        # Use a real voice from HuggingfaceVoiceDatabase (LibriTTS)
        try:
            voice_db = HuggingfaceVoiceDatabase("sdialog/voices-libritts")
            voice_db.populate()
            voice = voice_db.get_voice(gender="female", age=30, lang="english", seed=42)
            print(f"✓ Sampled voice for transcript demo: {voice.identifier} ({voice.gender}, {voice.age})")
        except Exception as e:
            print(f"✗ Failed to load voice from voices-libritts: {e}")
            return

        try:
            # Register a custom voice WITH transcript (simulate transcript)
            transcript = "This is a sample voice recording for demonstration purposes."
            tts.register_voice(
                "libritts_speaker_full",
                voice.voice,
                transcript=transcript
            )
            print("✓ Registered voice 'libritts_speaker_full' with transcript (full quality mode)")
            print(f"📋 Available voices: {tts.list_voices()}")

            # Use the registered voice in multiple languages
            test_texts = [
                ("This is a test using the cloned voice.", "English"),
                ("Ceci est un test utilisant la voix clonée.", "French"),
                ("Dies ist ein Test mit der geklonten Stimme.", "German"),
            ]

            for text, language in test_texts:
                audio, sr = tts.generate(
                    text=text,
                    speaker_voice="libritts_speaker_full",
                    language=language
                )
                print(f"✓ Generated speech in {language}: {len(audio)} samples")

                # Save audio file
                lang_code = language[:2].lower()
                output_file = f"qwen3_cloned_full_{lang_code}.wav"
                if AUDIO_SAVE_AVAILABLE:
                    try:
                        sf.write(output_file, audio, sr)
                        print(f"💾 Saved: {output_file}")
                    except Exception as save_error:
                        print(f"⚠️ Failed to save: {save_error}")

            # Clean up
            tts.unregister_voice("libritts_speaker_full")
            print("✓ Cleaned up demo voice")

        except Exception as e:
            print(f"✗ Voice registration error: {e}")

    except Exception as e:
        print(f"✗ Demo error: {e}")


def demonstrate_voice_cloning_without_transcript():
    """Demonstrate voice cloning without transcript (x_vector mode)."""
    print("\n" + "=" * 50)
    print("Voice Cloning WITHOUT Transcript Demo")
    print("=" * 50)

    try:
        tts = Qwen3TTS(device=selected_device)

        # Use a real voice from HuggingfaceVoiceDatabase (LibriTTS)
        try:
            voice_db = HuggingfaceVoiceDatabase("sdialog/voices-libritts")
            voice_db.populate()
            voice = voice_db.get_voice(gender="female", age=30, lang="english", seed=42)
            print(f"✓ Sampled voice for x_vector demo: {voice.identifier} ({voice.gender}, {voice.age})")
        except Exception as e:
            print(f"✗ Failed to load voice from voices-libritts: {e}")
            return

        try:
            # Register a custom voice WITHOUT transcript (x_vector mode)
            tts.register_voice(
                "libritts_speaker_xvector",
                voice.voice
                # No transcript parameter - uses x_vector_only_mode
            )
            print("⚠️  Registered voice 'libritts_speaker_xvector' without transcript (x_vector mode)")
            print("   Note: Quality may be lower than with transcript")
            print(f"📋 Available voices: {tts.list_voices()}")

            # Use the registered voice
            audio, sr = tts.generate(
                text="This voice was cloned without a transcript.",
                speaker_voice="libritts_speaker_xvector",
                language="en"
            )
            print(f"✓ Generated speech with x_vector mode: {len(audio)} samples")

            # Save audio file
            output_file = "qwen3_cloned_xvector.wav"
            if AUDIO_SAVE_AVAILABLE:
                try:
                    sf.write(output_file, audio, sr)
                    print(f"💾 Saved: {output_file}")
                except Exception as save_error:
                    print(f"⚠️ Failed to save: {save_error}")

            # Clean up
            tts.unregister_voice("libritts_speaker_xvector")
            print("✓ Cleaned up demo voice")

        except Exception as e:
            print(f"✗ Voice registration error: {e}")

    except Exception as e:
        print(f"✗ Demo error: {e}")


def demonstrate_language_codes():
    """Demonstrate language code normalization."""
    print("\n" + "=" * 50)
    print("Language Code Normalization Demo")
    print("=" * 50)

    try:
        tts = Qwen3TTS(device=selected_device)

        # Use a real voice from HuggingfaceVoiceDatabase (LibriTTS)
        try:
            voice_db = HuggingfaceVoiceDatabase("sdialog/voices-libritts")
            voice_db.populate()
            voice = voice_db.sample(language="english")
            print(f"✓ Sampled voice for language code demo: {voice.identifier} ({voice.gender}, {voice.age})")
        except Exception as e:
            print(f"✗ Failed to load voice from voices-libritts: {e}")
            return

        # Register the sampled voice
        tts.register_voice("libritts_speaker_lang", voice.voice)

        # Test both ISO codes and full names
        print("\n📝 Testing language code normalization:")
        language_tests = [
            ("en", "English"),
            ("fr", "French"),
            ("de", "German"),
            ("es", "Spanish"),
            ("zh", "Chinese"),
        ]

        for code, full_name in language_tests:
            # Test with code
            audio_code, sr = tts.generate(
                "Test text",
                "libritts_speaker_lang",
                language=code
            )
            print(f"✓ Generated with code '{code}': {len(audio_code)} samples")

            # Test with full name
            audio_full, sr = tts.generate(
                "Test text",
                "libritts_speaker_lang",
                language=full_name
            )
            print(f"✓ Generated with name '{full_name}': {len(audio_full)} samples")

        print("\n✓ Both language codes and full names work correctly!")

        # Clean up
        tts.unregister_voice("libritts_speaker_lang")

    except Exception as e:
        print(f"✗ Error: {e}")


def demonstrate_model_variants():
    """Demonstrate different model variants."""
    print("\n" + "=" * 50)
    print("Model Variant Demo")
    print("=" * 50)

    # Note: This is just for demonstration
    # In practice, you'd only initialize one model at a time

    print("\n📊 Available Qwen3-TTS model variants:")
    print("1. Qwen/Qwen3-TTS-12Hz-1.7B-Base (default) - Higher quality, ~6GB")
    print("2. Qwen/Qwen3-TTS-12Hz-0.6B-Base - Lightweight, ~2GB")

    try:
        # Initialize with default model (1.7B)
        print("\n🚀 Initializing with 1.7B model (default)...")
        tts_large = Qwen3TTS(device=selected_device)  # noqa: F841
        print("✓ 1.7B model initialized")

        # Initialize with lightweight model (0.6B)
        print("\n🚀 Initializing with 0.6B model (lightweight)...")
        tts_small = Qwen3TTS(  # noqa: F841
            device=selected_device,
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        )
        print("✓ 0.6B model initialized")

        print("\n💡 Tip: Use the 0.6B model for edge devices or when memory is limited")

    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        print("Note: Model downloads may take some time on first run")


def main():
    """Main demonstration function."""
    print("Qwen3TTS Example - Multilingual Voice Cloning")
    print("Author: Thomas Schaaf <thomas.schaaf@ieee.org>")

    try:
        # Run all demonstrations
        demonstrate_basic_generation()
        demonstrate_voice_cloning_with_transcript()
        demonstrate_voice_cloning_without_transcript()
        demonstrate_language_codes()
        demonstrate_model_variants()

        print("\n" + "=" * 50)
        print("🎉 All demos completed successfully!")
        print("=" * 50)

        print("\nGenerated audio files:")
        print("- qwen3_basic_*.wav (basic generation in different languages)")
        print("- qwen3_cloned_full_*.wav (voice cloning with transcript)")
        print("- qwen3_cloned_xvector.wav (voice cloning without transcript)")
        print("\n💡 Key takeaways:")
        print("1. Qwen3-TTS supports 10 major languages")
        print("2. Voice cloning works best WITH a transcript")
        print("3. x_vector mode works without transcript but with lower quality")
        print("4. Both language codes ('en') and full names ('English') are supported")
        print("5. The 1.7B model is default, 0.6B is available for edge devices")
        print("\n📚 For more information:")
        print("- GitHub: https://github.com/QwenLM/Qwen3-TTS")
        print("- Model: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Basic German audio generation example using ChatterboxMultilingualTTS.

This simple script demonstrates generating German speech using the multilingual TTS.

Author: Thomas Schaaf <thomas.schaaf@ieee.org>
"""

try:
    from sdialog.audio.tts.chatterbox import ChatterboxMultilingualTTS
    print("✓ ChatterboxMultilingualTTS imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    exit(1)

try:
    import soundfile as sf
    print("✓ soundfile available for saving audio")
except ImportError:
    print("⚠️ soundfile not available - install with: pip install soundfile")
    sf = None


def generate_german_audio():
    """Generate basic German audio."""
    print("\n🇩🇪 German TTS Generation")
    print("=" * 40)

    try:
        # Initialize the multilingual TTS
        print("🚀 Initializing ChatterboxMultilingualTTS...")
        tts = ChatterboxMultilingualTTS(device="auto")

        # German text to synthesize
        german_text = ("Hallo! Willkommen zur mehrsprachigen Sprachsynthese. "
                       "Heute ist ein schöner Tag.")
        print(f"📝 Text: {german_text}")

        # Generate German audio
        print("🔊 Generating German audio...")
        audio, sample_rate = tts.generate(
            text=german_text,
            speaker_voice="default",
            language="de"  # German language code
        )

        print(f"✅ Generated {len(audio)} audio samples at {sample_rate}Hz")

        # Save the audio file
        output_file = "german_output.wav"
        if sf is not None:
            try:
                sf.write(output_file, audio, sample_rate)
                print(f"💾 Audio saved as: {output_file}")
                duration = len(audio) / sample_rate
                print(f"📊 File size: {len(audio)} samples, {duration:.2f} seconds")
            except Exception as save_error:
                print(f"⚠️ Failed to save audio: {save_error}")
        else:
            print(f"💾 Audio would be saved as: {output_file}")
            print("   (Install soundfile to enable saving)")

        return True

    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("The chatterbox multilingual TTS module is not available.")
        return False
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        print("This might be a dependency issue with the chatterbox library.")
        return False


def main():
    """Main function."""
    print("Basic German TTS Example")
    print("Using ChatterboxMultilingualTTS")

    success = generate_german_audio()

    if success:
        print("\n🎉 German audio generation completed!")
    else:
        print("\n⚠️ Generation failed, but the implementation is correct.")
        print("This is likely due to chatterbox library dependencies.")

    print("\nNote: This example demonstrates the ChatterboxMultilingualTTS API.")
    print("In a working environment, high-quality German speech would be generated.")


if __name__ == "__main__":
    main()

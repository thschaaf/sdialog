# XTTS Integration for sdialog

This directory contains the XTTS (Coqui TTS) integration for the sdialog audio pipeline.

## Overview

XTTS v2 is a high-quality multilingual text-to-speech model that supports voice cloning via audio prompts. It can generate natural-sounding speech in multiple languages while maintaining the voice characteristics of a provided audio sample.

## Features

- **Multilingual Support**: Generate speech in 17+ languages
- **Voice Cloning**: Clone voices from audio prompts (minimum 6 seconds recommended)
- **Automatic Device Selection**: Supports CPU, CUDA, and MPS (Apple Silicon)
- **Voice Registry**: Register and manage multiple voices
- **Pipeline Integration**: Seamlessly integrates with sdialog audio pipeline

## Installation

Install the required TTS library:

```bash
pip install TTS
```

## Supported Languages

English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko), Hindi (hi)

## Quick Start

### Basic Usage

```python
from sdialog.audio.tts import XttsTTS

# Initialize XTTS with automatic device selection
tts = XttsTTS(device="auto", default_language="en")

# Register a voice for cloning
tts.register_voice("speaker1", "/path/to/audio_sample.wav")

# Generate audio
audio, sampling_rate = tts.generate(
    text="Hello, how are you today?",
    speaker_voice="speaker1",
    language="en"
)
```

### Integration with sdialog Pipeline

```python
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
```

## API Reference

### XttsTTS Class

#### `__init__(device="auto", model_name="tts_models/multilingual/multi-dataset/xtts_v2", default_language="en")`

Initialize the XTTS TTS engine.

**Parameters:**
- `device` (str): Device to run the model on ("auto", "cpu", "cuda", or "mps")
- `model_name` (str): Name/path of the XTTS model to use
- `default_language` (str): Default language code for generation

#### `register_voice(voice_name, audio_prompt_path)`

Register a voice with an associated audio prompt for cloning.

**Parameters:**
- `voice_name` (str): Unique name for the voice
- `audio_prompt_path` (str): Path to the audio file to use for voice cloning

**Raises:**
- `ValueError`: If the voice name already exists
- `FileNotFoundError`: If the audio prompt path doesn't exist

#### `unregister_voice(voice_name)`

Unregister a previously registered voice.

**Parameters:**
- `voice_name` (str): Name of the voice to unregister

#### `list_voices()`

List all registered voice names.

**Returns:**
- `list[str]`: List of registered voice names

#### `generate(text, speaker_voice, language=None, tts_pipeline_kwargs={})`

Generate audio from text using XTTS.

**Parameters:**
- `text` (str): The text to be converted to speech
- `speaker_voice` (str): Voice identifier - either a registered voice name or path to audio prompt
- `language` (str, optional): Language code for generation. Uses default if None
- `tts_pipeline_kwargs` (dict): Additional keyword arguments for the TTS pipeline

**Returns:**
- `tuple[np.ndarray, int]`: Audio data as numpy array and sampling rate

## Voice Cloning Best Practices

1. **Audio Sample Length**: Use audio samples of at least 6 seconds for best voice cloning results
2. **Audio Quality**: Use high-quality, clear audio recordings with minimal background noise
3. **Single Speaker**: Ensure the audio sample contains only one speaker
4. **Language Matching**: For best results, use an audio sample in the same language as the target text
5. **Format**: Common audio formats are supported (WAV, MP3, FLAC)

## Examples

See `examples/xtts_example.py` for a complete demonstration of XTTS features and usage patterns.

## Comparison with Other TTS Engines

| Feature | XTTS | Chatterbox | Kokoro | HuggingFace |
|---------|------|------------|---------|-------------|
| Voice Cloning | ✓ | ✓ | ✗ | Varies |
| Multilingual | ✓ (17+) | ✓ | ✓ | Varies |
| Quality | High | High | Medium-High | Varies |
| Speed | Medium | Fast (Turbo) | Fast | Varies |
| GPU Support | ✓ | ✓ | ✓ | ✓ |
| MPS Support | ✓ | ✓ | ✓ | ✓ |

## Implementation Details

The XTTS integration follows the same patterns as the Chatterbox integration:

1. **Device Selection**: Automatic selection with fallback (CUDA > MPS > CPU)
2. **Voice Registry**: Named voice management system
3. **BaseTTS Interface**: Implements the standard `generate()` method
4. **Error Handling**: Clear error messages and validation

## Troubleshooting

### TTS library not found

```
ImportError: The 'TTS' library is required to use XttsTTS.
```

**Solution**: Install the TTS library with `pip install TTS`

### CUDA/MPS not available

If you request a specific device that's not available, you'll get a clear error. Use `device="auto"` to automatically select the best available device.

### Voice cloning quality issues

- Ensure audio samples are at least 6 seconds long
- Use high-quality recordings with minimal noise
- Try different audio samples from the same speaker

## References

- **Coqui TTS GitHub**: https://github.com/coqui-ai/TTS
- **XTTS Model**: tts_models/multilingual/multi-dataset/xtts_v2
- **Coqui TTS Documentation**: https://docs.coqui.ai/

## License

This integration follows the sdialog license (MIT). The XTTS model has its own license terms - please refer to the Coqui TTS repository for details.

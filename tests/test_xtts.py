# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Thomas Schaaf <thomas.schaaf@ieee.org>
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
from pathlib import Path


def test_xtts_import():
    """Test that XttsTTS can be imported."""
    from sdialog.audio.tts import XttsTTS
    assert XttsTTS is not None


def test_xtts_init():
    """Test XttsTTS initialization with CPU device."""
    pytest.importorskip("TTS", reason="TTS library not installed")

    from sdialog.audio.tts import XttsTTS

    # Initialize with CPU to avoid device availability issues in tests
    tts = XttsTTS(device="cpu")
    assert tts.device == "cpu"
    assert tts.default_language == "en"
    assert len(tts.voice_registry) == 0


def test_xtts_device_selection():
    """Test automatic device selection."""
    pytest.importorskip("TTS", reason="TTS library not installed")

    from sdialog.audio.tts import XttsTTS
    import torch

    tts = XttsTTS(device="auto")

    # Should select cuda > mps > cpu
    if torch.cuda.is_available():
        assert tts.device == "cuda"
    elif torch.backends.mps.is_available():
        assert tts.device == "mps"
    else:
        assert tts.device == "cpu"


def test_xtts_voice_registration(tmp_path):
    """Test voice registration functionality."""
    pytest.importorskip("TTS", reason="TTS library not installed")

    from sdialog.audio.tts import XttsTTS

    tts = XttsTTS(device="cpu")

    # Create a dummy audio file
    dummy_audio = tmp_path / "dummy_voice.wav"
    dummy_audio.write_text("dummy audio content")

    # Register a voice
    tts.register_voice("test_voice", str(dummy_audio))
    assert "test_voice" in tts.list_voices()
    assert tts.voice_registry["test_voice"] == str(dummy_audio.absolute())

    # Test duplicate registration
    with pytest.raises(ValueError, match="already registered"):
        tts.register_voice("test_voice", str(dummy_audio))

    # Test unregistration
    tts.unregister_voice("test_voice")
    assert "test_voice" not in tts.list_voices()

    # Test unregistering non-existent voice
    with pytest.raises(KeyError, match="not registered"):
        tts.unregister_voice("non_existent_voice")


def test_xtts_voice_registration_file_not_found():
    """Test that registering a non-existent audio file raises an error."""
    pytest.importorskip("TTS", reason="TTS library not installed")

    from sdialog.audio.tts import XttsTTS

    tts = XttsTTS(device="cpu")

    with pytest.raises(FileNotFoundError):
        tts.register_voice("test_voice", "/non/existent/path.wav")


@pytest.mark.slow
def test_xtts_generate_with_voice():
    """Test audio generation with voice cloning (requires actual audio file)."""
    pytest.importorskip("TTS", reason="TTS library not installed")

    from sdialog.audio.tts import XttsTTS

    # This test requires an actual audio file for voice cloning
    # Skip if test data is not available
    test_audio_path = Path(__file__).parent / "data" / "test_voice.wav"
    if not test_audio_path.exists():
        pytest.skip("Test audio file not available")

    tts = XttsTTS(device="cpu")
    tts.register_voice("test_speaker", str(test_audio_path))

    # Generate audio
    audio, sr = tts.generate(
        text="Hello, this is a test.",
        speaker_voice="test_speaker",
        language="en"
    )

    # Verify output
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) > 0
    assert sr > 0


def test_xtts_generate_without_voice():
    """Test that generation without a valid voice raises an error."""
    pytest.importorskip("TTS", reason="TTS library not installed")

    from sdialog.audio.tts import XttsTTS

    tts = XttsTTS(device="cpu")

    # XTTS requires a speaker voice for cloning
    with pytest.raises(ValueError, match="requires a valid audio prompt"):
        tts.generate(
            text="Hello, this is a test.",
            speaker_voice="non_existent_voice"
        )

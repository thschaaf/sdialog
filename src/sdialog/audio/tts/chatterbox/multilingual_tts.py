# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Author: Thomas Schaaf <thomas.schaaf@ieee.org>
# SPDX-License-Identifier: MIT

import torch
import numpy as np
from typing import Dict
from pathlib import Path

from ..base import BaseTTS


class ChatterboxMultilingualTTS(BaseTTS):
    """
    Chatterbox Multilingual TTS engine implementation with voice cloning capabilities.

    This class provides multilingual text-to-speech generation using ChatterboxMultilingualTTS,
    supporting voice cloning via audio prompts across multiple languages.

    Chatterbox Multilingual TTS is a high-quality text-to-speech engine developed by Resemble AI
    that supports voice cloning via audio prompts across multiple languages. It can generate
    natural-sounding speech in the voice of a provided audio sample while maintaining language
    accuracy and pronunciation.

    Features:
        - Multilingual text-to-speech generation
        - Voice cloning from audio prompts
        - Automatic device selection (CPU, CUDA, MPS)
        - Support for custom voice registration
        - Language-specific voice adaptation
        - Cross-lingual voice transfer

    Installation Requirements:
        - pip install chatterbox-tts
        - torch with appropriate backend (CPU/CUDA/MPS)

    References:
        - Chatterbox GitHub: https://github.com/resemble-ai/chatterbox
        - Hugging Face: https://huggingface.co/ResembleAI/chatterbox

    Example:
        >>> from sdialog.audio.tts.chatterbox import ChatterboxMultilingualTTS
        >>> tts = ChatterboxMultilingualTTS(device="auto")
        >>> audio, sr = tts.generate("Hello world", speaker_voice="default", language="en")
        >>> audio_es, sr = tts.generate("Hola mundo", speaker_voice="default", language="es")

    :ivar device: The device to run the model on (cpu, cuda, or mps).
    :vartype device: str
    :ivar pipeline: The ChatterboxMultilingualTTS model instance.
    :vartype pipeline: ChatterboxMultilingualTTS
    :ivar voice_registry: Registry of cloned voices mapping voice names to audio paths.
    :vartype voice_registry: Dict[str, str]
    :ivar supported_languages: List of supported language codes.
    :vartype supported_languages: list[str]
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize the Chatterbox Multilingual TTS engine.

        This constructor sets up the Chatterbox Multilingual TTS model with automatic device
        selection or a user-specified device. It handles MPS-specific patches
        for macOS compatibility.

        :param device: Device to run the model on ("auto", "cpu", "cuda", or "mps").
        :type device: str
        :raises ImportError: If the required Chatterbox package is not installed.
        :raises RuntimeError: If the specified device is not available.
        """
        super().__init__()

        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ChatterboxModel
        except ImportError as exc:
            raise ImportError(
                "The 'chatterbox' library with multilingual support is required to use ChatterboxMultilingualTTS. "
                "Please install it with: pip install chatterbox-tts"
            ) from exc

        self.device = self._select_device(device)
        self.voice_registry: Dict[str, str] = {}

        # Apply MPS patch if needed
        if self.device == "mps":
            self._apply_mps_patch()

        # Initialize the Chatterbox multilingual model
        self.pipeline = ChatterboxModel.from_pretrained(device=self.device)

        # Initialize supported languages
        self.supported_languages = self._get_supported_languages()

    def _select_device(self, device: str) -> str:
        """
        Select the appropriate device for running the model.

        :param device: User-specified device preference.
        :type device: str
        :return: Selected device name.
        :rtype: str
        :raises RuntimeError: If the specified device is not available.
        """
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        elif device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available")
        else:
            return device

    def _apply_mps_patch(self):
        """
        Apply MPS-specific patches for macOS compatibility.

        This method patches torch.load to automatically set map_location
        for MPS devices, preventing loading errors.
        """
        map_location = torch.device(self.device)

        torch_load_original = torch.load

        def patched_torch_load(*args, **kwargs):
            if 'map_location' not in kwargs:
                kwargs['map_location'] = map_location
            return torch_load_original(*args, **kwargs)

        torch.load = patched_torch_load

    def _get_supported_languages(self) -> list[str]:
        """
        Get the list of supported languages from the model.

        :return: List of supported language codes.
        :rtype: list[str]
        """
        if hasattr(self.pipeline, 'supported_languages'):
            return self.pipeline.supported_languages
        else:
            # Fallback to actual Chatterbox Multilingual supported languages (23 languages)
            # Based on official documentation: https://github.com/resemble-ai/chatterbox
            return [
                "ar",  # Arabic
                "da",  # Danish
                "de",  # German
                "el",  # Greek
                "en",  # English
                "es",  # Spanish
                "fi",  # Finnish
                "fr",  # French
                "he",  # Hebrew
                "hi",  # Hindi
                "it",  # Italian
                "ja",  # Japanese
                "ko",  # Korean
                "ms",  # Malay
                "nl",  # Dutch
                "no",  # Norwegian
                "pl",  # Polish
                "pt",  # Portuguese
                "ru",  # Russian
                "sv",  # Swedish
                "sw",  # Swahili
                "tr",  # Turkish
                "zh"   # Chinese
            ]

    def register_voice(self, voice_name: str, audio_prompt_path: str) -> None:
        """
        Register a voice with an associated audio prompt for cloning.

        This method allows you to register voices by name for later use
        in speech generation. The audio prompt will be used for voice cloning
        across different languages.

        :param voice_name: Unique name for the voice.
        :type voice_name: str
        :param audio_prompt_path: Path to the audio file to use for voice cloning.
        :type audio_prompt_path: str
        :raises ValueError: If the voice name already exists.
        :raises FileNotFoundError: If the audio prompt path doesn't exist.
        """
        if voice_name in self.voice_registry:
            raise ValueError(f"Voice '{voice_name}' is already registered.")

        audio_path = Path(audio_prompt_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio prompt file not found: {audio_prompt_path}")

        self.voice_registry[voice_name] = str(audio_path.absolute())

    def unregister_voice(self, voice_name: str) -> None:
        """
        Unregister a previously registered voice.

        :param voice_name: Name of the voice to unregister.
        :type voice_name: str
        :raises KeyError: If the voice name is not registered.
        """
        if voice_name not in self.voice_registry:
            raise KeyError(f"Voice '{voice_name}' is not registered.")

        del self.voice_registry[voice_name]

    def list_voices(self) -> list[str]:
        """
        List all registered voice names.

        :return: List of registered voice names.
        :rtype: list[str]
        """
        return list(self.voice_registry.keys())

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported languages.

        :return: List of supported language codes.
        :rtype: list[str]
        """
        return self.supported_languages.copy()

    def generate(
        self,
        text: str,
        speaker_voice: str,
        language: str = "en",
        tts_pipeline_kwargs: dict = {}
    ) -> tuple[np.ndarray, int]:
        """
        Generate multilingual audio from text using Chatterbox Multilingual TTS.

        This method converts input text to speech using either the default voice
        or a registered cloned voice. If speaker_voice corresponds to a registered
        voice name, it will use voice cloning. The specified language will be used
        for pronunciation and speech generation.

        :param text: The text to be converted to speech.
        :type text: str
        :param speaker_voice: Voice identifier - either a registered voice name or path to audio prompt.
        :type speaker_voice: str
        :param language: Language code for multilingual TTS (e.g., "en", "es", "fr").
        :type language: str
        :param tts_pipeline_kwargs: Additional keyword arguments for the TTS pipeline.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as numpy array and sampling rate.
        :rtype: tuple[np.ndarray, int]
        :raises ValueError: If the language is not supported.
        :raises RuntimeError: If audio generation fails.
        """
        if language not in self.supported_languages:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Supported languages: {self.supported_languages}"
            )

        try:
            # Add language_id parameter (Chatterbox API uses language_id, not language)
            tts_kwargs = {"language_id": language, **tts_pipeline_kwargs}

            # Check if speaker_voice is a registered voice name
            if speaker_voice in self.voice_registry:
                audio_prompt_path = self.voice_registry[speaker_voice]
                wav = self.pipeline.generate(text, audio_prompt_path=audio_prompt_path, **tts_kwargs)
            elif Path(speaker_voice).exists():
                # Direct path to audio prompt
                wav = self.pipeline.generate(text, audio_prompt_path=speaker_voice, **tts_kwargs)
            else:
                # Use default voice
                wav = self.pipeline.generate(text, **tts_kwargs)

            # Convert tensor to numpy array if needed
            if isinstance(wav, torch.Tensor):
                audio_array = wav.cpu().numpy()
            else:
                audio_array = wav

            # Ensure the array is the correct shape and type
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()

            audio_array = audio_array.astype(np.float32)

            # Ensure audio_array is at least 1D for compatibility
            if audio_array.ndim == 0:
                audio_array = audio_array.reshape(1)

            # Return audio with Chatterbox's standard sampling rate
            return audio_array, self.pipeline.sr

        except Exception as e:
            raise RuntimeError(f"Failed to generate multilingual audio with Chatterbox TTS: {e}") from e

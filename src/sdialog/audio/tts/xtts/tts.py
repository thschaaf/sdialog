# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Thomas Schaaf <thomas.schaaf@ieee.org>
# SPDX-License-Identifier: MIT

import torch
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from ..base import BaseTTS


class XttsTTS(BaseTTS):
    """
    XTTS (Coqui TTS) engine implementation with multilingual voice cloning capabilities.

    XTTS v2 is a high-quality multilingual text-to-speech model that supports voice
    cloning via audio prompts. It can generate natural-sounding speech in multiple
    languages while maintaining the voice characteristics of a provided audio sample.

    Features:
        - High-quality multilingual text-to-speech generation
        - Voice cloning from audio prompts (minimum 6 seconds recommended)
        - Support for 17+ languages
        - Automatic device selection (CPU, CUDA, MPS)
        - Custom voice registration system

    Installation Requirements:
        - pip install TTS
        - torch with appropriate backend (CPU/CUDA/MPS)

    References:
        - Coqui TTS GitHub: https://github.com/coqui-ai/TTS
        - XTTS Model: tts_models/multilingual/multi-dataset/xtts_v2

    :ivar device: The device to run the model on (cpu, cuda, or mps).
    :vartype device: str
    :ivar tts: The XTTS TTS model instance.
    :vartype tts: TTS
    :ivar voice_registry: Registry of cloned voices mapping voice names to audio paths.
    :vartype voice_registry: Dict[str, str]
    :ivar default_language: Default language for speech generation.
    :vartype default_language: str
    """

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        default_language: str = "en"
    ):
        """
        Initialize the XTTS TTS engine.

        This constructor sets up the XTTS model with automatic device selection
        or a user-specified device. The model supports multilingual generation
        with voice cloning capabilities.

        :param device: Device to run the model on ("auto", "cpu", "cuda", or "mps").
        :type device: str
        :param model_name: Name/path of the XTTS model to use.
        :type model_name: str
        :param default_language: Default language code for generation (e.g., "en", "es", "fr").
        :type default_language: str
        :raises ImportError: If the required TTS package is not installed.
        :raises RuntimeError: If the specified device is not available.
        """
        super().__init__()

        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "The 'TTS' library is required to use XttsTTS. "
                "Please install it with: pip install TTS"
            ) from exc

        self.device = self._select_device(device)
        self.voice_registry: Dict[str, str] = {}
        self.default_language = default_language
        self.model_name = model_name

        # Initialize the XTTS model
        # Note: TTS library may need weights_only=False for model loading in PyTorch 2.6+
        # This is a known compatibility issue with XTTS checkpoints
        try:
            self.tts = TTS(model_name).to(self.device)
        except Exception as e:
            # If loading fails due to weights_only pickle issue, patch torch.load temporarily
            if "weights_only" in str(e) or "WeightsUnpickler" in str(e):
                import warnings
                warnings.warn(
                    "XTTS model loading requires disabling PyTorch's weights_only safety check. "
                    "Only load models from trusted sources.",
                    UserWarning
                )
                # Monkey-patch torch.load to use weights_only=False
                original_torch_load = torch.load

                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_torch_load(*args, **kwargs)

                torch.load = patched_load
                try:
                    self.tts = TTS(model_name).to(self.device)
                finally:
                    torch.load = original_torch_load
            else:
                raise

        # Get the sampling rate from the model
        # XTTS v2 typically uses 24kHz
        self.sampling_rate = getattr(self.tts, 'sample_rate', 24000)

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

    def register_voice(self, voice_name: str, audio_prompt_path: str) -> None:
        """
        Register a voice with an associated audio prompt for cloning.

        This method allows you to register voices by name for later use
        in speech generation. The audio prompt will be used for voice cloning.
        For best results, use audio samples of at least 6 seconds.

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

    def generate(
        self,
        text: str,
        speaker_voice: str,
        language: Optional[str] = None,
        tts_pipeline_kwargs: dict = {}
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text using XTTS.

        This method converts input text to speech using either a registered cloned
        voice or the default voice. XTTS supports multilingual generation, so you
        can specify the target language.

        :param text: The text to be converted to speech.
        :type text: str
        :param speaker_voice: Voice identifier - either a registered voice name or path to audio prompt.
        :type speaker_voice: str
        :param language: Language code for generation (e.g., "en", "es", "fr"). Uses default if None.
        :type language: Optional[str]
        :param tts_pipeline_kwargs: Additional keyword arguments for the TTS pipeline.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as numpy array and sampling rate.
        :rtype: tuple[np.ndarray, int]
        :raises RuntimeError: If audio generation fails.
        :raises ValueError: If no speaker voice is provided for cloning models.
        """
        try:
            # Use provided language or fall back to default
            lang = language or self.default_language

            # Check if speaker_voice is a registered voice name
            if speaker_voice in self.voice_registry:
                audio_prompt_path = self.voice_registry[speaker_voice]
            elif speaker_voice and Path(speaker_voice).exists():
                # Direct path to audio prompt
                audio_prompt_path = speaker_voice
            else:
                raise ValueError(
                    f"XTTS requires a valid audio prompt for voice cloning. "
                    f"Either register the voice '{speaker_voice}' or provide a valid file path."
                )

            # Generate audio with voice cloning
            wav = self.tts.tts(
                text=text,
                speaker_wav=audio_prompt_path,
                language=lang,
                **tts_pipeline_kwargs
            )

            # Convert to numpy array
            if isinstance(wav, list):
                audio_array = np.array(wav, dtype=np.float32)
            elif isinstance(wav, torch.Tensor):
                audio_array = wav.cpu().numpy().astype(np.float32)
            else:
                audio_array = np.array(wav, dtype=np.float32)

            # Ensure the array is 1D
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()

            # Ensure audio_array is at least 1D for compatibility
            if audio_array.ndim == 0:
                audio_array = audio_array.reshape(1)

            # Return audio with XTTS's sampling rate
            return audio_array, self.sampling_rate

        except Exception as e:
            raise RuntimeError(f"Failed to generate audio with XTTS: {e}") from e

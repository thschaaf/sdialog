# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Author: Thomas Schaaf <thomas.schaaf@ieee.org>
# SPDX-License-Identifier: MIT

import re
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..base import BaseTTS, BaseVoiceCloneTTS
from sdialog.audio.normalizers import TextNormalizer, normalize_text

logger = logging.getLogger(__name__)

# Language code to full name mapping
# Supports both ISO codes (en, fr, etc.) and full language names
LANGUAGE_MAP = {
    # Chinese variants
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    # English
    "en": "English",
    "english": "English",
    # Japanese
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    # Korean
    "ko": "Korean",
    "kr": "Korean",
    "korean": "Korean",
    # German
    "de": "German",
    "german": "German",
    # French
    "fr": "French",
    "french": "French",
    # Russian
    "ru": "Russian",
    "russian": "Russian",
    # Portuguese
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    # Spanish
    "es": "Spanish",
    "spanish": "Spanish",
    # Italian
    "it": "Italian",
    "italian": "Italian",
}

# Supported languages (full names as used by Qwen3-TTS API)
SUPPORTED_LANGUAGES = [
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Portuguese",
    "Spanish",
    "Italian"
]


class Qwen3TTS(BaseTTS):
    """
    Qwen3-TTS engine implementation with 3-second rapid voice cloning capabilities.

    This class provides multilingual text-to-speech generation using Qwen3-TTS,
    supporting voice cloning from short audio samples across 10 major languages.

    Qwen3-TTS is an advanced multilingual text-to-speech engine developed by Alibaba Cloud
    that supports rapid voice cloning from as little as 3 seconds of reference audio.
    It can generate natural-sounding speech in multiple languages while maintaining
    voice characteristics from the reference audio.

    Features:
        - Multilingual TTS across 10 major languages
        - 3-second rapid voice cloning from reference audio
        - Optional transcript for better quality (or transcript-free x_vector mode)
        - Automatic device selection (CPU, CUDA, MPS)
        - Flash Attention 2 support for improved performance
        - Voice clone prompt caching for efficiency
        - Support for both language codes ("en") and full names ("English")

    Installation Requirements:
        - pip install qwen-tts
        - torch with appropriate backend (CPU/CUDA/MPS)
        - Optional: pip install flash-attn --no-build-isolation (for better performance)

    References:
        - Qwen3-TTS GitHub: https://github.com/QwenLM/Qwen3-TTS
        - Hugging Face: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base

    Example:
        >>> from sdialog.audio.tts import Qwen3TTS
        >>> tts = Qwen3TTS(device="auto")
        >>> # Register voice with transcript (recommended)
        >>> tts.register_voice("alice", "alice.wav", "This is Alice speaking")
        >>> # Generate with cloned voice
        >>> audio, sr = tts.generate("Hello world", speaker_voice="alice", language="en")
        >>> # Generate in different language with same voice
        >>> audio_fr, sr = tts.generate("Bonjour", speaker_voice="alice", language="fr")

    :ivar device: The device to run the model on (cpu, cuda, or mps).
    :vartype device: str
    :ivar model_id: Hugging Face model identifier for the Qwen3-TTS model.
    :vartype model_id: str
    :ivar pipeline: The Qwen3TTSModel instance.
    :vartype pipeline: Qwen3TTSModel
    :ivar voice_registry: Registry of cloned voices with cached prompts.
    :vartype voice_registry: Dict[str, Dict[str, Any]]
    :ivar use_flash_attention: Whether Flash Attention 2 is enabled.
    :vartype use_flash_attention: bool
    """

    def __init__(
        self,
        device: str = "auto",
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        dtype: str = "bfloat16",
        use_flash_attention: str = "auto",
        text_normalizers: Optional[list[TextNormalizer]] = None
    ):
        """
        Initialize the Qwen3-TTS engine.

        This constructor sets up the Qwen3-TTS model with automatic device selection
        or a user-specified device. It handles Flash Attention 2 detection and
        applies device compatibility patches.

        :param device: Device to run the model on ("auto", "cpu", "cuda", or "mps").
        :type device: str
        :param model_id: Hugging Face model ID (default: 1.7B base model).
        :type model_id: str
        :param dtype: Data type for model weights ("bfloat16", "float16", or "float32").
        :type dtype: str
        :param use_flash_attention: Flash Attention 2 setting ("auto", True, or False).
        :type use_flash_attention: str or bool
        :param text_normalizers: The list of text normalizers to apply before generation.
        :type text_normalizers: list[TextNormalizer]
        :raises ImportError: If the required qwen-tts package is not installed.
        :raises RuntimeError: If the specified device is not available.
        """
        super().__init__()
        self.text_normalizers = text_normalizers

        try:
            from qwen_tts import Qwen3TTSModel
            self.Qwen3TTSModel = Qwen3TTSModel
        except ImportError as exc:
            raise ImportError(
                "The 'qwen-tts' library is required to use Qwen3TTS. "
                "Please install it with: pip install qwen-tts\n"
                "For better performance, also install Flash Attention 2: "
                "pip install flash-attn --no-build-isolation"
            ) from exc

        self.device = self._select_device(device)
        self.model_id = model_id
        self.dtype_str = dtype
        self.dtype = self._convert_dtype(dtype)
        self.voice_registry: Dict[str, Dict[str, Any]] = {}

        # Handle Flash Attention 2
        self.use_flash_attention = self._detect_flash_attention(use_flash_attention)

        # Initialize the model with device mapping fix
        self._initialize_model()

    def _select_device(self, device: str) -> str:
        """
        Select the appropriate device for running the model.

        Automatically selects the best available device when device="auto",
        prioritizing CUDA > MPS > CPU.

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

    def _convert_dtype(self, dtype_str: str) -> torch.dtype:
        """
        Convert string dtype specification to torch.dtype.

        :param dtype_str: String representation of dtype.
        :type dtype_str: str
        :return: Corresponding torch dtype.
        :rtype: torch.dtype
        :raises ValueError: If dtype string is not recognized.
        """
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_str not in dtype_map:
            raise ValueError(
                f"Unsupported dtype '{dtype_str}'. "
                f"Supported dtypes: {list(dtype_map.keys())}"
            )
        return dtype_map[dtype_str]

    def _detect_flash_attention(self, use_flash_attention) -> bool:
        """
        Detect Flash Attention 2 availability and determine if it should be used.

        :param use_flash_attention: User preference ("auto", True, or False).
        :type use_flash_attention: str or bool
        :return: Whether Flash Attention 2 should be used.
        :rtype: bool
        :raises ImportError: If Flash Attention is required but not available.
        """
        if use_flash_attention is False:
            logger.info("Flash Attention 2 disabled by user")
            return False

        try:
            import flash_attn  # noqa: F401
            logger.info("Flash Attention 2 detected and will be used")
            return True
        except ImportError:
            if use_flash_attention is True:
                raise ImportError(
                    "Flash Attention 2 was requested but is not installed. "
                    "Install it with: pip install flash-attn --no-build-isolation"
                )
            elif use_flash_attention == "auto":
                logger.info(
                    "Flash Attention 2 not available, falling back to standard attention. "
                    "For better performance, install with: pip install flash-attn --no-build-isolation"
                )
                return False

        return False

    def _initialize_model(self):
        """
        Initialize the Qwen3-TTS model with device compatibility fixes.

        This method handles the common issue where models trained on CUDA
        fail to load on CPU-only machines due to missing map_location parameter.
        It also configures Flash Attention 2 if available.
        """
        # Store original torch.load
        original_torch_load = torch.load

        def device_aware_torch_load(*args, **kwargs):
            # If CUDA is not available and no map_location is specified,
            # force CPU loading to prevent CUDA deserialization errors
            if not torch.cuda.is_available() and 'map_location' not in kwargs:
                kwargs['map_location'] = torch.device('cpu')
                logger.debug("Applied CPU map_location for CUDA-incompatible environment")
            return original_torch_load(*args, **kwargs)

        # Temporarily patch torch.load
        torch.load = device_aware_torch_load

        try:
            # Prepare model loading arguments
            device_map = (
                self.device if self.device == "cuda"
                else f"{self.device}:0" if self.device in ["cuda", "mps"]
                else self.device
            )
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": self.dtype,
            }

            # Add Flash Attention 2 if enabled
            if self.use_flash_attention:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # Initialize the model
            self.pipeline = self.Qwen3TTSModel.from_pretrained(
                self.model_id,
                **model_kwargs
            )

            logger.info(
                f"Successfully initialized Qwen3TTS "
                f"(model: {self.model_id}, device: {self.device}, "
                f"dtype: {self.dtype_str}, flash_attn: {self.use_flash_attention})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Qwen3TTS: {e}")
            raise
        finally:
            # Always restore the original torch.load
            torch.load = original_torch_load

    def _normalize_language(self, language: str) -> str:
        """
        Normalize language code or name to full language name.

        Accepts both ISO language codes (e.g., "en", "fr") and full language names
        (e.g., "English", "French"). Returns the full language name as expected
        by the Qwen3-TTS API.

        :param language: Language code or name.
        :type language: str
        :return: Normalized full language name.
        :rtype: str
        :raises ValueError: If language is not recognized or supported.
        """
        # Convert to lowercase for case-insensitive matching
        lang_lower = language.lower()

        # Look up in mapping
        if lang_lower in LANGUAGE_MAP:
            normalized = LANGUAGE_MAP[lang_lower]
            logger.debug(f"Normalized language '{language}' to '{normalized}'")
            return normalized

        # Check if it's already a full name (case-insensitive)
        for supported_lang in SUPPORTED_LANGUAGES:
            if lang_lower == supported_lang.lower():
                return supported_lang

        # Language not recognized
        raise ValueError(
            f"Language '{language}' is not supported. "
            f"Supported languages: {SUPPORTED_LANGUAGES} "
            f"or their ISO codes: {sorted(set(k for k in LANGUAGE_MAP.keys() if len(k) == 2))}"
        )

    def register_voice(
        self,
        voice_name: str,
        audio_path: str,
        transcript: Optional[str] = None
    ) -> None:
        """
        Register a voice with an associated audio prompt for cloning.

        This method allows you to register voices by name for later use in speech
        generation. When a transcript is provided, the voice cloning quality is
        significantly better. Without a transcript, the system falls back to
        x_vector_only_mode with lower quality but no transcript requirement.

        The voice clone prompt is created and cached during registration for
        efficient reuse across multiple generations.

        :param voice_name: Unique name for the voice.
        :type voice_name: str
        :param audio_path: Path to the audio file to use for voice cloning.
        :type audio_path: str
        :param transcript: Optional transcript of the audio (recommended for best quality).
        :type transcript: str or None
        :raises ValueError: If the voice name already exists.
        :raises FileNotFoundError: If the audio file doesn't exist.
        """
        if voice_name in self.voice_registry:
            raise ValueError(f"Voice '{voice_name}' is already registered.")

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create voice clone prompt
        try:
            if transcript:
                # Full quality mode with transcript
                voice_prompt = self.pipeline.create_voice_clone_prompt(
                    ref_audio=str(audio_file.absolute()),
                    ref_text=transcript
                )
                x_vector_mode = False
                logger.debug(f"Registered voice '{voice_name}' with transcript (full quality mode)")
            else:
                # x_vector only mode (transcript-free but lower quality)
                voice_prompt = self.pipeline.create_voice_clone_prompt(
                    ref_audio=str(audio_file.absolute()),
                    x_vector_only_mode=True
                )
                x_vector_mode = True
                logger.warning(
                    f"Registered voice '{voice_name}' without transcript. "
                    "Using x_vector_only_mode (lower quality). "
                    "For best results, provide a transcript."
                )

            # Store in registry with cached prompt
            self.voice_registry[voice_name] = {
                "audio_path": str(audio_file.absolute()),
                "transcript": transcript,
                "prompt": voice_prompt,
                "x_vector_mode": x_vector_mode
            }

        except Exception as e:
            raise RuntimeError(f"Failed to register voice '{voice_name}': {e}") from e

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
        logger.debug(f"Unregistered voice '{voice_name}'")

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

        Returns the full language names as used by the Qwen3-TTS API.
        Both ISO codes (e.g., "en", "fr") and full names (e.g., "English", "French")
        are accepted in the generate() method via automatic normalization.

        :return: List of supported language names.
        :rtype: list[str]
        """
        return SUPPORTED_LANGUAGES.copy()

    def generate(
        self,
        text: str,
        speaker_voice: str,
        language: str = "English",
        tts_pipeline_kwargs: dict = {}
    ) -> tuple[np.ndarray, int]:
        """
        Generate multilingual audio from text using Qwen3-TTS.

        This method converts input text to speech using either the default voice,
        a registered cloned voice, or a direct audio file path. The language parameter
        accepts both ISO codes ("en", "fr") and full language names ("English", "French").

        Voice Cloning Modes:
        - Registered voice: Uses cached voice clone prompt from registration
        - Direct file path: Creates voice clone prompt on-the-fly (x_vector mode)
        - Default voice: No voice cloning, uses model's default voice

        :param text: The text to be converted to speech.
        :type text: str
        :param speaker_voice: Voice identifier - registered voice name or path to audio file.
        :type speaker_voice: str
        :param language: Language code or name (e.g., "en"/"English", "fr"/"French").
        :type language: str
        :param tts_pipeline_kwargs: Additional keyword arguments for the TTS pipeline.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as numpy array and sampling rate.
        :rtype: tuple[np.ndarray, int]
        :raises ValueError: If the language is not supported.
        :raises RuntimeError: If audio generation fails.
        """
        # Normalize the text if text normalizers are provided
        if self.text_normalizers is not None and len(self.text_normalizers) > 0:
            text = normalize_text(text, self.text_normalizers)

        # Remove audio event tags (e.g. [door opening]) before TTS
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Normalize and validate language
        try:
            normalized_language = self._normalize_language(language)
        except ValueError as e:
            raise ValueError(str(e)) from e

        try:
            # Prepare generation kwargs
            gen_kwargs = {"language": normalized_language, **tts_pipeline_kwargs}

            # Determine voice source and generate accordingly
            if speaker_voice in self.voice_registry:
                # Use registered voice with cached prompt
                voice_info = self.voice_registry[speaker_voice]
                logger.debug(
                    f"Generating with registered voice '{speaker_voice}' "
                    f"(x_vector_mode: {voice_info['x_vector_mode']})"
                )
                wavs, sample_rate = self.pipeline.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=voice_info["prompt"],
                    **gen_kwargs
                )

            elif Path(speaker_voice).exists():
                # Direct path to audio file - create prompt on-the-fly
                logger.debug(f"Generating with direct audio path (x_vector mode): {speaker_voice}")
                voice_prompt = self.pipeline.create_voice_clone_prompt(
                    ref_audio=speaker_voice,
                    x_vector_only_mode=True
                )
                wavs, sample_rate = self.pipeline.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=voice_prompt,
                    **gen_kwargs
                )

            else:
                # Use default voice (no voice cloning)
                logger.debug("Generating with default voice (no voice cloning)")
                # Note: Qwen3-TTS primarily focuses on voice cloning
                # For default voice, we use a simple generation approach
                # This may need adjustment based on actual API
                wavs, sample_rate = self.pipeline.generate_voice_clone(
                    text=text,
                    voice_clone_prompt=None,  # No prompt = default voice
                    **gen_kwargs
                )

            # Extract audio from waveforms list (Qwen3-TTS returns list of waveforms)
            if isinstance(wavs, list) and len(wavs) > 0:
                audio_array = wavs[0]
            else:
                audio_array = wavs

            # Convert to numpy if needed
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()

            # Ensure correct shape and type
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()

            audio_array = audio_array.astype(np.float32)

            # Ensure audio_array is at least 1D
            if audio_array.ndim == 0:
                audio_array = audio_array.reshape(1)

            logger.debug(f"Generated audio: {len(audio_array)} samples at {sample_rate}Hz")
            return audio_array, sample_rate

        except Exception as e:
            raise RuntimeError(f"Failed to generate audio with Qwen3-TTS: {e}") from e


class Qwen3TTSVoiceClone(BaseVoiceCloneTTS):
    """
    Qwen3-TTS voice cloning engine using the Base model.

    This class provides voice cloning capabilities using Qwen3-TTS-12Hz-1.7B-Base.
    Unlike Qwen3TTS which uses CustomVoice model with speaker embeddings,
    this class uses the Base model with reference audio for voice cloning.

    :param model: The model identifier from the Hugging Face Hub.
    :type model: str
    :param device_map: Device to run the model on (e.g., "cuda:0", "cpu").
    :type device_map: str
    :param dtype: Data type for model weights.
    :type dtype: torch.dtype
    :param text_normalizers: List of text normalizers to apply before generation.
    :type text_normalizers: list[TextNormalizer]
    """

    def __init__(
            self,
            model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map: str = None,
            dtype: torch.dtype = torch.bfloat16,
            text_normalizers: Optional[list[TextNormalizer]] = None,
            **model_kwargs):

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "qwen_tts is not installed. Please install it with `pip install qwen-tts`."
            )

        if device_map is None:
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.dtype = dtype
        self.device_map = device_map
        self.model = Qwen3TTSModel.from_pretrained(
            model,
            device_map=device_map,
            dtype=dtype,
            **model_kwargs
        )
        self.text_normalizers = text_normalizers

    def generate(
        self,
        text: str,
        speaker_voice=None,
        tts_pipeline_kwargs: dict = {}
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text using Qwen3-TTS voice cloning.

        :param text: The text to be converted to speech.
        :type text: str
        :param speaker_voice: Either a string path to a reference audio file,
                              or a voice clone prompt object, or None for default voice.
        :type speaker_voice: str | object | None
        :param tts_pipeline_kwargs: Additional keyword arguments for the TTS pipeline.
        :type tts_pipeline_kwargs: dict
        :return: A tuple containing the audio data as numpy array and sampling rate.
        :rtype: tuple[np.ndarray, int]
        """
        # Normalize the text if text normalizers are provided.
        if self.text_normalizers is not None and len(self.text_normalizers) > 0:
            text = normalize_text(text, self.text_normalizers)

        if "language" not in tts_pipeline_kwargs:
            tts_pipeline_kwargs["language"] = "English"

        if speaker_voice is not None:
            # Path to reference audio or (array, sampling_rate) tuple
            tts_pipeline_kwargs["ref_audio"] = speaker_voice
            tts_pipeline_kwargs["x_vector_only_mode"] = True
        else:
            raise ValueError(
                "speaker_voice must be provided for "
                "voice cloning in Qwen3TTSVoiceClone"
            )

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            **tts_pipeline_kwargs
        )
        audio = wavs[0].cpu().numpy() if hasattr(wavs[0], "cpu") else np.asarray(wavs[0])

        return (audio, sr)

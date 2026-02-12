# ChangeLog

All notable changes to SDialog will be documented here.

---

## [Unreleased]

### Added
- **sdialog.audio.evaluation**: New audio evaluation framework (merged from qanastek/sdialog)
  - Audio quality evaluation (SNR, PESQ, STOI metrics)
  - Speaker consistency evaluation
  - Speech analytics evaluation
  - Speech signal evaluation
- **sdialog.audio.normalizers**: Text normalization utilities for TTS
  - LowercaseNormalizer, StageNormalizer, ReplaceCommaWithDotNormalizer, WhisperNormalizer
- **sdialog.audio**: Sound effects annotation and integration via LLM
  - `add_sound_effects()` method for automatic sound effect placement
  - `compute_overlapping_and_pausing_llm()` for realistic turn timing
- **sdialog.audio.tts**: Text normalizers support added to Kokoro, Index, and HuggingFace TTS
- **sdialog.audio.tts**: IndexTTS v2 support via `version` parameter
- **tutorials**: 4 new audio tutorials (evaluation, IndexTTS, overlaps/pauses, sound effects)

### Changed
- **sdialog.audio.dialog**: Added `RoomAcousticsConfig` class (Pydantic model) for room acoustics configuration
- **sdialog.audio.dialog**: `audio_step_3_filepaths` now uses `RoomAcousticsConfig` objects instead of dictionaries
- **sdialog.audio.dialog**: Added `get_dry_audio()` method (replaces `master_audio()`)
- **sdialog.audio.pipeline**: Removed `re_sampling_rate` parameter (use `sampling_rate` in constructor instead)
- **sdialog.audio.turn**: Added `gap_duration`, `text_with_tags`, and `sound_effects` attributes
- **requirements-audio.txt**: Updated dscaper requirement to >=1.7.4

- **sdialog.audio.tts**: Qwen3-TTS integration for rapid multilingual voice cloning
  - Support for 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)
  - 3-second rapid voice cloning from reference audio
  - Dual-mode voice cloning: full quality with transcript or transcript-free x_vector mode
  - Language code normalization (accepts both ISO codes like "en" and full names like "English")
  - Voice clone prompt caching for efficient reuse
  - Multiple model variants: 1.7B (default, higher quality) and 0.6B (lightweight for edge devices)
  - Flash Attention 2 support with auto-detection and graceful fallback
  - Automatic device selection (CUDA > MPS > CPU) with compatibility fixes
  - Example script demonstrating all features
- **sdialog.audio.tts**: XTTS (Coqui TTS) integration for multilingual voice cloning
  - Support for 17+ languages with voice cloning capabilities
  - Voice registration system for managing cloned voices
  - Automatic device selection (CUDA > MPS > CPU)
  - PyTorch 2.6 compatibility with model loading safety check handling
  - Example script and comprehensive documentation

---

## [0.4.0] 2025-10-30 🔊

### Added
- **sdialog.audio**: New audio generation module to transform text dialogs into audio conversations
  - `Dialog.to_audio()`: Convert any text dialog to audio with a single method call
  - **Text-to-Speech (TTS)**: Support for multiple TTS engines including Kokoro and Hugging Face models
  - **Voice databases**: Automatic or manual voice assignment from Hugging Face, local storage, or on-the-fly generation based on persona attributes (age, gender, language)
  - **Acoustic simulation**: Realistic room acoustics simulation with ray tracing technology for spatial audio
  - **Microphone simulation**: Professional microphone impulse responses from brands like Shure, Sennheiser, and Sony
  - **Room generation**: Define and generate diverse room types with customizable properties (dimensions, wall materials, furniture placement)
  - **Multiple formats**: Export to WAV, MP3, or FLAC with custom sampling rates
  - **Background/foreground effects**: Add environmental sounds and acoustic variations
  - 7 comprehensive tutorials covering audio generation, acoustic simulation, rooms, voice databases, and impulse responses
- **sdialog.interpretability**: 
  - Support for inspecting layer and component inputs with `Inspector(target="model.layers.15", inspect_input=True)`
  - Support for inspecting and steering input tokens (tokens given as input, not only generated ones), e.g., `inspector.input[i][j].act` where `i` is turn index and `j` is input token index
- **sdialog.config**: 
  - Support for Anthropic backend (#100)
  - Support for Azure OpenAI backend (#100)

### Fixed
- **sdialog.agents**: Agent memory reset when no prompt system and no persona is given

---

## [0.3.3] 2025-10-30 🚀

### Added
- **sdialog.server**:
  - New module to serve agents via an Ollama/OpenAI-compatible REST API (works with UIs like Open WebUI) (#92)
- **sdialog**:
  - `Dialog.from_huggingface()` to load/download dialogues directly from Hugging Face datasets (#59)

### Changed
- **sdialog.evaluation**:
  - LLM judge methods now accept additional user-defined template arguments (e.g., like `document` in [this example](https://sdialog.readthedocs.io/en/latest/examples/index.html#example-1-yes-no-relevance-judgment-with-reasoning)) (#86)
- **sdialog.agents**:
  - Improved `Agent` initialization so agents can act as a proxy for external conversational systems (#90, fa1d8f3)

### Fixed
- **sdialog.evaluation**:
  - Corrected Flesch Reading Ease and Gunning Fog score calculations (d1d4260)


---

## [0.3.0] 2025-10-03 ✨

### Added
- **sdialog**: 
  - `Context`: new class class to explicitly model the common/shared context of conversations (#73)
  - `Dialog`: merge functionality - Added option to merge consecutive turns of the same speaker when loading a dialog (#77)
  - `Dialog`: built-in string support - Added support to built-in str functions for `Dialog` class (#83)
- **sdialog.agents**: Added new `sdialog.agents` module and moved `Agent` class inside (#81)
  - `Agent`: thinking capabilities - Agents can now handle internal thinking processes (#95)
  - `Agent`: tools support - Added tools capabilities to Agents (e.g. RAG or any other function) (#84)
    - New tutorial for agents with tools and thoughts.
- **sdialog.generators**: 
  - `ContextGenerator`: new class added to explicitly model the common/shared context of conversations (#73)
  - `Paraphraser`: new class class to paraphrase dialogues (#76)
- **sdialog.evaluation**: 
  - `LinguisticFeatureScore`: new class added to compute Flesch reading ease, Gunning fog, Hesitation rate, and/or Mean turn length (#63)
- **sdialog.personas**: 
  - `Customer` and `SupportAgent`: new personas added for customer service dialogues (#85)
  - `Persona`: Added static method to get the list of all attributes in `Persona` class (#79)


### Changed
- **sdialog**: Improved metadata handling (#66)
- **sdialog.interpretability**: Improved and simplified the way inspection targets are defined in `interpretability` submodule (#78)
- **sdialog.evaluation.base**: 
  - `LLMJudgeYesNoOutput`: Renamed attribute `yes` to `positive` (#86)
  - `LLMJudgeScoreOutput`: Renamed attribute `feedback` to `reason` (#86)

### Fixed
- **sdialog.generators**: Fixed potential bug in `PersonaDialogGenerator` class (#67)


### Enhanced
- **sdialog.agents**: Added `base_model` attribute to `Agent` to direclty access the LLM's underlying model for mechanistic interpretability (#74)
- **sdialog.config**: Added `clear_cache()` method to config (#75)

### Documentation
- API Documentation: Refactored/cleaned all components and added docstrings with examples (#82, #88)
- Updated all tutorials to work with new code and added "Open in Colab" badges
- Completed API documentation for initial official release (#87)
- Automatic generation of `llm.txt` from API documentation (24f6ee6)

---

## [0.1.0] 2025-08-05 🌱

### Added
- Multi-backend support (Hugging Face, Ollama, OpenAI, AWS)
- Enhanced persona generation (beyond initial `PersonaDialogGenerator`)
- Interpretability module (`sdialog.interpretability`): inspectors, steerers, hooks, intruders
- Evaluation module (`sdialog.evaluation`): metrics, LLM-as-a-judge scoring, evaluators, dataset comparators

### Changed
- Standardized / improved dialog format

### Notes
- >500 commits since 0.0.2 (post-JSALT 2025 consolidation)

### Pending
- Audio module (`sdialog.audio`) integration
- Documentation updates

---

## [0.0.2] 2025-06-03 🔧

### Added
- `language` attribute to `Persona` class
- `PersonaDialogGenerator` to `generators` module to support persona-based dialogue generatin with single LLM:
  ```python
  from sdialog.generators import PersonaDialogGenerator

  dialog_generator = PersonaDialogGenerator(
      model=MODEL_NAME,
      persona_a=bob_persona,
      persona_b=alice_persona,
  )

  dialog_generator.generate().print()
  ```

### Fixed
- Python 2 and 3 compatibility problem with scikit-learn (using version 0.20.1 from now on)
- PyPi: setup.py: `long_description_content_type` set to `'text/markdown'`

---

## [0.0.1] 2025-05-22 🎉

_(initial release)_
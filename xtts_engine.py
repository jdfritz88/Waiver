"""XTTS v2 voice cloning engine for generating vocal sounds with TTS."""
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
import tempfile


class XTTSEngine:
    """Handles XTTS v2 voice cloning and TTS generation."""

    def __init__(self, sample_rate=44100):
        """Initialize the XTTS engine."""
        self.sample_rate = sample_rate
        self.model = None
        self.reference_audio = None
        self.reference_audio_path = None
        self.model_loaded = False
        self.device = None

        # Cache speaker latents for faster generation
        self.gpt_cond_latent = None
        self.speaker_embedding = None

        print("Initializing XTTS v2 engine...")
        self._load_model()

    def _load_model(self):
        """Load the XTTS v2 model from local installation."""
        try:
            # Try importing TTS modules directly
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts

            # Path to your existing XTTS model
            model_path = Path("F:/Apps/freedom_system/freedom_system_2000/text-generation-webui/extensions/alltalk_tts/models/xtts/xttsv2_2.0.3")

            if not model_path.exists():
                raise FileNotFoundError(f"XTTS model not found at {model_path}")

            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            print(f"Loading XTTS v2 from: {model_path}")

            # Load config
            config_path = model_path / "config.json"
            config = XttsConfig()
            config.load_json(str(config_path))

            # Initialize model
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(
                config,
                checkpoint_dir=str(model_path),
                use_deepspeed=False
            )
            self.model.to(device)

            self.model_loaded = True
            self.device = device
            print("XTTS v2 model loaded successfully from local installation!")

        except ImportError as e:
            print(f"TTS library not installed: {e}")
            print("Install with: pip install TTS")
            print("TTS features will be disabled")
            self.model_loaded = False
        except Exception as e:
            print(f"Error loading XTTS model: {e}")
            import traceback
            traceback.print_exc()
            print("TTS features will be disabled")
            self.model_loaded = False

    def load_reference_voice(self, audio_path):
        """
        Load a reference voice from a WAV file for voice cloning.

        Args:
            audio_path: Path to WAV file containing the voice to clone

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading reference voice from: {audio_path}")

            # Read the audio file
            audio, sr = sf.read(audio_path)

            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed (XTTS works best with 22050 Hz)
            if sr != 22050:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)

            # Save to temp file (XTTS needs file path)
            temp_dir = tempfile.gettempdir()
            self.reference_audio_path = str(Path(temp_dir) / "xtts_reference.wav")
            sf.write(self.reference_audio_path, audio, 22050)

            # Pre-compute speaker latents for faster generation
            if self.model_loaded:
                print("Computing speaker latents for voice cloning...")
                self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=[self.reference_audio_path]
                )
                print("Speaker latents cached!")

            print(f"Reference voice loaded successfully")
            print(f"Duration: {len(audio) / 22050:.2f} seconds")
            return True

        except Exception as e:
            print(f"Error loading reference voice: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_audio(self, text, language="en", speed=1.0):
        """
        Generate audio from text using voice cloning.

        Args:
            text: Text to convert to speech (can include phonetic vocalizations)
            language: Language code (default: "en")
            speed: Speech speed multiplier (default: 1.0)

        Returns:
            numpy array: Generated audio samples, or None if failed
        """
        if not self.model_loaded:
            print("XTTS model not loaded")
            return None

        if self.reference_audio_path is None:
            print("No reference voice loaded - please load a voice file first")
            return None

        try:
            print(f"Generating audio from text: '{text[:50]}...'")

            # Use cached speaker latents (computed when voice was loaded)
            if self.gpt_cond_latent is None or self.speaker_embedding is None:
                print("Computing speaker latents...")
                self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=[self.reference_audio_path]
                )

            # Generate audio using XTTS inference
            out = self.model.inference(
                text=text,
                language=language,
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                temperature=0.75,
                speed=speed
            )

            # Convert to numpy array
            audio = np.array(out["wav"], dtype=np.float32)

            # XTTS outputs at 24kHz, resample to target sample rate if needed
            xtts_sample_rate = 24000
            if xtts_sample_rate != self.sample_rate:
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=xtts_sample_rate,
                    target_sr=self.sample_rate
                )

            print(f"Generated {len(audio) / self.sample_rate:.2f} seconds of audio")
            return audio

        except Exception as e:
            print(f"Error generating audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_short_clip(self, text, language="en", speed=1.0):
        """
        Generate short audio clip optimized for streaming (3-5 seconds).
        Same as generate_audio but with optimized settings for speed.

        Args:
            text: Short phonetic text (3-5 seconds worth)
            language: Language code (default: "en")
            speed: Speech speed multiplier (default: 1.0)

        Returns:
            numpy array: Generated audio samples, or None if failed
        """
        # For now, just call generate_audio (XTTS is already fast enough)
        # In future, could add quality/speed tradeoffs here
        return self.generate_audio(text, language, speed)

    def is_available(self):
        """Check if XTTS is available and ready."""
        return self.model_loaded and self.reference_audio_path is not None

    def cleanup(self):
        """Clean up resources."""
        try:
            # Remove temp file if it exists
            if self.reference_audio_path and Path(self.reference_audio_path).exists():
                Path(self.reference_audio_path).unlink()
        except:
            pass

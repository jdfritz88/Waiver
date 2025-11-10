"""Audio generation and playback engine for vocal moaning sounds."""
import numpy as np
import soundfile as sf
import librosa
import pyaudio
from threading import Thread, Event, Lock
import time
import random
import pyttsx3
from io import BytesIO
import wave
from vocal_synthesizer import VocalSynthesizer
from voice_analyzer import VoiceAnalyzer
from xtts_engine import XTTSEngine
from vocalization_generator import VocalizationGenerator
from audio_processor import AudioProcessor
from prosody_settings import ProsodySettings
from app_state import AppState
from config import (
    SAMPLE_RATE, CHUNK_SIZE, MIN_BUILD_UP_DURATION,
    MAX_BUILD_UP_DURATION, CLIMAX_DURATION,
    MIN_TIME_BETWEEN_EVENTS, MAX_TIME_BETWEEN_EVENTS
)


class AudioEngine:
    """Handles real-time vocal sound generation and playback."""

    def __init__(self, mistral_client=None):
        """Initialize the audio engine."""
        self.mistral_client = mistral_client
        self.voice_analyzer = VoiceAnalyzer()
        self.vocal_synth = VocalSynthesizer(sample_rate=SAMPLE_RATE)
        self.sample_rate = SAMPLE_RATE
        self.is_playing = False
        self.stop_event = Event()
        self.playback_thread = None

        # Voice profile from analyzed WAV file
        self.voice_profile = None
        self.loaded_file_name = None

        # Audio parameters (controlled by UI)
        self.volume = 1.0
        self.pitch_shift = 0.0  # in semitones
        self.octave_shift = 0  # -2 to +2
        self.word_frequency = 0.3  # 0.0 to 1.0 (controls how often words are spoken)

        # Base frequency range for moaning (typical female vocal range)
        self.base_freq_min = 180  # Hz
        self.base_freq_max = 350  # Hz

        # State management
        self.current_state = "normal"  # normal, building, orgasm, post_orgasm_breathing
        self.state_start_time = 0
        self.next_event_time = 0
        self.next_word_time = 0
        self.add_post_orgasm_breathing = True  # Toggle for post-orgasm breathing

        # Random breathing control
        self.random_breathing_enabled = True
        self.last_breath_time = 0
        self.min_time_between_breaths = 5.0  # At least 5 seconds between random breaths

        # Mistral API caching to avoid rate limits
        self.cached_mistral_params = None
        self.last_mistral_call_time = 0
        self.mistral_cache_duration = 60.0  # Cache for 60 seconds (default)
        self.last_mistral_state = None

        # Audio recording
        self.is_recording = False
        self.recording_buffer = []
        self.recording_duration = 20.0  # Record for 20 seconds

        # Vocalization parameters
        self.current_vowel = 'ah'
        self.target_vowel = 'oh'
        self.vowel_transition_progress = 0.0

        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None

        # TTS engine for occasional words
        try:
            self.tts_engine = pyttsx3.init()
            # Set voice properties
            voices = self.tts_engine.getProperty('voices')
            # Try to set a female voice
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            self.tts_engine.setProperty('rate', 150)  # Slower speech rate
            self.tts_available = True
        except:
            print("TTS not available")
            self.tts_engine = None
            self.tts_available = False

        # Waveform buffer for visualization
        self.waveform_buffer = np.zeros(1000)

        # XTTS voice cloning engine
        self.xtts_engine = XTTSEngine(sample_rate=self.sample_rate)

        # Vocalization generator (creates phonetic text for TTS)
        self.vocalization_generator = VocalizationGenerator(mistral_client=self.mistral_client)

        # TTS generation buffer (stores last generated audio)
        self.last_generated_audio = None

        # Audio processor for advanced prosody control
        self.audio_processor = AudioProcessor(sample_rate=self.sample_rate)

        # Prosody settings manager
        self.prosody_settings = ProsodySettings()

        # App state persistence (saves last voice file)
        self.app_state = AppState()

    def load_audio_file(self, file_path):
        """
        Analyze a WAV file and extract voice characteristics.
        The app will synthesize new sounds using these characteristics.
        """
        try:
            print(f"Analyzing voice from: {file_path}")
            print("This may take a moment...")

            # Analyze the voice
            self.voice_profile = self.voice_analyzer.analyze_voice(file_path, self.sample_rate)

            # Get synthesis parameters
            synth_params = self.voice_analyzer.get_synthesis_params()

            if synth_params:
                print(f"Voice Analysis Complete:")
                print(f"  - Base Pitch: {synth_params['base_freq']:.1f} Hz")
                print(f"  - Pitch Range: {synth_params['freq_range'][0]:.1f} - {synth_params['freq_range'][1]:.1f} Hz")
                print(f"  - Formants (F1,F2,F3): {[f'{f:.0f}' for f in synth_params['formants']]}")
                print(f"  - Breathiness: {synth_params['breathiness_factor']:.2f}")

                # Update base frequency range from voice
                self.base_freq_min = synth_params['freq_range'][0]
                self.base_freq_max = synth_params['freq_range'][1]

                # Create new synthesizer with voice profile
                self.vocal_synth = VocalSynthesizer(
                    sample_rate=self.sample_rate,
                    voice_profile=synth_params
                )

                self.loaded_file_name = file_path.split('/')[-1].split('\\')[-1]

                # Also load into XTTS for voice cloning
                if self.xtts_engine.load_reference_voice(file_path):
                    print("Voice also loaded into XTTS for TTS generation!")

                # Save to app state for auto-load on next startup
                self.app_state.set_last_voice_file(file_path)

                print(f"Voice profile loaded successfully!")
                print("App will now generate sounds using this voice.")
                return True
            else:
                print("Could not extract voice parameters")
                return False

        except Exception as e:
            print(f"Error analyzing audio file: {e}")
            import traceback
            traceback.print_exc()
            return False

    def auto_load_last_voice_file(self):
        """
        Automatically load the last voice file from app state if it exists.

        Returns:
            tuple: (success, file_path) - success is True if loaded, file_path is the loaded file
        """
        if self.app_state.has_last_voice_file():
            last_file = self.app_state.get_last_voice_file()
            print(f"Auto-loading last voice file: {last_file}")
            success = self.load_audio_file(last_file)
            return (success, last_file if success else None)
        return (False, None)

    def unload_audio_file(self):
        """Unload the voice profile and use default synthesis."""
        self.voice_profile = None
        self.loaded_file_name = None

        # Reset to default synthesizer
        self.vocal_synth = VocalSynthesizer(sample_rate=self.sample_rate)

        # Reset frequency range to defaults
        self.base_freq_min = 180
        self.base_freq_max = 350

        print("Switched to default synthesis mode")

    def start_playback(self):
        """Start XTTS streaming audio generation and playback."""
        if self.is_playing:
            print("Already playing")
            return False

        # Check if XTTS is available
        if not self.xtts_engine.is_available():
            print("ERROR: XTTS not ready - please load a voice file first")
            return False

        self.is_playing = True
        self.stop_event.clear()
        self.current_state = "normal"
        self.schedule_next_event()

        self.playback_thread = Thread(target=self._xtts_streaming_loop, daemon=True)
        self.playback_thread.start()
        return True

    def stop_playback(self):
        """Stop audio generation."""
        self.is_playing = False
        self.stop_event.set()
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def _generation_loop(self):
        """Main audio generation loop running in separate thread."""
        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )

        chunk_duration = CHUNK_SIZE / self.sample_rate  # Duration of each chunk in seconds
        position = 0  # Position in audio file (for file mode)

        while self.is_playing and not self.stop_event.is_set():
            current_time = time.time()

            # Check if we need to trigger an event
            if current_time >= self.next_event_time:
                self.trigger_random_event()

            # Check if we should speak a word (based on word frequency)
            if current_time >= self.next_word_time and random.random() < self.word_frequency:
                self._trigger_word()

            # ALWAYS SYNTHESIZE - never playback
            # Voice characteristics are baked into the synthesizer
            chunk = self._generate_from_synthesis(chunk_duration)

            # Apply volume
            chunk = chunk * self.volume

            # Update waveform buffer for visualization
            self.waveform_buffer = np.roll(self.waveform_buffer, -len(chunk))
            self.waveform_buffer[-len(chunk):] = chunk[:min(len(chunk), len(self.waveform_buffer))]

            # If recording, add to recording buffer
            if self.is_recording:
                self.recording_buffer.append(chunk.copy())

            # Ensure correct format
            chunk = chunk.astype(np.float32)

            # Play chunk
            try:
                self.stream.write(chunk.tobytes())
            except:
                break

        # Cleanup
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def _xtts_streaming_loop(self):
        """Main XTTS streaming loop - generates and plays audio continuously with thread-safe buffering."""
        print("\n=== Starting XTTS Streaming Mode ===")
        print(f"Initial state: {self.current_state}")

        # Open audio stream for playback
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )

        # Clip duration depends on state
        base_clip_duration = 4.0  # Normal clips are 4 seconds

        # Thread-safe double buffering using a lock
        next_clip = None
        clip_lock = Lock()

        def generate_next_clip_async():
            """Generate the next audio clip in background thread with thread safety."""
            nonlocal next_clip

            current_time = time.time()

            # Check if we need to trigger a state transition
            if current_time >= self.next_event_time:
                self.trigger_random_event()

            # Adjust clip duration based on state
            if self.current_state == "building":
                clip_duration = self.prosody_settings.get('buildup_duration_min', 5)
            elif self.current_state == "orgasm":
                clip_duration = self.prosody_settings.get('orgasm_duration_min', 15)
            elif self.current_state == "post_orgasm_breathing":
                clip_duration = 2.0  # Short clips for fast breathing
            else:
                clip_duration = base_clip_duration

            # Generate phonetic text
            phonetic_text = self.vocalization_generator.generate_streaming_phonetics(
                current_state=self.current_state,
                duration=clip_duration
            )

            # Generate audio with XTTS
            audio_clip = self.xtts_engine.generate_short_clip(
                text=phonetic_text,
                language="en",
                speed=1.0
            )

            if audio_clip is None:
                print("ERROR: XTTS generation failed")
                return

            # Apply advanced prosody processing with REDUCED intensity
            audio_clip = self.audio_processor.process_for_state(
                audio_clip,
                self.current_state,
                self.prosody_settings.get_all()
            )

            # Apply volume
            audio_clip = audio_clip * self.volume

            # Aggressive normalization to prevent clipping
            max_val = np.max(np.abs(audio_clip))
            if max_val > 0.8:
                audio_clip = audio_clip * (0.8 / max_val)

            # Thread-safe assignment
            with clip_lock:
                next_clip = audio_clip

        # Pre-generate first clip synchronously
        print(f"[{self.current_state.upper()}] Pre-generating first clip...")
        generate_next_clip_async()

        with clip_lock:
            if next_clip is None:
                print("ERROR: Failed to generate initial clip")
                return

        while self.is_playing and not self.stop_event.is_set():
            # Thread-safe get of pre-generated clip
            with clip_lock:
                current_clip = next_clip
                next_clip = None

            # Start generating next clip in background WHILE playing current clip
            next_clip_generation_thread = Thread(target=generate_next_clip_async, daemon=True)
            next_clip_generation_thread.start()

            # Update waveform buffer for visualization
            self.waveform_buffer = np.roll(self.waveform_buffer, -len(current_clip))
            if len(current_clip) < len(self.waveform_buffer):
                self.waveform_buffer[-len(current_clip):] = current_clip
            else:
                self.waveform_buffer = current_clip[-len(self.waveform_buffer):]

            # If recording, add to recording buffer
            if self.is_recording:
                self.recording_buffer.append(current_clip.copy())

            # RANDOMLY insert breathing before main clip (based on breathing_frequency setting)
            breathing_freq = self.prosody_settings.get('breathing_frequency', 15)
            current_time = time.time()

            if (self.random_breathing_enabled and
                breathing_freq > 0 and
                random.random() < (breathing_freq / 100.0) and
                current_time - self.last_breath_time > self.min_time_between_breaths):

                # Generate short breathing clip
                breath_phonetics = random.choice(['hh...', 'hah...', 'mmm...', 'hhh... hah...'])
                print(f"[BREATHING] Inserting random breath: '{breath_phonetics}'")

                breath_audio = self.xtts_engine.generate_short_clip(
                    text=breath_phonetics,
                    language="en",
                    speed=1.2  # Slightly faster for breathing
                )

                if breath_audio is not None:
                    # Apply high breathiness effect
                    breath_audio = self.audio_processor.apply_breathiness(
                        breath_audio,
                        50  # High breathiness
                    )

                    # Apply volume and normalization
                    breath_audio = breath_audio * self.volume * 0.8  # Slightly quieter than main audio
                    max_val = np.max(np.abs(breath_audio))
                    if max_val > 0.8:
                        breath_audio = breath_audio * (0.8 / max_val)

                    # Play breath before main clip
                    self._play_audio_chunk(breath_audio)
                    self.last_breath_time = current_time

                    # Add to recording buffer if recording
                    if self.is_recording:
                        self.recording_buffer.append(breath_audio.copy())

            # Play the current clip (while next one generates in background)
            print(f"[{self.current_state.upper()}] Playing {len(current_clip)/self.sample_rate:.2f}s clip...")
            self._play_audio_chunk(current_clip)

            # Wait for next clip to finish generating
            if next_clip_generation_thread:
                next_clip_generation_thread.join()

            # Thread-safe check if generation failed
            with clip_lock:
                if next_clip is None:
                    print("WARNING: Background generation failed, generating synchronously...")
                    # Release lock before generating

            if next_clip is None:
                generate_next_clip_async()
                with clip_lock:
                    if next_clip is None:
                        print("ERROR: Could not generate audio, stopping...")
                        break

        # Cleanup
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        print("=== XTTS Streaming Stopped ===")

    def _play_audio_chunk(self, audio_chunk):
        """Play an audio chunk through PyAudio stream."""
        try:
            # Ensure correct format
            audio_chunk = audio_chunk.astype(np.float32)

            # Play chunk in small pieces to allow for interruption
            chunk_size = CHUNK_SIZE
            for i in range(0, len(audio_chunk), chunk_size):
                if not self.is_playing or self.stop_event.is_set():
                    break

                piece = audio_chunk[i:i+chunk_size]
                self.stream.write(piece.tobytes())

        except Exception as e:
            print(f"Error playing audio chunk: {e}")

    def _generate_from_synthesis(self, chunk_duration):
        """Generate audio chunk from synthesis."""
        # Get current synthesis parameters based on state
        params = self._get_synthesis_parameters()

        # Update vowel transition
        self._update_vowel_transition()

        # Generate audio chunk
        chunk = self.vocal_synth.generate_chunk(
            duration=chunk_duration,
            base_freq=params['base_freq'],
            vowel=self.current_vowel,
            breathiness=params['breathiness'],
            intensity=params['intensity'],
            vibrato_rate=params['vibrato_rate'],
            vibrato_depth=params['vibrato_depth']
        )

        return chunk

    def _get_synthesis_parameters(self):
        """
        Get synthesis parameters based on current state.

        Returns:
            dict with parameters for vocal synthesis
        """
        # Calculate base frequency with pitch and octave shifts
        base_freq = (self.base_freq_min + self.base_freq_max) / 2
        pitch_mult = 2 ** ((self.pitch_shift + self.octave_shift * 12) / 12.0)
        base_freq = base_freq * pitch_mult

        # Get state-specific parameters from Mistral (with caching to avoid rate limits)
        mistral_params = None
        if self.mistral_client:
            current_time = time.time()
            # Only call API if cache expired OR state changed
            if (self.cached_mistral_params is None or
                current_time - self.last_mistral_call_time > self.mistral_cache_duration or
                self.last_mistral_state != self.current_state):
                try:
                    mistral_params = self.mistral_client.generate_intensity_parameters(self.current_state)
                    # Cache the results
                    self.cached_mistral_params = mistral_params
                    self.last_mistral_call_time = current_time
                    self.last_mistral_state = self.current_state
                    print(f"[Mistral API] Called for state '{self.current_state}'")
                except Exception as e:
                    print(f"[Mistral API] Error: {e}")
                    mistral_params = self.cached_mistral_params  # Use cached value
            else:
                # Use cached value
                mistral_params = self.cached_mistral_params

        # Default parameters based on state
        if self.current_state == "climax":
            params = {
                'base_freq': base_freq * 1.3,  # Higher pitch
                'breathiness': 0.5,
                'intensity': 1.8,
                'vibrato_rate': 6.5,
                'vibrato_depth': 1.2
            }
        elif self.current_state == "building":
            progress = (time.time() - self.state_start_time) / MAX_BUILD_UP_DURATION
            progress = min(1.0, progress)
            params = {
                'base_freq': base_freq * (1.0 + progress * 0.3),
                'breathiness': 0.3 + progress * 0.2,
                'intensity': 1.0 + progress * 0.8,
                'vibrato_rate': 5.0 + progress * 1.5,
                'vibrato_depth': 0.5 + progress * 0.7
            }
        else:  # normal
            # Add natural variation
            variation = random.uniform(-0.05, 0.05)
            params = {
                'base_freq': base_freq * (1.0 + variation),
                'breathiness': 0.25 + random.uniform(-0.1, 0.1),
                'intensity': 1.0 + random.uniform(-0.2, 0.2),
                'vibrato_rate': 5.0 + random.uniform(-1.0, 1.0),
                'vibrato_depth': 0.6 + random.uniform(-0.2, 0.2)
            }

        # Apply Mistral modulation if available
        if mistral_params:
            params['intensity'] *= mistral_params.get('volume_mult', 1.0)
            freq_shift = mistral_params.get('pitch_shift', 0.0)
            params['base_freq'] *= 2 ** (freq_shift / 12.0)

        return params

    def _update_vowel_transition(self):
        """Update vowel sound transition for natural variation."""
        self.vowel_transition_progress += 0.01

        if self.vowel_transition_progress >= 1.0:
            # Start new transition
            self.current_vowel = self.target_vowel
            _, self.target_vowel = self.vocal_synth.get_random_vowel_transition()
            self.vowel_transition_progress = 0.0

        # Interpolate between vowels
        self.current_vowel = self.vocal_synth.interpolate_vowels(
            self.current_vowel,
            self.target_vowel,
            self.vowel_transition_progress
        )

    def schedule_next_event(self):
        """Schedule the next random build-up/climax event."""
        delay = random.uniform(MIN_TIME_BETWEEN_EVENTS, MAX_TIME_BETWEEN_EVENTS)
        self.next_event_time = time.time() + delay

    def schedule_next_word(self):
        """Schedule the next word/phrase utterance."""
        if self.word_frequency <= 0.01:
            # If frequency is very low, schedule far in the future
            delay = 9999
        else:
            # Base delay inversely proportional to frequency
            # word_frequency 0.0-1.0 maps to 60-10 seconds
            base_delay = 60 - (self.word_frequency * 50)
            delay = random.uniform(base_delay * 0.7, base_delay * 1.3)
        self.next_word_time = time.time() + delay

    def trigger_random_event(self):
        """Trigger a random build-up and orgasm sequence."""
        self.current_state = "building"
        self.state_start_time = time.time()

        # Get build-up duration from settings
        build_up_min = self.prosody_settings.get('buildup_duration_min', 5)
        build_up_max = self.prosody_settings.get('buildup_duration_max', 15)
        build_up_duration = random.uniform(build_up_min, build_up_max)

        print(f"[BUILD-UP TRIGGERED] Duration: {build_up_duration:.1f} seconds")

        def trigger_orgasm():
            time.sleep(build_up_duration)
            if self.is_playing:
                self.current_state = "orgasm"
                self.state_start_time = time.time()

                # Get orgasm duration from settings (extended: 15-30 seconds)
                orgasm_min = self.prosody_settings.get('orgasm_duration_min', 15)
                orgasm_max = self.prosody_settings.get('orgasm_duration_max', 30)
                orgasm_duration = random.uniform(orgasm_min, orgasm_max)

                print(f"[ORGASM TRIGGERED] Duration: {orgasm_duration:.1f} seconds")

                time.sleep(orgasm_duration)

                # 70% chance of post-orgasm fast breathing
                if self.add_post_orgasm_breathing and random.random() < 0.7:
                    self.current_state = "post_orgasm_breathing"
                    breathing_duration = random.uniform(3, 6)  # 3-6 seconds of fast breathing
                    print(f"[POST-ORGASM BREATHING] Duration: {breathing_duration:.1f} seconds")
                    time.sleep(breathing_duration)

                self.current_state = "normal"
                print("[AUTO TRANSITION] Returned to normal state")
                self.schedule_next_event()

        Thread(target=trigger_orgasm, daemon=True).start()

    def trigger_build_up_manual(self):
        """Manually trigger a build-up."""
        if self.current_state == "normal":
            self.current_state = "building"
            self.state_start_time = time.time()

            def reset_to_normal():
                time.sleep(random.uniform(MIN_BUILD_UP_DURATION, MAX_BUILD_UP_DURATION))
                if self.current_state == "building":
                    self.current_state = "normal"

            Thread(target=reset_to_normal, daemon=True).start()

    def trigger_orgasm_manual(self):
        """Manually trigger an orgasm."""
        print("[MANUAL TRIGGER] Orgasm triggered!")
        self.current_state = "orgasm"
        self.state_start_time = time.time()

        # Get orgasm duration from settings (extended: 15-30 seconds)
        orgasm_min = self.prosody_settings.get('orgasm_duration_min', 15)
        orgasm_max = self.prosody_settings.get('orgasm_duration_max', 30)
        orgasm_duration = random.uniform(orgasm_min, orgasm_max)

        print(f"[MANUAL ORGASM] Duration: {orgasm_duration:.1f} seconds")

        def reset_to_normal():
            time.sleep(orgasm_duration)
            if self.current_state == "orgasm":
                self.current_state = "normal"
                print("[AUTO TRANSITION] Returned to normal state")

        Thread(target=reset_to_normal, daemon=True).start()

    def _trigger_word(self):
        """Trigger a spoken word or phrase using TTS or Mistral."""
        self.schedule_next_word()

        if not self.tts_available:
            return

        # Get word from Mistral if available
        word = self._get_word_from_mistral()

        # Speak the word in a separate thread to not block audio
        def speak_word():
            try:
                # Temporarily lower volume for spoken word
                original_volume = self.volume
                self.volume = self.volume * 0.3  # Reduce moaning volume
                time.sleep(0.1)

                # Speak the word
                self.tts_engine.say(word)
                self.tts_engine.runAndWait()

                time.sleep(0.1)
                self.volume = original_volume
            except:
                pass

        Thread(target=speak_word, daemon=True).start()

    def _get_word_from_mistral(self):
        """Get a word or phrase from Mistral based on current state."""
        if self.mistral_client:
            try:
                prompt = f"""Generate a single short exclamation or word that would be used during sexual activity in the {self.current_state} state.
                Respond with ONLY the word or phrase, nothing else. Maximum 2-3 words.
                Examples: "yes", "oh god", "more", "right there", "don't stop", "oh yes"
                Respond with just the text, no quotes or explanation."""

                print(f"[Mistral API] Requesting word for state '{self.current_state}'")
                response = self.mistral_client.client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": prompt}]
                )
                word = response.choices[0].message.content.strip().strip('"\'')
                print(f"[Mistral API] Generated word: '{word}'")
                return word
            except Exception as e:
                print(f"[Mistral API] Word generation error: {e}")
                pass

        # Fallback words based on state
        if self.current_state == "climax":
            words = ["oh", "yes", "oh god", "yes yes", "don't stop"]
        elif self.current_state == "building":
            words = ["mmm", "oh yes", "more", "right there", "yes"]
        else:
            words = ["mmm", "oh", "yes", "mmm hmm"]

        return random.choice(words)

    def get_waveform_data(self, num_points=1000):
        """Get waveform data for visualization."""
        if len(self.waveform_buffer) == 0:
            return np.zeros(num_points)

        # Return the current waveform buffer (live generated audio)
        if len(self.waveform_buffer) >= num_points:
            return self.waveform_buffer[-num_points:]
        else:
            # Pad if necessary
            padded = np.zeros(num_points)
            padded[-len(self.waveform_buffer):] = self.waveform_buffer
            return padded

    def start_recording(self):
        """Start recording audio for the specified duration."""
        if not self.is_playing:
            print("Cannot record - stream is not playing")
            return False

        self.recording_buffer = []
        self.is_recording = True
        print(f"Recording started - will capture {self.recording_duration} seconds")

        # Schedule stop recording
        def stop_after_duration():
            time.sleep(self.recording_duration)
            self.stop_recording()

        Thread(target=stop_after_duration, daemon=True).start()
        return True

    def stop_recording(self):
        """Stop recording and save to file."""
        if not self.is_recording:
            return

        self.is_recording = False

        if len(self.recording_buffer) == 0:
            print("No audio recorded")
            return None

        # Combine all chunks
        recording = np.concatenate(self.recording_buffer)

        # Generate filename with timestamp
        from datetime import datetime
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to the app directory with absolute path
        app_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(app_dir, f"recording_{timestamp}.wav")

        # Save to file
        try:
            sf.write(filename, recording, self.sample_rate)
            print(f"Recording saved: {filename}")
            print(f"Recording length: {len(recording)/self.sample_rate:.2f} seconds")
            print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")
            return filename
        except Exception as e:
            print(f"Error saving recording: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_audio_from_prompt(self, prompt, duration):
        """
        Generate moaning audio using XTTS voice cloning and LLM-generated phonetics.

        Args:
            prompt: Text description of desired intensity/style
            duration: Duration in seconds to generate

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"\n=== Generating TTS Audio with Voice Cloning ===")
            print(f"Prompt: {prompt}")
            print(f"Duration: {duration} seconds")

            # Check if XTTS is ready
            if not self.xtts_engine.is_available():
                print("ERROR: XTTS not ready - please load a voice file first")
                return False

            # Parse prompt to determine intensity and progression
            intensity, progression = self._parse_prompt_for_intensity(prompt)

            # Generate phonetic vocalization text using LLM or templates
            print(f"Generating phonetic text ({intensity}, {progression})...")
            phonetic_text = self.vocalization_generator.generate_vocalization_text(
                intensity=intensity,
                duration_seconds=duration,
                progression=progression
            )

            print(f"Phonetic text: '{phonetic_text[:100]}...'")

            # Generate audio using XTTS with voice cloning
            print("Synthesizing audio with XTTS v2...")
            audio = self.xtts_engine.generate_audio(
                text=phonetic_text,
                language="en",
                speed=1.0
            )

            if audio is None:
                print("ERROR: XTTS generation failed")
                return False

            # Store the generated audio
            self.last_generated_audio = audio

            # Apply volume
            self.last_generated_audio *= self.volume

            # Normalize to prevent clipping
            max_val = np.max(np.abs(self.last_generated_audio))
            if max_val > 0.95:
                self.last_generated_audio = self.last_generated_audio * (0.95 / max_val)

            print(f"✓ Generated {len(self.last_generated_audio) / self.sample_rate:.2f} seconds of audio")
            return True

        except Exception as e:
            print(f"Error generating audio: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _parse_prompt_for_intensity(self, prompt):
        """
        Parse user prompt to determine intensity and progression.

        Returns:
            tuple: (intensity, progression)
        """
        prompt_lower = prompt.lower()

        # Determine intensity
        if any(word in prompt_lower for word in ['soft', 'gentle', 'quiet', 'subtle', 'whisper']):
            intensity = 'soft'
        elif any(word in prompt_lower for word in ['intense', 'strong', 'powerful', 'passionate', 'loud', 'extreme']):
            intensity = 'intense'
        else:
            intensity = 'moderate'

        # Determine progression
        if any(word in prompt_lower for word in ['building', 'growing', 'increasing', 'escalating', 'gradual']):
            progression = 'building'
        elif any(word in prompt_lower for word in ['climax', 'peak', 'maximum', 'apex', 'culminat']):
            progression = 'climaxing'
        else:
            progression = 'steady'

        return intensity, progression

    def export_generated_audio(self, file_path):
        """
        Export the last generated audio to a WAV file.

        Args:
            file_path: Path where to save the WAV file

        Returns:
            bool: True if successful, False otherwise
        """
        if self.last_generated_audio is None:
            print("No audio to export - generate audio first")
            return False

        try:
            sf.write(file_path, self.last_generated_audio, self.sample_rate)
            print(f"Audio exported to: {file_path}")
            return True
        except Exception as e:
            print(f"Error exporting audio: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Clean up resources."""
        self.stop_playback()
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        if self.xtts_engine:
            self.xtts_engine.cleanup()
        self.p.terminate()

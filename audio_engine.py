"""Audio generation and playback engine for vocal moaning sounds."""
import numpy as np
import soundfile as sf
import librosa
import pyaudio
from threading import Thread, Event
import time
import random
import pyttsx3
from io import BytesIO
import wave
from vocal_synthesizer import VocalSynthesizer
from voice_analyzer import VoiceAnalyzer
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
        self.current_state = "normal"  # normal, building, climax
        self.state_start_time = 0
        self.next_event_time = 0
        self.next_word_time = 0

        # Mistral API caching to avoid rate limits
        self.cached_mistral_params = None
        self.last_mistral_call_time = 0
        self.mistral_cache_duration = 60.0  # Cache for 60 seconds (default)
        self.last_mistral_state = None

        # Audio recording
        self.is_recording = False
        self.recording_buffer = []
        self.recording_duration = 5.0  # Record for 5 seconds

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
        """Start audio generation and playback."""
        if self.is_playing:
            print("Already playing")
            return False

        self.is_playing = True
        self.stop_event.clear()
        self.current_state = "normal"
        self.schedule_next_event()
        self.schedule_next_word()

        self.playback_thread = Thread(target=self._generation_loop, daemon=True)
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
        """Trigger a random build-up and climax sequence."""
        self.current_state = "building"
        self.state_start_time = time.time()
        build_up_duration = random.uniform(MIN_BUILD_UP_DURATION, MAX_BUILD_UP_DURATION)

        def trigger_climax():
            time.sleep(build_up_duration)
            if self.is_playing:
                self.current_state = "climax"
                self.state_start_time = time.time()
                time.sleep(CLIMAX_DURATION)
                self.current_state = "normal"
                self.schedule_next_event()

        Thread(target=trigger_climax, daemon=True).start()

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

    def trigger_climax_manual(self):
        """Manually trigger a climax."""
        self.current_state = "climax"
        self.state_start_time = time.time()

        def reset_to_normal():
            time.sleep(CLIMAX_DURATION)
            if self.current_state == "climax":
                self.current_state = "normal"

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"

        # Save to file
        try:
            sf.write(filename, recording, self.sample_rate)
            print(f"Recording saved: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving recording: {e}")
            return None

    def cleanup(self):
        """Clean up resources."""
        self.stop_playback()
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
        self.p.terminate()

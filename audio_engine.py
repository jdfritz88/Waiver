"""Audio processing and playback engine."""
import numpy as np
import soundfile as sf
import librosa
import pyaudio
from threading import Thread, Event
import time
import random
from config import (
    SAMPLE_RATE, CHUNK_SIZE, MIN_BUILD_UP_DURATION,
    MAX_BUILD_UP_DURATION, CLIMAX_DURATION,
    MIN_TIME_BETWEEN_EVENTS, MAX_TIME_BETWEEN_EVENTS
)


class AudioEngine:
    """Handles audio loading, processing, and playback."""

    def __init__(self, mistral_client=None):
        """Initialize the audio engine."""
        self.mistral_client = mistral_client
        self.audio_data = None
        self.sample_rate = SAMPLE_RATE
        self.is_playing = False
        self.stop_event = Event()
        self.playback_thread = None

        # Audio parameters (controlled by UI)
        self.volume = 1.0
        self.pitch_shift = 0.0  # in semitones
        self.octave_shift = 0  # -2 to +2

        # State management
        self.current_state = "normal"  # normal, building, climax
        self.state_start_time = 0
        self.next_event_time = 0

        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = None

    def load_audio_file(self, file_path):
        """Load a WAV file."""
        try:
            data, sr = sf.read(file_path)
            # Resample if necessary
            if sr != self.sample_rate:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)

            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            self.audio_data = data
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False

    def process_audio(self, audio_chunk, state_params=None):
        """
        Apply audio effects to a chunk.

        Args:
            audio_chunk: numpy array of audio samples
            state_params: dict with volume_mult, pitch_shift, speed_mult from Mistral

        Returns:
            processed numpy array
        """
        processed = audio_chunk.copy()

        # Apply base volume
        processed = processed * self.volume

        # Apply state-based volume multiplier
        if state_params:
            processed = processed * state_params.get('volume_mult', 1.0)

        # Apply pitch and octave shift
        total_pitch_shift = self.pitch_shift + (self.octave_shift * 12)
        if state_params:
            total_pitch_shift += state_params.get('pitch_shift', 0.0)

        if total_pitch_shift != 0:
            processed = librosa.effects.pitch_shift(
                processed,
                sr=self.sample_rate,
                n_steps=total_pitch_shift
            )

        return processed

    def start_playback(self):
        """Start audio playback in a separate thread."""
        if self.audio_data is None:
            print("No audio file loaded")
            return False

        if self.is_playing:
            print("Already playing")
            return False

        self.is_playing = True
        self.stop_event.clear()
        self.current_state = "normal"
        self.schedule_next_event()

        self.playback_thread = Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()
        return True

    def stop_playback(self):
        """Stop audio playback."""
        self.is_playing = False
        self.stop_event.set()
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def _playback_loop(self):
        """Main playback loop running in separate thread."""
        # Open audio stream
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        )

        position = 0
        audio_length = len(self.audio_data)

        while self.is_playing and not self.stop_event.is_set():
            # Check if we need to trigger an event
            current_time = time.time()
            if current_time >= self.next_event_time:
                self.trigger_random_event()

            # Update state-based parameters
            state_params = self._get_state_parameters()

            # Get chunk of audio
            end_pos = min(position + CHUNK_SIZE, audio_length)
            chunk = self.audio_data[position:end_pos]

            # Loop audio if we reach the end
            if end_pos >= audio_length:
                position = 0
                continue

            # Process audio with current parameters
            processed_chunk = self.process_audio(chunk, state_params)

            # Ensure correct format
            processed_chunk = processed_chunk.astype(np.float32)

            # Play chunk
            self.stream.write(processed_chunk.tobytes())

            position = end_pos

        # Cleanup
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def _get_state_parameters(self):
        """Get current state parameters from Mistral or defaults."""
        if self.mistral_client:
            try:
                return self.mistral_client.generate_intensity_parameters(self.current_state)
            except:
                pass

        # Default parameters based on state
        if self.current_state == "climax":
            return {"volume_mult": 1.5, "pitch_shift": 2.0, "speed_mult": 1.1}
        elif self.current_state == "building":
            return {"volume_mult": 1.2, "pitch_shift": 1.0, "speed_mult": 1.05}
        else:
            return {"volume_mult": 1.0, "pitch_shift": 0.0, "speed_mult": 1.0}

    def schedule_next_event(self):
        """Schedule the next random event."""
        delay = random.uniform(MIN_TIME_BETWEEN_EVENTS, MAX_TIME_BETWEEN_EVENTS)
        self.next_event_time = time.time() + delay

    def trigger_random_event(self):
        """Trigger a random build-up and climax sequence."""
        # Start build-up
        self.current_state = "building"
        build_up_duration = random.uniform(MIN_BUILD_UP_DURATION, MAX_BUILD_UP_DURATION)

        # Schedule climax after build-up
        def trigger_climax():
            time.sleep(build_up_duration)
            if self.is_playing:
                self.current_state = "climax"
                time.sleep(CLIMAX_DURATION)
                self.current_state = "normal"
                self.schedule_next_event()

        Thread(target=trigger_climax, daemon=True).start()

    def trigger_build_up_manual(self):
        """Manually trigger a build-up."""
        if self.current_state == "normal":
            self.current_state = "building"

            def reset_to_normal():
                time.sleep(random.uniform(MIN_BUILD_UP_DURATION, MAX_BUILD_UP_DURATION))
                if self.current_state == "building":
                    self.current_state = "normal"

            Thread(target=reset_to_normal, daemon=True).start()

    def trigger_climax_manual(self):
        """Manually trigger a climax."""
        self.current_state = "climax"

        def reset_to_normal():
            time.sleep(CLIMAX_DURATION)
            if self.current_state == "climax":
                self.current_state = "normal"

        Thread(target=reset_to_normal, daemon=True).start()

    def get_waveform_data(self, num_points=1000):
        """Get waveform data for visualization."""
        if self.audio_data is None:
            return np.zeros(num_points)

        # Downsample for visualization
        step = max(1, len(self.audio_data) // num_points)
        return self.audio_data[::step][:num_points]

    def cleanup(self):
        """Clean up resources."""
        self.stop_playback()
        self.p.terminate()

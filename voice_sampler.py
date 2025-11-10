"""Voice sampler that extracts and manages audio grains from voice samples."""
import numpy as np
import librosa
import soundfile as sf
from scipy import signal


class VoiceSampler:
    """
    Extracts vowel-like segments from voice samples for concatenative synthesis.
    This preserves the actual voice characteristics instead of synthesizing.
    """

    def __init__(self, sample_rate=44100):
        """Initialize the voice sampler."""
        self.sample_rate = sample_rate
        self.grains = []  # List of extracted audio segments
        self.grain_pitches = []  # Detected pitch for each grain
        self.loaded_audio = None

    def load_voice_sample(self, file_path):
        """
        Load a voice sample and extract usable grains.

        Args:
            file_path: Path to WAV file

        Returns:
            bool: Success status
        """
        try:
            print(f"Loading voice sample: {file_path}")

            # Load audio
            audio, sr = sf.read(file_path)

            # Resample if necessary
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            self.loaded_audio = audio

            # Extract grains
            self._extract_grains(audio)

            print(f"Extracted {len(self.grains)} usable voice segments")
            return True

        except Exception as e:
            print(f"Error loading voice sample: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_grains(self, audio):
        """
        Extract vowel-like segments (grains) from audio.
        Focus on sustained, voiced segments (moans, vowels).
        """
        # Parameters for grain extraction
        grain_size = int(0.3 * self.sample_rate)  # 300ms grains
        hop_size = int(0.1 * self.sample_rate)    # 100ms hop

        # Compute RMS energy for each frame
        rms = librosa.feature.rms(y=audio, frame_length=grain_size, hop_length=hop_size)[0]

        # Compute zero crossing rate (lower = more vowel-like)
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=grain_size, hop_length=hop_size)[0]

        # Compute spectral centroid (stable = voiced)
        cent = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate,
                                                  n_fft=grain_size, hop_length=hop_size)[0]

        # Find frames with good characteristics for moaning:
        # - High energy (loud)
        # - Low zero crossing rate (sustained, not noisy)
        # - Moderate spectral centroid (voiced, not breath)

        energy_threshold = np.percentile(rms, 50)  # Above median energy
        zcr_threshold = np.percentile(zcr, 40)     # Below 40th percentile (sustained)

        for i in range(len(rms)):
            start_sample = i * hop_size
            end_sample = start_sample + grain_size

            if end_sample > len(audio):
                break

            # Check if this frame meets criteria
            if rms[i] > energy_threshold and zcr[i] < zcr_threshold:
                # Extract grain
                grain = audio[start_sample:end_sample].copy()

                # Apply fade in/out to avoid clicks
                grain = self._apply_envelope(grain)

                # Detect pitch for this grain
                pitch = self._detect_pitch(grain)

                # Store grain
                self.grains.append(grain)
                self.grain_pitches.append(pitch if pitch > 0 else 220.0)  # Default to 220 Hz if no pitch

        print(f"  - Grain size: {grain_size/self.sample_rate:.2f}s")
        print(f"  - Energy threshold: {energy_threshold:.4f}")
        print(f"  - Found {len(self.grains)} good segments")

    def _apply_envelope(self, grain):
        """Apply smooth fade in/out envelope to grain to avoid clicks."""
        fade_length = int(0.01 * self.sample_rate)  # 10ms fade
        envelope = np.ones(len(grain))

        # Fade in
        envelope[:fade_length] = np.linspace(0, 1, fade_length)

        # Fade out
        envelope[-fade_length:] = np.linspace(1, 0, fade_length)

        return grain * envelope

    def _detect_pitch(self, grain):
        """
        Detect the fundamental frequency of a grain.

        Args:
            grain: Audio segment

        Returns:
            float: Detected pitch in Hz, or 0 if not detected
        """
        # Use autocorrelation for pitch detection
        corr = np.correlate(grain, grain, mode='full')
        corr = corr[len(corr)//2:]

        # Find peaks in autocorrelation
        # Limit search to reasonable pitch range (80-400 Hz)
        min_period = int(self.sample_rate / 400)
        max_period = int(self.sample_rate / 80)

        if max_period >= len(corr):
            return 220.0

        # Find peak in autocorrelation
        peak_idx = np.argmax(corr[min_period:max_period]) + min_period

        if peak_idx > 0:
            pitch = self.sample_rate / peak_idx
            return pitch
        else:
            return 220.0

    def get_random_grain(self, target_pitch=None, target_duration=None):
        """
        Get a random grain, optionally pitch-shifted and time-stretched.

        Args:
            target_pitch: Desired pitch in Hz (None = no shift)
            target_duration: Desired duration in seconds (None = no stretch)

        Returns:
            numpy array: Audio segment
        """
        if len(self.grains) == 0:
            # Return silence if no grains
            duration = target_duration if target_duration else 0.3
            return np.zeros(int(duration * self.sample_rate))

        # Pick a random grain
        idx = np.random.randint(0, len(self.grains))
        grain = self.grains[idx].copy()
        grain_pitch = self.grain_pitches[idx]

        # Pitch shift if requested
        if target_pitch is not None and grain_pitch > 0:
            # Calculate pitch shift in semitones
            pitch_ratio = target_pitch / grain_pitch
            n_steps = 12 * np.log2(pitch_ratio)

            # Apply pitch shift
            grain = librosa.effects.pitch_shift(grain, sr=self.sample_rate,
                                               n_steps=n_steps, n_fft=2048)

        # Time stretch if requested
        if target_duration is not None:
            current_duration = len(grain) / self.sample_rate
            stretch_ratio = current_duration / target_duration

            if stretch_ratio != 1.0:
                grain = librosa.effects.time_stretch(grain, rate=stretch_ratio)

        return grain

    def generate_continuous_stream(self, duration, base_pitch=220, pitch_variation=0.1):
        """
        Generate a continuous stream of audio by blending grains.

        Args:
            duration: Duration in seconds
            base_pitch: Base pitch in Hz
            pitch_variation: Pitch variation factor (0.0 to 1.0)

        Returns:
            numpy array: Generated audio
        """
        if len(self.grains) == 0:
            return np.zeros(int(duration * self.sample_rate))

        output = []
        total_samples = 0
        target_samples = int(duration * self.sample_rate)

        grain_duration = 0.3  # 300ms grains
        overlap = 0.15        # 150ms overlap

        while total_samples < target_samples:
            # Vary pitch slightly
            pitch = base_pitch * (1.0 + np.random.uniform(-pitch_variation, pitch_variation))

            # Get grain with target pitch
            grain = self.get_random_grain(target_pitch=pitch, target_duration=grain_duration)

            output.append(grain)
            total_samples += int(len(grain) * (1.0 - overlap))  # Account for overlap

        # Concatenate with crossfading
        return self._crossfade_grains(output, overlap_ratio=0.5)

    def _crossfade_grains(self, grains, overlap_ratio=0.5):
        """
        Concatenate grains with crossfading for smooth transitions.

        Args:
            grains: List of audio segments
            overlap_ratio: How much to overlap (0.0 to 1.0)

        Returns:
            numpy array: Blended audio
        """
        if len(grains) == 0:
            return np.array([])

        if len(grains) == 1:
            return grains[0]

        # Calculate overlap in samples
        avg_grain_len = int(np.mean([len(g) for g in grains]))
        overlap_samples = int(avg_grain_len * overlap_ratio)

        # Estimate output length
        step_size = avg_grain_len - overlap_samples
        output_len = avg_grain_len + step_size * (len(grains) - 1)
        output = np.zeros(output_len)

        position = 0

        for i, grain in enumerate(grains):
            grain_len = len(grain)

            if position + grain_len > len(output):
                # Extend output if needed
                output = np.pad(output, (0, position + grain_len - len(output)))

            if i == 0:
                # First grain: just add it
                output[position:position + grain_len] = grain
            else:
                # Create crossfade
                fade_len = min(overlap_samples, grain_len)
                fade_out = np.linspace(1, 0, fade_len)
                fade_in = np.linspace(0, 1, fade_len)

                # Crossfade overlapping region
                output[position:position + fade_len] *= fade_out
                output[position:position + fade_len] += grain[:fade_len] * fade_in

                # Add rest of grain
                if grain_len > fade_len:
                    output[position + fade_len:position + grain_len] = grain[fade_len:]

            position += step_size

        return output

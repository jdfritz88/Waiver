"""Vocal sound synthesizer for generating moaning sounds."""
import numpy as np
from scipy import signal
import random


class VocalSynthesizer:
    """Synthesizes realistic vocal moaning sounds using formant synthesis."""

    # Formant frequencies for different vowel sounds (F1, F2, F3 in Hz)
    # These create different vocal timbres
    # These are BASE formants that will be shifted based on voice profile
    VOWEL_FORMANTS = {
        'ah': (700, 1220, 2600),   # Open mouth sound
        'oh': (570, 840, 2410),     # Rounded sound
        'uh': (640, 1190, 2390),    # Neutral sound
        'eh': (530, 1840, 2480),    # Mid-open sound
        'aa': (850, 1610, 2850),    # Very open sound
    }

    def __init__(self, sample_rate=44100, voice_profile=None):
        """Initialize the vocal synthesizer."""
        self.sample_rate = sample_rate
        self.phase = 0.0  # Phase accumulator for continuous pitch
        self.voice_profile = voice_profile  # Voice characteristics from analyzer

        # Calculate formant shift ratio if we have a voice profile
        if voice_profile and 'formants' in voice_profile:
            # Shift all formants proportionally to match the voice
            # Use ratio of voice F1 to standard female F1 (700 Hz)
            self.formant_shift_ratio = voice_profile['formants'][0] / 700.0
        else:
            self.formant_shift_ratio = 1.0

    def generate_chunk(self, duration, base_freq, vowel='ah', breathiness=0.3,
                      intensity=1.0, vibrato_rate=5.0, vibrato_depth=0.5):
        """
        Generate a chunk of vocal sound.

        Args:
            duration: Duration in seconds
            base_freq: Base frequency in Hz (pitch)
            vowel: Vowel sound ('ah', 'oh', 'uh', 'eh', 'aa')
            breathiness: Amount of breath noise (0.0 to 1.0)
            intensity: Overall volume (0.0 to 2.0)
            vibrato_rate: Vibrato frequency in Hz
            vibrato_depth: Vibrato depth in semitones

        Returns:
            numpy array of audio samples
        """
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples, endpoint=False)

        # Apply vibrato (pitch modulation)
        vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
        freq_modulated = base_freq * (2 ** (vibrato / 12.0))

        # Generate the glottal source (vocal fold vibration)
        # Using a richer waveform than pure sine for more realistic vocal quality
        source = self._generate_glottal_source(t, freq_modulated)

        # Apply formant filtering (vocal tract resonance)
        formants = self.VOWEL_FORMANTS.get(vowel, self.VOWEL_FORMANTS['ah'])
        vocal_sound = self._apply_formants(source, formants)

        # Add breathiness (aspiration noise)
        if breathiness > 0:
            noise = np.random.normal(0, breathiness * 0.15, num_samples)
            # Filter noise to make it more realistic
            noise = self._apply_breath_filter(noise)
            vocal_sound = vocal_sound + noise

        # Apply intensity
        vocal_sound = vocal_sound * intensity

        # Normalize to prevent clipping
        max_val = np.max(np.abs(vocal_sound))
        if max_val > 0:
            vocal_sound = vocal_sound / max_val * 0.8

        return vocal_sound.astype(np.float32)

    def _generate_glottal_source(self, t, freq):
        """
        Generate the glottal source signal (vocal fold vibration).

        Uses additive synthesis with multiple harmonics for richer sound.
        """
        # Phase accumulation for smooth frequency transitions
        phase_increment = 2 * np.pi * freq / self.sample_rate
        phases = self.phase + np.cumsum(phase_increment)
        self.phase = phases[-1] % (2 * np.pi)

        # Fundamental frequency
        source = np.sin(phases)

        # Add harmonics (overtones) with decreasing amplitude
        num_harmonics = 8
        for n in range(2, num_harmonics + 1):
            harmonic_amp = 1.0 / (n ** 1.5)  # Amplitude decreases with harmonic number
            source += harmonic_amp * np.sin(n * phases)

        # Normalize
        source = source / (num_harmonics / 2)

        return source

    def _apply_formants(self, source, formants):
        """
        Apply formant filtering to create vowel sounds.

        Formants are resonant frequencies of the vocal tract.
        Uses voice profile to adjust formants to match the loaded voice.
        """
        filtered = source.copy()

        # Shift formants based on voice profile
        shifted_formants = tuple(f * self.formant_shift_ratio for f in formants[:3])

        # Apply bandpass filters for each formant
        for i, formant_freq in enumerate(shifted_formants):
            # Bandwidth increases with formant number
            bandwidth = 50 + i * 30

            # Create bandpass filter
            nyquist = self.sample_rate / 2
            low = (formant_freq - bandwidth) / nyquist
            high = (formant_freq + bandwidth) / nyquist

            # Ensure filter parameters are valid
            low = max(0.01, min(0.99, low))
            high = max(0.01, min(0.99, high))

            if low < high:
                b, a = signal.butter(2, [low, high], btype='band')
                # Apply filter with initial conditions to avoid transients
                formant_component = signal.lfilter(b, a, source)
                filtered += formant_component * (2.0 - i * 0.5)

        return filtered

    def _apply_breath_filter(self, noise):
        """Apply filtering to breath noise to make it more realistic."""
        # High-pass filter to remove low rumble
        nyquist = self.sample_rate / 2
        b, a = signal.butter(2, 500 / nyquist, btype='high')
        filtered_noise = signal.lfilter(b, a, noise)

        # Bandpass to emphasize breath frequencies
        b2, a2 = signal.butter(2, [1000 / nyquist, 8000 / nyquist], btype='band')
        filtered_noise = signal.lfilter(b2, a2, filtered_noise)

        return filtered_noise

    def get_random_vowel_transition(self):
        """
        Get a random vowel transition for natural variation.

        Returns:
            tuple: (start_vowel, end_vowel)
        """
        vowels = list(self.VOWEL_FORMANTS.keys())

        # Common transitions for moaning sounds
        common_transitions = [
            ('ah', 'oh'),
            ('oh', 'ah'),
            ('uh', 'ah'),
            ('ah', 'aa'),
            ('oh', 'uh'),
        ]

        if random.random() < 0.7:  # 70% use common transitions
            return random.choice(common_transitions)
        else:
            return random.choice(vowels), random.choice(vowels)

    def interpolate_vowels(self, vowel1, vowel2, position):
        """
        Interpolate between two vowels.

        Args:
            vowel1: Starting vowel
            vowel2: Ending vowel
            position: Interpolation position (0.0 to 1.0)

        Returns:
            str: Vowel sound to use (snaps to nearest)
        """
        # Simple approach: snap to nearest vowel
        return vowel1 if position < 0.5 else vowel2

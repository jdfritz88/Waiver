"""Advanced audio post-processing for emotional prosody control."""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import random
from scipy import signal

# Optional imports for advanced pitch processing
try:
    import pyrubberband as pyrb
    RUBBERBAND_AVAILABLE = True
except ImportError:
    RUBBERBAND_AVAILABLE = False
    print("[AudioProcessor] pyrubberband not available - using fallback methods")

try:
    import pyworld as pw
    PYWORLD_AVAILABLE = True
except ImportError:
    PYWORLD_AVAILABLE = False
    print("[AudioProcessor] pyworld not available - using fallback methods")


class AudioProcessor:
    """
    Applies advanced prosody controls to audio for emotional expression.

    Supports 7 key prosody controls:
    1. Pitch Variation - Dynamic frequency control
    2. Speaking Rate/Tempo - Variable speed
    3. Volume/Intensity - Dynamic loudness with crescendo
    4. Emphasis/Stress - Syllable/word emphasis
    5. Pause/Timing - Breath breaks and gasps
    6. Voice Quality/Timbre - Texture changes (breathiness, roughness)
    7. Intonation Patterns - Melodic pitch curves
    """

    def __init__(self, sample_rate=44100):
        """Initialize the audio processor."""
        self.sample_rate = sample_rate

        # Pitch processing method selection
        # Options: 'rubberband', 'pyworld', 'lower_then_shift', 'hybrid'
        self.pitch_method = 'hybrid'  # Default to hybrid (no extra software needed)

        # Design anti-aliasing low-pass filter to remove artifacts above 10kHz
        # This prevents buzz from pitch shifting and time stretching artifacts
        nyquist = sample_rate / 2
        cutoff_freq = 10000  # Hz - remove content above 10kHz
        normalized_cutoff = cutoff_freq / nyquist
        # 6th order Butterworth filter for smooth rolloff
        self.lpf_b, self.lpf_a = signal.butter(6, normalized_cutoff, btype='low')

    def set_pitch_method(self, method):
        """
        Set the pitch processing method to use.

        Args:
            method: 'rubberband', 'pyworld', 'lower_then_shift', or 'hybrid'
        """
        valid_methods = ['rubberband', 'pyworld', 'lower_then_shift', 'hybrid']
        if method in valid_methods:
            self.pitch_method = method
            print(f"[AudioProcessor] Pitch method set to: {method}")
        else:
            print(f"[AudioProcessor] Invalid method '{method}', keeping '{self.pitch_method}'")

    # ========== OPTION 1: Rubberband Formant-Preserving Pitch Shift ==========
    def _pitch_shift_rubberband(self, audio, semitones):
        """
        Option 1: Formant-preserving pitch shift using rubberband.

        Best quality, but requires external rubberband-cli binary.
        """
        if not RUBBERBAND_AVAILABLE:
            print("[AudioProcessor] Rubberband not available, falling back to librosa")
            return self._pitch_shift_librosa(audio, semitones)

        try:
            # pyrubberband preserves formants automatically
            shifted = pyrb.pitch_shift(audio, self.sample_rate, semitones)
            return shifted.astype(np.float32)
        except Exception as e:
            print(f"[AudioProcessor] Rubberband error: {e}, falling back")
            return self._pitch_shift_librosa(audio, semitones)

    # ========== OPTION 2: WORLD Vocoder Resynthesis ==========
    def _pitch_shift_pyworld(self, audio, semitones):
        """
        Option 2: Resynthesis with WORLD vocoder.

        Professional quality, preserves formants, pure Python.
        More complex but very high quality.
        """
        if not PYWORLD_AVAILABLE:
            print("[AudioProcessor] pyworld not available, falling back to librosa")
            return self._pitch_shift_librosa(audio, semitones)

        try:
            # Ensure float64 for pyworld
            audio = audio.astype(np.float64)

            # Extract F0 (pitch), spectral envelope, and aperiodicity
            f0, sp, ap = pw.wav2world(audio, self.sample_rate)

            # Modify F0 (pitch) by semitone shift
            pitch_ratio = 2 ** (semitones / 12.0)
            f0_shifted = f0 * pitch_ratio

            # Resynthesize with new pitch but original spectral envelope (formants)
            shifted = pw.synthesize(f0_shifted, sp, ap, self.sample_rate)

            return shifted.astype(np.float32)
        except Exception as e:
            print(f"[AudioProcessor] pyworld error: {e}, falling back")
            return self._pitch_shift_librosa(audio, semitones)

    # ========== OPTION 3: Generate Lower, Then Shift Up ==========
    # Note: This is handled at the XTTS generation level, not here
    # This method just does standard shift (upward shifts sound better)
    def _pitch_shift_lower_first(self, audio, semitones):
        """
        Option 3: Standard pitch shift (assumes XTTS generated lower).

        This method is a placeholder - the actual "generate lower" happens
        in the XTTS engine by requesting lower pitch phonetics.
        Here we just apply upward shift.
        """
        # For upward shifts, librosa works reasonably well
        return self._pitch_shift_librosa(audio, semitones)

    # ========== OPTION 4: Hybrid Approach ==========
    def _pitch_shift_hybrid(self, audio, semitones, state='normal'):
        """
        Option 4: Hybrid approach - state-dependent processing.

        - Normal/Building: Apply pitch variation
        - Orgasm: No pitch shift (use XTTS raw output)
        - This avoids distortion at high pitches

        Returns audio with or without pitch shift based on state.
        """
        # If orgasm state or very small shift, skip pitch processing
        if state == 'orgasm' or abs(semitones) < 0.5:
            return audio

        # For other states, apply standard pitch shift
        return self._pitch_shift_librosa(audio, semitones)

    # ========== Standard librosa pitch shift (fallback) ==========
    def _pitch_shift_librosa(self, audio, semitones):
        """Standard librosa pitch shift (fallback method)."""
        if abs(semitones) < 0.01:
            return audio

        audio = audio.astype(np.float64)
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=semitones
        )
        return shifted.astype(np.float32)

    # ========== Unified pitch shift interface ==========
    def apply_pitch_shift(self, audio, semitones, state='normal'):
        """
        Apply pitch shift using the currently selected method.

        Args:
            audio: Input audio
            semitones: Pitch shift in semitones
            state: Current emotional state (for hybrid mode)

        Returns:
            Pitch-shifted audio
        """
        if abs(semitones) < 0.01:
            return audio

        if self.pitch_method == 'rubberband':
            return self._pitch_shift_rubberband(audio, semitones)
        elif self.pitch_method == 'pyworld':
            return self._pitch_shift_pyworld(audio, semitones)
        elif self.pitch_method == 'lower_then_shift':
            return self._pitch_shift_lower_first(audio, semitones)
        elif self.pitch_method == 'hybrid':
            return self._pitch_shift_hybrid(audio, semitones, state)
        else:
            return self._pitch_shift_librosa(audio, semitones)

    def apply_glissando(self, audio, start_semitones=-6, end_semitones=6, curve='exponential'):
        """
        Apply continuous pitch sweep (glissando) from low to high pitch.

        Args:
            audio: Input audio array
            start_semitones: Starting pitch shift in semitones (negative = lower)
            end_semitones: Ending pitch shift in semitones (positive = higher)
            curve: 'linear', 'exponential', or 'logarithmic'

        Returns:
            Audio with smooth pitch ramp applied
        """
        # Ensure audio is float64 for librosa compatibility
        audio = audio.astype(np.float64)

        duration = len(audio) / self.sample_rate
        num_samples = len(audio)

        # Create pitch curve over time
        if curve == 'exponential':
            # Slow start, fast finish - builds intensity
            t = np.linspace(0, 1, num_samples)
            pitch_curve = start_semitones + (end_semitones - start_semitones) * (t ** 2)
        elif curve == 'logarithmic':
            # Fast start, slow finish
            t = np.linspace(0, 1, num_samples)
            pitch_curve = start_semitones + (end_semitones - start_semitones) * np.sqrt(t)
        else:  # linear
            pitch_curve = np.linspace(start_semitones, end_semitones, num_samples)

        # Apply time-varying pitch shift
        # We'll process in overlapping chunks for smooth transitions
        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        hop_size = chunk_size // 2
        output = np.zeros_like(audio)
        window = np.hanning(chunk_size)

        for i in range(0, len(audio) - chunk_size, hop_size):
            chunk = audio[i:i + chunk_size]
            # Get average pitch shift for this chunk
            chunk_pitch = np.mean(pitch_curve[i:i + chunk_size])

            # Apply pitch shift to chunk
            shifted_chunk = librosa.effects.pitch_shift(
                chunk,
                sr=self.sample_rate,
                n_steps=chunk_pitch
            )

            # Overlap-add with window
            output[i:i + chunk_size] += shifted_chunk * window

        # Normalize
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output))

        # Convert back to float32
        return output.astype(np.float32)

    def apply_pitch_variation(self, audio, variation_percent=10, state='normal'):
        """
        Apply random pitch variation throughout audio using selected method.

        Args:
            variation_percent: Percentage of random pitch variation (0-100)
            state: Current emotional state (for hybrid mode)

        Returns:
            Audio with pitch variation applied
        """
        if variation_percent <= 0:
            return audio

        # Convert percentage to semitone range - REDUCED to prevent hollow sound
        # Reduced from 0-12 semitones to 0-4 semitones max
        max_semitones = (variation_percent / 100.0) * 4

        # Random pitch shift
        pitch_shift = random.uniform(-max_semitones, max_semitones)

        # Skip pitch shift if it's too small
        if abs(pitch_shift) < 0.5:
            return audio

        # Use the unified pitch shift interface
        return self.apply_pitch_shift(audio, pitch_shift, state)

    def apply_tempo_modulation(self, audio, tempo_percent=100, state='normal'):
        """
        Modify speaking rate/tempo.

        Args:
            audio: Input audio
            tempo_percent: Base tempo percentage (0-200%, 100=normal)
            state: 'normal', 'building', or 'orgasm' - affects tempo

        Returns:
            Time-stretched audio
        """
        # Ensure audio is float64 for librosa compatibility
        audio = audio.astype(np.float64)

        # State-based tempo adjustments - REDUCED to prevent distortion
        if state == 'building':
            # Slightly speed up during build-up (reduced from 30% to 10%)
            tempo_mult = 1.0 + (tempo_percent / 100.0) * 0.10
        elif state == 'orgasm':
            # Moderate speed up during orgasm (reduced from 50% to 15%)
            tempo_mult = 1.0 + (tempo_percent / 100.0) * 0.15
        else:  # normal
            tempo_mult = tempo_percent / 100.0

        # Clamp tempo to safe range to prevent severe distortion
        tempo_mult = max(0.8, min(1.25, tempo_mult))

        # Apply time stretch (tempo change without pitch change)
        stretched = librosa.effects.time_stretch(audio, rate=tempo_mult)

        # Convert back to float32
        return stretched.astype(np.float32)

    def apply_volume_dynamics(self, audio, intensity_percent=100, state='normal'):
        """
        Apply dynamic volume control with crescendo effects.

        Args:
            audio: Input audio
            intensity_percent: Volume intensity (0-200%)
            state: Current emotional state

        Returns:
            Audio with volume dynamics applied
        """
        # Base volume multiplier
        base_volume = intensity_percent / 100.0

        # Create volume envelope based on state
        # Updated based on reference audio analysis: orgasms are QUIETER (0.25x energy)
        if state == 'building':
            # Crescendo - gradual increase (reduced from 0.5-1.5 to 0.7-1.2)
            envelope = np.linspace(0.7, 1.2, len(audio))
        elif state == 'orgasm':
            # REDUCED volume during orgasm (was 1.15x, now 0.85x)
            # Reference audio shows orgasms are 0.25x energy of buildups
            # Intensity comes from breathiness and rapid vocalizations, not volume
            envelope = np.ones(len(audio)) * 0.85
            # Slight variation instead of peaks
            for _ in range(5):
                variation_pos = random.randint(0, len(audio) - 1)
                variation_width = int(self.sample_rate * 0.2)  # 200ms variation
                start = max(0, variation_pos - variation_width)
                end = min(len(audio), variation_pos + variation_width)
                envelope[start:end] *= 1.05  # Subtle variation
        else:  # normal
            # Gentle variation
            envelope = np.ones(len(audio))
            # Add some random gentle swells (reduced from 1.2x to 1.1x)
            for _ in range(3):
                swell_pos = random.randint(0, len(audio) - 1)
                swell_width = int(self.sample_rate * 0.5)
                start = max(0, swell_pos - swell_width)
                end = min(len(audio), swell_pos + swell_width)
                envelope[start:end] *= 1.05

        # Apply volume envelope
        output = audio * envelope * base_volume

        # Aggressive normalization to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0.85:
            output = output * (0.85 / max_val)

        return output

    def add_breath_pauses(self, audio, pause_percent=50):
        """
        Add breath pauses and timing variations.

        Args:
            audio: Input audio
            pause_percent: Likelihood and duration of pauses (0-100%)

        Returns:
            Audio with pauses inserted
        """
        if pause_percent <= 0:
            return audio

        # Determine number of pauses based on percentage
        num_pauses = int((pause_percent / 100.0) * 3)  # 0-3 pauses

        if num_pauses == 0:
            return audio

        # Insert pauses at random positions
        segments = []
        last_pos = 0

        pause_positions = sorted([random.randint(0, len(audio)) for _ in range(num_pauses)])

        for pause_pos in pause_positions:
            # Add segment before pause
            segments.append(audio[last_pos:pause_pos])

            # Add pause (silence)
            pause_duration = random.uniform(0.1, 0.5) * (pause_percent / 100.0)  # 0.1-0.5s
            pause_samples = int(pause_duration * self.sample_rate)
            segments.append(np.zeros(pause_samples))

            last_pos = pause_pos

        # Add final segment
        segments.append(audio[last_pos:])

        return np.concatenate(segments)

    def apply_breathiness(self, audio, breathiness_percent=0):
        """
        Add breathy/airy quality to voice.

        Args:
            audio: Input audio
            breathiness_percent: Amount of breathiness (0-100%)

        Returns:
            Audio with breathy texture
        """
        if breathiness_percent <= 0:
            return audio

        # Add subtle high-frequency noise to simulate breathiness
        # Reduced from 0.02 to 0.005 to prevent audible noise artifacts
        noise = np.random.normal(0, 0.005, len(audio))
        breath_amount = breathiness_percent / 100.0

        # Mix noise with audio at reduced level
        output = audio + (noise * breath_amount * 0.3)

        # Gentle normalization to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val

        return output

    def apply_roughness(self, audio, roughness_percent=0):
        """
        Add rough/raspy quality to voice.

        Args:
            audio: Input audio
            roughness_percent: Amount of roughness (0-100%)

        Returns:
            Audio with rough texture
        """
        if roughness_percent <= 0:
            return audio

        # Add low-frequency distortion for roughness
        rough_amount = roughness_percent / 100.0

        # Apply VERY subtle soft clipping (reduced multiplier from 2 to 0.5)
        output = np.tanh(audio * (1.0 + rough_amount * 0.5))

        # Normalize
        if np.max(np.abs(output)) > 0:
            output = output / np.max(np.abs(output))

        return output

    def apply_anti_aliasing_filter(self, audio):
        """
        Apply low-pass filter to remove high-frequency artifacts and buzz.

        This removes aliasing artifacts from pitch shifting and time stretching.

        Args:
            audio: Input audio

        Returns:
            Filtered audio with reduced high-frequency artifacts
        """
        # Apply zero-phase filtering (forward-backward) to avoid phase distortion
        filtered = signal.filtfilt(self.lpf_b, self.lpf_a, audio)

        return filtered.astype(np.float32)

    def process_for_state(self, audio, state, settings):
        """
        Apply all prosody controls based on current state and settings.

        Args:
            audio: Input audio from XTTS
            state: 'normal', 'building', or 'orgasm'
            settings: Dictionary of control percentages

        Returns:
            Fully processed audio with all effects applied
        """
        # Extract settings with defaults
        pitch_var = settings.get('pitch_variation', 10)
        tempo = settings.get('tempo', 100)
        volume = settings.get('volume', 100)
        pauses = settings.get('pauses', 30)
        breathiness = settings.get('breathiness', 20)
        roughness = settings.get('roughness', 10)

        # Add randomization within percentage range
        def randomize(value, variation=0.2):
            """Add ±20% random variation"""
            return value * random.uniform(1.0 - variation, 1.0 + variation)

        # Apply state-specific processing based on reference audio analysis
        if state == 'building':
            # Build-up: Apply glissando pitch ramp
            # Updated from ±4 to -13/+19 based on reference audio analysis
            audio = self.apply_glissando(
                audio,
                start_semitones=settings.get('glissando_start_semitones', -13),
                end_semitones=settings.get('glissando_end_semitones', 19),
                curve=settings.get('glissando_curve', 'exponential')
            )
            # Increase tempo gradually
            audio = self.apply_tempo_modulation(audio, randomize(tempo, 0.15), state)
            # Add crescendo
            audio = self.apply_volume_dynamics(audio, randomize(volume, 0.1), state)
            # More breathiness during build (reduced from 1.5x to 1.2x)
            audio = self.apply_breathiness(audio, randomize(breathiness * 1.2, 0.2))

        elif state == 'orgasm':
            # Orgasm: Based on reference audio analysis
            # - Orgasms are QUIETER (0.25x energy vs buildups)
            # - Intensity from breathiness (0.0473-0.0600 vs 0.0462-0.0521)
            # - Rapid vocalizations, not volume

            # DISABLED pitch variation - XTTS already generates high-pitched vocalizations
            # audio = self.apply_pitch_variation(audio, randomize(pitch_var * 1.5, 0.2))

            # Minimal tempo increase (reduced from 1.15x to 1.05x)
            audio = self.apply_tempo_modulation(audio, randomize(tempo * 1.05, 0.1), state)

            # REDUCED volume (0.85x) - orgasms are quieter in reference audio
            audio = self.apply_volume_dynamics(audio, randomize(volume, 0.1), state)

            # INCREASED breathiness (2.0x) - key to orgasm intensity
            # Reference shows orgasms have much higher breathiness
            audio = self.apply_breathiness(audio, randomize(breathiness * 2.0, 0.2))

            # Roughness disabled by default (roughness=0)
            audio = self.apply_roughness(audio, randomize(roughness * 1.5, 0.2))

            # Short breath gasps
            audio = self.add_breath_pauses(audio, randomize(pauses * 0.5, 0.3))

        elif state == 'post_orgasm_breathing':
            # Post-orgasm: Fast, heavy breathing recovery
            # Slightly increased tempo for rapid breathing
            audio = self.apply_tempo_modulation(audio, randomize(tempo * 1.2, 0.1), 'normal')
            # Normal volume
            audio = self.apply_volume_dynamics(audio, randomize(volume, 0.05), 'normal')
            # High breathiness for exhausted breathing
            audio = self.apply_breathiness(audio, randomize(breathiness * 2.5, 0.2))
            # Frequent short pauses between breaths
            audio = self.add_breath_pauses(audio, randomize(pauses * 1.5, 0.2))

        else:  # normal
            # Normal: Subtle variations
            audio = self.apply_pitch_variation(audio, randomize(pitch_var, 0.2), state)
            audio = self.apply_tempo_modulation(audio, randomize(tempo, 0.1), state)
            audio = self.apply_volume_dynamics(audio, randomize(volume, 0.1), state)
            audio = self.add_breath_pauses(audio, randomize(pauses, 0.2))
            audio = self.apply_breathiness(audio, randomize(breathiness, 0.2))

        # FINAL STEP: Apply anti-aliasing filter to remove artifacts and buzz
        audio = self.apply_anti_aliasing_filter(audio)

        return audio

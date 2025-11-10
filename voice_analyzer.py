"""Voice characteristic analyzer for extracting features from audio files."""
import numpy as np
import librosa
import soundfile as sf


class VoiceAnalyzer:
    """Analyzes voice characteristics from audio samples."""

    def __init__(self):
        """Initialize the voice analyzer."""
        self.voice_profile = None

    def analyze_voice(self, file_path, sample_rate=44100):
        """
        Analyze a voice sample and extract characteristics.

        Args:
            file_path: Path to WAV file
            sample_rate: Target sample rate

        Returns:
            dict: Voice profile with extracted characteristics
        """
        # Load audio
        audio, sr = sf.read(file_path)

        # Resample if necessary
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Extract voice characteristics
        profile = {}

        # 1. Extract fundamental frequency (pitch)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        # Get average pitch (filter out zeros)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if pitch_values:
            profile['avg_pitch'] = np.median(pitch_values)
            profile['pitch_std'] = np.std(pitch_values)
            profile['pitch_min'] = np.percentile(pitch_values, 10)
            profile['pitch_max'] = np.percentile(pitch_values, 90)
        else:
            # Fallback to reasonable defaults
            profile['avg_pitch'] = 220.0
            profile['pitch_std'] = 30.0
            profile['pitch_min'] = 180.0
            profile['pitch_max'] = 280.0

        # 2. Extract spectral characteristics
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        profile['spectral_centroid'] = np.median(spectral_centroids)

        # 3. Extract formants using LPC (Linear Predictive Coding)
        # This approximates the vocal tract resonances
        formants = self._estimate_formants(audio, sr=sample_rate)
        profile['formants'] = formants

        # 4. Extract timbre characteristics
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        profile['mfcc_mean'] = np.mean(mfccs, axis=1)
        profile['mfcc_std'] = np.std(mfccs, axis=1)

        # 5. Analyze breathiness (high frequency content)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        profile['breathiness'] = np.median(spectral_rolloff) / sample_rate

        # 6. Analyze intensity/energy
        rms = librosa.feature.rms(y=audio)[0]
        profile['avg_energy'] = np.median(rms)
        profile['energy_std'] = np.std(rms)

        self.voice_profile = profile
        return profile

    def _estimate_formants(self, audio, sr=44100, n_formants=4):
        """
        Estimate formant frequencies using LPC.

        Args:
            audio: Audio signal
            sr: Sample rate
            n_formants: Number of formants to extract

        Returns:
            list: Estimated formant frequencies
        """
        # Pre-emphasis filter (enhance high frequencies)
        pre_emphasized = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # Window the signal
        window_size = int(0.025 * sr)  # 25ms window
        hop_size = int(0.010 * sr)      # 10ms hop

        formant_estimates = []

        # Process each frame
        for i in range(0, len(pre_emphasized) - window_size, hop_size):
            frame = pre_emphasized[i:i + window_size]

            # Apply Hamming window
            windowed = frame * np.hamming(len(frame))

            # LPC analysis
            # Order = 2 + sample_rate/1000 (rule of thumb)
            lpc_order = 2 + int(sr / 1000)

            # Compute autocorrelation
            r = np.correlate(windowed, windowed, mode='full')
            r = r[len(r)//2:]
            r = r[:lpc_order + 1]

            # Levinson-Durbin recursion to get LPC coefficients
            try:
                a = self._levinson_durbin(r, lpc_order)

                # Find roots of LPC polynomial
                roots = np.roots(a)

                # Convert roots to frequencies
                # Only keep roots inside unit circle (stable)
                roots = roots[np.abs(roots) < 1]

                # Convert to angles
                angles = np.angle(roots)

                # Convert to frequencies
                freqs = np.abs(angles * (sr / (2 * np.pi)))

                # Sort and keep positive frequencies
                freqs = sorted([f for f in freqs if f > 0])

                if len(freqs) >= n_formants:
                    formant_estimates.append(freqs[:n_formants])
            except:
                pass

        # Average formants across frames
        if formant_estimates:
            formants = np.median(formant_estimates, axis=0)
            return formants.tolist()
        else:
            # Return typical formants as fallback
            return [700, 1220, 2600, 3500]

    def _levinson_durbin(self, r, order):
        """
        Levinson-Durbin recursion for LPC coefficients.

        Args:
            r: Autocorrelation sequence
            order: LPC order

        Returns:
            LPC coefficients
        """
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, order + 1):
            lambda_val = 0
            for j in range(1, i):
                lambda_val -= a[j] * r[i - j]
            lambda_val -= r[i]

            # Avoid division by zero
            if abs(e) < 1e-10:
                break

            lambda_val /= e

            a[1:i+1] += lambda_val * a[i-1::-1]
            e *= (1 - lambda_val ** 2)

        return a

    def get_synthesis_params(self):
        """
        Get parameters suitable for synthesis.

        Returns:
            dict: Synthesis parameters
        """
        if self.voice_profile is None:
            return None

        return {
            'base_freq': self.voice_profile['avg_pitch'],
            'freq_range': (self.voice_profile['pitch_min'], self.voice_profile['pitch_max']),
            'formants': self.voice_profile['formants'][:3],  # F1, F2, F3
            'breathiness_factor': min(1.0, self.voice_profile['breathiness'] * 2),
            'timbre': self.voice_profile['mfcc_mean'],
            'avg_energy': self.voice_profile['avg_energy']
        }

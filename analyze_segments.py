"""Analyze different segments of audio to identify where distortion occurs."""
import numpy as np
import soundfile as sf
from scipy import signal

# Load audio
audio, sr = sf.read('recording_20251109_231433.wav')

print(f"Total duration: {len(audio)/sr:.2f}s\n")

# Split into 3-second segments
segment_duration = 3.0  # seconds
segment_samples = int(segment_duration * sr)

print("Analyzing segments for distortion:\n")
print("Segment | Time Range | Peak | RMS  | THD   | Notes")
print("--------|------------|------|------|-------|------")

for i in range(0, len(audio), segment_samples):
    segment = audio[i:i+segment_samples]
    if len(segment) < sr:  # Skip very short segments
        continue

    segment_num = i // segment_samples + 1
    start_time = i / sr
    end_time = min((i + segment_samples) / sr, len(audio) / sr)

    # Calculate peak and RMS
    peak = np.max(np.abs(segment))
    rms = np.sqrt(np.mean(segment**2))

    # Calculate THD for this segment
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1/sr)

    # Find fundamental
    low_freq_range = np.where((freqs > 100) & (freqs < 500))[0]
    if len(low_freq_range) > 0:
        fundamental_idx = low_freq_range[np.argmax(spectrum[low_freq_range])]
        fundamental_freq = freqs[fundamental_idx]
        fundamental_amp = spectrum[fundamental_idx]

        # Check harmonics
        harmonic_energy = 0
        for h in range(2, 6):
            harmonic_freq = fundamental_freq * h
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            if abs(freqs[harmonic_idx] - harmonic_freq) < 20:
                harmonic_amp = spectrum[harmonic_idx]
                harmonic_energy += harmonic_amp**2

        # Calculate THD
        if fundamental_amp > 0:
            thd = 100 * np.sqrt(harmonic_energy) / fundamental_amp
        else:
            thd = 0
    else:
        thd = 0
        fundamental_freq = 0

    # Determine likely state based on characteristics
    if peak > 0.35 or fundamental_freq > 250:
        state = "HIGH (orgasm?)"
    elif rms < 0.04:
        state = "QUIET"
    else:
        state = "normal"

    print(f"   {segment_num:2d}   | {start_time:4.1f}s-{end_time:4.1f}s | {peak:.3f} | {rms:.3f} | {thd:5.1f}% | {state}")

print("\n=== Segment Analysis Complete ===")
print("\nHigh THD segments indicate distortion. Look for 'HIGH' segments with THD > 10%")

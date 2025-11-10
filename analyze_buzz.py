"""Analyze audio file for buzz/distortion artifacts."""
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load audio
audio, sr = sf.read('recording_20251109_231433.wav')

print(f"Sample rate: {sr}")
print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Peak amplitude: {np.max(np.abs(audio)):.4f}")
print(f"RMS energy: {np.sqrt(np.mean(audio**2)):.4f}")
print(f"DC offset: {np.mean(audio):.6f}")

# Analyze first 2 seconds for frequency content
segment = audio[:sr*2]
spectrum = np.abs(np.fft.rfft(segment))
freqs = np.fft.rfftfreq(len(segment), 1/sr)

# Find top frequency components
top_indices = np.argsort(spectrum)[-20:]
print("\nTop 20 frequency components:")
for idx in top_indices[::-1]:
    print(f"  {freqs[idx]:.1f} Hz: amplitude {spectrum[idx]:.0f}")

# Check for distortion indicators
# 1. Clipping check
clipping_threshold = 0.99
clipped_samples = np.sum(np.abs(audio) > clipping_threshold)
print(f"\nClipped samples: {clipped_samples} ({100*clipped_samples/len(audio):.2f}%)")

# 2. High frequency noise check
high_freq_start = 8000  # Hz
high_freq_idx = np.where(freqs > high_freq_start)[0]
high_freq_energy = np.sum(spectrum[high_freq_idx]**2)
total_energy = np.sum(spectrum**2)
print(f"High frequency (>{high_freq_start}Hz) energy: {100*high_freq_energy/total_energy:.2f}%")

# 3. Check for specific buzz frequencies
buzz_candidates = [50, 60, 100, 120, 150, 180, 200, 240, 300]
print("\nChecking for buzz frequencies:")
for buzz_freq in buzz_candidates:
    # Find closest frequency bin
    idx = np.argmin(np.abs(freqs - buzz_freq))
    if freqs[idx] < buzz_freq + 5:  # Within 5 Hz
        energy = spectrum[idx]
        print(f"  {buzz_freq}Hz: {energy:.0f}")

# 4. Calculate Total Harmonic Distortion (THD) estimation
# Find fundamental frequency (assume it's the strongest low frequency)
low_freq_range = np.where((freqs > 100) & (freqs < 500))[0]
if len(low_freq_range) > 0:
    fundamental_idx = low_freq_range[np.argmax(spectrum[low_freq_range])]
    fundamental_freq = freqs[fundamental_idx]
    fundamental_amp = spectrum[fundamental_idx]

    print(f"\nFundamental frequency: {fundamental_freq:.1f} Hz")

    # Check harmonics
    harmonic_energy = 0
    print("Harmonics:")
    for h in range(2, 6):
        harmonic_freq = fundamental_freq * h
        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
        if abs(freqs[harmonic_idx] - harmonic_freq) < 20:
            harmonic_amp = spectrum[harmonic_idx]
            harmonic_energy += harmonic_amp**2
            print(f"  {h}x ({harmonic_freq:.1f}Hz): {harmonic_amp:.0f}")

    # THD calculation
    if fundamental_amp > 0:
        thd = 100 * np.sqrt(harmonic_energy) / fundamental_amp
        print(f"\nEstimated THD: {thd:.2f}%")

        if thd > 10:
            print("WARNING: High harmonic distortion detected!")

# 5. Check for inter-sample peaks (can cause aliasing buzz)
# Upsample and check
from scipy import signal
upsampled = signal.resample_poly(audio[:sr], 4, 1)
peak_upsampled = np.max(np.abs(upsampled))
peak_original = np.max(np.abs(audio[:sr]))
print(f"\nInter-sample peaks:")
print(f"  Original peak: {peak_original:.4f}")
print(f"  Upsampled peak: {peak_upsampled:.4f}")
print(f"  Difference: {peak_upsampled - peak_original:.4f}")

if peak_upsampled > 0.95:
    print("WARNING: Upsampled signal shows near-clipping - possible aliasing!")

print("\n=== Analysis complete ===")

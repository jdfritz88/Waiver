"""Analyze reference audio for vocalization patterns (moans, not words)."""
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from scipy.ndimage import label as nd_label

# Load reference audio
audio_file = "F:/Chest/Torrent/518881__the_power_of_sound__slowly-making-love.wav"
print(f"Loading: {audio_file}")
audio, sr = sf.read(audio_file)

print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f}s ({len(audio)/sr/60:.1f} minutes)")

# Convert to mono if stereo
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

print(f"\n=== OVERALL STATISTICS ===")
print(f"Peak amplitude: {np.max(np.abs(audio)):.4f}")
print(f"RMS energy: {np.sqrt(np.mean(audio**2)):.4f}")

# User-specified segments (corrected: start < end)
segments = {
    'buildups': [
        (25*60+30, 27*60, "27:00-25:30"),  # Swapped to make start < end
        (23*60, 23*60+30, "23:30-23:00"),
        (22*60, 23*60, "23:00-22:00"),
        (18*60+15, 19*60+15, "19:15-18:15"),
        (16*60, 17*60+12, "17:12-16:00"),
    ],
    'orgasms': [
        (14*60+38, 15*60+16, "15:16-14:38"),  # Swapped to make start < end
        (14*60+19, 14*60+27, "14:27-14:19"),
        (10*60+20, 10*60+49, "10:49-10:20"),
        (9*60+15, 9*60+35, "9:35-9:15"),
        (8*60+15, 8*60+38, "8:38-8:15"),
    ]
}

def analyze_vocalization_segment(audio_seg, sr, label, seg_type):
    """Analyze a single segment for vocalization characteristics."""
    if len(audio_seg) < sr:
        print(f"  Segment too short, skipping")
        return None

    results = {
        'label': label,
        'type': seg_type,
        'duration': len(audio_seg) / sr
    }

    # Energy/intensity analysis
    rms = librosa.feature.rms(y=audio_seg, hop_length=512)[0]
    results['avg_energy'] = float(np.mean(rms))
    results['peak_energy'] = float(np.max(rms))
    results['energy_range'] = (float(np.min(rms)), float(np.max(rms)))

    # Fundamental frequency (F0) tracking using YIN algorithm
    # YIN is better for vocal pitch tracking than piptrack
    f0 = librosa.yin(audio_seg, fmin=80, fmax=800, sr=sr)

    # Filter out unvoiced frames (f0 == fmin or very low confidence)
    voiced_f0 = f0[f0 > 100]  # Remove very low/unvoiced estimates

    if len(voiced_f0) > 10:
        results['pitch_median'] = float(np.median(voiced_f0))
        results['pitch_mean'] = float(np.mean(voiced_f0))
        results['pitch_min'] = float(np.percentile(voiced_f0, 5))
        results['pitch_max'] = float(np.percentile(voiced_f0, 95))
        results['pitch_std'] = float(np.std(voiced_f0))
    else:
        results['pitch_median'] = None

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_seg, sr=sr)[0]
    results['spectral_centroid_mean'] = float(np.mean(spectral_centroid))

    # Tempo/rhythm (onset detection for vocalization events)
    onset_env = librosa.onset.onset_strength(y=audio_seg, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    results['tempo_bpm'] = float(tempo)

    # Detect vocalization events (onsets)
    onsets = librosa.onset.onset_detect(y=audio_seg, sr=sr, units='time')
    results['num_vocalizations'] = len(onsets)
    if len(onsets) > 1:
        onset_intervals = np.diff(onsets)
        results['avg_vocalization_interval'] = float(np.mean(onset_intervals))

    # Zero crossing rate (indicates breathiness/noisiness)
    zcr = librosa.feature.zero_crossing_rate(audio_seg, hop_length=512)[0]
    results['avg_zcr'] = float(np.mean(zcr))

    return results

print(f"\n{'='*60}")
print(f"=== ANALYZING BUILDUPS ===")
print(f"{'='*60}")

buildup_results = []
for start, end, label in segments['buildups'][:5]:
    segment = audio[int(start*sr):int(end*sr)]
    print(f"\n[BUILDUP] {label}")
    print(f"  Duration: {end-start:.1f}s")

    result = analyze_vocalization_segment(segment, sr, label, 'buildup')
    if result and result['pitch_median']:
        buildup_results.append(result)

        print(f"  Energy: avg={result['avg_energy']:.4f}, peak={result['peak_energy']:.4f}")
        print(f"  Pitch (F0):")
        print(f"    Median: {result['pitch_median']:.1f} Hz")
        print(f"    Range: {result['pitch_min']:.1f} - {result['pitch_max']:.1f} Hz")
        print(f"    Span: {result['pitch_max'] - result['pitch_min']:.1f} Hz")
        print(f"    Std Dev: {result['pitch_std']:.1f} Hz")
        print(f"  Vocalizations: {result['num_vocalizations']} events")
        if 'avg_vocalization_interval' in result:
            print(f"  Avg interval: {result['avg_vocalization_interval']:.2f}s")
        print(f"  Tempo: {result['tempo_bpm']:.1f} BPM")
        print(f"  Spectral Centroid: {result['spectral_centroid_mean']:.1f} Hz")
        print(f"  Zero Crossing Rate: {result['avg_zcr']:.4f} (breathiness indicator)")

print(f"\n{'='*60}")
print(f"=== ANALYZING ORGASMS ===")
print(f"{'='*60}")

orgasm_results = []
for start, end, label in segments['orgasms'][:5]:
    segment = audio[int(start*sr):int(end*sr)]
    print(f"\n[ORGASM] {label}")
    print(f"  Duration: {start-end:.1f}s")

    result = analyze_vocalization_segment(segment, sr, label, 'orgasm')
    if result and result['pitch_median']:
        orgasm_results.append(result)

        print(f"  Energy: avg={result['avg_energy']:.4f}, peak={result['peak_energy']:.4f}")
        print(f"  Pitch (F0):")
        print(f"    Median: {result['pitch_median']:.1f} Hz")
        print(f"    Range: {result['pitch_min']:.1f} - {result['pitch_max']:.1f} Hz")
        print(f"    Span: {result['pitch_max'] - result['pitch_min']:.1f} Hz")
        print(f"    Std Dev: {result['pitch_std']:.1f} Hz")
        print(f"  Vocalizations: {result['num_vocalizations']} events")
        if 'avg_vocalization_interval' in result:
            print(f"  Avg interval: {result['avg_vocalization_interval']:.2f}s")
        print(f"  Tempo: {result['tempo_bpm']:.1f} BPM")
        print(f"  Spectral Centroid: {result['spectral_centroid_mean']:.1f} Hz")
        print(f"  Zero Crossing Rate: {result['avg_zcr']:.4f} (breathiness indicator)")

# Summary comparison
print(f"\n{'='*60}")
print(f"=== SUMMARY: BUILDUPS vs ORGASMS ===")
print(f"{'='*60}")

if buildup_results:
    buildup_pitch_min = min(r['pitch_min'] for r in buildup_results)
    buildup_pitch_max = max(r['pitch_max'] for r in buildup_results)
    buildup_pitch_median = np.median([r['pitch_median'] for r in buildup_results])
    buildup_energy_avg = np.mean([r['avg_energy'] for r in buildup_results])

    print(f"\nBUILDUPS (n={len(buildup_results)}):")
    print(f"  Pitch Range: {buildup_pitch_min:.1f} - {buildup_pitch_max:.1f} Hz")
    print(f"  Median Pitch: {buildup_pitch_median:.1f} Hz")
    print(f"  Avg Energy: {buildup_energy_avg:.4f}")

if orgasm_results:
    orgasm_pitch_min = min(r['pitch_min'] for r in orgasm_results)
    orgasm_pitch_max = max(r['pitch_max'] for r in orgasm_results)
    orgasm_pitch_median = np.median([r['pitch_median'] for r in orgasm_results])
    orgasm_energy_avg = np.mean([r['avg_energy'] for r in orgasm_results])

    print(f"\nORGASMS (n={len(orgasm_results)}):")
    print(f"  Pitch Range: {orgasm_pitch_min:.1f} - {orgasm_pitch_max:.1f} Hz")
    print(f"  Median Pitch: {orgasm_pitch_median:.1f} Hz")
    print(f"  Avg Energy: {orgasm_energy_avg:.4f}")
    print(f"  Peak pitch can reach: {orgasm_pitch_max:.1f} Hz")

if buildup_results and orgasm_results:
    print(f"\nCOMPARISON:")
    print(f"  Pitch increase (median): {orgasm_pitch_median - buildup_pitch_median:+.1f} Hz")
    print(f"  Energy increase: {orgasm_energy_avg / buildup_energy_avg:.2f}x")

# Breathing analysis
print(f"\n{'='*60}")
print(f"=== BREATHING PATTERN ANALYSIS ===")
print(f"{'='*60}")

# Detect quiet segments (likely breaths/pauses)
rms_full = librosa.feature.rms(y=audio, hop_length=2048)[0]
rms_threshold = np.percentile(rms_full, 15)  # Bottom 15% = quiet/breathing

quiet_segments = rms_full < rms_threshold

# Find breath clusters
labeled, num_breaths = nd_label(quiet_segments)
print(f"Detected ~{num_breaths} potential breath/pause segments")

# Sample breath durations
breath_durations = []
for i in range(1, min(100, num_breaths)):
    breath_length = np.sum(labeled == i)
    duration = (breath_length * 2048) / sr
    if 0.1 < duration < 3.0:  # Filter realistic breath durations
        breath_durations.append(duration)

if breath_durations:
    print(f"Typical breath duration: {np.median(breath_durations):.2f}s")
    print(f"Breath range: {np.min(breath_durations):.2f} - {np.max(breath_durations):.2f}s")
    print(f"Breathing occurs roughly every: {len(audio)/sr/len(breath_durations):.1f}s")

print(f"\n{'='*60}")
print("=== RECOMMENDATIONS FOR APP ===")
print(f"{'='*60}")

if buildup_results and orgasm_results:
    # Calculate semitone offsets
    ref_pitch = 220.0  # A3 reference

    buildup_semitones_low = 12 * np.log2(buildup_pitch_min / ref_pitch)
    buildup_semitones_high = 12 * np.log2(buildup_pitch_max / ref_pitch)
    orgasm_semitones_low = 12 * np.log2(orgasm_pitch_min / ref_pitch)
    orgasm_semitones_high = 12 * np.log2(orgasm_pitch_max / ref_pitch)

    print(f"\nPitch Range Settings (relative to {ref_pitch} Hz):")
    print(f"  Buildup range: {buildup_semitones_low:.1f} to {buildup_semitones_high:.1f} semitones")
    print(f"  Orgasm range: {orgasm_semitones_low:.1f} to {orgasm_semitones_high:.1f} semitones")
    print(f"\nSuggested glissando settings:")
    print(f"  glissando_start_semitones: {buildup_semitones_low:.0f}")
    print(f"  glissando_end_semitones: {orgasm_semitones_high:.0f}")

if breath_durations:
    breathing_freq = (len(breath_durations) / (len(audio)/sr)) * 100
    print(f"\nBreathing frequency: ~{breathing_freq:.1f}% of time")
    print(f"Suggested breathing_frequency setting: {min(30, breathing_freq):.0f}%")

print("\n=== Analysis Complete ===")

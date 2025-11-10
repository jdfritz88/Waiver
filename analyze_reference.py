"""Analyze reference audio for interjection patterns, pitch range, tempo, and breathing."""
import numpy as np
import soundfile as sf
import librosa
from scipy import signal

# Load reference audio
audio_file = "F:/Chest/Torrent/518881__the_power_of_sound__slowly-making-love.wav"
print(f"Loading: {audio_file}")
audio, sr = sf.read(audio_file)

print(f"Sample rate: {sr}")
print(f"Duration: {len(audio)/sr:.2f}s ({len(audio)/sr/60:.1f} minutes)")
print(f"Channels: {audio.shape}")

# Convert to mono if stereo
if len(audio.shape) > 1:
    audio = np.mean(audio, axis=1)

print(f"\n=== OVERALL STATISTICS ===")
print(f"Peak amplitude: {np.max(np.abs(audio)):.4f}")
print(f"RMS energy: {np.sqrt(np.mean(audio**2)):.4f}")

# Analyze user-specified segments
segments = {
    'buildups': [
        (27*60, 25*60+30, "27:00-25:30"),
        (23*60+30, 23*60, "23:30-23:00"),
        (23*60, 22*60, "23:00-22:00 (>1min)"),
        (19*60+15, 18*60+15, "19:15-18:15"),
        (17*60+12, 16*60, "17:12-16:00"),
        (15*60+34, 15*60+17, "15:34-15:17"),
        (14*60+37, 14*60+28, "14:37-14:28"),
        (11*60+7, 10*60+50, "11:07-10:50"),
        (9*60+48, 9*60+36, "9:48-9:36"),
        (9*60, 8*60+39, "9:00-8:39"),
        (5*60, 4*60+10, "5:00-4:10"),
    ],
    'orgasms': [
        (15*60+16, 14*60+38, "15:16-14:38"),
        (14*60+27, 14*60+19, "14:27-14:19"),
        (10*60+49, 10*60+20, "10:49-10:20"),
        (9*60+35, 9*60+15, "9:35-9:15"),
        (8*60+38, 8*60+15, "8:38-8:15"),
        (4*60+9, 4*60+1, "4:09-4:01"),
    ]
}

print(f"\n=== ANALYZING BUILDUPS ===")
buildup_pitches = []
buildup_tempos = []

for start, end, label in segments['buildups'][:3]:  # Analyze first 3 buildups
    segment = audio[int(start*sr):int(end*sr)]
    if len(segment) < sr:
        continue

    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    if pitch_values:
        avg_pitch = np.median(pitch_values)
        pitch_range = (np.percentile(pitch_values, 10), np.percentile(pitch_values, 90))
        buildup_pitches.append((avg_pitch, pitch_range))

        print(f"\nBuildup {label}:")
        print(f"  Duration: {end-start:.1f}s")
        print(f"  Avg Pitch: {avg_pitch:.1f} Hz")
        print(f"  Pitch Range: {pitch_range[0]:.1f} - {pitch_range[1]:.1f} Hz")
        print(f"  Pitch Span: {pitch_range[1] - pitch_range[0]:.1f} Hz")

    # Tempo/rhythm analysis (onset detection)
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    buildup_tempos.append(tempo)
    print(f"  Tempo: {tempo:.1f} BPM")

    # Energy dynamics
    rms = librosa.feature.rms(y=segment)[0]
    print(f"  Avg Energy: {np.mean(rms):.4f}")
    print(f"  Energy Range: {np.min(rms):.4f} - {np.max(rms):.4f}")

print(f"\n=== ANALYZING ORGASMS ===")
orgasm_pitches = []

for start, end, label in segments['orgasms'][:3]:  # Analyze first 3 orgasms
    segment = audio[int(start*sr):int(end*sr)]
    if len(segment) < sr:
        continue

    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=segment, sr=sr)
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:
            pitch_values.append(pitch)

    if pitch_values:
        avg_pitch = np.median(pitch_values)
        pitch_range = (np.percentile(pitch_values, 10), np.percentile(pitch_values, 90))
        orgasm_pitches.append((avg_pitch, pitch_range))

        print(f"\nOrgasm {label}:")
        print(f"  Duration: {start-end:.1f}s")
        print(f"  Avg Pitch: {avg_pitch:.1f} Hz")
        print(f"  Pitch Range: {pitch_range[0]:.1f} - {pitch_range[1]:.1f} Hz")
        print(f"  Pitch Span: {pitch_range[1] - pitch_range[0]:.1f} Hz")

    # Tempo/rhythm
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    print(f"  Tempo: {tempo:.1f} BPM")

    # Energy
    rms = librosa.feature.rms(y=segment)[0]
    print(f"  Avg Energy: {np.mean(rms):.4f}")
    print(f"  Peak Energy: {np.max(rms):.4f}")

# Overall pitch range comparison
print(f"\n=== PITCH RANGE SUMMARY ===")
if buildup_pitches:
    buildup_low = min(p[1][0] for p in buildup_pitches)
    buildup_high = max(p[1][1] for p in buildup_pitches)
    print(f"Buildup Range: {buildup_low:.1f} - {buildup_high:.1f} Hz")

if orgasm_pitches:
    orgasm_low = min(p[1][0] for p in orgasm_pitches)
    orgasm_high = max(p[1][1] for p in orgasm_pitches)
    print(f"Orgasm Range: {orgasm_low:.1f} - {orgasm_high:.1f} Hz")
    print(f"Orgasm peak can reach: {orgasm_high:.1f} Hz")

# Breathing pattern detection
print(f"\n=== BREATHING PATTERN ANALYSIS ===")
# Detect quiet segments (likely breaths/pauses)
rms_full = librosa.feature.rms(y=audio, hop_length=512)[0]
rms_threshold = np.percentile(rms_full, 20)  # Bottom 20% = quiet
quiet_segments = rms_full < rms_threshold

# Find breath clusters
from scipy.ndimage import label
labeled, num_breaths = label(quiet_segments)
print(f"Detected ~{num_breaths} potential breath/pause segments")

# Sample breath durations
breath_durations = []
for i in range(1, min(50, num_breaths)):
    breath_length = np.sum(labeled == i)
    duration = (breath_length * 512) / sr
    if 0.1 < duration < 3.0:  # Filter realistic breath durations
        breath_durations.append(duration)

if breath_durations:
    print(f"Typical breath duration: {np.median(breath_durations):.2f}s")
    print(f"Breath range: {np.min(breath_durations):.2f} - {np.max(breath_durations):.2f}s")

print("\n=== Analysis Complete ===")

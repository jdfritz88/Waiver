# Proposed Changes for Waiver App Enhancement
## Based on Reference Audio Analysis & waiver_standards.md

**Date:** 2025-11-09
**Status:** AWAITING USER APPROVAL
**Reference:** F:\Chest\Torrent\518881__the_power_of_sound__slowly-making-love.wav

---

## 🎯 OBJECTIVES

1. **Wider Pitch Range** - Implement low-to-high pitch emotions matching reference audio
2. **Interjection Templates** - Extract and use realistic vocal patterns from reference
3. **Breathing Patterns** - Random breathing between moans/buildups/orgasms
4. **Quality Improvement** - Formant-preserving pitch shift to eliminate distortion
5. **Tempo Dynamics** - Variable speed matching emotional intensity

---

## 📊 REFERENCE AUDIO ANALYSIS (Preliminary)

### Current Findings:
- **Duration:** 37.4 minutes
- **Sample Rate:** 48kHz (stereo)
- **Peak Amplitude:** 0.9451
- **Breathing Pattern:** ~1601 breath segments, typical duration 0.52s, range 0.14-2.74s

### Build-up Timestamps Provided:
- 27:00-25:30, 23:30-23:00, 23:00-22:00 (>1min long)
- 19:15-18:15, 17:12-16:00, 15:34-15:17
- 14:37-14:28, 11:07-10:50, 9:48-9:36
- 9:00-8:39, 5:00-4:10

### Orgasm Timestamps Provided:
- 15:16-14:38 (38s), 14:27-14:19 (8s)
- 10:49-10:20 (29s), 9:35-9:15 (20s)
- 8:38-8:15 (23s), 4:09-4:01 (8s)

### What We Need to Extract:
- [ ] Pitch range for each segment type (normal, buildup, orgasm)
- [ ] Tempo/rhythm patterns (BPM)
- [ ] Common interjections/phonetic patterns
- [ ] Breathing frequency and placement
- [ ] Volume dynamics (crescendo patterns)

---

## 🔧 PROPOSED CHANGES

### **CHANGE 1: Implement Formant-Preserving Pitch Shift**

**Problem:** Current pitch shifting distorts high-pitched orgasm audio
- XTTS generates high pitch → We add pitch shift → 22.3% THD distortion

**Solution:** Use formant-preserving algorithm

**Implementation:**
```python
# Add to audio_processor.py
import pyrubberband as pyrb  # Formant-preserving pitch shift

def apply_pitch_variation_formant_preserving(self, audio, variation_percent=10):
    """Apply pitch shift while preserving formants (voice character)."""
    if variation_percent <= 0:
        return audio

    max_semitones = (variation_percent / 100.0) * 4
    pitch_shift = random.uniform(-max_semitones, max_semitones)

    if abs(pitch_shift) < 0.5:
        return audio

    # Use rubberband for formant-preserving pitch shift
    shifted = pyrb.pitch_shift(audio, self.sample_rate, pitch_shift,
                               rbargs={'--formant': True})
    return shifted.astype(np.float32)
```

**Benefits:**
- Eliminates distortion from pitch shifting
- Maintains voice quality at high pitches
- Allows wider pitch range without artifacts

**Dependencies:**
- Requires `pyrubberband` library: `pip install pyrubberband`
- Requires `rubberband-cli` system binary

**Alternative if rubberband unavailable:**
- Use pyworld vocoder for pitch modification
- Or disable pitch variation during orgasm (current approach)

---

### **CHANGE 2: Wider Pitch Range Implementation**

**Current State:**
- Pitch variation: ±4 semitones max (±6.9% frequency)
- Glissando: ±6 semitones (build-up)
- Orgasm: pitch shift disabled to prevent distortion

**Proposed:**
Based on reference audio analysis:
```python
# In prosody_settings.py - Update defaults
DEFAULT_SETTINGS = {
    'pitch_variation': 15,  # Increased from 10 to 15%
    'pitch_variation_range_semitones': 6,  # Explicitly define max range

    # Low pitch for normal/quiet moments
    'low_pitch_offset': -8,  # Can go 8 semitones below base

    # High pitch for orgasm peaks
    'high_pitch_offset': +12,  # Can go 12 semitones above base

    # Glissando settings (already exist, may adjust)
    'glissando_start_semitones': -8,
    'glissando_end_semitones': +12,  # Increased from +8
}
```

**Implementation in audio_processor.py:**
```python
def process_for_state(self, audio, state, settings):
    # Extract new settings
    low_pitch = settings.get('low_pitch_offset', -8)
    high_pitch = settings.get('high_pitch_offset', +12)
    pitch_range = settings.get('pitch_variation_range_semitones', 6)

    if state == 'building':
        # Glissando from low to high
        audio = self.apply_glissando_formant_preserving(
            audio,
            start_semitones=low_pitch,
            end_semitones=high_pitch,
            curve='exponential'
        )

    elif state == 'orgasm':
        # Peak high pitches with formant preservation
        audio = self.apply_pitch_variation_formant_preserving(
            audio,
            randomize(pitch_var * 1.5, 0.2)
        )

    elif state == 'normal':
        # Subtle variations, occasional low dips
        if random.random() < 0.3:  # 30% chance of low moment
            pitch_target = random.uniform(low_pitch * 0.5, 0)
        else:
            pitch_target = random.uniform(-pitch_range, pitch_range)

        audio = self.apply_pitch_shift_formant_preserving(audio, pitch_target)
```

---

### **CHANGE 3: Extract & Use Reference Interjections**

**Goal:** Create phonetic templates based on actual audio patterns

**Implementation Steps:**

1. **Transcribe Reference Audio Segments**
   - Use speech recognition to extract phonetic patterns
   - Manual annotation of key interjections

2. **Create Interjection Database**
```python
# New file: interjection_templates.py
REFERENCE_INTERJECTIONS = {
    'normal': [
        'mmm...',
        'ahh...',
        'oh...',
        'mmm ahh...',
        # Extracted from reference 0:00-4:00
    ],

    'building': [
        'mmm... ahh...',
        'ohhh... ahhh...',
        'mmm... ohhh... AHHH...',
        'oh... OH... OHH...',
        # Extracted from reference buildups
    ],

    'orgasm': [
        'AHHH!',
        'OH! OH! OHHH!',
        'YES... YES!',
        'AHHH... mmm... AHHH!',
        # Extracted from reference orgasms
    ],

    'post_orgasm_breathing': [
        'hah... hah... hah...',
        'mmm... hah... hah...',
        'hh... hh... mmm...',
        # Extracted from post-orgasm moments
    ]
}
```

3. **Update vocalization_generator.py**
```python
from interjection_templates import REFERENCE_INTERJECTIONS

def _generate_streaming_with_templates(self, state, duration):
    # Use reference-based templates
    templates = REFERENCE_INTERJECTIONS.get(state, REFERENCE_INTERJECTIONS['normal'])
    selected = random.choice(templates)

    # Possibly combine multiple for longer durations
    if duration > 3.0:
        num_patterns = int(duration / 2)
        patterns = [random.choice(templates) for _ in range(num_patterns)]
        return " ... ".join(patterns)

    return selected
```

---

### **CHANGE 4: Random Breathing Between Segments**

**Current:** Breathing only in specific states
**Proposed:** Random breathing can occur anytime

**Reference Pattern:** ~0.52s average breath, range 0.14-2.74s

**Implementation:**
```python
# In audio_engine.py
class AudioEngine:
    def __init__(self, ...):
        # Add breathing control
        self.random_breathing_enabled = True
        self.breathing_frequency = 0.15  # 15% chance per clip
        self.last_breath_time = 0
        self.min_time_between_breaths = 5.0  # At least 5s between breaths

    def _xtts_streaming_loop(self):
        # ... existing code ...

        while self.is_playing:
            # ... generate audio_clip ...

            # RANDOMLY insert breathing
            if (self.random_breathing_enabled and
                random.random() < self.breathing_frequency and
                time.time() - self.last_breath_time > self.min_time_between_breaths):

                # Generate short breathing clip
                breath_phonetics = random.choice(['hh...', 'hah...', 'mmm...'])
                breath_audio = self.xtts_engine.generate_short_clip(
                    text=breath_phonetics,
                    language="en",
                    speed=1.2  # Slightly faster
                )

                # Apply breathiness effect
                breath_audio = self.audio_processor.apply_breathiness(
                    breath_audio,
                    randomize(50, 0.3)  # High breathiness
                )

                # Play breath before main clip
                self._play_audio_chunk(breath_audio)
                self.last_breath_time = time.time()

            # Play main audio clip
            self._play_audio_chunk(audio_clip)
```

---

### **CHANGE 5: Tempo Dynamics Based on Reference**

**Current:** Fixed tempo multipliers per state
**Proposed:** Variable tempo matching emotional intensity

```python
# In audio_processor.py - Enhanced tempo function
def apply_tempo_modulation_dynamic(self, audio, tempo_percent=100, state='normal', intensity=0.5):
    """
    Apply tempo with intensity awareness.

    Args:
        intensity: 0.0 (calm) to 1.0 (peak intensity)
    """
    audio = audio.astype(np.float64)

    if state == 'building':
        # Tempo increases with intensity during buildup
        tempo_mult = 1.0 + (intensity * 0.15)  # 1.0 to 1.15x
    elif state == 'orgasm':
        # Fast but not too extreme
        tempo_mult = 1.0 + (tempo_percent / 100.0) * 0.08  # Up to 1.08x
    elif state == 'post_orgasm_breathing':
        # Rapid breathing tempo
        tempo_mult = 1.15
    else:  # normal
        # Slight variations
        tempo_mult = tempo_percent / 100.0

    # Clamp to safe range
    tempo_mult = max(0.85, min(1.2, tempo_mult))

    stretched = librosa.effects.time_stretch(audio, rate=tempo_mult)
    return stretched.astype(np.float32)
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Analysis & Extraction (MANUAL)
- [ ] Listen to reference audio segments
- [ ] Manually transcribe key interjections/phonetics
- [ ] Note pitch characteristics (high moments, low moments)
- [ ] Document tempo changes during buildups
- [ ] Identify breathing placement patterns

### Phase 2: Infrastructure Changes
- [ ] Install pyrubberband library (`pip install pyrubberband`)
- [ ] Create `interjection_templates.py` with reference patterns
- [ ] Add formant-preserving pitch shift methods
- [ ] Update prosody_settings.py with wider range defaults

### Phase 3: Audio Processing Updates
- [ ] Implement `apply_pitch_shift_formant_preserving()`
- [ ] Implement `apply_glissando_formant_preserving()`
- [ ] Update `process_for_state()` to use new pitch methods
- [ ] Add dynamic tempo modulation with intensity

### Phase 4: Behavior Changes
- [ ] Add random breathing insertion logic
- [ ] Update vocalization_generator to use reference templates
- [ ] Add intensity tracking for dynamic tempo
- [ ] Test with various states

### Phase 5: Validation
- [ ] Record test clips for each state
- [ ] Analyze THD (should be <5%)
- [ ] Compare pitch range to reference audio
- [ ] Validate breathing sounds natural
- [ ] Get user approval

---

## ⚠️ RISKS & CONSIDERATIONS

### Dependencies
- **pyrubberband:** Requires system `rubberband-cli` binary
  - Windows: Download from https://breakfastquay.com/rubberband/
  - May need to add to PATH
  - Alternative: Use pyworld if unavailable

### Performance
- Formant-preserving pitch shift is slower than librosa
- May increase audio generation time by 20-30%
- Consider caching or pre-processing where possible

### Backwards Compatibility
- New settings in prosody_settings.py (with defaults)
- Existing recordings may sound different with new algorithms
- Consider keeping old pitch shift as fallback option

---

## 🔍 QUESTIONS FOR USER

1. **Formant Preservation:** Are you willing to install rubberband-cli for better quality? Or should I use alternative methods?

2. **Interjection Extraction:** Should I manually transcribe segments, or would you prefer to provide specific phonetic patterns?

3. **Breathing Frequency:** How often should random breathing occur? Current proposal: 15% chance per clip (roughly every 30-40 seconds)

4. **Pitch Range:** The reference likely has wider range than current ±6 semitones. Should I analyze exact values or use wider presets (±8-12 semitones)?

5. **Priority:** Which change is most important?
   - A) Wider pitch range
   - B) Reference interjections
   - C) Random breathing
   - D) Quality improvement (formant preservation)
   - E) Tempo dynamics

---

## 📊 EXPECTED RESULTS

### Before Changes:
- Pitch range: ±4-6 semitones
- Orgasm THD: 8-22%
- No random breathing
- Generic interjections

### After Changes:
- Pitch range: -8 to +12 semitones (wider emotional expression)
- Orgasm THD: <5% (with formant preservation)
- Natural random breathing throughout
- Reference-based realistic interjections
- Dynamic tempo matching intensity

---

**AWAITING USER DECISION:** Please review and approve/modify before implementation.

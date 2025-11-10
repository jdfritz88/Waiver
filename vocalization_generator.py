"""LLM-based phonetic vocalization generator for creating realistic moaning text."""
import random


class VocalizationGenerator:
    """Generates phonetic text for TTS that sounds like moaning."""

    def __init__(self, mistral_client=None):
        """Initialize the vocalization generator."""
        self.mistral_client = mistral_client

        # Phonetic building blocks for different intensities
        self.phonetics = {
            'soft': {
                'breaths': ['mmm', 'hh', 'ah', 'uh'],
                'moans': ['mmm ah', 'uh huh', 'oh', 'mm oh'],
                'gasps': ['ah', 'oh', 'uh'],
                'duration_mult': '...',  # Ellipsis adds slight pause
            },
            'moderate': {
                'breaths': ['hhh', 'ahh', 'uhh', 'ohh'],
                'moans': ['mmm ahh', 'oh yes', 'ah ah', 'ohh god'],
                'gasps': ['ahh!', 'ohh!', 'yes!'],
                'duration_mult': '...',
            },
            'intense': {
                'breaths': ['hhhh', 'ahhh', 'uhhh', 'ohhh'],
                'moans': ['ahhh yes', 'oh god yes', 'ahhh ahhh', 'yes yes yes'],
                'gasps': ['AHHH!', 'OHHH!', 'YES!', 'OH GOD!'],
                'duration_mult': '......',  # Longer pauses for emphasis
            },
        }

    def generate_vocalization_text(self, intensity='moderate', duration_seconds=5, progression='steady'):
        """
        Generate phonetic text that TTS will speak as moaning sounds.

        Args:
            intensity: 'soft', 'moderate', or 'intense'
            duration_seconds: Target duration for the vocalization
            progression: 'steady', 'building', or 'climaxing'

        Returns:
            str: Phonetically-crafted text for TTS
        """
        if self.mistral_client:
            # Use LLM to generate creative phonetic vocalizations
            return self._generate_with_llm(intensity, duration_seconds, progression)
        else:
            # Use template-based generation
            return self._generate_with_templates(intensity, duration_seconds, progression)

    def _generate_with_llm(self, intensity, duration_seconds, progression):
        """Generate phonetic text using Mistral AI."""
        try:
            prompt = f"""Generate phonetic text that represents {intensity} vocal moaning sounds for text-to-speech.

Requirements:
- Duration: approximately {duration_seconds} seconds when spoken
- Intensity: {intensity} (soft/moderate/intense)
- Progression: {progression} (steady/building/climaxing)

Use phonetic spellings that TTS can pronounce naturally:
- Breaths: "mmm", "hhh", "ahh", "uhh", "ohh"
- Extended sounds: "ahhhhh", "ohhhhh", "mmmmm" (repeat letters for duration)
- Gasps: "ah!", "oh!", "yes!"
- Pauses: use "..." or "," for breath breaks
- Emphasis: capitalize for volume (e.g., "AHHH")

Progression guidance:
- steady: maintain consistent intensity throughout
- building: start soft (mmm, ah) and gradually increase to louder (AHHH, YES)
- climaxing: peak intensity sounds (AHHH! OH GOD! YES YES YES!)

Respond with ONLY the phonetic text, no explanations.

Example outputs:
- Soft steady: "mmm... ahh... oh... mmm ahh... oh yes..."
- Building: "mmm... ahh... ohhh... ahhh... AHHH... YES!"
- Climaxing: "AHHH! OH GOD! YES! YES! AHHH! DON'T STOP!"

Generate {intensity} {progression} moaning sounds now:"""

            print("[Mistral API] Generating phonetic vocalizations...")
            response = self.mistral_client.client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}]
            )

            phonetic_text = response.choices[0].message.content.strip()
            print(f"[Mistral API] Generated: '{phonetic_text[:100]}...'")
            return phonetic_text

        except Exception as e:
            print(f"[Mistral API] Error: {e}, falling back to templates")
            return self._generate_with_templates(intensity, duration_seconds, progression)

    def _generate_with_templates(self, intensity, duration_seconds, progression):
        """Generate phonetic text using templates (fallback)."""
        phonetics = self.phonetics.get(intensity, self.phonetics['moderate'])

        # Estimate how many vocalizations fit in duration (roughly 1 per second)
        num_vocalizations = max(3, int(duration_seconds * 0.8))

        vocalizations = []

        for i in range(num_vocalizations):
            progress = i / max(1, num_vocalizations - 1)

            if progression == 'building':
                # Start soft, build to intense
                if progress < 0.3:
                    current_phonetics = self.phonetics['soft']
                elif progress < 0.7:
                    current_phonetics = self.phonetics['moderate']
                else:
                    current_phonetics = self.phonetics['intense']
            elif progression == 'climaxing':
                # Stay at peak intensity
                current_phonetics = self.phonetics['intense']
            else:  # steady
                current_phonetics = phonetics

            # Mix different types of sounds
            if i % 4 == 0:
                sound = random.choice(current_phonetics['breaths'])
            elif i % 4 == 3:
                sound = random.choice(current_phonetics['gasps'])
            else:
                sound = random.choice(current_phonetics['moans'])

            vocalizations.append(sound)

        # Join with appropriate pauses
        pause = phonetics['duration_mult']
        return f" {pause} ".join(vocalizations) + pause

    def generate_streaming_phonetics(self, current_state, duration=3.0):
        """
        Generate short phonetic clips for continuous streaming.

        Args:
            current_state: 'normal', 'building', or 'orgasm'
            duration: Target duration in seconds (3-5 seconds typical)

        Returns:
            str: Phonetic text optimized for streaming
        """
        if self.mistral_client:
            return self._generate_streaming_with_llm(current_state, duration)
        else:
            return self._generate_streaming_with_templates(current_state, duration)

    def _generate_streaming_with_llm(self, state, duration):
        """Generate streaming phonetics using Mistral AI."""
        try:
            prompt = f"""Generate phonetic moaning sounds for a {duration}-second audio clip.

State: {state}
- normal: Soft, baseline moaning (mmm, ahh, oh)
- building: Gradually increasing intensity (mmm, ahh, ohhh, AHHH)
- orgasm: Peak climax sounds (AHHH! OH GOD! YES! YES!)
- post_orgasm_breathing: Fast heavy breathing, exhausted (hah... hah... hah... mmm...)

Requirements:
- Use phonetic spellings TTS can pronounce
- Duration: approximately {duration} seconds when spoken
- Natural breathing pauses with "..."
- Capitalize for emphasis in building/orgasm states
- For post_orgasm_breathing: rapid short breaths with pauses

Respond with ONLY the phonetic text, no explanations.

Examples:
- Normal: "mmm... ahh... oh... mmm ahh..."
- Building: "mmm... ahh... ohhh... ahhh... AHHH..."
- Orgasm: "AHHH! YES! OH GOD! YES! AHHH!"
- Post-orgasm breathing: "hah... hah... hah... mmm... hah... hah..."

Generate {state} state sounds now:"""

            print(f"[Mistral API] Generating {state} streaming phonetics...")
            response = self.mistral_client.client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}]
            )

            phonetic_text = response.choices[0].message.content.strip()
            print(f"[Mistral API] Generated: '{phonetic_text[:80]}...'")
            return phonetic_text

        except Exception as e:
            print(f"[Mistral API] Error: {e}, falling back to templates")
            return self._generate_streaming_with_templates(state, duration)

    def _generate_streaming_with_templates(self, state, duration):
        """Generate streaming phonetics using templates (fallback)."""
        # Estimate number of vocalizations (roughly 0.5-1 second each)
        num_vocalizations = max(3, int(duration * 1.5))

        if state == 'normal':
            phonetics = self.phonetics['soft']
            patterns = [
                random.choice(phonetics['breaths']),
                random.choice(phonetics['moans']),
                random.choice(phonetics['breaths']),
                random.choice(phonetics['moans'])
            ]
        elif state == 'building':
            # Mix soft and moderate, trending toward intense
            soft = self.phonetics['soft']
            moderate = self.phonetics['moderate']
            intense = self.phonetics['intense']

            patterns = []
            for i in range(num_vocalizations):
                progress = i / max(1, num_vocalizations - 1)
                if progress < 0.3:
                    patterns.append(random.choice(soft['moans']))
                elif progress < 0.7:
                    patterns.append(random.choice(moderate['moans']))
                else:
                    patterns.append(random.choice(intense['moans']))

        elif state == 'orgasm':
            phonetics = self.phonetics['intense']
            patterns = [
                random.choice(phonetics['gasps']),
                random.choice(phonetics['moans']),
                random.choice(phonetics['gasps']),
                random.choice(phonetics['gasps'])
            ]

        elif state == 'post_orgasm_breathing':
            # Fast breathing sounds
            breathing_sounds = ['hah', 'hh', 'ah hah', 'mmm', 'hhh']
            patterns = [random.choice(breathing_sounds) for _ in range(num_vocalizations * 2)]

        else:  # fallback to normal
            phonetics = self.phonetics['soft']
            patterns = [
                random.choice(phonetics['breaths']),
                random.choice(phonetics['moans'])
            ]

        # Join with pauses
        return " ... ".join(patterns[:num_vocalizations]) + " ..."

    def create_ssml_formatted_text(self, phonetic_text, pitch_adjustment=0, rate=1.0):
        """
        Format phonetic text with SSML tags for better control.

        Note: XTTS doesn't support SSML natively, but this prepares text
        for potential post-processing or future SSML-compatible engines.

        Args:
            phonetic_text: The phonetic text to format
            pitch_adjustment: Pitch adjustment in semitones (-12 to +12)
            rate: Speech rate multiplier (0.5 to 2.0)

        Returns:
            str: Text formatted for better TTS output
        """
        # For now, XTTS doesn't support SSML, so just return cleaned text
        # But we can add special formatting that helps pronunciation

        formatted = phonetic_text

        # Add commas for natural pauses
        formatted = formatted.replace('...', ', ')

        # Ensure exclamation points have space before them for emphasis
        formatted = formatted.replace('!', ' !')

        return formatted

"""Mistral AI client for dynamic parameter generation."""
import random
from mistralai import Mistral
from config import MISTRAL_API_KEY, MISTRAL_MODEL


class MistralClient:
    """Client for interacting with Mistral AI API."""

    def __init__(self):
        """Initialize the Mistral client."""
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        self.client = Mistral(api_key=MISTRAL_API_KEY)

    def generate_intensity_parameters(self, current_state="normal"):
        """
        Generate audio parameters based on current state.

        Args:
            current_state: One of "normal", "building", "climax"

        Returns:
            dict: Parameters including volume_mult, pitch_shift, speed_mult
        """
        prompt = f"""Generate audio modulation parameters for a {current_state} state.
        Respond with ONLY a JSON object with these keys:
        - volume_mult: float between 0.8 and 1.5
        - pitch_shift: float between -2 and 2 (semitones)
        - speed_mult: float between 0.9 and 1.1

        For 'building' state, increase all values gradually.
        For 'climax' state, maximize all values.
        For 'normal' state, keep values moderate with natural variation.

        Respond ONLY with valid JSON, no other text."""

        try:
            response = self.client.chat.complete(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract JSON from response
            content = response.choices[0].message.content
            # Try to parse JSON
            import json
            # Find JSON object in response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                params = json.loads(json_str)
                return params
            else:
                # Fallback to default values
                return self._get_default_parameters(current_state)

        except Exception as e:
            print(f"Mistral API error: {e}")
            return self._get_default_parameters(current_state)

    def _get_default_parameters(self, current_state):
        """Generate default parameters if API fails."""
        if current_state == "climax":
            return {
                "volume_mult": 1.5,
                "pitch_shift": 2.0,
                "speed_mult": 1.1
            }
        elif current_state == "building":
            return {
                "volume_mult": 1.2,
                "pitch_shift": 1.0,
                "speed_mult": 1.05
            }
        else:  # normal
            return {
                "volume_mult": 1.0,
                "pitch_shift": random.uniform(-0.5, 0.5),
                "speed_mult": 1.0
            }

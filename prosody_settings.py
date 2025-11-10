"""Prosody control settings with JSON persistence."""
import json
from pathlib import Path


class ProsodySettings:
    """
    Manages prosody control settings with JSON persistence.

    Each control has a default percentage value that determines
    how much random variation is applied.
    """

    DEFAULT_SETTINGS = {
        # Pitch Variation (0-100%)
        # Controls how much the pitch randomly varies
        'pitch_variation': 10,

        # Tempo/Speaking Rate (50-150%)
        # 100 = normal speed, <100 = slower, >100 = faster
        'tempo': 100,

        # Volume/Intensity (0-200%)
        # 100 = normal volume, higher = louder
        'volume': 100,

        # Pause/Timing (0-100%)
        # Controls frequency and duration of breath pauses
        'pauses': 30,

        # Breathiness (0-100%)
        # Adds airy, breathy quality to voice
        'breathiness': 20,

        # Roughness (0-100%)
        # Adds raspy, rough texture to voice
        # REDUCED from 10 to 0 to prevent harmonic distortion/buzz
        'roughness': 0,

        # Emphasis/Stress (0-100%)
        # Controls syllable emphasis variation
        'emphasis': 30,

        # Build-up duration (seconds)
        'buildup_duration_min': 5,
        'buildup_duration_max': 15,

        # Orgasm duration (seconds)
        'orgasm_duration_min': 15,
        'orgasm_duration_max': 30,

        # Glissando settings for build-up
        # Updated based on reference audio analysis (518881__the_power_of_sound__slowly-making-love.wav)
        'glissando_start_semitones': -13,  # Start pitch (negative = lower) - was -8
        'glissando_end_semitones': 19,      # End pitch (positive = higher) - was 8
        'glissando_curve': 'exponential',   # 'linear', 'exponential', 'logarithmic'

        # Random breathing frequency (0-100%)
        # Controls how often random breaths occur between audio clips
        # Updated based on reference audio analysis: breathing occurs every ~31.6s
        'breathing_frequency': 3,  # 3% chance per clip - was 15%
    }

    def __init__(self, settings_file='prosody_settings.json'):
        """Initialize settings manager."""
        self.settings_file = Path(settings_file)
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.load()

    def load(self):
        """Load settings from JSON file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    # Update settings with loaded values
                    self.settings.update(loaded)
                    print(f"Loaded prosody settings from {self.settings_file}")
            except Exception as e:
                print(f"Error loading settings: {e}, using defaults")
        else:
            # Create default settings file
            self.save()

    def save(self):
        """Save current settings to JSON file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
                print(f"Saved prosody settings to {self.settings_file}")
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get(self, key, default=None):
        """Get a setting value."""
        return self.settings.get(key, default)

    def set(self, key, value):
        """Set a setting value."""
        self.settings[key] = value

    def update(self, updates):
        """Update multiple settings at once."""
        self.settings.update(updates)

    def reset_to_defaults(self):
        """Reset all settings to defaults."""
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.save()

    def get_all(self):
        """Get all current settings."""
        return self.settings.copy()

    def validate_value(self, key, value):
        """
        Validate that a value is within acceptable range for its control.

        Args:
            key: Setting key
            value: Proposed value

        Returns:
            Validated and clamped value
        """
        # Define ranges for each control
        ranges = {
            'pitch_variation': (0, 100),
            'tempo': (50, 200),
            'volume': (0, 200),
            'pauses': (0, 100),
            'breathiness': (0, 100),
            'roughness': (0, 100),
            'emphasis': (0, 100),
            'buildup_duration_min': (1, 60),
            'buildup_duration_max': (1, 60),
            'orgasm_duration_min': (1, 60),
            'orgasm_duration_max': (1, 60),
            'glissando_start_semitones': (-24, 24),
            'glissando_end_semitones': (-24, 24),
        }

        if key in ranges:
            min_val, max_val = ranges[key]
            return max(min_val, min(max_val, value))

        return value

    def set_validated(self, key, value):
        """Set a value with validation."""
        validated_value = self.validate_value(key, value)
        self.set(key, validated_value)
        return validated_value

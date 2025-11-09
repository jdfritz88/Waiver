"""Configuration and constants for the audio streaming application."""
import os
from dotenv import load_dotenv

load_dotenv()

# Mistral API Configuration
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
MISTRAL_MODEL = "mistral-large-latest"

# Audio Configuration
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
CHANNELS = 2

# Control Ranges
VOLUME_RANGE = (0.0, 2.0)  # 0% to 200%
PITCH_RANGE = (-12, 12)  # -12 to +12 semitones
OCTAVE_RANGE = (-2, 2)  # -2 to +2 octaves

# Trigger Configuration
MIN_BUILD_UP_DURATION = 3  # seconds
MAX_BUILD_UP_DURATION = 10  # seconds
CLIMAX_DURATION = 2  # seconds
MIN_TIME_BETWEEN_EVENTS = 15  # seconds
MAX_TIME_BETWEEN_EVENTS = 60  # seconds

# UI Configuration
WINDOW_TITLE = "Audio Stream Generator"
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

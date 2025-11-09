"""Main entry point for the Audio Stream Generator application."""
import sys
from audio_engine import AudioEngine
from mistral_client import MistralClient
from ui import AudioStreamUI
from config import MISTRAL_API_KEY


def main():
    """Initialize and run the application."""
    print("Starting Audio Stream Generator...")

    # Initialize Mistral client (optional)
    mistral_client = None
    if MISTRAL_API_KEY:
        try:
            mistral_client = MistralClient()
            print("Mistral AI client initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize Mistral client: {e}")
            print("Continuing without AI-powered parameter generation")
    else:
        print("Warning: MISTRAL_API_KEY not found in .env file")
        print("Continuing without AI-powered parameter generation")

    # Initialize audio engine
    audio_engine = AudioEngine(mistral_client=mistral_client)

    # Create and run UI
    app = AudioStreamUI(audio_engine)
    print("Application ready!")
    app.run()


if __name__ == "__main__":
    main()

"""Generate sample audio files for testing."""
import numpy as np
import soundfile as sf


def generate_sample_wav(filename="sample_audio.wav", duration=5, sample_rate=44100):
    """
    Generate a simple test audio file.

    Args:
        filename: Output filename
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create a simple tone with some variation
    # Base frequency around 200-400 Hz (typical vocal range)
    frequency = 300 + 50 * np.sin(2 * np.pi * 0.5 * t)

    # Generate sine wave
    audio = np.sin(2 * np.pi * frequency * t)

    # Add some harmonics for richer sound
    audio += 0.5 * np.sin(2 * np.pi * 2 * frequency * t)
    audio += 0.3 * np.sin(2 * np.pi * 3 * frequency * t)

    # Add amplitude envelope (fade in/out)
    envelope = np.ones_like(t)
    fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    audio = audio * envelope

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    # Save as WAV file
    sf.write(filename, audio, sample_rate)
    print(f"Generated sample audio file: {filename}")


if __name__ == "__main__":
    generate_sample_wav("sample_audio.wav", duration=5)
    print("Sample audio file created successfully!")

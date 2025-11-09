# Vocal Sound Generator

A real-time vocal sound synthesis application with AI-powered dynamics using Mistral LLM and formant synthesis.

## Features

- **Real-time vocal synthesis** using formant synthesis (no audio files needed)
- **Continuous moaning sounds** with natural variation in pitch, breathiness, and intensity
- **AI-powered dynamics** with Mistral integration for intelligent parameter modulation
- **Occasional spoken words** using text-to-speech (contextually appropriate)
- **Interactive controls** with sliders for volume, pitch, and octave
- **Live waveform visualization** of generated audio
- **Automatic events**: Random build-ups and climaxes
- **Manual triggers**: On-demand build-up and climax controls

## How It Works

The application generates vocal sounds in real-time using:

1. **Formant Synthesis**: Models the human vocal tract using resonant frequencies to create different vowel sounds (ah, oh, uh, etc.)
2. **Additive Synthesis**: Combines multiple sine waves with harmonics for rich, realistic vocal timbre
3. **Breathiness**: Adds filtered noise for natural breathy quality
4. **Dynamic Modulation**: Continuously varies pitch, intensity, vibrato, and vowel shapes
5. **Mistral AI**: Generates intelligent parameter adjustments and contextual words/phrases
6. **Text-to-Speech**: Occasionally speaks appropriate words using pyttsx3

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Mistral API key (optional but recommended):
```
MISTRAL_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python main.py
```

## Usage

1. **Start**: Click "Start Stream" to begin real-time audio generation
2. **Adjust Controls**: Use sliders to control:
   - **Volume**: Overall loudness (0-200%)
   - **Pitch**: Base pitch shift (-12 to +12 semitones)
   - **Octave**: Octave shift (-2 to +2 octaves)
3. **Manual Triggers**:
   - **Trigger Build-up**: Gradually increases intensity
   - **Trigger Climax**: Immediate peak intensity
4. **Automatic Events**: The app randomly triggers build-ups and climaxes every 15-60 seconds
5. **Spoken Words**: Occasionally speaks contextual words/phrases (every 20-45 seconds)

## Technical Details

### Vocal Synthesis Parameters

- **Base Frequency**: 180-350 Hz (typical vocal range)
- **Formants**: F1 and F2 frequencies shape vowel sounds
- **Vibrato**: Natural pitch modulation (5-7 Hz)
- **Breathiness**: Controlled noise addition (0-50%)
- **Vowel Transitions**: Smooth morphing between different vowel sounds

### States

- **Normal**: Baseline moaning with natural variation
- **Building**: Gradually increasing pitch, intensity, and breathiness
- **Climax**: Peak intensity with higher pitch and strong vibrato

## Dependencies

- **numpy**: Numerical operations
- **scipy**: Signal processing and filtering
- **pyaudio**: Real-time audio playback
- **mistralai**: AI-powered parameter generation
- **pyttsx3**: Text-to-speech for spoken words
- **matplotlib**: Waveform visualization
- **python-dotenv**: Environment variable management

## Notes

- The application works without Mistral API key but uses fallback parameters
- TTS functionality requires system speech engines (SAPI5 on Windows, etc.)
- All audio is generated in real-time - no pre-recorded files needed

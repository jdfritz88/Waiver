# Vocal Sound Generator

A hybrid audio streaming application combining WAV file playback OR real-time vocal synthesis with AI-powered dynamics using Mistral LLM.

## Features

- **Dual Mode Operation**:
  - **File Mode**: Load and play your own WAV files with dynamic modulation
  - **Synthesis Mode**: Generate vocal sounds from scratch using formant synthesis
- **Your Voice**: Use your favorite WAV file - the app will enhance it with dynamic effects
- **Continuous sound** with natural variation in pitch, breathiness, and intensity
- **AI-powered dynamics** with Mistral integration for intelligent parameter modulation
- **Controllable spoken words** using text-to-speech with frequency slider (0-100%)
- **Interactive controls** with sliders for volume, pitch, octave, and word frequency
- **Live waveform visualization** of audio
- **Automatic events**: Random build-ups and climaxes
- **Manual triggers**: On-demand build-up and climax controls

## How It Works

The application supports two modes:

### File Mode (Recommended - Use Your Own Voice!)
1. **Load WAV File**: Select your favorite moaning audio WAV file
2. **Dynamic Effects**: The app applies real-time modulation:
   - State-based pitch shifting (higher during climax)
   - Intensity variations (louder during build-ups)
   - Breathiness enhancement during intense states
3. **Continuous Loop**: Your audio file loops seamlessly
4. **Layered Words**: TTS words are layered on top of your audio

### Synthesis Mode
If no file is loaded, the app generates sounds using:

1. **Formant Synthesis**: Models the human vocal tract using resonant frequencies to create different vowel sounds (ah, oh, uh, etc.)
2. **Additive Synthesis**: Combines multiple sine waves with harmonics for rich, realistic vocal timbre
3. **Breathiness**: Adds filtered noise for natural breathy quality
4. **Dynamic Modulation**: Continuously varies pitch, intensity, vibrato, and vowel shapes

### Common Features
- **Mistral AI**: Generates intelligent parameter adjustments and contextual words/phrases
- **Text-to-Speech**: Speaks appropriate words using pyttsx3 (controllable frequency)

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

1. **Choose Mode**:
   - Click **"Select WAV File"** to load your own audio (recommended!)
   - OR leave empty to use synthesis mode
   - Click **"Clear (Use Synthesis)"** to switch back to synthesis

2. **Start**: Click "Start Stream" to begin playback/generation

3. **Adjust Controls**: Use sliders to control:
   - **Volume**: Overall loudness (0-200%)
   - **Pitch**: Base pitch shift (-12 to +12 semitones)
   - **Octave**: Octave shift (-2 to +2 octaves)
   - **Words**: Frequency of spoken words (0% = off, 100% = very frequent)

4. **Manual Triggers**:
   - **Trigger Build-up**: Gradually increases intensity
   - **Trigger Climax**: Immediate peak intensity

5. **Automatic Events**: The app randomly triggers build-ups and climaxes every 15-60 seconds

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

- **Best Experience**: Use your own WAV file for the most realistic voice!
- The application works without Mistral API key but uses fallback parameters
- TTS functionality requires system speech engines (SAPI5 on Windows, etc.)
- Both file playback and synthesis modes support all controls and triggers
- Word frequency slider lets you control how often TTS words are spoken (0% = silent)

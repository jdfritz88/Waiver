# Audio Stream Generator

An audio streaming application with AI-powered dynamic control using Mistral LLM.

## Features

- Load and play WAV audio files
- Real-time audio manipulation (volume, pitch, octave)
- Waveform visualization
- Random and manual trigger controls for build-ups and climaxes
- Mistral AI integration for dynamic parameter generation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your Mistral API key:
```
MISTRAL_API_KEY=your_api_key_here
```

3. Run the application:
```bash
python main.py
```

## Usage

1. Click "Select Audio File" to choose a WAV file
2. Adjust sliders for volume, pitch, and octave
3. Click "Start Stream" to begin playback
4. Use "Trigger Build-up" and "Trigger Climax" for manual control
5. The app will automatically trigger random events during playback

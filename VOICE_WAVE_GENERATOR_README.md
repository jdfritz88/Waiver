# Voice Wave Generator

A Python application that processes wave audio files to generate human sounds with realistic variations and emotional dynamics.

## Features

The application applies 7 audio specifications to transform input audio:

1. **Pitch Variation** - Randomly adjusts pitch ±10% to simulate natural vocal fluctuations
2. **Breath Timing** - Inserts variable-length pauses (0.5-2 seconds) between phrases to mimic real breathing
3. **Volume Dynamics** - Gradually increases volume during climactic moments, then drops sharply for relaxation
4. **Stutter & Repetition** - Occasionally repeats short phrases ("Oh-oh-oh") or inserts stutters for realism
5. **Moan Length** - Stretches vowels randomly (1-3 seconds long)
6. **Whisper & Gasps** - Lowers volume for whispered lines and adds sharp, sudden gasps
7. **Speed Fluctuation** - Speeds up speech during intense moments, then slows down afterward

## Requirements

- Python 3.6 or higher
- NumPy library

## Installation

1. Ensure Python 3 is installed:
```bash
python3 --version
```

2. Install NumPy if not already installed:
```bash
pip install numpy
```

## Usage

### Running the Application

```bash
python3 voice_wave_generator.py
```

Or make it executable and run directly:
```bash
chmod +x voice_wave_generator.py
./voice_wave_generator.py
```

### Using the GUI

1. **Select Input File**
   - Click the "Browse" button
   - Select a .wav file from your system
   - The file name will appear in the interface

2. **Generate Audio**
   - Click the "Generate Audio" button
   - Wait for processing to complete (status updates will show progress)
   - Choose where to save the output file
   - The generated audio will be saved with all effects applied

### Processing Time

Processing time depends on:
- Input file length (longer files take more time)
- Audio sample rate (higher rates take more time)
- System performance

Typical processing times:
- 10-second file: ~5-10 seconds
- 30-second file: ~15-30 seconds
- 60-second file: ~30-60 seconds

## Technical Details

### Audio Processing Pipeline

The application processes audio in the following order:

1. **Pitch Variation**: Segments audio into 300ms chunks and randomly resamples each segment
2. **Breath Pauses**: Detects phrase boundaries and inserts silence
3. **Volume Dynamics**: Creates volume envelopes for crescendo/decrescendo effects
4. **Stutters**: Randomly repeats 500ms segments 2-3 times
5. **Moan Stretching**: Detects sustained sounds and stretches them 1.5-3x
6. **Whispers/Gasps**: Reduces volume for sections and adds sharp amplitude spikes
7. **Speed Fluctuation**: Adjusts playback speed based on audio variance

### Supported Formats

- **Input**: WAV files (8-bit, 16-bit, or 32-bit)
- **Output**: WAV files (same format as input)

### Algorithm Details

- **Pitch shifting** is achieved through sample rate manipulation
- **Time stretching** uses interpolation to maintain audio quality
- **Volume envelopes** are applied using NumPy array multiplication
- **Randomization** ensures each generation is unique

## Troubleshooting

### "No module named 'numpy'" Error

Install NumPy:
```bash
pip install numpy
```

### "Permission denied" Error

Make the script executable:
```bash
chmod +x voice_wave_generator.py
```

### "Error reading wave file"

Ensure the input file is a valid WAV file:
- Check file extension (.wav)
- Try opening the file in an audio player first
- Some compressed WAV formats may not be supported

### Output audio sounds distorted

This is often due to:
- Very short input files (less than 5 seconds)
- Input files with very low quality
- Multiple random effects compounding

Try:
- Using longer input files
- Using higher quality source audio
- Reducing the number of effects (modify the code)

## Customization

To adjust the effects, edit the following parameters in `voice_wave_generator.py`:

- **Pitch variation range**: Modify `random.uniform(0.9, 1.1)` in `apply_pitch_variation()`
- **Breath pause duration**: Change `random.uniform(0.5, 2.0)` in `add_breath_pauses()`
- **Stutter probability**: Adjust `random.random() < 0.15` in `add_stutters()`
- **Stretch factor**: Modify `random.uniform(1.5, 3.0)` in `stretch_moans()`

## Notes

- The application creates unique output each time due to randomization
- Processing is done in memory, so very large files may consume significant RAM
- Original input files are never modified
- All effects are applied automatically; individual effects cannot be toggled in the GUI

## License

This software is provided as-is for personal use.

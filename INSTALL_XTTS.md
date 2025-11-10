# XTTS Installation Instructions

## Your XTTS Model
Your existing XTTS v2.0.3 model is located at:
```
F:\Apps\freedom_system\freedom_system_2000\text-generation-webui\extensions\alltalk_tts\models\xtts\xttsv2_2.0.3
```

The code has been configured to use this existing model - **no need to download it again!**

## Install TTS Library

You just need to install the TTS Python library to use the model:

### Option 1: Direct Install (Recommended)
```bash
pip install TTS
```

### Option 2: Install with Requirements
```bash
pip install -r requirements.txt
```

## Verify Installation

After installing, test that it works:

```bash
python -c "from TTS.tts.configs.xtts_config import XttsConfig; print('TTS library installed successfully!')"
```

## What's Configured

The app has been updated to:
1. ✅ Load XTTS v2 directly from your local installation
2. ✅ Use the low-level XTTS API for better control
3. ✅ Cache speaker latents for faster generation
4. ✅ Support GPU acceleration if available
5. ✅ Handle graceful fallback if TTS not installed

## Performance Optimizations

- **GPU**: If you have a CUDA-compatible GPU, it will be used automatically
- **Speaker Latent Caching**: Voice characteristics are computed once and reused
- **24kHz Output**: XTTS generates at 24kHz, automatically resampled to 44.1kHz

## Next Steps

1. Install TTS library: `pip install TTS`
2. Run the app: `python main.py`
3. Load a voice file for cloning
4. Click "Start Streaming"
5. Enjoy continuous moaning audio with voice cloning!

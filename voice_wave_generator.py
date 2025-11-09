#!/usr/bin/env python3
"""
Voice Wave File Generator
Creates human sounds with realistic variations using audio manipulation
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import wave
import struct
import random
import os
from pathlib import Path

class VoiceWaveGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Wave Generator")
        self.root.geometry("600x400")

        self.selected_file = None
        self.output_file = None

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(main_frame, text="Voice Wave Generator",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # File selection
        ttk.Label(main_frame, text="Select Input Wave File:").grid(row=1, column=0, sticky=tk.W, pady=5)

        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.file_label = ttk.Label(file_frame, text="No file selected",
                                    relief=tk.SUNKEN, width=50)
        self.file_label.grid(row=0, column=0, padx=(0, 5))

        select_btn = ttk.Button(file_frame, text="Browse", command=self.select_file)
        select_btn.grid(row=0, column=1)

        # Audio specifications display
        specs_label = ttk.Label(main_frame, text="Audio Specifications:",
                                font=("Arial", 10, "bold"))
        specs_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))

        specs_text = tk.Text(main_frame, height=10, width=70, wrap=tk.WORD)
        specs_text.grid(row=4, column=0, columnspan=2, pady=5)

        specs_content = """1. Pitch Variation - Randomly adjust pitch ±10%
2. Breath Timing - Variable pauses (0.5-2 seconds) between phrases
3. Volume Dynamics - Crescendo during climactic moments, drop for relaxation
4. Stutter & Repetition - Repeat short phrases ("Oh-oh-oh") or stutters
5. Moan Length - Stretch vowels randomly (1-3 seconds)
6. Whisper & Gasps - Lower volume for whispers, sharp gasps
7. Speed Fluctuation - Speed up during intense moments, slow down after"""

        specs_text.insert("1.0", specs_content)
        specs_text.config(state=tk.DISABLED)

        # Generate button
        self.generate_btn = ttk.Button(main_frame, text="Generate Audio",
                                       command=self.generate_audio, state=tk.DISABLED)
        self.generate_btn.grid(row=5, column=0, columnspan=2, pady=20)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready",
                                      foreground="blue")
        self.status_label.grid(row=6, column=0, columnspan=2)

    def select_file(self):
        """Open file dialog to select a wave file"""
        filename = filedialog.askopenfilename(
            title="Select a Wave File",
            filetypes=(("Wave files", "*.wav"), ("All files", "*.*"))
        )

        if filename:
            self.selected_file = filename
            self.file_label.config(text=os.path.basename(filename))
            self.generate_btn.config(state=tk.NORMAL)
            self.status_label.config(text=f"Selected: {os.path.basename(filename)}",
                                    foreground="green")

    def read_wave_file(self, filename):
        """Read wave file and return audio data and parameters"""
        try:
            with wave.open(filename, 'rb') as wav_file:
                params = wav_file.getparams()
                frames = wav_file.readframes(params.nframes)

                # Convert bytes to numpy array
                if params.sampwidth == 1:
                    dtype = np.uint8
                elif params.sampwidth == 2:
                    dtype = np.int16
                else:
                    dtype = np.int32

                audio_data = np.frombuffer(frames, dtype=dtype)

                # Convert to float for processing
                audio_data = audio_data.astype(np.float32)

                return audio_data, params
        except Exception as e:
            raise Exception(f"Error reading wave file: {str(e)}")

    def write_wave_file(self, filename, audio_data, params):
        """Write processed audio data to wave file"""
        try:
            # Convert back to original dtype
            if params.sampwidth == 1:
                dtype = np.uint8
            elif params.sampwidth == 2:
                dtype = np.int16
            else:
                dtype = np.int32

            # Clip and convert
            audio_data = np.clip(audio_data, np.iinfo(dtype).min, np.iinfo(dtype).max)
            audio_data = audio_data.astype(dtype)

            with wave.open(filename, 'wb') as wav_file:
                wav_file.setparams(params)
                wav_file.writeframes(audio_data.tobytes())

        except Exception as e:
            raise Exception(f"Error writing wave file: {str(e)}")

    def apply_pitch_variation(self, audio_data, sample_rate):
        """Apply random pitch variation ±10%"""
        # Split audio into segments for pitch variation
        segment_length = int(sample_rate * 0.3)  # 300ms segments
        result = []

        for i in range(0, len(audio_data), segment_length):
            segment = audio_data[i:i+segment_length]
            if len(segment) == 0:
                continue

            # Random pitch shift ±10%
            pitch_factor = random.uniform(0.9, 1.1)

            # Resample to change pitch
            indices = np.arange(0, len(segment), pitch_factor)
            indices = indices[indices < len(segment)].astype(int)

            if len(indices) > 0:
                pitched_segment = segment[indices]
                result.extend(pitched_segment)

        return np.array(result, dtype=np.float32)

    def add_breath_pauses(self, audio_data, sample_rate):
        """Insert variable-length pauses (0.5-2 seconds) to simulate breathing"""
        # Detect phrase boundaries (silence or low amplitude)
        threshold = np.max(np.abs(audio_data)) * 0.1

        # Split into phrases
        phrases = []
        current_phrase = []
        silence_count = 0
        min_silence = int(sample_rate * 0.1)  # 100ms minimum silence to detect phrase end

        for sample in audio_data:
            if abs(sample) < threshold:
                silence_count += 1
            else:
                if silence_count > min_silence and len(current_phrase) > 0:
                    phrases.append(np.array(current_phrase))
                    current_phrase = []
                silence_count = 0

            current_phrase.append(sample)

        if current_phrase:
            phrases.append(np.array(current_phrase))

        # Reconstruct with breath pauses
        result = []
        for i, phrase in enumerate(phrases):
            result.extend(phrase)

            if i < len(phrases) - 1:
                # Add breath pause
                pause_duration = random.uniform(0.5, 2.0)
                pause_samples = int(sample_rate * pause_duration)
                result.extend([0] * pause_samples)

        return np.array(result, dtype=np.float32)

    def apply_volume_dynamics(self, audio_data, sample_rate):
        """Apply volume crescendo and drops"""
        # Create volume envelope
        length = len(audio_data)
        envelope = np.ones(length)

        # Find segments for crescendo
        segment_length = int(sample_rate * 2)  # 2-second segments
        num_segments = length // segment_length

        for i in range(num_segments):
            start = i * segment_length
            end = min(start + segment_length, length)

            # Random decision for crescendo
            if random.random() < 0.3:  # 30% chance of crescendo
                # Crescendo
                crescendo = np.linspace(0.7, 1.3, end - start)
                envelope[start:end] = crescendo
            elif random.random() < 0.2:  # 20% chance of drop
                # Sudden drop (post-climax)
                drop = np.linspace(1.0, 0.5, end - start)
                envelope[start:end] = drop

        return audio_data * envelope

    def add_stutters(self, audio_data, sample_rate):
        """Add occasional stutters and repetitions"""
        # Split into segments
        segment_length = int(sample_rate * 0.5)  # 500ms segments
        result = []

        i = 0
        while i < len(audio_data):
            segment = audio_data[i:i+segment_length]

            # Random chance to stutter
            if random.random() < 0.15 and len(segment) > 0:  # 15% chance
                # Repeat segment 2-3 times
                repeats = random.randint(2, 3)
                for _ in range(repeats):
                    result.extend(segment)
                    # Small gap between repeats
                    result.extend([0] * int(sample_rate * 0.05))
            else:
                result.extend(segment)

            i += segment_length

        return np.array(result, dtype=np.float32)

    def stretch_moans(self, audio_data, sample_rate):
        """Randomly stretch vowel sounds (1-3 seconds)"""
        # Detect sustained sounds (relatively constant amplitude)
        window_size = int(sample_rate * 0.1)  # 100ms window
        result = []

        i = 0
        while i < len(audio_data):
            window = audio_data[i:i+window_size]

            if len(window) == 0:
                break

            # Check if this is a sustained sound
            variance = np.var(window)
            avg_amplitude = np.mean(np.abs(window))

            # Low variance and high amplitude = sustained sound
            if variance < avg_amplitude * 0.3 and avg_amplitude > np.max(np.abs(audio_data)) * 0.3:
                # This might be a vowel/moan, stretch it
                if random.random() < 0.3:  # 30% chance
                    stretch_factor = random.uniform(1.5, 3.0)
                    stretched_length = int(len(window) * stretch_factor)

                    # Interpolate to stretch
                    indices = np.linspace(0, len(window)-1, stretched_length)
                    stretched = np.interp(indices, np.arange(len(window)), window)
                    result.extend(stretched)
                    i += window_size
                    continue

            result.extend(window)
            i += window_size

        return np.array(result, dtype=np.float32)

    def add_whispers_gasps(self, audio_data, sample_rate):
        """Add whisper effects (lower volume) and sharp gasps"""
        result = audio_data.copy()

        # Randomly add whisper sections (reduce volume)
        segment_length = int(sample_rate * 1.0)  # 1-second segments

        for i in range(0, len(result), segment_length):
            segment_end = min(i + segment_length, len(result))

            # 20% chance for whisper
            if random.random() < 0.2:
                result[i:segment_end] *= 0.3

            # 10% chance for gasp (sharp, sudden increase)
            if random.random() < 0.1:
                gasp_length = int(sample_rate * 0.2)  # 200ms gasp
                gasp_end = min(i + gasp_length, len(result))

                # Create gasp envelope (quick rise and fall)
                gasp_env = np.concatenate([
                    np.linspace(1.0, 2.0, gasp_length // 2),
                    np.linspace(2.0, 1.0, gasp_length - gasp_length // 2)
                ])

                actual_length = min(len(gasp_env), gasp_end - i)
                result[i:i+actual_length] *= gasp_env[:actual_length]

        return result

    def apply_speed_fluctuation(self, audio_data, sample_rate):
        """Vary playback speed - faster during intense moments, slower after"""
        segment_length = int(sample_rate * 1.5)  # 1.5-second segments
        result = []

        for i in range(0, len(audio_data), segment_length):
            segment = audio_data[i:i+segment_length]

            if len(segment) == 0:
                continue

            # Determine if this is an "intense" moment (higher amplitude variance)
            variance = np.var(segment)
            avg_variance = np.var(audio_data)

            if variance > avg_variance * 1.2:
                # Intense moment - speed up
                speed_factor = random.uniform(1.1, 1.3)
            else:
                # Calm moment - slow down slightly
                speed_factor = random.uniform(0.9, 1.0)

            # Resample to change speed
            indices = np.arange(0, len(segment), speed_factor)
            indices = indices[indices < len(segment)].astype(int)

            if len(indices) > 0:
                resampled = segment[indices]
                result.extend(resampled)

        return np.array(result, dtype=np.float32)

    def generate_audio(self):
        """Main function to process audio with all effects"""
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a wave file first")
            return

        try:
            self.status_label.config(text="Processing audio...", foreground="orange")
            self.root.update()

            # Read input file
            audio_data, params = self.read_wave_file(self.selected_file)
            sample_rate = params.framerate

            # Apply all effects in sequence
            self.status_label.config(text="Applying pitch variation...")
            self.root.update()
            audio_data = self.apply_pitch_variation(audio_data, sample_rate)

            self.status_label.config(text="Adding breath pauses...")
            self.root.update()
            audio_data = self.add_breath_pauses(audio_data, sample_rate)

            self.status_label.config(text="Applying volume dynamics...")
            self.root.update()
            audio_data = self.apply_volume_dynamics(audio_data, sample_rate)

            self.status_label.config(text="Adding stutters and repetitions...")
            self.root.update()
            audio_data = self.add_stutters(audio_data, sample_rate)

            self.status_label.config(text="Stretching moans...")
            self.root.update()
            audio_data = self.stretch_moans(audio_data, sample_rate)

            self.status_label.config(text="Adding whispers and gasps...")
            self.root.update()
            audio_data = self.add_whispers_gasps(audio_data, sample_rate)

            self.status_label.config(text="Applying speed fluctuation...")
            self.root.update()
            audio_data = self.apply_speed_fluctuation(audio_data, sample_rate)

            # Ask for output location
            output_file = filedialog.asksaveasfilename(
                title="Save Generated Audio",
                defaultextension=".wav",
                filetypes=(("Wave files", "*.wav"), ("All files", "*.*")),
                initialfile="generated_voice.wav"
            )

            if output_file:
                self.status_label.config(text="Saving audio file...")
                self.root.update()

                # Update params with new frame count
                params = wave._wave_params(
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                    len(audio_data),
                    params.comptype,
                    params.compname
                )

                self.write_wave_file(output_file, audio_data, params)

                self.status_label.config(
                    text=f"Success! Generated: {os.path.basename(output_file)}",
                    foreground="green"
                )

                messagebox.showinfo("Success",
                    f"Audio generated successfully!\n\nSaved to:\n{output_file}")
            else:
                self.status_label.config(text="Save cancelled", foreground="blue")

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", foreground="red")
            messagebox.showerror("Error", f"Failed to generate audio:\n\n{str(e)}")


def main():
    root = tk.Tk()
    app = VoiceWaveGenerator(root)
    root.mainloop()


if __name__ == "__main__":
    main()

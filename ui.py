"""User interface for the audio streaming application."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from config import (
    WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT,
    VOLUME_RANGE, PITCH_RANGE, OCTAVE_RANGE
)


class AudioStreamUI:
    """Main UI for the audio stream generator."""

    def __init__(self, audio_engine):
        """Initialize the UI."""
        self.audio_engine = audio_engine
        self.root = tk.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(True, True)

        self.current_file = None
        self.is_streaming = False

        self._create_ui()

    def _create_ui(self):
        """Create all UI elements."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for responsive layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Vocal Sound Generator",
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10))

        # File selection section
        file_frame = ttk.LabelFrame(main_frame, text="Audio Source", padding="10")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)

        self.file_label = ttk.Label(file_frame, text="No file loaded - using synthesis mode")
        self.file_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        select_btn = ttk.Button(
            button_frame,
            text="Select WAV File",
            command=self._select_file
        )
        select_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))

        self.clear_btn = ttk.Button(
            button_frame,
            text="Clear (Use Synthesis)",
            command=self._clear_file,
            state=tk.DISABLED
        )
        self.clear_btn.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))

        # Waveform visualization
        waveform_frame = ttk.LabelFrame(main_frame, text="Waveform", padding="10")
        waveform_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(2, weight=1)

        self._create_waveform_plot(waveform_frame)

        # Audio controls section
        controls_frame = ttk.LabelFrame(main_frame, text="Audio Controls", padding="10")
        controls_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)

        # Volume slider
        ttk.Label(controls_frame, text="Volume:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.volume_slider = ttk.Scale(
            controls_frame,
            from_=VOLUME_RANGE[0],
            to=VOLUME_RANGE[1],
            orient=tk.HORIZONTAL,
            command=self._on_volume_change
        )
        self.volume_slider.set(1.0)
        self.volume_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        self.volume_value = ttk.Label(controls_frame, text="100%")
        self.volume_value.grid(row=0, column=2, padx=(10, 0), pady=5)

        # Pitch slider
        ttk.Label(controls_frame, text="Pitch:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.pitch_slider = ttk.Scale(
            controls_frame,
            from_=PITCH_RANGE[0],
            to=PITCH_RANGE[1],
            orient=tk.HORIZONTAL,
            command=self._on_pitch_change
        )
        self.pitch_slider.set(0.0)
        self.pitch_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        self.pitch_value = ttk.Label(controls_frame, text="0 st")
        self.pitch_value.grid(row=1, column=2, padx=(10, 0), pady=5)

        # Octave slider
        ttk.Label(controls_frame, text="Octave:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.octave_slider = ttk.Scale(
            controls_frame,
            from_=OCTAVE_RANGE[0],
            to=OCTAVE_RANGE[1],
            orient=tk.HORIZONTAL,
            command=self._on_octave_change
        )
        self.octave_slider.set(0)
        self.octave_slider.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        self.octave_value = ttk.Label(controls_frame, text="0")
        self.octave_value.grid(row=2, column=2, padx=(10, 0), pady=5)

        # Word frequency slider
        ttk.Label(controls_frame, text="Words:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.word_freq_slider = ttk.Scale(
            controls_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            command=self._on_word_freq_change
        )
        self.word_freq_slider.set(0.3)
        self.word_freq_slider.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        self.word_freq_value = ttk.Label(controls_frame, text="30%")
        self.word_freq_value.grid(row=3, column=2, padx=(10, 0), pady=5)

        # Playback controls
        playback_frame = ttk.LabelFrame(main_frame, text="Playback", padding="10")
        playback_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        playback_frame.columnconfigure(0, weight=1)
        playback_frame.columnconfigure(1, weight=1)

        self.start_btn = ttk.Button(
            playback_frame,
            text="Start Stream",
            command=self._toggle_stream
        )
        self.start_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))

        # Manual triggers
        trigger_frame = ttk.LabelFrame(main_frame, text="Manual Triggers", padding="10")
        trigger_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        trigger_frame.columnconfigure(0, weight=1)
        trigger_frame.columnconfigure(1, weight=1)

        self.buildup_btn = ttk.Button(
            trigger_frame,
            text="Trigger Build-up",
            command=self._trigger_buildup,
            state=tk.DISABLED
        )
        self.buildup_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))

        self.climax_btn = ttk.Button(
            trigger_frame,
            text="Trigger Climax",
            command=self._trigger_climax,
            state=tk.DISABLED
        )
        self.climax_btn.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))

        # Status bar
        self.status_label = ttk.Label(
            main_frame,
            text="Ready - Select a WAV file or use synthesis mode",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=6, column=0, sticky=(tk.W, tk.E))

    def _create_waveform_plot(self, parent):
        """Create the matplotlib waveform visualization."""
        # Create figure and axis
        self.fig = Figure(figsize=(8, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Audio Waveform')
        self.ax.grid(True, alpha=0.3)

        # Initialize with empty plot
        self.waveform_line, = self.ax.plot([], [], linewidth=0.5)
        self.ax.set_ylim(-1, 1)

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _update_waveform(self):
        """Update the waveform visualization."""
        waveform_data = self.audio_engine.get_waveform_data()
        if len(waveform_data) > 0:
            x = np.linspace(0, len(waveform_data), len(waveform_data))
            self.waveform_line.set_data(x, waveform_data)
            self.ax.set_xlim(0, len(waveform_data))

            # Auto-scale y-axis
            if np.max(np.abs(waveform_data)) > 0:
                max_val = np.max(np.abs(waveform_data))
                self.ax.set_ylim(-max_val * 1.1, max_val * 1.1)

            self.canvas.draw()

    def _select_file(self):
        """Handle file selection."""
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if file_path:
            if self.audio_engine.load_audio_file(file_path):
                self.current_file = file_path
                filename = file_path.split('/')[-1].split('\\')[-1]  # Handle both / and \
                self.file_label.config(text=f"Loaded: {filename}")
                self._update_waveform()
                self.clear_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Audio file loaded - ready to stream")
            else:
                messagebox.showerror("Error", "Failed to load audio file")
                self.status_label.config(text="Error loading file")

    def _clear_file(self):
        """Clear the loaded file and switch to synthesis mode."""
        self.audio_engine.unload_audio_file()
        self.current_file = None
        self.file_label.config(text="No file loaded - using synthesis mode")
        self.clear_btn.config(state=tk.DISABLED)
        self._update_waveform()
        self.status_label.config(text="Switched to synthesis mode")

    def _start_waveform_updates(self):
        """Start updating the waveform display periodically."""
        if self.is_streaming:
            self._update_waveform()
            # Schedule next update
            self.root.after(100, self._start_waveform_updates)  # Update every 100ms

    def _on_volume_change(self, value):
        """Handle volume slider change."""
        volume = float(value)
        self.audio_engine.volume = volume
        self.volume_value.config(text=f"{int(volume * 100)}%")

    def _on_pitch_change(self, value):
        """Handle pitch slider change."""
        pitch = float(value)
        self.audio_engine.pitch_shift = pitch
        self.pitch_value.config(text=f"{pitch:.1f} st")

    def _on_octave_change(self, value):
        """Handle octave slider change."""
        octave = int(float(value))
        self.audio_engine.octave_shift = octave
        self.octave_value.config(text=f"{octave:+d}")

    def _on_word_freq_change(self, value):
        """Handle word frequency slider change."""
        freq = float(value)
        self.audio_engine.word_frequency = freq
        if freq < 0.01:
            self.word_freq_value.config(text="Off")
        else:
            self.word_freq_value.config(text=f"{int(freq * 100)}%")

    def _toggle_stream(self):
        """Toggle streaming on/off."""
        if not self.is_streaming:
            if self.audio_engine.start_playback():
                self.is_streaming = True
                self.start_btn.config(text="Stop Stream")
                self.buildup_btn.config(state=tk.NORMAL)
                self.climax_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Generating audio stream...")
                self._start_waveform_updates()  # Start updating waveform
            else:
                messagebox.showerror("Error", "Failed to start audio generation")
        else:
            self.audio_engine.stop_playback()
            self.is_streaming = False
            self.start_btn.config(text="Start Stream")
            self.buildup_btn.config(state=tk.DISABLED)
            self.climax_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Stopped")

    def _trigger_buildup(self):
        """Trigger manual build-up."""
        self.audio_engine.trigger_build_up_manual()
        self.status_label.config(text="Build-up triggered")

    def _trigger_climax(self):
        """Trigger manual climax."""
        self.audio_engine.trigger_climax_manual()
        self.status_label.config(text="Climax triggered")

    def run(self):
        """Start the UI main loop."""
        try:
            self.root.mainloop()
        finally:
            self.audio_engine.cleanup()

"""User interface for the audio streaming application."""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import time
from threading import Thread
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
        # Configure root grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create canvas with scrollbar
        canvas = tk.Canvas(self.root, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)

        # Scrollable frame inside canvas
        scrollable_frame = ttk.Frame(canvas, padding="10")

        # Configure canvas scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Grid layout for canvas and scrollbar
        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Main content frame (was main_frame, now scrollable_frame)
        main_frame = scrollable_frame
        main_frame.columnconfigure(0, weight=1)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Vocal Sound Generator",
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10))

        # Voice Profile Selection
        file_frame = ttk.LabelFrame(main_frame, text="Voice Profile", padding="10")
        file_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)

        info_text = "Load a WAV file to analyze the voice characteristics.\nThe app will synthesize new sounds using that voice."
        info_label = ttk.Label(file_frame, text=info_text, font=("Arial", 8), foreground="gray")
        info_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.file_label = ttk.Label(file_frame, text="No voice loaded - using default voice")
        self.file_label.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        button_frame = ttk.Frame(file_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        select_btn = ttk.Button(
            button_frame,
            text="Load Voice from WAV",
            command=self._select_file
        )
        select_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))

        self.clear_btn = ttk.Button(
            button_frame,
            text="Use Default Voice",
            command=self._clear_file,
            state=tk.DISABLED
        )
        self.clear_btn.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))

        # Current State Indicator
        state_frame = ttk.LabelFrame(main_frame, text="Current State", padding="10")
        state_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        self.state_label = ttk.Label(
            state_frame,
            text="NORMAL",
            font=("Arial", 16, "bold"),
            foreground="green"
        )
        self.state_label.pack()

        # Status bar (moved above waveform)
        self.status_label = ttk.Label(
            main_frame,
            text="Ready - Load a voice file to start streaming",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # Waveform visualization
        waveform_frame = ttk.LabelFrame(main_frame, text="Waveform", padding="10")
        waveform_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(4, weight=1)

        self._create_waveform_plot(waveform_frame)

        # Audio controls section (simplified - just volume)
        controls_frame = ttk.LabelFrame(main_frame, text="Volume Control", padding="10")
        controls_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        controls_frame.columnconfigure(1, weight=1)

        # Volume slider
        ttk.Label(controls_frame, text="Volume:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.volume_slider = ttk.Scale(
            controls_frame,
            from_=VOLUME_RANGE[0],
            to=VOLUME_RANGE[1],
            orient=tk.HORIZONTAL
        )
        self.volume_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        self.volume_value = ttk.Label(controls_frame, text="100%")
        self.volume_value.grid(row=0, column=2, padx=(10, 0), pady=5)

        # Configure slider and set initial value
        self.volume_slider.config(command=self._on_volume_change)
        self.volume_slider.set(1.0)

        # Prosody Controls section
        prosody_frame = ttk.LabelFrame(main_frame, text="Prosody Controls (Emotional Expression)", padding="10")
        prosody_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        prosody_frame.columnconfigure(1, weight=1)

        # Create sliders for all 7 prosody controls
        self.prosody_sliders = {}
        self.prosody_labels = {}
        self.prosody_waveforms = {}

        prosody_controls = [
            ('pitch_variation', 'Pitch Variation', 0, 100, '%'),
            ('tempo', 'Tempo/Speed', 50, 200, '%'),
            ('pauses', 'Breath Pauses', 0, 100, '%'),
            ('breathiness', 'Breathiness', 0, 100, '%'),
            ('roughness', 'Roughness', 0, 100, '%'),
            ('emphasis', 'Emphasis/Stress', 0, 100, '%'),
        ]

        # Each control takes 2 rows: one for slider, one for mini waveform
        for control_idx, (key, label, min_val, max_val, unit) in enumerate(prosody_controls):
            row_idx = control_idx * 2  # Each control uses 2 rows

            # Label
            ttk.Label(prosody_frame, text=f"{label}:").grid(row=row_idx, column=0, sticky=tk.W, pady=(3, 0))

            # Slider
            slider = ttk.Scale(
                prosody_frame,
                from_=min_val,
                to=max_val,
                orient=tk.HORIZONTAL,
                command=lambda val, k=key: self._on_prosody_change(k, val)
            )
            slider.grid(row=row_idx, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=(3, 0))

            # Value label
            value_label = ttk.Label(prosody_frame, text=f"{min_val}{unit}")
            value_label.grid(row=row_idx, column=2, padx=(10, 0), pady=(3, 0))

            # Store references
            self.prosody_sliders[key] = slider
            self.prosody_labels[key] = value_label

            # Set initial value from settings
            initial_value = self.audio_engine.prosody_settings.get(key, min_val)
            slider.set(initial_value)
            value_label.config(text=f"{int(initial_value)}{unit}")

            # Mini waveform visualization beneath slider
            waveform_fig = Figure(figsize=(4, 0.5), dpi=80)
            waveform_ax = waveform_fig.add_subplot(111)
            waveform_ax.set_ylim(-1, 1)
            waveform_ax.set_xlim(0, 100)
            waveform_ax.axis('off')  # Hide axes for cleaner look
            waveform_fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Initialize with flat line
            waveform_line, = waveform_ax.plot([], [], linewidth=1.0, color='blue')

            # Embed in tkinter
            waveform_canvas = FigureCanvasTkAgg(waveform_fig, master=prosody_frame)
            waveform_canvas.draw()
            waveform_canvas.get_tk_widget().grid(
                row=row_idx + 1,
                column=1,
                sticky=(tk.W, tk.E),
                padx=(10, 0),
                pady=(0, 5)
            )

            # Store references for updates
            self.prosody_waveforms[key] = {
                'fig': waveform_fig,
                'ax': waveform_ax,
                'line': waveform_line,
                'canvas': waveform_canvas
            }

        # Add reset and save buttons (after all sliders and waveforms)
        button_row = len(prosody_controls) * 2  # Each control uses 2 rows
        reset_btn = ttk.Button(
            prosody_frame,
            text="Reset to Defaults",
            command=self._reset_prosody_settings
        )
        reset_btn.grid(row=button_row, column=0, pady=(10, 0), sticky=tk.W)

        save_btn = ttk.Button(
            prosody_frame,
            text="Save Settings",
            command=self._save_prosody_settings
        )
        save_btn.grid(row=button_row, column=1, pady=(10, 0), padx=(10, 0), sticky=tk.W)

        # Start updating prosody waveforms
        self._update_prosody_waveforms()

        # Pitch Processing Method section
        pitch_method_frame = ttk.LabelFrame(main_frame, text="Pitch Processing Method", padding="10")
        pitch_method_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        pitch_method_frame.columnconfigure(0, weight=1)

        info_text = "Select the pitch processing algorithm (can be changed in real-time during streaming)"
        info_label = ttk.Label(pitch_method_frame, text=info_text, font=("Arial", 8), foreground="gray")
        info_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        self.pitch_method_var = tk.StringVar(value='hybrid')

        pitch_methods = [
            ('hybrid', 'Hybrid (Recommended) - No pitch shift during orgasm, prevents distortion'),
            ('rubberband', 'Rubberband - Formant-preserving (requires pyrubberband + binary)'),
            ('pyworld', 'WORLD Vocoder - Professional quality resynthesis (requires pyworld)'),
            ('lower_then_shift', 'Lower-First Method - Generate lower, shift upward')
        ]

        for idx, (value, label) in enumerate(pitch_methods):
            rb = ttk.Radiobutton(
                pitch_method_frame,
                text=label,
                variable=self.pitch_method_var,
                value=value,
                command=self._update_pitch_method
            )
            rb.grid(row=idx + 1, column=0, sticky=tk.W, pady=2)

        # Breathing Frequency section
        breathing_frame = ttk.LabelFrame(main_frame, text="Random Breathing Control", padding="10")
        breathing_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        breathing_frame.columnconfigure(1, weight=1)

        info_text = "Control how often random breathing occurs between audio clips (0% = never, 100% = always)"
        info_label = ttk.Label(breathing_frame, text=info_text, font=("Arial", 8), foreground="gray")
        info_label.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Label(breathing_frame, text="Breathing Frequency:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.breathing_slider = ttk.Scale(
            breathing_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self._on_breathing_change
        )
        self.breathing_slider.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        self.breathing_value = ttk.Label(breathing_frame, text="3%")
        self.breathing_value.grid(row=1, column=2, padx=(10, 0), pady=5)

        # Set initial value (updated from 15% to 3% based on reference audio analysis)
        initial_breathing = self.audio_engine.prosody_settings.get('breathing_frequency', 3)
        self.breathing_slider.set(initial_breathing)
        self.breathing_value.config(text=f"{int(initial_breathing)}%")

        # Streaming control
        streaming_frame = ttk.LabelFrame(main_frame, text="Streaming Control", padding="10")
        streaming_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        streaming_frame.columnconfigure(0, weight=1)

        self.start_btn = ttk.Button(
            streaming_frame,
            text="▶ Start Streaming",
            command=self._toggle_stream
        )
        self.start_btn.pack(fill=tk.X, pady=5)

        # Manual triggers
        trigger_frame = ttk.LabelFrame(main_frame, text="Manual State Triggers", padding="10")
        trigger_frame.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        trigger_frame.columnconfigure(0, weight=1)
        trigger_frame.columnconfigure(1, weight=1)
        trigger_frame.columnconfigure(2, weight=1)

        self.buildup_btn = ttk.Button(
            trigger_frame,
            text="📈 Trigger Build-up",
            command=self._trigger_buildup,
            state=tk.DISABLED
        )
        self.buildup_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))

        self.orgasm_btn = ttk.Button(
            trigger_frame,
            text="💥 Trigger Orgasm",
            command=self._trigger_orgasm,
            state=tk.DISABLED
        )
        self.orgasm_btn.grid(row=0, column=1, padx=(5, 5), sticky=(tk.W, tk.E))

        self.record_btn = ttk.Button(
            trigger_frame,
            text="⏺ Record 20 Sec",
            command=self._record_audio,
            state=tk.DISABLED
        )
        self.record_btn.grid(row=0, column=2, padx=(5, 0), sticky=(tk.W, tk.E))

        # Start state update timer
        self._update_state_indicator()

        # Auto-load last voice file after UI is initialized
        self.root.after(100, self._auto_load_voice_file)

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

    def _auto_load_voice_file(self):
        """Automatically load the last voice file if it exists."""
        success, file_path = self.audio_engine.auto_load_last_voice_file()
        if success and file_path:
            # Update status
            file_name = file_path.split('/')[-1].split('\\')[-1]
            self.status_label.config(text=f"✓ Auto-loaded voice file: {file_name}")
            self.file_label.config(text=f"Loaded: {file_name}")
            print(f"Auto-loaded voice file: {file_path}")

    def _select_file(self):
        """Handle voice file selection and analysis."""
        file_path = filedialog.askopenfilename(
            title="Select Voice Sample (WAV File)",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )

        if file_path:
            self.status_label.config(text="Analyzing voice... please wait")
            self.root.update()  # Force UI update

            if self.audio_engine.load_audio_file(file_path):
                self.current_file = file_path
                filename = file_path.split('/')[-1].split('\\')[-1]  # Handle both / and \
                self.file_label.config(text=f"Voice Profile: {filename}")
                self.clear_btn.config(state=tk.NORMAL)
                self.status_label.config(text="Voice analyzed - ready to generate!")
                messagebox.showinfo(
                    "Voice Loaded",
                    f"Voice profile loaded from {filename}\n\n"
                    "The app will now synthesize sounds using this voice's characteristics."
                )
            else:
                messagebox.showerror("Error", "Failed to analyze voice from file")
                self.status_label.config(text="Error analyzing voice")

    def _clear_file(self):
        """Clear the voice profile and use default voice."""
        self.audio_engine.unload_audio_file()
        self.current_file = None
        self.file_label.config(text="No voice loaded - using default voice")
        self.clear_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Using default voice synthesis")

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

    def _on_prosody_change(self, key, value):
        """Handle prosody slider change."""
        float_value = float(value)
        # Update the engine's prosody settings
        self.audio_engine.prosody_settings.set_validated(key, float_value)
        # Update the label
        self.prosody_labels[key].config(text=f"{int(float_value)}%")

    def _update_pitch_method(self):
        """Handle pitch method radio button change."""
        method = self.pitch_method_var.get()
        self.audio_engine.audio_processor.set_pitch_method(method)
        self.status_label.config(text=f"✓ Pitch method changed to: {method}")
        print(f"Pitch processing method changed to: {method}")

    def _on_breathing_change(self, value):
        """Handle breathing frequency slider change."""
        int_value = int(float(value))
        self.audio_engine.prosody_settings.set('breathing_frequency', int_value)
        self.breathing_value.config(text=f"{int_value}%")
        print(f"Breathing frequency changed to: {int_value}%")

    def _save_prosody_settings(self):
        """Save current prosody settings to JSON."""
        self.audio_engine.prosody_settings.save()
        self.status_label.config(text="✓ Prosody settings saved!")

    def _reset_prosody_settings(self):
        """Reset prosody settings to defaults."""
        self.audio_engine.prosody_settings.reset_to_defaults()
        # Update UI sliders
        for key, slider in self.prosody_sliders.items():
            value = self.audio_engine.prosody_settings.get(key)
            slider.set(value)
            self.prosody_labels[key].config(text=f"{int(value)}%")
        self.status_label.config(text="✓ Prosody settings reset to defaults")

    def _update_prosody_waveforms(self):
        """Update mini waveform visualizations for prosody controls."""
        if not hasattr(self, 'prosody_waveforms'):
            return

        # Generate example waveforms based on current settings
        x = np.linspace(0, 100, 200)

        for key, waveform_data in self.prosody_waveforms.items():
            value = self.audio_engine.prosody_settings.get(key, 0)

            # Generate waveform based on control type
            if key == 'pitch_variation':
                # Show pitch variation as sine wave with varying amplitude
                amplitude = value / 100.0
                y = amplitude * np.sin(x / 5) * np.sin(x / 2)
            elif key == 'tempo':
                # Show tempo as frequency of oscillation
                freq = (value / 100.0) * 3  # 50-200% maps to different frequencies
                y = 0.5 * np.sin(x * freq / 5)
            elif key == 'pauses':
                # Show pauses as gaps in waveform
                y = np.sin(x / 5)
                if value > 30:
                    # Add gaps
                    y[40:50] = 0
                    y[80:90] = 0
            elif key == 'breathiness':
                # Show breathiness as noisy signal
                noise_amount = value / 100.0
                y = 0.5 * np.sin(x / 5) + noise_amount * np.random.normal(0, 0.2, len(x))
            elif key == 'roughness':
                # Show roughness as distorted wave
                rough_amount = value / 100.0
                y = np.tanh(2 * np.sin(x / 5) * (1 + rough_amount))
            elif key == 'emphasis':
                # Show emphasis as amplitude spikes
                y = np.sin(x / 5)
                emphasis_amount = value / 100.0
                y[50:60] *= (1 + emphasis_amount)
            else:
                y = np.sin(x / 5)

            # Update the plot
            waveform_data['line'].set_data(x, y)
            waveform_data['canvas'].draw()

        # Schedule next update
        self.root.after(500, self._update_prosody_waveforms)

    def _update_state_indicator(self):
        """Update the state indicator label."""
        if self.is_streaming:
            current_state = self.audio_engine.current_state.upper()
            self.state_label.config(text=current_state)

            # Update color based on state
            if current_state == "NORMAL":
                self.state_label.config(foreground="green")
            elif current_state == "BUILDING":
                self.state_label.config(foreground="orange")
            elif current_state == "ORGASM":
                self.state_label.config(foreground="red")

        # Schedule next update
        self.root.after(200, self._update_state_indicator)

    def _toggle_stream(self):
        """Toggle streaming on/off."""
        if not self.is_streaming:
            # Check if voice file is loaded
            if not self.audio_engine.xtts_engine.is_available():
                messagebox.showerror(
                    "No Voice Loaded",
                    "Please load a voice file first for XTTS voice cloning.\n\n"
                    "Click 'Load Voice from WAV' and select a voice sample."
                )
                return

            if self.audio_engine.start_playback():
                self.is_streaming = True
                self.start_btn.config(text="⏹ Stop Streaming")
                self.buildup_btn.config(state=tk.NORMAL)
                self.orgasm_btn.config(state=tk.NORMAL)
                self.record_btn.config(state=tk.NORMAL)
                self.status_label.config(text="🔴 Streaming with XTTS voice cloning...")
                self._start_waveform_updates()  # Start updating waveform
            else:
                messagebox.showerror("Error", "Failed to start XTTS streaming")
        else:
            self.audio_engine.stop_playback()
            self.is_streaming = False
            self.start_btn.config(text="▶ Start Streaming")
            self.buildup_btn.config(state=tk.DISABLED)
            self.orgasm_btn.config(state=tk.DISABLED)
            self.record_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Stopped")
            self.state_label.config(text="STOPPED", foreground="gray")

    def _trigger_buildup(self):
        """Trigger manual build-up."""
        self.audio_engine.trigger_build_up_manual()
        self.status_label.config(text="📈 Build-up triggered manually")

    def _trigger_orgasm(self):
        """Trigger manual orgasm."""
        self.audio_engine.trigger_orgasm_manual()
        self.status_label.config(text="💥 Orgasm triggered manually")

    def _record_audio(self):
        """Record 20 seconds of audio."""
        if self.audio_engine.start_recording():
            self.record_btn.config(state=tk.DISABLED, text="Recording...")
            self.status_label.config(text="Recording 20 seconds...")

            # Re-enable button after recording completes
            def enable_button():
                time.sleep(self.audio_engine.recording_duration + 0.5)
                self.root.after(0, lambda: self._recording_complete())

            Thread(target=enable_button, daemon=True).start()

    def _recording_complete(self):
        """Called when recording is complete."""
        self.record_btn.config(state=tk.NORMAL, text="⏺ Record 20 Sec")
        self.status_label.config(text="✓ Recording saved!")
        messagebox.showinfo("Recording Saved", "20 second audio clip saved to file!")

    def run(self):
        """Start the UI main loop."""
        try:
            self.root.mainloop()
        finally:
            self.audio_engine.cleanup()

"""Application state persistence for voice file and settings."""
import json
from pathlib import Path


class AppState:
    """
    Manages persistent application state across sessions.

    Saves and loads:
    - Last loaded voice file path
    - Window position/size (future)
    - Recent files list (future)
    """

    DEFAULT_STATE = {
        'last_voice_file': None,  # Path to last loaded voice file
    }

    def __init__(self, state_file='app_state.json'):
        """Initialize state manager."""
        self.state_file = Path(state_file)
        self.state = self.DEFAULT_STATE.copy()
        self.load()

    def load(self):
        """Load state from JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    loaded = json.load(f)
                    # Update state with loaded values
                    self.state.update(loaded)
                    print(f"Loaded app state from {self.state_file}")
            except Exception as e:
                print(f"Error loading app state: {e}, using defaults")
        else:
            # Create default state file
            self.save()

    def save(self):
        """Save current state to JSON file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=4)
                print(f"Saved app state to {self.state_file}")
        except Exception as e:
            print(f"Error saving app state: {e}")

    def get(self, key, default=None):
        """Get a state value."""
        return self.state.get(key, default)

    def set(self, key, value):
        """Set a state value."""
        self.state[key] = value

    def update(self, updates):
        """Update multiple state values at once."""
        self.state.update(updates)

    def get_last_voice_file(self):
        """Get the last loaded voice file path."""
        return self.get('last_voice_file')

    def set_last_voice_file(self, file_path):
        """Set the last loaded voice file path and save immediately."""
        self.set('last_voice_file', str(file_path) if file_path else None)
        self.save()

    def has_last_voice_file(self):
        """Check if there's a last voice file saved."""
        last_file = self.get_last_voice_file()
        if last_file:
            # Verify file still exists
            return Path(last_file).exists()
        return False

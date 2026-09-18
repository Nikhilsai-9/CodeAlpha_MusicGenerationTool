"""Genre/mood presets for the music generation app.

Each preset defines:
  - A short primer (MIDI pitch list) representing a characteristic motif.
  - A suggested temperature (randomness).
  - A description string for the UI.
"""

# ---------------------------------------------------------------------------
# Genre presets (primer is in MIDI note numbers; middle C = 60)
# ---------------------------------------------------------------------------
PRESETS = {
    "Twinkle (C major)": {
        "pitches": [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60],
        "temperature": 0.8,
        "description": "Classic nursery melody in C major - safe starting point.",
    },
    "Ode to Joy": {
        "pitches": [64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 64, 62, 62],
        "temperature": 0.9,
        "description": "Beethoven's famous melody - recognisable and uplifting.",
    },
    "Jingle Bells": {
        "pitches": [60, 60, 60, 60, 60, 60, 60, 62, 55, 60, 64, 57, 60, 60, 60],
        "temperature": 1.0,
        "description": "Festive Christmas motif - works for upbeat variations.",
    },
    "Chromatic Run (A)": {
        "pitches": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
        "temperature": 1.2,
        "description": "Ascending chromatic scale - exploratory and modern.",
    },
    "Blues Riff (A)": {
        "pitches": [57, 60, 62, 65, 64, 62, 60, 57, 60, 62, 64, 65, 67, 65],
        "temperature": 1.0,
        "description": "A minor pentatonic-ish blues feel in A.",
    },
    "Middle-C Drone": {
        "pitches": [60, 60, 60, 60],
        "temperature": 1.2,
        "description": "Just middle C repeated - lets the AI improvise freely.",
    },
    "Custom (enter below)": {
        "pitches": [],
        "temperature": 1.0,
        "description": "Enter your own sequence of MIDI notes (0-127) below.",
    },
}


def get_preset_names() -> list[str]:
    """Return the list of preset names (stable order)."""
    return list(PRESETS.keys())


def get_preset(name: str) -> dict:
    """Return the preset dict, or raise KeyError if unknown."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {get_preset_names()}")
    return PRESETS[name]


# ---------------------------------------------------------------------------
# Note-name helpers (for the UI)
# ---------------------------------------------------------------------------
NOTE_NAMES_SHARP = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


def midi_to_name(pitch: int) -> str:
    """Convert MIDI pitch (e.g. 60) to a note name (e.g. 'C4')."""
    if not 0 <= pitch <= 127:
        return f"?{pitch}"
    name = NOTE_NAMES_SHARP[pitch % 12]
    octave = (pitch // 12) - 1
    return f"{name}{octave}"


def pitches_to_names(pitches: list[int]) -> list[str]:
    """Convert a list of MIDI pitches to note names."""
    return [midi_to_name(p) for p in pitches]

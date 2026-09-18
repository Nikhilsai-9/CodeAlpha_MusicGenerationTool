"""Audio synthesis and file output for generated NoteSequences.

Wraps pretty_midi (synthesis) and soundfile (WAV export) so callers can
turn a Magenta ``NoteSequence`` into a playable WAV file with one call.

Public API:
    sequence_to_wav(note_sequence, out_path, fs=44100, instrument=0) -> Path
    sequence_to_midi(note_sequence, out_path) -> Path
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pretty_midi

log = logging.getLogger("audio")


def sequence_to_midi(note_sequence, out_path) -> Path:
    """Write a NoteSequence to a MIDI file. Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    for note in note_sequence.notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity) if note.velocity else 80,
                pitch=int(note.pitch),
                start=float(note.start_time),
                end=float(note.end_time),
            )
        )
    midi_data.instruments.append(instrument)
    midi_data.write(str(out_path))
    log.info("Wrote MIDI: %s (%d notes)", out_path.name, len(instrument.notes))
    return out_path


def sequence_to_wav(
    note_sequence,
    out_path,
    fs: int = 44100,
) -> Path:
    """Synthesize a NoteSequence to a WAV file. Returns the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
    for note in note_sequence.notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity) if note.velocity else 80,
                pitch=int(note.pitch),
                start=float(note.start_time),
                end=float(note.end_time),
            )
        )
    midi_data.instruments.append(instrument)
    audio = midi_data.synthesize(fs=fs)
    # Clamp and convert to 16-bit PCM (WAV standard)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    import soundfile as sf
    sf.write(str(out_path), pcm, fs, subtype="PCM_16")
    log.info(
        "Wrote WAV: %s (%.2fs, %d samples)",
        out_path.name,
        len(pcm) / fs,
        len(pcm),
    )
    return out_path

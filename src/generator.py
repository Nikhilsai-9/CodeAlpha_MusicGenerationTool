"""Magenta music generation wrapper.

Thin, robust wrapper around Magenta's MelodyRNN sequence generator that
handles bundle loading, primer (seed) construction, options building, and
generation in a single call. Uses TensorFlow 2.x with TF1 compatibility.

Public API:
    MelodyGenerator(model_name: str = "attention_rnn")
        .load(bundle_path: Path) -> None
        .generate(seed_pitches, length_steps, temperature, qpm) -> NoteSequence
    BundleNotFoundError
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List

# Suppress noisy TF logs (only show ERROR+)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Force TF1 compatibility for Magenta (which uses TF1-style graphs)
import tensorflow.compat.v1 as tf  # noqa: E402

tf.disable_v2_behavior()

from note_seq.protobuf import generator_pb2, music_pb2  # noqa: E402
from magenta.models.melody_rnn import (  # noqa: E402
    melody_rnn_sequence_generator,
)
from magenta.models.shared import (  # noqa: E402
    sequence_generator_bundle,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("generator")


class BundleNotFoundError(FileNotFoundError):
    """Raised when the .mag bundle file is missing."""


class MelodyGenerator:
    """Wrap Magenta's MelodyRNN to produce melodic sequences."""

    AVAILABLE_MODELS = (
        "basic_rnn",
        "mono_rnn",
        "lookback_rnn",
        "attention_rnn",
    )


    def __init__(self, model_name: str = "attention_rnn") -> None:
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from "
                f"{self.AVAILABLE_MODELS}."
            )
        self.model_name: str = model_name
        self._bundle = None
        self._generator = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, bundle_path) -> None:
        """Load the .mag bundle from disk and initialize the generator."""
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise BundleNotFoundError(
                f"Bundle file not found: {bundle_path}. "
                f"Place the .mag file in the 'bundles/' directory."
            )
        if bundle_path.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Bundle file '{bundle_path}' is suspiciously small "
                f"({bundle_path.stat().st_size} bytes). "
                f"Re-download from the official Magenta mirror."
            )

        log.info("Loading bundle: %s", bundle_path.name)
        try:
            self._bundle = sequence_generator_bundle.read_bundle_file(
                str(bundle_path)
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to parse bundle: {exc}") from exc

        try:
            generator_map = melody_rnn_sequence_generator.get_generator_map()
            self._generator = generator_map[self.model_name](
                checkpoint=None, bundle=self._bundle
            )
            self._generator.initialize()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize {self.model_name} generator: {exc}"
            ) from exc

        log.info("Generator '%s' ready.", self.model_name)

    @property
    def is_loaded(self) -> bool:
        """Return True if a bundle has been successfully loaded."""
        return self._generator is not None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        seed_pitches: List[int],
        length_steps: int = 32,
        temperature: float = 1.0,
        qpm: int = 120,
    ):
        """Generate a continuation from a melodic seed.

        Args:
            seed_pitches: List of MIDI pitch numbers (0-127).
            length_steps: 16th-note steps to generate (16=1 bar, 32=2 bars).
            temperature: Sampling temperature. 1.0 = balanced.
            qpm: Quarter notes per minute (tempo).

        Returns:
            A ``note_seq`` ``NoteSequence`` protobuf.

        Raises:
            RuntimeError: If the generator has not been loaded.
            ValueError: If inputs are out of valid ranges.
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Generator not loaded. Call load(bundle_path) first."
            )
        if not seed_pitches:
            raise ValueError("seed_pitches cannot be empty.")
        if any(p < 0 or p > 127 for p in seed_pitches):
            raise ValueError(
                "seed_pitches must be in MIDI range 0-127."
            )
        if length_steps < 1 or length_steps > 512:
            raise ValueError(
                f"length_steps must be 1..512 (got {length_steps})."
            )
        if not 0.1 <= temperature <= 2.0:
            raise ValueError(
                f"temperature must be 0.1..2.0 (got {temperature})."
            )

        primer = self._make_primer(seed_pitches, qpm)

        options = generator_pb2.GeneratorOptions()
        options.input_sections.add(
            start_time=0.0,
            end_time=primer.total_time,
        )
        options.generate_sections.add(
            start_time=primer.total_time,
            end_time=primer.total_time
            + length_steps * (60.0 / (qpm * 4)),
        )
        options.args["temperature"].float_value = float(temperature)

        log.info(
            "Generating %d steps at qpm=%d, temperature=%.2f...",
            length_steps, qpm, temperature,
        )
        start = time.time()
        try:
            output = self._generator.generate(primer, options)
        except Exception as exc:
            raise RuntimeError(
                f"Magenta generation failed: {exc}"
            ) from exc
        elapsed = time.time() - start
        log.info(
            "Generated %d notes in %.2fs",
            len(output.notes), elapsed,
        )
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_primer(seed_pitches: List[int], qpm: int):
        """Convert a flat list of pitches into a NoteSequence primer.

        Each pitch lasts one quarter note (4 sixteenths) at the given qpm,
        starting at time 0.
        """
        primer = music_pb2.NoteSequence()
        step_sec = 60.0 / qpm
        t = 0.0
        for pitch in seed_pitches:
            primer.notes.add(
                pitch=int(pitch),
                start_time=t,
                end_time=t + step_sec,
                velocity=80,
            )
            t += step_sec
        primer.total_time = t
        primer.tempos.add(qpm=qpm)
        return primer


# ----------------------------------------------------------------------
# Convenience: model-specific bundle filenames
# ----------------------------------------------------------------------
BUNDLE_FILENAMES = {
    "basic_rnn": "basic_rnn.mag",
    "mono_rnn": "mono_rnn.mag",
    "lookback_rnn": "lookback_rnn.mag",
    "attention_rnn": "attention_rnn.mag",
}

# Mirror the class attribute at module level for easy `from src.generator
# import AVAILABLE_MODELS`
AVAILABLE_MODELS = MelodyGenerator.AVAILABLE_MODELS


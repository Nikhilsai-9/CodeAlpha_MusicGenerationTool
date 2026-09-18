"""CodeAlpha AI Internship - Task 3: Music Generation Tool.

A Streamlit app that uses Google's Magenta project (MelodyRNN) to generate
melodic continuations from a user-provided seed.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import streamlit as st

# Make src/ importable when run from project root
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from src.audio import sequence_to_midi, sequence_to_wav  # noqa: E402
from src.generator import (  # noqa: E402
    AVAILABLE_MODELS,
    BUNDLE_FILENAMES,
    BundleNotFoundError,
    MelodyGenerator,
)
from src.presets import (  # noqa: E402
    get_preset,
    get_preset_names,
    midi_to_name,
    pitches_to_names,
)

# ---------------------------------------------------------------------------
# Page config + constants
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MelodyForge | AI Music Generator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLES_DIR = ROOT / "bundles"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def load_generator(model_name: str) -> MelodyGenerator:
    """Load a Magenta model bundle. Cached across Streamlit reruns."""
    bundle_path = BUNDLES_DIR / BUNDLE_FILENAMES[model_name]
    gen = MelodyGenerator(model_name=model_name)
    gen.load(bundle_path)
    return gen


_defaults = {
    "model_name": "attention_rnn",
    "preset": "Twinkle (C major)",
    "custom_pitches": "60, 62, 64, 65, 67, 69, 71, 72",
    "length_steps": 32,
    "qpm": 120,
    "temperature": 1.0,
}
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎵 MelodyForge")
    st.caption("AI melody generator — Magenta MelodyRNN")
    st.markdown("---")
    st.markdown("### Model")
    st.session_state.model_name = st.selectbox(
        "RNN architecture",
        options=list(AVAILABLE_MODELS),
        index=list(AVAILABLE_MODELS).index(st.session_state.model_name),
        help=(
            "attention_rnn: best coherence. "
            "basic_rnn: smallest/fastest. "
            "lookback_rnn: uses recent context. "
            "mono_rnn: classical style."
        ),
    )
    bundle_file = BUNDLES_DIR / BUNDLE_FILENAMES[st.session_state.model_name]
    if bundle_file.exists():
        size_mb = round(bundle_file.stat().st_size / 1e6, 2)
        st.success(f"Bundle loaded ✓ ({size_mb} MB)")
    else:
        st.error("Bundle MISSING")
        st.caption(f"Expected: {bundle_file}")
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "**MelodyForge** generates melodic continuations using Google's "
        "[Magenta](https://magenta.tensorflow.org/) project."
    )
    st.markdown(
        "Built with TensorFlow 2.13 + Magenta 2.1.4 + Streamlit. "
        "Task 3 of the **CodeAlpha AI Internship**."
    )
    st.markdown("---")
    st.caption(f"Project: `{ROOT.name}`")
    st.caption(f"Bundle dir: `{BUNDLES_DIR.name}/`")


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("🎵 MelodyForge — AI Music Generator")
st.markdown(
    "Generate a melodic continuation from any MIDI note sequence using "
    "Google's Magenta MelodyRNN models."
)

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown(
        """
**Magenta MelodyRNN** is a recurrent neural network trained on thousands
of MIDI melodies. It learns which notes tend to follow which, and can
generate plausible continuations of a seed melody.

**Pipeline:**
1. You provide a **seed** (a short list of MIDI pitches).
2. The RNN **samples** new pitches one 16th-note at a time.
3. The result is **rendered to MIDI** (musician-friendly) and **WAV**.

**Tips:**
- A **shorter seed** (1-4 notes) gives the model more freedom.
- **Lower temperature** (0.5-0.9) = safer, more musical choices.
- **Higher temperature** (1.1-1.5) = more surprising.
        """
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# Generation logic (defined before column blocks so it can be called)
# ---------------------------------------------------------------------------
def _run_generation(custom_pitches_text, preset) -> None:
    """Generate, render, and display the result. Errors are shown in-page."""
    # Resolve seed pitches
    if preset["pitches"]:
        seed_pitches = preset["pitches"]
    else:
        try:
            parsed = [
                int(p.strip())
                for p in (custom_pitches_text or "").replace(",", " ").split()
                if p.strip()
            ]
        except ValueError:
            st.error("Seed contains non-integer values.")
            return
        if not parsed:
            st.error("Seed is empty. Enter at least one MIDI pitch.")
            return
        if any(p < 0 or p > 127 for p in parsed):
            st.error("MIDI pitches must be 0..127 (0=C-1, 60=C4, 127=G9).")
            return
        seed_pitches = parsed

    bundle_path = BUNDLES_DIR / BUNDLE_FILENAMES[st.session_state.model_name]
    if not bundle_path.exists():
        st.error(
            f"**Bundle file missing:** `{bundle_path.name}` not found. "
            f"Download the bundle. See README."
        )
        return

    with st.spinner(f"Loading {st.session_state.model_name} model..."):
        try:
            generator = load_generator(st.session_state.model_name)
        except (BundleNotFoundError, RuntimeError) as exc:
            st.error(f"Failed to load model: {exc}")
            return

    with st.spinner("Generating melody..."):
        try:
            start = time.time()
            sequence = generator.generate(
                seed_pitches=seed_pitches,
                length_steps=st.session_state.length_steps,
                temperature=st.session_state.temperature,
                qpm=st.session_state.qpm,
            )
            elapsed = time.time() - start
        except ValueError as exc:
            st.error(f"Invalid parameters: {exc}")
            return
        except RuntimeError as exc:
            st.error(f"Generation failed: {exc}")
            return

    if not sequence.notes:
        st.error("Model produced 0 notes. Try different seed or higher temperature.")
        return

    job_id = uuid.uuid4().hex[:8]
    midi_path = OUTPUTS_DIR / f"melody_{job_id}.mid"
    wav_path = OUTPUTS_DIR / f"melody_{job_id}.wav"
    try:
        sequence_to_midi(sequence, midi_path)
        sequence_to_wav(sequence, wav_path)
    except Exception as exc:
        st.error(f"Failed to save audio: {exc}")
        return

    st.session_state.last_generation = {
        "midi_path": str(midi_path),
        "wav_path": str(wav_path),
        "note_count": len(sequence.notes),
        "duration": sequence.total_time,
        "elapsed": elapsed,
        "seed": seed_pitches,
        "model": st.session_state.model_name,
    }

    st.success(
        f"Generated **{len(sequence.notes)} notes** in **{elapsed:.2f}s** "
        f"({sequence.total_time:.1f}s of music)."
    )
    st.audio(str(wav_path), format="audio/wav")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Notes", len(sequence.notes))
    c2.metric("Duration", f"{sequence.total_time:.1f}s")
    c3.metric("Gen time", f"{elapsed:.2f}s")
    c4.metric("Model", st.session_state.model_name)

    d1, d2 = st.columns(2)
    with open(midi_path, "rb") as f:
        d1.download_button(
            "💾 Download MIDI",
            data=f.read(),
            file_name=midi_path.name,
            mime="audio/midi",
            width="stretch",
        )
    with open(wav_path, "rb") as f:
        d2.download_button(
            "💾 Download WAV",
            data=f.read(),
            file_name=wav_path.name,
            mime="audio/wav",
            width="stretch",
        )

    with st.expander("🎼 See the notes (raw data)", expanded=False):
        st.markdown(
            f"**Seed:** {' · '.join(pitches_to_names(seed_pitches))}  (MIDI: {seed_pitches})"
        )
        generated_pitches = [n.pitch for n in sequence.notes[len(seed_pitches):]]
        if generated_pitches:
            generated_str = " · ".join(pitches_to_names(generated_pitches))
        else:
            generated_str = "(none)"
        st.markdown(f"**Generated:** {generated_str}")
        note_table = [
            {
                "#": i + 1,
                "Pitch (MIDI)": n.pitch,
                "Note": midi_to_name(n.pitch),
                "Start (s)": round(n.start_time, 2),
                "End (s)": round(n.end_time, 2),
                "Velocity": n.velocity,
            }
            for i, n in enumerate(sequence.notes)
        ]
        st.dataframe(note_table, width='stretch', hide_index=True)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
col_controls, col_outputs = st.columns([1, 1], gap="large")

with col_controls:
    st.markdown("### 🎼 1. Choose your seed")
    st.session_state.preset = st.selectbox(
        "Genre preset",
        options=get_preset_names(),
        index=get_preset_names().index(st.session_state.preset),
        help="Start with a known motif, or choose Custom.",
    )
    preset = get_preset(st.session_state.preset)
    st.caption(preset["description"])

    if preset["pitches"]:
        seed_pitches = preset["pitches"]
        st.info(
            "**Seed notes:** "
            + " · ".join(pitches_to_names(seed_pitches))
            + f"   (raw: {seed_pitches})"
        )
        custom_pitches_text = None
    else:
        st.markdown("**Enter MIDI pitches** (0-127, comma/space separated):")
        custom_pitches_text = st.text_input(
            "MIDI pitches",
            value=st.session_state.custom_pitches,
            label_visibility="collapsed",
            help="e.g. '60, 62, 64, 65, 67, 69, 71, 72' for a C major scale",
        )

    st.markdown("### ⚙️ 2. Generation parameters")

    st.session_state.length_steps = st.slider(
        "Length (16th notes)",
        min_value=8,
        max_value=128,
        value=st.session_state.length_steps,
        step=4,
        help="16 = 1 bar (4/4), 32 = 2 bars, 64 = 4 bars.",
    )
    st.session_state.temperature = st.slider(
        "Temperature (creativity)",
        min_value=0.1,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Lower = more predictable. Higher = more random.",
    )
    st.session_state.qpm = st.slider(
        "Tempo (BPM)",
        min_value=60,
        max_value=200,
        value=st.session_state.qpm,
        step=5,
    )

    seed_count = len(preset["pitches"]) if preset["pitches"] else 1
    bar_count = (seed_count + st.session_state.length_steps) / 16
    st.caption(
        f"Will produce ≈ **{bar_count:.1f} bars** "
        f"(≈ **{bar_count * 4 * 60 / st.session_state.qpm:.1f} seconds**)."
    )

    generate_btn = st.button(
        "🎶 Generate melody",
        type="primary",
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Right column: output placeholder + generation trigger
# ---------------------------------------------------------------------------
with col_outputs:
    st.markdown("### 🔊 Result")
    if not generate_btn:
        st.info("Click **Generate melody** on the left to create music.")
        st.markdown("*(No generation yet — outputs will appear here.)*")
    else:
        _run_generation(custom_pitches_text, preset)


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📜 Recent generations")

outputs = sorted(
    OUTPUTS_DIR.glob("melody_*.wav"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if not outputs:
    st.caption("No generations yet.")
else:
    cols = st.columns(min(3, len(outputs)))
    for i, wav in enumerate(outputs[:3]):
        with cols[i]:
            st.caption(wav.name)
            st.audio(str(wav), format="audio/wav")

# 🎵 MelodyForge — AI Music Generation Tool

> **CodeAlpha AI Internship — Task 3** | Built with **Google Magenta MelodyRNN** + **Streamlit**

Generate original monophonic melodies in your browser. Pick a seed, choose a temperature, click **Generate**, and hear the model continue your tune.

![status](https://img.shields.io/badge/python-3.10-blue) ![status](https://img.shields.io/badge/tensorflow-2.13-orange) ![status](https://img.shields.io/badge/magenta-2.1.4-red) ![status](https://img.shields.io/badge/streamlit-1.32-ff4b4b)

---

## ✨ Features

- 🎼 **6 curated presets** — Twinkle, Ode to Joy, Jingle Bells, Chromatic, Blues, Drone — plus a **Custom** MIDI-note picker
- 🤖 **Magenta MelodyRNN** with attention model (loaded from a `.mag` bundle)
- 🎚️ **Temperature & length controls** — explore how randomness changes the output
- 🎧 **In-browser audio playback** of generated WAV files (synthesized via pretty_midi + soundfile)
- 📥 **Dual download** of the MIDI sequence and the rendered WAV
- 📜 **History** — last 3 generations auto-saved to `outputs/` and replayable
- 🖥️ **Clean Streamlit UI** — sidebar status, single-click generation, error-safe pipeline

---

## 📦 Project Structure

```
CodeAlpha_MusicGenerationTool/
├── app.py                  # Streamlit UI (entrypoint)
├── requirements.txt        # Pinned dependencies
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── generator.py        # MelodyGenerator — wraps Magenta MelodyRNN
│   ├── audio.py            # MIDI → WAV conversion (pretty_midi + soundfile)
│   └── presets.py          # 6 seed-melody presets + note-name helpers
├── bundles/
│   └── attention_rnn.mag   # ~20MB (download instructions below)
├── outputs/                # Generated .mid / .wav files
└── screenshots/            # UI screenshots for the README
```

---

## 🚀 Setup

### 1. Create a virtual environment & install deps

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The `requirements.txt` pins known-compatible versions of TensorFlow 2.13, Magenta 2.1.4, note-seq 0.0.5, pretty_midi, and soundfile.

### 2. Get the model bundle (~20 MB)

The Magenta project's official bundle URLs were retired when the GitHub repo was archived in early 2026. Use one of these mirrors:

**Recommended — Wayback Machine snapshot:**

```powershell
Invoke-WebRequest -Uri "https://web.archive.org/web/2023/http://download.magenta.tensorflow.org/models/attention_rnn.mag" -OutFile "bundles\attention_rnn.mag"
```

> If the link is broken, try `https://storage.googleapis.com/magenta-models/melody_rnn/attention_rnn.mag` directly.

### 3. Run the app

```powershell
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🎹 Usage

1. Wait for the **sidebar** to show **Bundle loaded ✓**.
2. Pick a preset seed (or write your own in **Custom**).
3. Adjust:
   - **Steps** (8–512) — how many notes to generate
   - **Temperature** (0.1–2.0) — low = conservative, high = wild
4. Click **🎶 Generate melody**.
5. Listen in-browser, download `.mid` and `.wav`, see it appear in **Recent generations**.

---

## 🧠 How it works

Magenta's **MelodyRNN** is a recurrent neural network with attention trained on the
[Lakh MIDI](https://colinraffel.com/projects/lmd/) piano-roll corpus. Given a seed
melody, the model samples note-by-note, conditioned on the previous notes and a
softmax temperature.

This app uses the `attention_rnn` configuration:

- **Input:** a `NoteSequence` with a short seed of pitches (0–127).
- **Output:** an extended `NoteSequence` with the seed + generated continuation.
- **Sample API:** `melody_rnn_sequence_generator` → `sequence_generator.sample(...)`.

The bundle is loaded once via `@st.cache_resource` so the TensorFlow graph stays
warm across reruns.

---

## 🧪 Quick smoke-test (no UI)

```powershell
.\venv\Scripts\python.exe -c "from src.generator import MelodyGenerator; from src.audio import sequence_to_midi, sequence_to_wav; g = MelodyGenerator('bundles/attention_rnn.mag'); seq = g.generate(steps=32, temperature=1.0, seed_pitches=[60,62,64,65,67,69,71,72]); sequence_to_midi(seq, 'outputs/_smoke.mid'); sequence_to_wav(seq, 'outputs/_smoke.wav'); print('OK notes:', len(seq.notes))"
```

Expected: `OK notes: 32` in ~1 second.

---

## 🩹 Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'magenta'` | Run `pip install magenta==2.1.4 --no-deps` then `pip install note-seq dm-tree cloudpickle tf-slim`. |
| `Bundle not found` | Re-download from the Wayback URL above. |
| Empty audio | The model produced 0 notes — lower the temperature or change seed. |
| `return` syntax error in app.py | Pull latest — earlier versions had a stray `return` outside `_run_generation`. |
| GPU OOM | We use TF2 CPU graph — should not OOM; otherwise set `CUDA_VISIBLE_DEVICES=""`. |

---

## 📜 License

This project is for the **CodeAlpha AI Internship**. Magenta is Apache-2.0, TensorFlow is Apache-2.0, Streamlit is Apache-2.0.

---

## 🙏 Credits

- **Google Magenta** team for the MelodyRNN model & training data.
- **CodeAlpha** for the internship task brief.
- Built with ❤️ using Streamlit.

# 🧠 LSTM Music Generator

A Tkinter desktop app that **trains an LSTM neural network on a short melody, generates new music, creates a MIDI file, and plays it** — the same pipeline as the original Colab script, but running on your own computer with no installation problems.

> Only `numpy` is needed. There is **no TensorFlow, midiutil, FluidSynth or soundfont** to install, so there is nothing to break.

## ✨ Features

- Type your own training notes (like `C4 D4 E4 G4`) or pick a preset (your original melody, Twinkle Twinkle, Ode to Joy)
- Live **training-loss curve** while the LSTM learns
- **Piano roll** of the generated music with a moving playhead
- Console-style output panel (original music, available notes, encoded notes, model, generated music, file paths)
- Settings: sequence length, LSTM units, epochs, number of notes, note length, creativity
- Creates `generated_music.mid` and `generated_music.wav` and plays the music automatically
- **Save MIDI / Save WAV** buttons to export anywhere

## 📁 Project Structure

```
LSTMMusicGenerator/
├── main.py            # Tkinter app (window, buttons, plots, playback)
├── lstm_music.py      # LSTM neural network written in NumPy (+ note parsing)
├── audio_engine.py    # Built-in piano synthesizer, WAV and MIDI writers
├── requirements.txt
├── run_windows.bat    # Double-click launcher (Windows)
├── run_mac_linux.sh   # Launcher (macOS / Linux)
├── README.md
└── output/            # created on first run: generated_music.mid / .wav
```

## 🚀 Getting Started

**Requirements:** Python 3.8 or newer and `numpy`.

```bash
pip install numpy
python main.py
```

On Windows you can also double-click `run_windows.bat`.

The app trains on your original melody and starts playing right after it opens (about 1–2 seconds of training).

**Linux only:** if Tkinter is missing run `sudo apt install python3-tk`. For sound, have one of `paplay`, `aplay` or `ffplay` installed (or use **Save WAV**).

## 🎹 How to Use

1. Choose a **preset** or type your own notes in the training box.
2. Adjust the **settings** if you like.
3. Click **Train & Generate Music**. The LSTM trains, then the new music plays.
4. Use **Play / Stop**, or **Save MIDI / Save WAV** to keep the result.

### Writing notes
Use a letter `A`–`G`, an optional `#` (sharp) or `b` (flat), and an octave number: `C4`, `F#4`, `Bb3`.
Separate notes with spaces, commas, or arrows (`C4 → D4 → E4`). You need at least 2 different notes and at least *sequence length + 2* notes in total.

### Settings

| Setting | Meaning |
|---|---|
| Sequence length | How many previous notes the LSTM looks at to predict the next one (original script: 4) |
| LSTM units | Size of the network's memory (original script: 64) |
| Epochs | How many times it practises on the data (original script: 300) |
| Notes to generate | Length of the new melody (original script: 30) |
| Note length | Seconds per note (original script: 0.5) |
| Creativity | `0` always picks the most likely note, exactly like the original `argmax`. Higher values add variety. Default `0.8` |

## 🔧 How It Works

| Step | What happens | Where |
|---|---|---|
| 1 | Training music is a list of notes | `main.py` |
| 2 | Notes → numbers (`C4`→0, `D4`→1, …) | `main.py` |
| 3 | Sliding windows: *4 notes → next note* | `make_sequences` |
| 4 | Model: `LSTM(64) → Dense(softmax)` | `NumpyLSTM` |
| 5 | Training with Adam + cross-entropy loss | `NumpyLSTM.fit` |
| 6 | Generate: predict a note, slide the window, repeat | `NumpyLSTM.generate` |
| 7–9 | Notes → MIDI numbers → `.mid` file | `audio_engine.save_midi` |
| 10 | MIDI notes → piano-like audio → `.wav` → play | `audio_engine.synthesize` |

**Differences from the Colab script**
- The LSTM is written in NumPy instead of Keras. It has the same structure and was verified against numerical gradients, so it learns correctly.
- Notes are fed as one-hot vectors instead of a single normalized number, which lets the network tell notes apart more clearly.
- Sound comes from a built-in synthesizer, so FluidSynth and a soundfont download are not needed.
- The seed notes are played first, followed by the generated notes.

## 🛠 Troubleshooting

| Problem | Fix |
|---|---|
| `No module named numpy` | `pip install numpy` |
| `No module named tkinter` (Linux) | `sudo apt install python3-tk` |
| "Not a valid note" message | Use names like `C4`, `F#4`, `Bb3` |
| "Need at least N notes" message | Add more training notes or lower the sequence length |
| No sound on Linux | Install `pulseaudio-utils`, `alsa-utils` or `ffmpeg`, or use **Save WAV** |
| Music repeats the same pattern | Increase **Creativity**, or train on a longer melody |

## 📜 License

Free to use for learning and personal projects.

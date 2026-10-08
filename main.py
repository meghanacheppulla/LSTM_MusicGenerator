"""
LSTM Music Generator  -  Tkinter desktop app
Run:  python main.py

Pipeline (same as the original Colab script):
  1. training music (notes)  ->  2. notes to numbers  ->  3. training sequences
  4. LSTM model              ->  5. train             ->  6. generate new notes
  7. notes to MIDI numbers   ->  8-9. create + save MIDI file  ->  10. play audio

Only `numpy` is required (the LSTM is written in NumPy, audio is built in).
"""

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


import audio_engine as ae
from lstm_music import NumpyLSTM, make_sequences, parse_notes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRESETS = {
    "Original melody (from your script)":
        "C4 D4 E4 G4 E4 D4 C4 E4 G4 A4 G4 E4 D4 C4 D4 E4 G4 E4 D4 C4",
    "Twinkle Twinkle Little Star":
        "C4 C4 G4 G4 A4 A4 G4 F4 F4 E4 E4 D4 D4 C4 "
        "G4 G4 F4 F4 E4 E4 D4 G4 G4 F4 F4 E4 E4 D4 "
        "C4 C4 G4 G4 A4 A4 G4 F4 F4 E4 E4 D4 D4 C4",
    "Ode to Joy":
        "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 E4 D4 D4 "
        "E4 E4 F4 G4 G4 F4 E4 D4 C4 C4 D4 E4 D4 C4 C4",
}


# ----------------------------------------------------------------------
# Audio playback (no extra libraries)
# ----------------------------------------------------------------------
class Player:
    def __init__(self):
        self.proc = None

    def play(self, wav_path):
        self.stop()
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        if sys.platform == "darwin":
            cmd = ["afplay", wav_path]
        else:
            cmd = None
            for tool, args in (("paplay", []), ("aplay", ["-q"]), ("play", ["-q"]),
                               ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"])):
                if shutil.which(tool):
                    cmd = [tool] + args + [wav_path]
                    break
        if not cmd:
            return False
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        return True

    def stop(self):
        if sys.platform.startswith("win"):
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None


def output_folder():
    """Folder for generated_music.mid / .wav (falls back to temp if read-only)."""
    for folder in (os.path.join(BASE_DIR, "output"),
                   os.path.join(tempfile.gettempdir(), "lstm_music_output")):
        try:
            os.makedirs(folder, exist_ok=True)
            test = os.path.join(folder, ".write_test")
            with open(test, "w") as f:
                f.write("ok")
            os.remove(test)
            return folder
        except OSError:
            continue
    return tempfile.gettempdir()


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("LSTM Music Generator")
        root.geometry("940x690")
        root.minsize(800, 600)

        self.player = Player()
        self.out_dir = output_folder()
        self.wav_path = os.path.join(self.out_dir, "generated_music.wav")
        self.mid_path = os.path.join(self.out_dir, "generated_music.mid")

        self.notes = []          # (pitch, start, end, velocity, instrument)
        self.seed_count = 0
        self.bpm = 120
        self.audio = None
        self.duration = 0.0
        self.losses = []
        self.play_started = None
        self.busy = False
        self.q = queue.Queue()

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._poll)
        root.after(400, self.start)  # train + play something straight away

    # ---------------- UI ----------------
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Big.TButton", font=("Segoe UI", 10, "bold"), padding=6)

        ttk.Label(self.root, text="\U0001F9E0 LSTM Music Generator", style="Title.TLabel",
                  padding=(14, 10, 14, 0)).pack(anchor="w")

        # --- training music
        box = ttk.LabelFrame(self.root, text=" 1. Training music (notes like C4 D4 F#4 Bb3) ",
                             padding=8)
        box.pack(fill="x", padx=14, pady=(8, 4))
        row = ttk.Frame(box)
        row.pack(fill="x")
        ttk.Label(row, text="Preset:").pack(side="left")
        self.preset = tk.StringVar(value=list(PRESETS)[0])
        cb = ttk.Combobox(row, textvariable=self.preset, values=list(PRESETS),
                          state="readonly", width=38)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", self._on_preset)
        self.text = tk.Text(box, height=2, wrap="word", font=("Consolas", 10))
        self.text.pack(fill="x", pady=(6, 0))
        self.text.insert("1.0", PRESETS[self.preset.get()])

        # --- settings
        st = ttk.LabelFrame(self.root, text=" 2. Settings ", padding=8)
        st.pack(fill="x", padx=14, pady=4)
        self.seq_len = tk.IntVar(value=4)
        self.hidden = tk.IntVar(value=64)
        self.epochs = tk.IntVar(value=300)
        self.count = tk.IntVar(value=30)
        self.temp = tk.DoubleVar(value=0.8)
        self.note_len = tk.DoubleVar(value=0.5)

        def spin(parent, label, var, lo, hi, inc, col):
            ttk.Label(parent, text=label).grid(row=0, column=col * 2, sticky="e", padx=(8, 3))
            ttk.Spinbox(parent, from_=lo, to=hi, increment=inc, textvariable=var,
                        width=7).grid(row=0, column=col * 2 + 1, sticky="w")

        spin(st, "Sequence length", self.seq_len, 2, 8, 1, 0)
        spin(st, "LSTM units", self.hidden, 8, 128, 8, 1)
        spin(st, "Epochs", self.epochs, 50, 2000, 50, 2)
        spin(st, "Notes to generate", self.count, 8, 100, 2, 3)
        ttk.Label(st, text="Note length (s)").grid(row=1, column=0, sticky="e", padx=(8, 3), pady=(8, 0))
        ttk.Combobox(st, textvariable=self.note_len, values=[0.25, 0.4, 0.5, 0.75, 1.0],
                     width=6, state="readonly").grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(st, text="Creativity (0 = most likely note)").grid(
            row=1, column=2, columnspan=2, sticky="e", padx=(8, 3), pady=(8, 0))
        ttk.Scale(st, from_=0.0, to=2.0, variable=self.temp, length=170).grid(
            row=1, column=4, columnspan=2, sticky="w", pady=(8, 0))
        self.temp_label = ttk.Label(st, text="0.80", width=5)
        self.temp_label.grid(row=1, column=6, sticky="w", pady=(8, 0))
        self.temp.trace_add("write", lambda *a: self.temp_label.config(
            text=f"{self.temp.get():.2f}"))

        # --- buttons
        btns = ttk.Frame(self.root, padding=(14, 8))
        btns.pack(fill="x")
        self.b_train = ttk.Button(btns, text="\U0001F3BC Train & Generate Music",
                                  style="Big.TButton", command=self.start)
        self.b_train.pack(side="left", padx=(0, 6))
        self.b_play = ttk.Button(btns, text="\u25B6 Play", style="Big.TButton", command=self.play)
        self.b_play.pack(side="left", padx=6)
        self.b_stop = ttk.Button(btns, text="\u25A0 Stop", style="Big.TButton", command=self.stop)
        self.b_stop.pack(side="left", padx=6)
        self.b_mid = ttk.Button(btns, text="Save MIDI", command=self.save_midi)
        self.b_mid.pack(side="left", padx=6)
        self.b_wav = ttk.Button(btns, text="Save WAV", command=self.save_wav)
        self.b_wav.pack(side="left", padx=6)

        self.status = tk.StringVar(value="Starting...")
        ttk.Label(self.root, textvariable=self.status, padding=(14, 0)).pack(anchor="w")
        self.progress = ttk.Progressbar(self.root, mode="determinate", maximum=1000)
        self.progress.pack(fill="x", padx=14, pady=(4, 6))

        # --- plots
        plots = ttk.Frame(self.root)
        plots.pack(fill="x", padx=14)
        left = ttk.LabelFrame(plots, text=" Training loss ", padding=4)
        left.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.loss_canvas = tk.Canvas(left, height=105, bg="#fdf6e3", highlightthickness=0)
        self.loss_canvas.pack(fill="both", expand=True)
        self.loss_canvas.bind("<Configure>", lambda e: self.draw_loss())
        right = ttk.LabelFrame(plots, text=" Generated music (blue = seed, orange = new) ",
                               padding=4)
        right.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.roll = tk.Canvas(right, height=105, bg="#14213d", highlightthickness=0)
        self.roll.pack(fill="both", expand=True)
        self.roll.bind("<Configure>", lambda e: self.draw_roll())

        # --- log
        logf = ttk.LabelFrame(self.root, text=" Output ", padding=4)
        logf.pack(fill="both", expand=True, padx=14, pady=(6, 12))
        self.log = tk.Text(logf, height=5, wrap="word", state="disabled",
                           font=("Consolas", 9), bg="#f7f7f7")
        sb = ttk.Scrollbar(logf, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ---------------- helpers ----------------
    def _on_preset(self, _event=None):
        self.text.delete("1.0", "end")
        self.text.insert("1.0", PRESETS[self.preset.get()])

    def write_log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def set_busy(self, busy, text=None):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.b_train, self.b_play, self.b_mid, self.b_wav):
            b.configure(state=state)
        if text:
            self.status.set(text)

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self.write_log(payload)
                elif kind == "progress":
                    self.progress["value"] = payload * 1000
                elif kind == "loss":
                    self.losses.append(payload)
                    if len(self.losses) % 10 == 0:
                        self.draw_loss()
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "done":
                    self._on_done(payload)
                elif kind == "error":
                    self.set_busy(False, "Something went wrong.")
                    self.progress["value"] = 0
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        if self.play_started is not None:
            elapsed = time.time() - self.play_started
            if elapsed >= self.duration:
                self.play_started = None
                self.progress["value"] = 0
                self.status.set("Finished playing.")
                self.draw_roll()
            else:
                self.progress["value"] = elapsed / self.duration * 1000
                self.draw_roll(playhead=elapsed)
        self.root.after(100, self._poll)

    # ---------------- training + generation ----------------
    def start(self):
        if self.busy:
            return
        # read + validate everything here (main thread) so errors show nicely
        try:
            seq_len = int(self.seq_len.get())
            hidden = int(self.hidden.get())
            epochs = int(self.epochs.get())
            count = int(self.count.get())
            temperature = float(self.temp.get())
            note_len = float(self.note_len.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Settings", "Please enter whole numbers in the settings boxes.")
            return
        try:
            pitches = parse_notes(self.text.get("1.0", "end"))
        except ValueError as exc:
            messagebox.showerror("Training music", str(exc))
            return
        if len(set(pitches)) < 2:
            messagebox.showerror("Training music", "Use at least 2 different notes.")
            return
        if len(pitches) < seq_len + 2:
            messagebox.showerror(
                "Training music",
                f"Need at least {seq_len + 2} notes for sequence length {seq_len} "
                f"(you have {len(pitches)}).\nAdd more notes or lower the sequence length.")
            return
        seq_len = max(2, min(8, seq_len))
        hidden = max(4, min(256, hidden))
        epochs = max(10, min(5000, epochs))
        count = max(4, min(200, count))

        self.stop()
        self.losses = []
        self.draw_loss()
        self.clear_log()
        self.set_busy(True, "Training the LSTM... please wait")
        self.progress["value"] = 0
        threading.Thread(
            target=self._worker,
            args=(pitches, seq_len, hidden, epochs, count, temperature, note_len),
            daemon=True).start()

    def _worker(self, pitches, seq_len, hidden, epochs, count, temperature, note_len):
        q = self.q
        try:
            names = [ae.note_name(p) for p in pitches]
            q.put(("log", "Original music:\n" + " \u2192 ".join(names)))

            # 2. notes -> numbers
            vocab = sorted(set(pitches))
            number_of = {p: i for i, p in enumerate(vocab)}
            encoded = [number_of[p] for p in pitches]
            q.put(("log", "\nAvailable notes:\n" + str([ae.note_name(p) for p in vocab])))
            q.put(("log", "\nEncoded notes:\n" + str(encoded)))

            # 3. training sequences
            X, y = make_sequences(encoded, seq_len)
            q.put(("log", f"\nTraining sequences: {X.shape[0]} (each {seq_len} notes -> next note)"))

            # 4. model
            model = NumpyLSTM(len(vocab), hidden)
            q.put(("log", f"\nModel: LSTM({hidden}) \u2192 Dense({len(vocab)}, softmax)"
                          f"   |  parameters: {model.count_params():,}"))

            # 5. train
            q.put(("log", "\nTraining LSTM..."))

            def on_epoch(epoch, total, loss):
                q.put(("loss", loss))
                if epoch % 5 == 0 or epoch == total:
                    q.put(("progress", epoch / total * 0.8))
                    q.put(("status", f"Training... epoch {epoch}/{total}   loss {loss:.4f}"))

            losses = model.fit(X, y, epochs=epochs, lr=0.01, callback=on_epoch)
            q.put(("log", f"Training completed!  (final loss: {losses[-1]:.4f})"))

            # 6. generate
            seed = encoded[:seq_len]
            generated = model.generate(seed, count, temperature)
            gen_names = [ae.note_name(vocab[i]) for i in generated]
            q.put(("log", "\nSeed notes:\n" + " \u2192 ".join(ae.note_name(vocab[i]) for i in seed)))
            q.put(("log", "\nGenerated Music:\n" + " \u2192 ".join(gen_names)))

            # 7. notes -> MIDI notes (seed first, then generated)
            all_ids = list(seed) + generated
            notes, t = [], 0.0
            for idx in all_ids:
                notes.append((vocab[idx], t, t + note_len * 0.96, 100, "piano"))
                t += note_len
            bpm = 60.0 / note_len

            # 8-9. create + save MIDI
            q.put(("status", "Creating MIDI and audio..."))
            ae.save_midi(notes, bpm, self.mid_path)
            q.put(("log", f"\nMIDI file created:\n{self.mid_path}"))

            # 10. audio
            audio = ae.synthesize(notes, progress=lambda f: q.put(("progress", 0.8 + f * 0.2)))
            ae.save_wav(audio, self.wav_path)
            q.put(("log", f"WAV file created:\n{self.wav_path}"))
            q.put(("done", (notes, len(seed), bpm, audio)))
        except Exception as exc:  # show any problem in a dialog
            q.put(("error", f"{type(exc).__name__}: {exc}"))

    def _on_done(self, payload):
        self.notes, self.seed_count, self.bpm, self.audio = payload
        self.duration = len(self.audio) / ae.SAMPLE_RATE
        self.draw_loss()
        self.set_busy(False, f"Ready: {len(self.notes)} notes, {self.duration:.0f} sec")
        self.progress["value"] = 0
        self.draw_roll()
        self.play()

    # ---------------- playback / saving ----------------
    def play(self):
        if self.audio is None or self.busy:
            return
        if not os.path.exists(self.wav_path):
            ae.save_wav(self.audio, self.wav_path)
        if self.player.play(self.wav_path):
            self.play_started = time.time()
            self.status.set("Playing generated music...")
        else:
            messagebox.showinfo(
                "No audio player found",
                "Could not find a sound player on this system.\n"
                "Use 'Save WAV' and open the file with any media player.")

    def stop(self):
        self.player.stop()
        self.play_started = None
        self.progress["value"] = 0
        if self.audio is not None:
            self.status.set("Stopped.")
        self.draw_roll()

    def save_midi(self):
        if not self.notes:
            return
        path = filedialog.asksaveasfilename(defaultextension=".mid",
                                            initialfile="generated_music.mid",
                                            filetypes=[("MIDI file", "*.mid")])
        if path:
            ae.save_midi(self.notes, self.bpm, path)
            messagebox.showinfo("Saved", f"Saved:\n{path}")

    def save_wav(self):
        if self.audio is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                            initialfile="generated_music.wav",
                                            filetypes=[("WAV audio", "*.wav")])
        if path:
            ae.save_wav(self.audio, path)
            messagebox.showinfo("Saved", f"Saved:\n{path}")

    # ---------------- drawing ----------------
    def draw_loss(self):
        c = self.loss_canvas
        c.delete("all")
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 60)
        if len(self.losses) < 2:
            c.create_text(w / 2, h / 2, text="(loss curve appears while training)",
                          fill="#888888")
            return
        top = max(self.losses)
        n = len(self.losses)
        pts = []
        for i, v in enumerate(self.losses):
            pts += [10 + i / (n - 1) * (w - 20), h - 14 - v / top * (h - 28)]
        c.create_line(*pts, fill="#d62828", width=2)
        c.create_text(12, 8, anchor="nw", text=f"{top:.2f}", fill="#555555", font=("Segoe UI", 8))
        c.create_text(w - 10, h - 4, anchor="se", text=f"final {self.losses[-1]:.3f}",
                      fill="#555555", font=("Segoe UI", 8))

    def draw_roll(self, playhead=None):
        c = self.roll
        c.delete("all")
        if not self.notes:
            return
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 60)
        total = max(self.duration, max(n[2] for n in self.notes), 1.0)
        lo = min(n[0] for n in self.notes)
        hi = max(n[0] for n in self.notes)
        span = max(hi - lo, 7)
        for i, (pitch, start, end, vel, inst) in enumerate(self.notes):
            x1 = start / total * w
            x2 = max(x1 + 3, end / total * w)
            y = h - 10 - (pitch - lo) / span * (h - 20)
            c.create_rectangle(x1, y - 4, x2, y + 4,
                               fill="#4cc9f0" if i < self.seed_count else "#f77f00",
                               outline="")
        if playhead is not None:
            x = min(playhead / total, 1.0) * w
            c.create_line(x, 0, x, h, fill="#ffffff", width=2)

    def on_close(self):
        self.player.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()

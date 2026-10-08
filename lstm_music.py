"""
lstm_music.py
-------------
A small LSTM neural network written in pure NumPy.

It does the same job as the Keras model in the original script:

    Sequential([ LSTM(64), Dense(num_notes, activation="softmax") ])
    loss = sparse categorical cross-entropy, optimizer = Adam

...but needs no TensorFlow, so it installs and runs everywhere.
"""

import re

import numpy as np

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(\d)$")
_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# ----------------------------------------------------------------------
# Note text  ->  MIDI numbers
# ----------------------------------------------------------------------
def parse_notes(text):
    """'C4 D4 F#4 -> Bb3' -> [60, 62, 66, 58].  Raises ValueError if invalid."""
    tokens = [t for t in re.split(r"[\s,;|]+|\u2192|->", text) if t]
    pitches = []
    for tok in tokens:
        m = _NOTE_RE.match(tok)
        if not m:
            raise ValueError(
                f"'{tok}' is not a valid note.\n"
                "Use names like C4, D4, F#4, Bb3 (letter A-G, optional # or b, octave 0-9).")
        letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
        pitch = 12 * (octave + 1) + _BASE[letter]
        if accidental == "#":
            pitch += 1
        elif accidental == "b":
            pitch -= 1
        if not 0 <= pitch <= 127:
            raise ValueError(f"Note '{tok}' is outside the MIDI range.")
        pitches.append(pitch)
    return pitches


def make_sequences(encoded, seq_len):
    """Sliding windows: seq_len notes -> the next note."""
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i:i + seq_len])
        y.append(encoded[i + seq_len])
    return np.array(X, dtype=int), np.array(y, dtype=int)


# ----------------------------------------------------------------------
# Math helpers
# ----------------------------------------------------------------------
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------
# The LSTM
# ----------------------------------------------------------------------
class NumpyLSTM:
    """LSTM(hidden) -> Dense(vocab, softmax), trained with Adam."""

    def __init__(self, vocab, hidden=64, seed=42):
        rng = np.random.default_rng(seed)
        self.V, self.H = vocab, hidden
        d = vocab + hidden
        self.params = {
            "W": rng.normal(0, 1.0 / np.sqrt(d), (4 * hidden, d)),   # gates i,f,o,g
            "b": np.zeros(4 * hidden),
            "Wy": rng.normal(0, 1.0 / np.sqrt(hidden), (vocab, hidden)),
            "by": np.zeros(vocab),
        }
        self.params["b"][hidden:2 * hidden] = 1.0  # forget-gate bias = 1 (helps memory)
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._t = 0

    def count_params(self):
        return sum(p.size for p in self.params.values())

    # ---------- forward ----------
    def forward(self, X):
        """X: (N, T) integer note ids -> probabilities (N, V)."""
        p, H, V = self.params, self.H, self.V
        N, T = X.shape
        h = np.zeros((N, H))
        c = np.zeros((N, H))
        cache = []
        for t in range(T):
            x = np.zeros((N, V))
            x[np.arange(N), X[:, t]] = 1.0
            z = np.concatenate([x, h], axis=1)
            a = z @ p["W"].T + p["b"]
            i = _sigmoid(a[:, :H])
            f = _sigmoid(a[:, H:2 * H])
            o = _sigmoid(a[:, 2 * H:3 * H])
            g = np.tanh(a[:, 3 * H:])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            cache.append((z, i, f, o, g, c_prev, tc))
        probs = _softmax(h @ p["Wy"].T + p["by"])
        return probs, h, cache

    # ---------- loss + gradients ----------
    def loss_and_grads(self, X, y):
        p, H, V = self.params, self.H, self.V
        N, T = X.shape
        probs, h, cache = self.forward(X)
        loss = float(-np.mean(np.log(probs[np.arange(N), y] + 1e-12)))

        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1.0
        dlogits /= N

        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["Wy"] = dlogits.T @ h
        grads["by"] = dlogits.sum(axis=0)

        dh = dlogits @ p["Wy"]
        dc = np.zeros((N, H))
        for t in reversed(range(T)):
            z, i, f, o, g, c_prev, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1.0 - tc ** 2)
            di, dg, df = dc * g, dc * i, dc * c_prev
            da = np.concatenate([di * i * (1 - i),
                                 df * f * (1 - f),
                                 do * o * (1 - o),
                                 dg * (1 - g ** 2)], axis=1)
            grads["W"] += da.T @ z
            grads["b"] += da.sum(axis=0)
            dz = da @ p["W"]
            dh = dz[:, V:]
            dc = dc * f
        return loss, grads

    # ---------- training ----------
    def _adam_step(self, grads, lr, clip=5.0):
        norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in grads.values()))
        scale = clip / norm if norm > clip else 1.0
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, param in self.params.items():
            g = grads[k] * scale
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            m_hat = self._m[k] / (1 - b1 ** self._t)
            v_hat = self._v[k] / (1 - b2 ** self._t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)  # in-place update

    def fit(self, X, y, epochs=300, lr=0.01, callback=None):
        losses = []
        for epoch in range(1, epochs + 1):
            loss, grads = self.loss_and_grads(X, y)
            self._adam_step(grads, lr)
            losses.append(loss)
            if callback:
                callback(epoch, epochs, loss)
        return losses

    # ---------- generation ----------
    def generate(self, seed, count, temperature=0.8, rng=None):
        """Predict `count` notes after `seed` (list of note ids)."""
        rng = rng or np.random.default_rng()
        seq = list(seed)
        seq_len = len(seed)
        out = []
        for _ in range(count):
            probs, _, _ = self.forward(np.array([seq[-seq_len:]]))
            probs = probs[0]
            if temperature <= 0.01:
                nxt = int(np.argmax(probs))          # most probable note (as in the original)
            else:
                logits = np.log(probs + 1e-9) / temperature
                logits -= logits.max()
                q = np.exp(logits)
                q /= q.sum()
                nxt = int(rng.choice(len(q), p=q))   # a little creativity
            out.append(nxt)
            seq.append(nxt)
        return out

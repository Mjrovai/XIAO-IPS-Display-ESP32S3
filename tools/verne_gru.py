#!/usr/bin/env python3
"""The VerneBot GRU in numpy: the reference for the board, and the exporter of its 8-bit weights.

The weights are the ones of the VerneBot project (docs/assets/models/rnn.bin and manifest.json: float16,
every matrix as (out, in)). The arithmetic is the one of its JavaScript demo (docs/assets/demo/js/gru.js),
which follows Keras with reset_after=True:

    z  = sigmoid(Wz x + Rz h + bz + rbz)
    r  = sigmoid(Wr x + Rr h + br + rbr)
    hh = tanh(Wh x + bh + r * (Rh h + rbh))
    h' = z * h + (1 - z) * hh

Usage:
  verne_gru.py check  PROJECT_DIR                 compare the float16 model with tests/reference.json
  verne_gru.py export PROJECT_DIR OUT.bin         write the 8-bit file that the board reads
  verne_gru.py greedy PROJECT_DIR "seed text" [N] [--int8]   greedy text of N characters (default 64)
  verne_gru.py agree  PROJECT_DIR BOOK.txt [N]    how much 8-bit weights change the model on N characters of a book
"""
import json, struct, sys
from pathlib import Path
import numpy as np

def load(project):
    m = json.load(open(Path(project) / "docs/assets/models/manifest.json"))
    info = m["models"]["rnn"]
    raw = np.fromfile(Path(project) / "docs/assets/models" / info["file"], dtype="<f2")
    t = {k: raw[v["offset"]:v["offset"] + v["length"]].astype(np.float32).reshape(v["shape"]) for k, v in info["tensors"].items()}
    return t, m["vocab"], info

def int8_rows(w):
    s = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127.0
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    return q, s.astype(np.float32)

def quantized(t):
    """Same tensors, but the six GRU matrices and the dense one rounded to 8 bits with one scale per row."""
    out = dict(t)
    for k in ("kernel_z", "kernel_r", "kernel_h", "recurrent_z", "recurrent_r", "recurrent_h", "dense_w"):
        q, s = int8_rows(t[k]); out[k] = (q.astype(np.float32) * s[:, None])
    return out

def sig(x): return 1.0 / (1.0 + np.exp(-x))

class GRU:
    def __init__(self, t): self.t = t; self.h = np.zeros(1024, np.float32)
    def step(self, c):
        t = self.t; x = t["wte"][c]; h = self.h
        z = sig(t["kernel_z"] @ x + t["recurrent_z"] @ h + t["input_bias_z"] + t["recurrent_bias_z"])
        r = sig(t["kernel_r"] @ x + t["recurrent_r"] @ h + t["input_bias_r"] + t["recurrent_bias_r"])
        hh = np.tanh(t["kernel_h"] @ x + t["input_bias_h"] + r * (t["recurrent_h"] @ h + t["recurrent_bias_h"]))
        self.h = z * h + (1 - z) * hh
        return t["dense_w"] @ self.h + t["dense_b"]

def encode(vocab, text):
    idx = {c: i for i, c in enumerate(vocab)}
    return [idx[c] for c in text if c in idx]

def cmd_check(project):
    t, vocab, _ = load(project)
    ref = json.load(open(Path(project) / "tests/reference.json"))
    g = GRU(t); lg = None
    for c in encode(vocab, ref["prompt"]): lg = g.step(c)
    r = np.array(ref["models"]["rnn"]["logits"], np.float32)
    print("prompt:", ref["prompt"], "| max |logit difference| vs the reference:", float(np.abs(lg - r).max()))
    print("top 8 here :", "".join(vocab[i] if vocab[i] != "\n" else "\\n" for i in np.argsort(-lg)[:8]))
    print("top 8 ref  :", "".join(ref["models"]["rnn"]["top_chars"]).replace("\n", "\\n"))

def cmd_export(project, out):
    t, vocab, info = load(project)
    f = open(out, "wb")
    vb = [c.encode("utf-8") for c in vocab]
    f.write(b"VGRU" + struct.pack("<7I", 1, len(vocab), 256, 1024, 120, 0, 0))
    for b in vb: f.write(struct.pack("<B", len(b)) + b)
    f.write(t["wte"].astype("<f4").tobytes())
    # biases first (float32), then each matrix as scales (float32) followed by int8 rows; gates in the order z, r, h
    for k in ("input_bias_z", "input_bias_r", "input_bias_h", "recurrent_bias_z", "recurrent_bias_r", "recurrent_bias_h", "dense_b"):
        f.write(t[k].astype("<f4").tobytes())
    mats = [np.concatenate([t["kernel_z"], t["kernel_r"], t["kernel_h"]]),
            np.concatenate([t["recurrent_z"], t["recurrent_r"], t["recurrent_h"]]), t["dense_w"]]
    for w in mats:
        q, s = int8_rows(w); f.write(s.astype("<f4").tobytes()); f.write(q.tobytes())
    f.close(); print("wrote", out, Path(out).stat().st_size, "bytes")

def sample_greedy(g, vocab, seed, n):
    lg = None
    for c in encode(vocab, seed): lg = g.step(c)
    out = []
    for _ in range(n):
        c = int(np.argmax(lg)); out.append(vocab[c]); lg = g.step(c)
    return "".join(out)

def cmd_agree(project, book, n):
    t, vocab, _ = load(project); q = quantized(t)
    text = open(book, encoding="utf-8", errors="ignore").read()
    ids = encode(vocab, text)[200000:200000 + n]
    a, b = GRU(t), GRU(q); la = lb = 0.0; agree = 0
    for i in range(len(ids) - 1):
        pa, pb = a.step(ids[i]), b.step(ids[i])
        la -= (pa - np.log(np.exp(pa - pa.max()).sum()) - pa.max())[ids[i + 1]]
        lb -= (pb - np.log(np.exp(pb - pb.max()).sum()) - pb.max())[ids[i + 1]]
        agree += int(np.argmax(pa) == np.argmax(pb))
    m = len(ids) - 1
    print(f"{m} characters: loss float16 {la / m:.4f}, loss 8-bit {lb / m:.4f}; same most likely next character {agree / m * 100:.1f}%")

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "check": cmd_check(a[1])
    elif a[0] == "export": cmd_export(a[1], a[2])
    elif a[0] == "greedy":
        t, vocab, _ = load(a[1])
        if "--int8" in a: t = quantized(t)
        n = int(a[3]) if len(a) > 3 and a[3].isdigit() else 64
        print(sample_greedy(GRU(t), vocab, a[2], n))
    elif a[0] == "agree": cmd_agree(a[1], a[2], int(a[3]) if len(a) > 3 else 1500)

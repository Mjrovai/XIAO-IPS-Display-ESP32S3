#!/usr/bin/env python3
"""The VerneBot Transformer in numpy: the reference for the board, and the exporter of its 8-bit weights.

The weights are the ones of the VerneBot project (docs/assets/models/tx-paired.bin, window 256, or
tx-paired-ctx120.bin, window 120, with manifest.json: float16, every matrix as (out, in)). The arithmetic is the
one of its JavaScript demo (docs/assets/demo/js/transformer.js): decoder-only, pre-norm, learned positions,
exact (erf) GELU, an output head tied to the embedding, and a key-value cache that never wraps: when the window
is full, the oldest 64 characters are dropped and the others are run through the network again.

Usage:
  verne_transformer.py check  PROJECT_DIR [tx-paired|tx-paired-ctx120]     compare with tests/reference.json
  verne_transformer.py export PROJECT_DIR OUT.bin [model]                   write the 8-bit file for the board
  verne_transformer.py greedy PROJECT_DIR "seed" [N] [--int8] [--model M]   greedy text of N characters (default 64)
  verne_transformer.py agree  PROJECT_DIR BOOK.txt [N]                      cost of 8-bit weights on N characters
"""
import json, math, struct, sys
from pathlib import Path
import numpy as np

REFRESH_EVERY = 64
MATS = ("wq_w", "wk_w", "wv_w", "wo_w", "fc_w", "proj_w")

def load(project, key="tx-paired"):
    m = json.load(open(Path(project) / "docs/assets/models/manifest.json"))
    info = m["models"][key]
    raw = np.fromfile(Path(project) / "docs/assets/models" / info["file"], dtype="<f2")
    t = {k: raw[v["offset"]:v["offset"] + v["length"]].astype(np.float32).reshape(v["shape"]) for k, v in info["tensors"].items()}
    return t, m["vocab"], info

def int8_rows(w):
    s = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127.0
    return np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32)

def quantized(t, layers):
    out = dict(t)
    for l in range(layers):
        for k in MATS:
            q, s = int8_rows(t[f"{k}_{l}"]); out[f"{k}_{l}"] = q.astype(np.float32) * s[:, None]
    return out

_erf = np.vectorize(math.erf, otypes=[np.float64])
def gelu(x): return (0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))).astype(np.float32)

def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(); v = ((x - m) ** 2).mean()
    return ((x - m) / np.sqrt(v + eps) * g + b).astype(np.float32)

class Transformer:
    def __init__(self, t, info):
        self.t, self.L, self.H, self.D, self.ctx = t, info["n_layers"], info["n_heads"], info["d_model"], info["context"]
        self.reset()
    def reset(self):
        self.k = np.zeros((self.L, self.ctx, self.D), np.float32); self.v = np.zeros_like(self.k)
    def step(self, c, pos):
        t, D, H = self.t, self.D, self.H; dh = D // H
        x = t["wte"][c] + t["wpe"][pos]
        for l in range(self.L):
            n = layer_norm(x, t[f"ln1_g_{l}"], t[f"ln1_b_{l}"])
            q = t[f"wq_w_{l}"] @ n + t[f"wq_b_{l}"]; k = t[f"wk_w_{l}"] @ n + t[f"wk_b_{l}"]; v = t[f"wv_w_{l}"] @ n + t[f"wv_b_{l}"]
            self.k[l, pos] = k; self.v[l, pos] = v
            att = np.zeros(D, np.float32)
            for h in range(H):
                s = self.k[l, :pos + 1, h * dh:(h + 1) * dh] @ q[h * dh:(h + 1) * dh] / math.sqrt(dh)
                e = np.exp(s - s.max()); e /= e.sum()
                att[h * dh:(h + 1) * dh] = e @ self.v[l, :pos + 1, h * dh:(h + 1) * dh]
            res = x + t[f"wo_w_{l}"] @ att + t[f"wo_b_{l}"]
            n2 = layer_norm(res, t[f"ln2_g_{l}"], t[f"ln2_b_{l}"])
            x = res + t[f"proj_w_{l}"] @ gelu(t[f"fc_w_{l}"] @ n2 + t[f"fc_b_{l}"]) + t[f"proj_b_{l}"]
        return t["head"] @ layer_norm(x, t["lnf_g"], t["lnf_b"])

def encode(vocab, text):
    idx = {c: i for i, c in enumerate(vocab)}
    return [idx[c] for c in text if c in idx]

def greedy(model, vocab, seed, n):
    """Greedy writing with the window rebuild of the demo; returns the text."""
    prompt = (encode(vocab, seed) or [0])[-(model.ctx - 1):]
    model.reset(); lg = None
    for p, c in enumerate(prompt): lg = model.step(c, p)
    history, pos, out = list(prompt), len(prompt), []
    for _ in range(n):
        nxt = int(np.argmax(lg)); out.append(vocab[nxt]); history.append(nxt)
        history = history[-model.ctx:]
        if pos + 1 >= model.ctx:
            tail = history[-max(1, model.ctx - REFRESH_EVERY):]
            model.reset()
            for p, c in enumerate(tail): lg = model.step(c, p)
            pos = len(tail)
        else:
            lg = model.step(nxt, pos); pos += 1
    return "".join(out)

def cmd_check(project, key):
    t, vocab, info = load(project, key); ref = json.load(open(Path(project) / "tests/reference.json"))
    if key not in ref["models"]: print("no reference for", key); return
    m = Transformer(t, info); lg = None
    for p, c in enumerate(encode(vocab, ref["prompt"])): lg = m.step(c, p)
    r = np.array(ref["models"][key]["logits"], np.float32)
    print(f"{key}: prompt {ref['prompt']!r}, max |logit difference| vs the reference: {float(np.abs(lg - r).max()):.5f}; head equals the embedding: {bool(np.array_equal(t['head'], t['wte']))}")
    print("top 8 here:", "".join(vocab[i] if vocab[i] != "\n" else "\\n" for i in np.argsort(-lg)[:8]), "| reference:", "".join(ref["models"][key]["top_chars"]).replace("\n", "\\n"))

def cmd_export(project, out, key):
    t, vocab, info = load(project, key)
    assert np.array_equal(t["head"], t["wte"]), "the head is not tied to the embedding"
    f = open(out, "wb")
    f.write(b"VTRF" + struct.pack("<7I", 1, len(vocab), info["d_model"], info["n_layers"], info["n_heads"], info["d_ff"], info["context"]))
    for c in vocab:
        b = c.encode("utf-8"); f.write(struct.pack("<B", len(b)) + b)
    for k in ("wte", "wpe", "lnf_g", "lnf_b"): f.write(t[k].astype("<f4").tobytes())
    for l in range(info["n_layers"]):
        for k in ("ln1_g", "ln1_b", "wq_b", "wk_b", "wv_b", "wo_b", "ln2_g", "ln2_b", "fc_b", "proj_b"):
            f.write(t[f"{k}_{l}"].astype("<f4").tobytes())
        for k in MATS:
            q, s = int8_rows(t[f"{k}_{l}"]); f.write(s.astype("<f4").tobytes()); f.write(q.tobytes())
    f.close(); print("wrote", out, Path(out).stat().st_size, "bytes")

def cmd_agree(project, book, n, key):
    t, vocab, info = load(project, key); q = quantized(t, info["n_layers"])
    ids = encode(vocab, open(book, encoding="utf-8", errors="ignore").read())[200000:200000 + n]
    ids = ids[:info["context"]]            # one window: no rebuild involved
    a, b = Transformer(t, info), Transformer(q, info); la = lb = 0.0; agree = 0
    for i in range(len(ids) - 1):
        pa, pb = a.step(ids[i], i), b.step(ids[i], i)
        la -= (pa - pa.max() - np.log(np.exp(pa - pa.max()).sum()))[ids[i + 1]]
        lb -= (pb - pb.max() - np.log(np.exp(pb - pb.max()).sum()))[ids[i + 1]]
        agree += int(np.argmax(pa) == np.argmax(pb))
    m = len(ids) - 1
    print(f"{m} characters: loss float16 {la / m:.4f}, loss 8-bit {lb / m:.4f}; same most likely next character {agree / m * 100:.1f}%")

if __name__ == "__main__":
    a = sys.argv[1:]; key = "tx-paired"
    if "--model" in a: i = a.index("--model"); key = a[i + 1]; del a[i:i + 2]
    if a[0] == "check": cmd_check(a[1], a[2] if len(a) > 2 else key)
    elif a[0] == "export": cmd_export(a[1], a[2], a[3] if len(a) > 3 else key)
    elif a[0] == "greedy":
        t, vocab, info = load(a[1], key)
        if "--int8" in a: t = quantized(t, info["n_layers"])
        n = int(a[3]) if len(a) > 3 and a[3].isdigit() else 64
        print(greedy(Transformer(t, info), vocab, a[2], n))
    elif a[0] == "agree": cmd_agree(a[1], a[2], int(a[3]) if len(a) > 3 else 120, key)

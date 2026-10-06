#!/usr/bin/env python3
"""A second, independent implementation of the stories260K model, in numpy, for checking the board.

Usage:
  python3 tools/slm_reference.py MODEL.bin TOKENIZER.bin ["prompt"] [--steps 64]

Reads the same files as part3_slm/07_slm_stories, runs greedy decoding (always the most
likely token) and prints the token numbers and the text. The sketch's serial command
`g <prompt>` prints the token numbers it gets, which must be the same, one by one.
This script follows the llama2.c checkpoint format (https://github.com/karpathy/llama2.c, MIT).
"""
import struct, sys
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 64
if "--steps" in sys.argv: args.remove(sys.argv[sys.argv.index("--steps") + 1])
model, tok = args[0], args[1]
prompt = args[2] if len(args) > 2 else ""

raw = open(model, "rb").read()
dim, hidden, L, nh, nkv, vocab, seq = struct.unpack("7i", raw[:28])
shared = vocab > 0
vocab = abs(vocab)
hs = dim // nh
kv = dim * nkv // nh
w = np.frombuffer(raw, dtype="<f4", offset=28)
pos = 0
def take(*shape):
    global pos
    n = int(np.prod(shape)); a = w[pos:pos + n].reshape(shape); pos += n; return a
emb = take(vocab, dim); rms_att = take(L, dim)
wq = take(L, dim, dim); wk = take(L, kv, dim); wv = take(L, kv, dim); wo = take(L, dim, dim)
rms_ffn = take(L, dim); w1 = take(L, hidden, dim); w2 = take(L, dim, hidden); w3 = take(L, hidden, dim)
rms_fin = take(dim); take(seq, hs // 2); take(seq, hs // 2)
wcls = emb if shared else take(vocab, dim)

# Tokenizer
t = open(tok, "rb").read()
maxlen = struct.unpack("i", t[:4])[0]; o = 4
pieces, scores = [], []
for _ in range(vocab):
    sc, ln = struct.unpack("fi", t[o:o + 8]); o += 8
    pieces.append(t[o:o + ln].decode("utf-8", "replace") if ln else ""); o += ln; scores.append(sc)

def encode(text):
    ids = [1]
    if text: ids.append(pieces.index(" "))
    for ch in text:
        ids.append(pieces.index(ch) if ch in pieces else ord(ch) + 3)
    while True:
        best, bi, bidx = -1e10, -1, -1
        for i in range(len(ids) - 1):
            s = pieces[ids[i]] + pieces[ids[i + 1]]
            if s in pieces and scores[pieces.index(s)] > best:
                best, bi, bidx = scores[pieces.index(s)], pieces.index(s), i
        if bidx < 0: break
        ids[bidx:bidx + 2] = [bi]
    return ids

def rms(x, wt): return wt * (x / np.sqrt(np.mean(x * x) + 1e-5))
kc = np.zeros((L, seq, kv), np.float32); vc = np.zeros((L, seq, kv), np.float32)

def forward(token, p):
    x = emb[token].astype(np.float32).copy()
    for l in range(L):
        xb = rms(x, rms_att[l])
        q = wq[l] @ xb; k = wk[l] @ xb; v = wv[l] @ xb
        for i in range(0, dim, 2):
            f = 1.0 / 10000 ** ((i % hs) / hs); c, s = np.cos(p * f), np.sin(p * f)
            vecs = (q, k) if i < kv else (q,)
            for vec in vecs:
                a, b = vec[i], vec[i + 1]; vec[i] = a * c - b * s; vec[i + 1] = a * s + b * c
        kc[l, p] = k; vc[l, p] = v
        out = np.zeros(dim, np.float32)
        for h in range(nh):
            g = h // (nh // nkv)
            sc = kc[l, :p + 1, g * hs:(g + 1) * hs] @ q[h * hs:(h + 1) * hs] / np.sqrt(hs)
            e = np.exp(sc - sc.max()); e /= e.sum()
            out[h * hs:(h + 1) * hs] = e @ vc[l, :p + 1, g * hs:(g + 1) * hs]
        x = x + wo[l] @ out
        xb = rms(x, rms_ffn[l])
        a = w1[l] @ xb; b = w3[l] @ xb
        x = x + w2[l] @ (a / (1 + np.exp(-a)) * b)
    return wcls @ rms(x, rms_fin)

ids = encode(prompt)
tok_in, out_ids, text, prev = ids[0], [], "", 0
for p in range(min(steps, seq)):
    lg = forward(tok_in, p)
    nxt = ids[p + 1] if p < len(ids) - 1 else int(np.argmax(lg))
    if nxt == 1 and p + 1 >= len(ids): break
    out_ids.append(nxt)
    piece = pieces[nxt]
    if tok_in == 1 and piece.startswith(" "): piece = piece[1:]
    text += piece; tok_in = nxt
print("IDS", " ".join(map(str, out_ids)))
print(text)

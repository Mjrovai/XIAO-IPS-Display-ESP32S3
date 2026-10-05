#!/usr/bin/env python3
"""Live keyword test with ground truth, using audible cues (macOS: needs `afplay`).

  python3 tools/kws_cued_test.py run PORT run.json [--per-class 10] [--seed 7]
  python3 tools/kws_cued_test.py analyze run.json [--threshold 0.8]

`run` builds a sound file of beeps and plays it on the computer's speakers, while it logs
what the board (running part2_tinyml/04_kws_inference) answers, with the time of every answer.
The beep pattern tells you which word to say, right after the beeps:
    1 low beep   -> say YES
    2 mid beeps  -> say NO
    3 high beeps -> say any OTHER word (not yes or no)
The cues come in a random order, 5 seconds apart, so the sequence you were asked for is
known exactly. `analyze` matches every detection (YES or NO winning with at least the
threshold, the rule of 04_kws_inference) to the word that was asked, and counts hits,
misses, wrong words, and false alarms, with 95% confidence intervals.

The beeps are heard by the board's microphone too. Detections that happen during the
beeps are counted separately, so you can see if the cue itself fools the model.
Standard library only.
"""
import argparse
import json
import math
import os
import random
import re
import select
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import wave

SR = 44100
BEEP, BEEP_GAP = 0.18, 0.12
INTRO, SLOT, TAIL = 5.0, 5.0, 3.0
WORD_AFTER_BEEPS, WORD_LEN = 0.15, 2.2
PATTERNS = {"yes": (1, 600.0), "no": (2, 900.0), "other": (3, 1300.0)}
LATENCY = 0.04  # a slice's answer arrives about this long after the slice ended


def make_plan(per_class, seed):
    rng = random.Random(seed)
    order = ["yes"] * per_class + ["no"] * per_class + ["other"] * per_class
    for _ in range(2000):  # no more than two of the same cue in a row
        rng.shuffle(order)
        if all(not (order[i] == order[i + 1] == order[i + 2]) for i in range(len(order) - 2)):
            break
    cues, t = [], INTRO
    for c in order:
        k, _ = PATTERNS[c]
        beeps = [[t + i * (BEEP + BEEP_GAP), t + i * (BEEP + BEEP_GAP) + BEEP] for i in range(k)]
        ws = beeps[-1][1] + WORD_AFTER_BEEPS
        cues.append({"class": c, "beeps": beeps, "word": [ws, ws + WORD_LEN]})
        t += SLOT
    return cues, t + TAIL


def render_wav(cues, total, path):
    n = int(total * SR)
    buf = [0.0] * n
    for cue in cues:
        freq = PATTERNS[cue["class"]][1]
        for b0, b1 in cue["beeps"]:
            i0, i1 = int(b0 * SR), int(b1 * SR)
            for i in range(i0, i1):
                fade = min(1.0, (i - i0) / (0.01 * SR), (i1 - i) / (0.01 * SR))
                buf[i] += 0.30 * fade * math.sin(2 * math.pi * freq * (i - i0) / SR)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", max(-32767, min(32767, int(x * 32767)))) for x in buf))


RX = re.compile(r"slice (\d+) \| no ([\d.]+) noise ([\d.]+) unknown ([\d.]+) yes ([\d.]+)(?: \| dsp \d+ ms nn \d+ ms \| level (-?\d+) dBFS)?")


class Capture(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(self.fd); tty.cfmakeraw(a); a[4] = a[5] = termios.B115200
        termios.tcsetattr(self.fd, termios.TCSANOW, a)
        self.records, self.seen, self.stop_flag, self.text = [], set(), False, ""

    def run(self):
        while not self.stop_flag:
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if not ready:
                continue
            try:
                chunk = os.read(self.fd, 8192)
            except BlockingIOError:
                continue
            now = time.monotonic()
            self.text += chunk.decode(errors="replace")
            for m in RX.finditer(self.text):
                n = int(m[1])
                if n in self.seen:
                    continue
                self.seen.add(n)
                self.records.append([now, n, float(m[2]), float(m[3]), float(m[4]), float(m[5]),
                                     int(m[6]) if m[6] is not None else None])
            self.text = self.text[-300:]


def run(port, out, per_class, seed):
    cues, total = make_plan(per_class, seed)
    wav = os.path.join(tempfile.mkdtemp(), "cues.wav")
    render_wav(cues, total, wav)
    cap = Capture(port)
    cap.start()
    time.sleep(2.0)
    print(f"{len(cues)} cues, {total:.0f} s. Say the word right after the beeps: 1 low = YES, 2 mid = NO, 3 high = OTHER.")
    t0 = time.monotonic()
    subprocess.run(["afplay", wav])
    time.sleep(1.5)
    cap.stop_flag = True
    cap.join(1.0)
    recs = [[r[0] - t0] + r[1:] for r in cap.records]
    json.dump({"plan": cues, "total": total, "records": recs, "seed": seed}, open(out, "w"))
    print(f"saved {len(recs)} answers to {out}")


def pearson(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0.0


def overlap(a0, a1, ints):
    return sum(max(0.0, min(a1, b1) - max(a0, b0)) for b0, b1 in ints)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def analyze(path, thr):
    run_ = json.load(open(path))
    cues, recs = run_["plan"], run_["records"]
    beeps = [b for c in cues for b in c["beeps"]]
    # Find how late the answers are, by matching the loudness of each slice with the beeps.
    best = (-2.0, 0.0)
    lv = [r for r in recs if r[6] is not None]
    for k in range(0, 80):
        shift = k * 0.02
        xs = [r[6] for r in lv]
        ys = [overlap(r[0] - shift - LATENCY - 0.25, r[0] - shift - LATENCY, beeps) / 0.25 for r in lv]
        c = pearson(xs, ys)
        if c > best[0]:
            best = (c, shift)
    corr, shift = best
    print(f"time alignment: the answers lag the sound by {shift:.2f} s (correlation of loudness with the beeps: {corr:.2f})")
    if corr < 0.3:
        print("  warning: weak alignment, the beeps may have been too quiet for the microphone; results may be unreliable")
    end = lambda r: r[0] - shift - LATENCY
    out = {"yes": {}, "no": {}, "other": {}}
    rows, claimed = [], []
    beep_alarms = 0
    for i, c in enumerate(cues):
        ws, we = c["word"]
        win = (ws + 0.40, we + 0.80)          # answers that can reflect this word
        beepwin = (c["beeps"][0][0], ws + 0.40)  # answers that still contain the beeps
        claimed.append((beepwin[0], win[1]))
        fired, best_score = set(), {"yes": 0.0, "no": 0.0}
        for r in recs:
            if win[0] <= end(r) <= win[1]:
                sc = {"no": r[2], "noise": r[3], "unknown": r[4], "yes": r[5]}
                top = max(sc, key=sc.get)
                if top in ("yes", "no") and sc[top] >= thr:
                    fired.add(top); best_score[top] = max(best_score[top], sc[top])
        for r in recs:
            if beepwin[0] <= end(r) < beepwin[1]:
                sc = {"no": r[2], "noise": r[3], "unknown": r[4], "yes": r[5]}
                top = max(sc, key=sc.get)
                if top in ("yes", "no") and sc[top] >= thr:
                    beep_alarms += 1; break
        word = "both" if len(fired) == 2 else (next(iter(fired)) if fired else "nothing")
        rows.append((i + 1, c["class"], word, best_score))
        tbl = out[c["class"]]
        tbl[word] = tbl.get(word, 0) + 1
    stray = 0
    for r in recs:
        if not any(a <= end(r) <= b for a, b in claimed):
            sc = {"no": r[2], "noise": r[3], "unknown": r[4], "yes": r[5]}
            top = max(sc, key=sc.get)
            if top in ("yes", "no") and sc[top] >= thr:
                stray += 1
    print(f"\nthreshold {thr}: for each cue, the word the board showed (YES or NO winning with >= {thr})\n")
    print(f"{'asked for':10s} {'YES':>5s} {'NO':>5s} {'both':>5s} {'nothing':>8s}   result")
    for cls in ("yes", "no", "other"):
        t = out[cls]; n = sum(t.values())
        if n == 0:
            continue
        cols = [t.get("yes", 0), t.get("no", 0), t.get("both", 0), t.get("nothing", 0)]
        if cls == "other":
            k = t.get("nothing", 0); lo, hi = wilson(k, n)
            res = f"correct (no yes/no shown) {k}/{n} = {100 * k / n:.0f}%  [95% CI {100 * lo:.0f}-{100 * hi:.0f}%]   false alarms {n - k}"
        else:
            k = t.get(cls, 0); lo, hi = wilson(k, n)
            res = f"correct {k}/{n} = {100 * k / n:.0f}%  [95% CI {100 * lo:.0f}-{100 * hi:.0f}%]"
        print(f"{cls.upper():10s} " + "".join(f"{v:5d}" if j < 3 else f"{v:8d}" for j, v in enumerate(cols)) + "   " + res)
    # A looser reading: when both words flashed, take the one with the highest score.
    print("\nIf, when both words flashed, the one with the highest score counts as the answer:")
    for cls in ("yes", "no", "other"):
        mine = [r for r in rows if r[1] == cls]
        if not mine:
            continue
        def decide(r):
            w = r[2]
            if w == "both":
                w = "yes" if r[3]["yes"] >= r[3]["no"] else "no"
            return w
        ok = sum(1 for r in mine if (decide(r) == cls if cls != "other" else decide(r) == "nothing"))
        lo, hi = wilson(ok, len(mine))
        print(f"   {cls.upper():6s} correct {ok}/{len(mine)} = {100 * ok / len(mine):.0f}%  [95% CI {100 * lo:.0f}-{100 * hi:.0f}%]")
    print(f"\ndetections during the beeps themselves (the cue fooling the model): {beep_alarms} of {len(cues)} cues")
    print(f"stray detections outside any cue (the quiet gaps and the intro/tail): {stray}")
    bad = [r for r in rows if (r[1] == "yes" and r[2] != "yes") or (r[1] == "no" and r[2] != "no") or (r[1] == "other" and r[2] != "nothing")]
    print(f"\ncues that did not go as asked ({len(bad)}):")
    for n, cls, word, sc in bad:
        print(f"   cue {n:2d}: asked {cls.upper():5s} -> showed {word:7s} (best yes {sc['yes']:.2f}, no {sc['no']:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("port"); r.add_argument("out")
    r.add_argument("--per-class", type=int, default=10); r.add_argument("--seed", type=int, default=7)
    a = sub.add_parser("analyze"); a.add_argument("run"); a.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.port, args.out, args.per_class, args.seed)
    else:
        analyze(args.run, args.threshold)

#!/usr/bin/env python3
"""Review your own keyword clips before uploading them to Edge Impulse.

Usage: python3 tools/kws_clip_review.py ~/datasets/edge-impulse/own_by_class

Expects one sub-folder per class (yes, no, unknown, noise) of 1-second WAV clips
recorded by part2_tinyml/03_kws_recorder. For each word clip it estimates where the
word starts and ends inside the second, and it flags clips that look wrong: a
word that starts too early or is cut off at the end, a clip that clips (peaks at
full scale), one that is very quiet, or one with two separate bursts of sound.
It also reports the level and the headroom, to judge how much digital gain a
quiet microphone could take. Standard library only.
"""
import array
import math
import os
import statistics
import sys
import wave

FRAME = 320  # 20 ms at 16 kHz
root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/edge-impulse/own_by_class")


def load(path):
    with wave.open(path) as w:
        return array.array("h", w.readframes(w.getnframes()))


def frame_db(a):
    out = []
    for i in range(0, len(a) - FRAME + 1, FRAME):
        f = a[i:i + FRAME]
        m = sum(f) / FRAME
        rms = max(math.sqrt(sum((x - m) ** 2 for x in f) / FRAME), 1.0)
        out.append(20 * math.log10(rms / 32768))
    return out


def pct(sorted_vals, p):
    return sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))]


for cls in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
    files = sorted(f for f in os.listdir(os.path.join(root, cls)) if f.endswith(".wav"))
    rms_all, peak_all, onsets, ends, flagged = [], [], [], [], []
    for name in files:
        a = load(os.path.join(root, cls, name))
        m = sum(a) / len(a)
        rms = max(math.sqrt(sum((x - m) ** 2 for x in a) / len(a)), 1.0)
        rms_db = 20 * math.log10(rms / 32768)
        peak_db = 20 * math.log10(max(max(abs(x) for x in a), 1) / 32768)
        rms_all.append(rms_db)
        peak_all.append(peak_db)
        if cls == "noise":
            continue
        fdb = frame_db(a)
        top = max(fdb)                      # the loudest 20 ms frame: the heart of the word
        floor = pct(sorted(fdb), 0.10)
        # A frame belongs to the word if it is within 20 dB of the loudest frame and
        # at least 12 dB above the room. Measuring against the word's own peak keeps
        # room noise from counting as part of the word.
        limit = max(top - 20, floor + 12)
        active = [i for i, d in enumerate(fdb) if d > limit]
        why = []
        if top < floor + 12:
            why.append("no clear word")
        else:
            on, off = active[0] * 20, (active[-1] + 1) * 20
            onsets.append(on)
            ends.append(off)
            if fdb[0] > top - 15:
                why.append(f"already loud at the start ({on} ms): the word may begin before the clip")
            if fdb[-1] > top - 15:
                why.append("still loud at the end: the word may be cut off")
            # two bursts: a quiet gap of 200 ms or more between two loud stretches
            loud = [i for i, d in enumerate(fdb) if d > top - 15]
            if any((b - a_) * 20 >= 200 for a_, b in zip(loud, loud[1:])):
                why.append("two bursts")
        if peak_db >= -0.5:
            why.append("clipped")
        if rms_db < -55:
            why.append(f"very quiet ({rms_db:.0f} dBFS)")
        if why:
            flagged.append((name, ", ".join(why)))
    rs, ps = sorted(rms_all), sorted(peak_all)
    print(f"\n{cls}: {len(files)} clips")
    print(f"  RMS dBFS   p10 {pct(rs, .1):6.1f}  median {statistics.median(rs):6.1f}  p90 {pct(rs, .9):6.1f}")
    print(f"  peak dBFS  p10 {pct(ps, .1):6.1f}  median {statistics.median(ps):6.1f}  p90 {pct(ps, .9):6.1f}  max {ps[-1]:.1f}")
    if onsets:
        print(f"  word start (ms) median {statistics.median(onsets):.0f}  (the recorder aims for about 300)")
        print(f"  word end   (ms) median {statistics.median(ends):.0f}")
    print(f"  flagged: {len(flagged)}")
    for n, w in flagged:
        print(f"    {n}: {w}")

#!/usr/bin/env python3
"""Check a keyword-spotting dataset folder before uploading it to Edge Impulse.

Usage: python3 tools/kws_dataset_check.py ~/datasets/edge-impulse/keywords2

Expects one sub-folder per class (yes, no, noise, unknown), each with .wav files
named <label>.<anything>.wav. Reports, per class: the number of files, empty or
unreadable files, the formats found, the durations found, and the level (RMS in
dBFS) of a random sample. Uses only the Python standard library.
"""
import array
import collections
import glob
import math
import os
import random
import statistics
import sys
import wave

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/datasets/edge-impulse/keywords2")
classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and not d.startswith("_"))
random.seed(1)
problems = 0

for cls in classes:
    files = sorted(glob.glob(os.path.join(root, cls, "*.wav")))
    formats, durations = collections.Counter(), collections.Counter()
    empty = unreadable = wrong_label = 0
    for f in files:
        if os.path.getsize(f) == 0:
            empty += 1
            continue
        if not os.path.basename(f).startswith(cls + "."):
            wrong_label += 1
        try:
            with wave.open(f) as w:
                formats[(w.getframerate(), w.getsampwidth() * 8, w.getnchannels())] += 1
                durations[round(w.getnframes() / w.getframerate(), 2)] += 1
        except Exception:
            unreadable += 1
    levels, clipped = [], 0
    for f in random.sample(files, min(300, len(files))):
        if os.path.getsize(f) == 0:
            continue
        with wave.open(f) as w:
            a = array.array("h", w.readframes(w.getnframes()))
        mean = sum(a) / len(a)
        rms = max(math.sqrt(sum((x - mean) ** 2 for x in a) / len(a)), 1.0)
        levels.append(20 * math.log10(rms / 32768))
        clipped += max(abs(x) for x in a) >= 32700
    levels.sort()
    print(f"{cls}: {len(files)} files, {empty} empty, {unreadable} unreadable, {wrong_label} with a name that does not start with '{cls}.'")
    print(f"  formats (Hz, bits, channels): {dict(formats)}")
    print(f"  durations (s): {dict(sorted(durations.items()))}")
    if levels:
        n = len(levels)
        print(f"  level of a {n}-file sample, RMS dBFS: p10 {levels[n // 10]:.1f}, median {statistics.median(levels):.1f}, "
              f"p90 {levels[n * 9 // 10]:.1f}; clipped files {clipped}")
    problems += empty + unreadable + wrong_label

print("\nOK: nothing to fix" if problems == 0 else f"\n{problems} problem file(s) found")
sys.exit(1 if problems else 0)

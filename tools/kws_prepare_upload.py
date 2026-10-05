#!/usr/bin/env python3
"""Gather your own keyword clips into train / test / review folders for Edge Impulse.

Usage:
  python3 tools/kws_prepare_upload.py OUT --source DIR [--source DIR ...] \
      [--review name1.wav name2.wav ...] [--test-fraction 0.33] [--seed 1]

Each --source holds one sub-folder per class (yes, no, unknown, noise) of WAV
clips. The clips are COPIED, never moved, into
  OUT/train/<class>/   about two thirds of the clips
  OUT/test/<class>/    about one third, to measure the model on your own voice
  OUT/review/<class>/  clips you listed with --review, to listen to first
When two sources use the same file name (the recorder numbers clips from 001 in
every session), the later one gets its folder name in the file name, e.g.
noise.own2.001.wav. The part before the first dot is still the label, which is how
the Studio infers it. Standard library only.
"""
import argparse
import os
import random
import shutil
import sys

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--source", action="append", required=True)
ap.add_argument("--review", nargs="*", default=[])
ap.add_argument("--test-fraction", type=float, default=1 / 3)
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

out = os.path.expanduser(args.out)
review_names = set(args.review)
if os.path.exists(out) and os.listdir(out):
    sys.exit(f"{out} is not empty; choose a new folder so nothing is overwritten")

clips = {}  # class -> list of (final name, source path, original name)
used = set()
for src in args.source:
    src = os.path.expanduser(src)
    tag = os.path.basename(src.rstrip("/"))
    for cls in sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))):
        for name in sorted(f for f in os.listdir(os.path.join(src, cls)) if f.endswith(".wav")):
            final = name
            if (cls, final) in used:  # same name from an earlier source: add the source tag
                label, rest = name.split(".", 1)
                final = f"{label}.{tag}.{rest.split('.', 1)[1] if rest.startswith('own.') else rest}"
            used.add((cls, final))
            clips.setdefault(cls, []).append((final, os.path.join(src, cls, name), name))

rng = random.Random(args.seed)
print(f"{'class':9s} {'train':>6s} {'test':>6s} {'review':>7s} {'total':>6s}")
for cls, items in sorted(clips.items()):
    review = [c for c in items if c[2] in review_names]
    rest = [c for c in items if c[2] not in review_names]
    rng.shuffle(rest)
    n_test = round(len(rest) * args.test_fraction)
    parts = {"test": rest[:n_test], "train": rest[n_test:], "review": review}
    for part, group in parts.items():
        d = os.path.join(out, part, cls)
        os.makedirs(d, exist_ok=True)
        for final, path, _ in group:
            shutil.copy2(path, os.path.join(d, final))
    print(f"{cls:9s} {len(parts['train']):6d} {len(parts['test']):6d} {len(parts['review']):7d} {len(items):6d}")

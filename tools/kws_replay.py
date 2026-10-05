#!/usr/bin/env python3
"""Send WAV clips to the board running part2_tinyml/05_kws_replay and score the answers.

Usage:
  python3 tools/kws_replay.py PORT FOLDER [FOLDER ...] [--gain-db 0 6 12] [--limit 100]
                              [--csv results.csv]

Each FOLDER holds one sub-folder per class (yes, no, unknown, noise) of 1-second,
16 kHz, 16-bit mono WAV clips. The class is the sub-folder name. For every gain the
clips are scaled in software (and limited to the 16-bit range) before they are sent,
so you can ask: would a louder microphone signal help? Prints, for each gain, the
accuracy per class and the confusion matrix. Standard library only (macOS/Linux).
"""
import argparse
import array
import collections
import os
import random
import select
import sys
import termios
import time
import tty
import wave

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("folders", nargs="+")
ap.add_argument("--gain-db", nargs="*", type=float, default=[0.0])
ap.add_argument("--limit", type=int, default=0, help="at most this many clips per class and folder (0 = all)")
ap.add_argument("--csv", default="")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--host", default="", help="path of a program built from the same library for this computer (see libs/ei-host-test); PORT is then ignored")
args = ap.parse_args()

LABELS = ["no", "noise", "unknown", "yes"]
fd = None
if not args.host:
    fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    tty.cfmakeraw(attrs)
    attrs[4] = attrs[5] = termios.B115200
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def read_until(token, timeout):
    buf, end = b"", time.time() + timeout
    while time.time() < end:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if ready:
            try:
                buf += os.read(fd, 4096)
            except BlockingIOError:
                pass
            if token in buf and buf.rstrip().endswith(b"\n") or (token in buf and b"\n" in buf[buf.index(token):]):
                return buf.decode(errors="replace")
    return buf.decode(errors="replace")


def send(data):
    view, sent = memoryview(data), 0
    while sent < len(data):
        try:
            sent += os.write(fd, view[sent:sent + 4096])
        except BlockingIOError:
            select.select([], [fd], [], 0.2)


def load(path):
    with wave.open(path) as w:
        assert (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (16000, 2, 1), path
        return array.array("h", w.readframes(w.getnframes()))


rng = random.Random(args.seed)
clips = []
for folder in args.folders:
    folder = os.path.expanduser(folder)
    for cls in sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))):
        files = sorted(f for f in os.listdir(os.path.join(folder, cls)) if f.endswith(".wav"))
        if args.limit and len(files) > args.limit:
            files = rng.sample(files, args.limit)
        clips += [(cls, os.path.join(folder, cls, f)) for f in files]

# make sure the board is listening
if fd is not None:
    os.write(fd, b"\n")
print("clips:", collections.Counter(c for c, _ in clips))
rows = []
corrupted = 0
for gain_db in args.gain_db:
    g = 10 ** (gain_db / 20)
    conf = collections.defaultdict(collections.Counter)
    t0 = time.time()
    if args.host:  # run the same clips through the program built for this computer
        import subprocess
        out = subprocess.run([args.host, str(gain_db)], input="\n".join(p for _, p in clips) + "\n",
                             capture_output=True, text=True).stdout
        answers = {}
        for l in out.splitlines():
            path, _, rest = l.partition("\t")
            t = rest.split()
            if len(t) >= 8 and t[0] != "ERROR":
                answers[path] = {t[i]: float(t[i + 1]) for i in range(0, 8, 2)}
        for cls, path in clips:
            sc = answers.get(path)
            if not sc:
                conf[cls]["(no answer)"] += 1
                continue
            top = max(sc, key=sc.get)
            conf[cls][top] += 1
            rows.append((gain_db, cls, os.path.basename(path), top, *(sc[k] for k in LABELS)))
        clips_for_loop = []
    else:
        clips_for_loop = clips
    for cls, path in clips_for_loop:
        a = load(path)
        if len(a) != 16000:
            continue
        if gain_db:
            a = array.array("h", (max(-32768, min(32767, int(x * g))) for x in a))
        send(b"WAVE" + a.tobytes())
        line = ""
        end = time.time() + 5
        buf = b""
        while time.time() < end and b"\n" not in buf:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                try:
                    buf += os.read(fd, 4096)
                except BlockingIOError:
                    pass
        for l in buf.decode(errors="replace").splitlines():
            if l.startswith("RESULT"):
                line = l
        if not line:
            conf[cls]["(no answer)"] += 1
            continue
        parts = line.split()[1:]
        scores = {parts[i]: float(parts[i + 1]) for i in range(0, 8, 2)}
        extra = dict(zip(parts[8::2], parts[9::2]))
        if "sum" in extra and (int(extra["sum"]), int(extra["peak"])) != (sum(a), max(abs(x) for x in a)):
            corrupted += 1
        top = max(scores, key=scores.get)
        conf[cls][top] += 1
        rows.append((gain_db, cls, os.path.basename(path), top, *(scores[k] for k in LABELS)))
    total = sum(sum(c.values()) for c in conf.values())
    right = sum(conf[c][c] for c in conf)
    print(f"\n(clips whose checksum differed on the board so far: {corrupted})")
    print(f"=== gain +{gain_db:g} dB: {right}/{total} correct = {100 * right / max(total, 1):.1f}%   ({time.time() - t0:.0f} s)")
    print(f"{'true \\ predicted':18s}" + "".join(f"{l:>9s}" for l in LABELS) + "   recall")
    for cls in sorted(conf):
        n = sum(conf[cls].values())
        print(f"{cls:18s}" + "".join(f"{conf[cls][l]:9d}" for l in LABELS) + f"   {100 * conf[cls][cls] / max(n, 1):5.1f}%  (n={n})")
if args.csv:
    with open(args.csv, "w") as f:
        f.write("gain_db,true,file,predicted," + ",".join(LABELS) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
if fd is not None:
    os.close(fd)

#!/usr/bin/env python3
"""Score the keyword model in CONTINUOUS mode, the way the live sketch uses it.

Usage: python3 tools/kws_stream_eval.py HOST_TEST NOISE.wav FOLDER [FOLDER ...]
                                       [--threshold 0.8] [--limit 100] [--gain-db 0] [--port PORT]

With --port, the slices are sent to the board running part2_tinyml/05_kws_replay instead,
so the continuous classifier is measured on the device (HOST_TEST is then ignored).

HOST_TEST is the program built from libs/ei-host-test (see tools/ei_host_test). Each clip
is placed between two copies of NOISE.wav and fed in 250 ms slices to the continuous
classifier. A clip is "detected as yes/no" if, in any slice, yes or no wins with at least
the threshold (the rule in 04_kws_inference). Reports, per class, how many clips fired
which word. For the yes and no folders the right answer is that word; for noise and
unknown folders, any detection is a false alarm. Standard library only.
"""
import argparse
import collections
import os
import random
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("host")
ap.add_argument("noise")
ap.add_argument("folders", nargs="+")
ap.add_argument("--threshold", type=float, default=0.8)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gain-db", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--port", default="")
args = ap.parse_args()

rng = random.Random(args.seed)
clips = []
for folder in args.folders:
    folder = os.path.expanduser(folder)
    for cls in sorted(d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))):
        files = sorted(f for f in os.listdir(os.path.join(folder, cls)) if f.endswith(".wav"))
        if args.limit and len(files) > args.limit:
            files = rng.sample(files, args.limit)
        clips += [(cls, os.path.join(folder, cls, f)) for f in files]

def board_stream():
    import array, select, termios, time, tty, wave
    fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    a = termios.tcgetattr(fd); tty.cfmakeraw(a); a[4] = a[5] = termios.B115200; termios.tcsetattr(fd, termios.TCSANOW, a)

    def wav(path):
        with wave.open(path) as w:
            return array.array("h", w.readframes(w.getnframes()))

    def ask(payload, token, timeout=5.0):
        view, sent = memoryview(payload), 0
        while sent < len(payload):
            try:
                sent += os.write(fd, view[sent:sent + 4096])
            except BlockingIOError:
                select.select([], [fd], [], 0.2)
        buf, end = b"", time.time() + timeout
        while time.time() < end and token not in buf.decode(errors="replace").split("\n")[-2:-1] and b"\n" not in buf:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                try: buf += os.read(fd, 4096)
                except BlockingIOError: pass
        return buf.decode(errors="replace")

    noise = wav(os.path.expanduser(args.noise))
    g = 10 ** (args.gain_db / 20)
    lines = []
    for cls, path in clips:
        stream = noise + wav(path) + noise
        if args.gain_db:
            stream = array.array("h", (max(-32768, min(32767, int(x * g))) for x in stream))
        ask(b"RSET", "RESET")
        got = []
        for off in range(0, len(stream) - 3999, 4000):
            reply = ask(b"SLCE" + stream[off:off + 4000].tobytes(), "SLICE")
            for l in reply.splitlines():
                if l.startswith("SLICE"):
                    t = l.split()[1:]
                    got.append("[" + " ".join(t) + "]")
        lines.append(path + "\t" + " ".join(got[3:]))   # drop the slices before the first full window
    os.close(fd)
    return "\n".join(lines)


if args.port:
    out = board_stream()
else:
    out = subprocess.run([args.host, str(args.gain_db), "stream", os.path.expanduser(args.noise)],
                         input="\n".join(p for _, p in clips) + "\n", capture_output=True, text=True).stdout
answers = {}
for line in out.splitlines():
    path, _, rest = line.partition("\t")
    slices = re.findall(r"\[(.*?)\]", rest)
    best = {"yes": 0.0, "no": 0.0}
    for s in slices:
        t = s.split()
        sc = {t[i]: float(t[i + 1]) for i in range(0, len(t), 2)}
        top = max(sc, key=sc.get)
        if top in best and sc[top] >= args.threshold:
            best[top] = max(best[top], sc[top])
    fired = [w for w in best if best[w] > 0]
    answers[path] = "both" if len(fired) == 2 else (fired[0] if fired else "nothing")

table = collections.defaultdict(collections.Counter)
for cls, path in clips:
    table[cls][answers.get(path, "no answer")] += 1
cols = ["yes", "no", "both", "nothing"]
print(f"threshold {args.threshold}, gain {args.gain_db:+g} dB, clips: {sum(sum(c.values()) for c in table.values())}")
print(f"{'class':10s}" + "".join(f"{c:>9s}" for c in cols) + "   result")
for cls in sorted(table):
    n = sum(table[cls].values())
    if cls in ("yes", "no"):
        res = f"detected correctly {100 * table[cls][cls] / n:5.1f}%   (missed {100 * table[cls]['nothing'] / n:5.1f}%, wrong word {100 * (table[cls]['no' if cls == 'yes' else 'yes'] + table[cls]['both']) / n:5.1f}%)"
    else:
        res = f"false alarms {100 * (n - table[cls]['nothing']) / n:5.1f}%"
    print(f"{cls:10s}" + "".join(f"{table[cls][c]:9d}" for c in cols) + f"   {res}   (n={n})")

#!/usr/bin/env python3
"""Send windows from the motion CSV files to the board running part2_tinyml/06_motion_inference.

Usage:
  python3 tools/motion_replay.py PORT FOLDER [--stride 50] [--files 4] [--csv out.csv]

FOLDER holds one sub-folder per class (idle, lift, maritime, terrestrial) with the CSV
files written by 01_imu_logger. Each file is cut into 2-second windows (100 samples of
accX, accY, accZ, gyrX, gyrY, gyrZ) every STRIDE samples (50 = 1 s) and each window is
sent to the board, which answers with its class scores. The class is the folder name.
Prints the confusion matrix and the accuracy per class. Standard library only.
"""
import argparse, array, collections, csv, os, select, struct, sys, termios, time, tty
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("folder")
ap.add_argument("--stride", type=int, default=50)
ap.add_argument("--files", type=int, default=0, help="at most this many files per class (0 = all)")
ap.add_argument("--csv", default="")
args = ap.parse_args()

LABELS = ["idle", "lift", "maritime", "terrestrial"]
fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
a = termios.tcgetattr(fd); tty.cfmakeraw(a); a[4] = a[5] = termios.B115200
termios.tcsetattr(fd, termios.TCSANOW, a)


def read_line(token, timeout):
    buf, end = b"", time.time() + timeout
    while time.time() < end:
        if select.select([fd], [], [], 0.1)[0]:
            try: buf += os.read(fd, 4096)
            except BlockingIOError: pass
            for l in buf.decode(errors="replace").split("\n")[:-1]:
                if l.startswith(token): return l.strip()
    return None


def send(data):
    off = 0
    while off < len(data):
        select.select([], [fd], [], 1)
        try: off += os.write(fd, data[off:off + 512])
        except BlockingIOError: time.sleep(0.005)


os.write(fd, b"\n"); time.sleep(0.3)
try: os.read(fd, 65536)
except BlockingIOError: pass

conf = collections.defaultdict(collections.Counter)
rows = []
for cls in LABELS:
    files = sorted((Path(args.folder).expanduser() / cls).glob("*.csv"))
    if args.files: files = files[:args.files]
    for f in files:
        data = [[float(v) for v in r[1:7]] for r in list(csv.reader(open(f)))[1:]]
        for s in range(0, len(data) - 100 + 1, args.stride):
            win = [x for row in data[s:s + 100] for x in row]
            send(b"WIND" + struct.pack("<600f", *win))
            line = read_line("RESULT", 5)
            if not line: print("no answer for", f.name, s); continue
            p = line.split()
            sc = {p[i]: float(p[i + 1]) for i in range(1, 8, 2)}
            guess = max(sc, key=sc.get)
            conf[cls][guess] += 1
            rows.append([cls, f.name, s] + [sc[l] for l in LABELS] + [line.split("anomaly")[1].split()[0]])
print("\nrows = true class, columns = answer of the board")
print(f"{'':12}" + "".join(f"{l:>13}" for l in LABELS) + f"{'accuracy':>10}")
for cls in LABELS:
    n = sum(conf[cls].values())
    print(f"{cls:12}" + "".join(f"{conf[cls][l]:13d}" for l in LABELS) + f"{(conf[cls][cls] / n * 100 if n else 0):9.1f}%")
tot = sum(sum(c.values()) for c in conf.values()); ok = sum(conf[c][c] for c in conf)
print(f"\nwindows {tot}, correct {ok} ({ok / tot * 100:.1f}%)")
if args.csv:
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["class", "file", "start"] + LABELS + ["anomaly"]); w.writerows(rows)

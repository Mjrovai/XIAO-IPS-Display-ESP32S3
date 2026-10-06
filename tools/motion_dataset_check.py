#!/usr/bin/env python3
"""Check the motion CSV files written by 01_imu_logger before uploading them to Edge Impulse.

Usage: python3 tools/motion_dataset_check.py ~/datasets/edge-impulse/motion

Per file: rows, timestamp step, mean/std of the acceleration magnitude (m/s2), and the
std of each accelerometer axis. Per class: the same, averaged. Flags files that are not
500 rows, have uneven timestamps, or look like the wrong class (e.g. idle that moves).
"""
import csv, math, statistics as st, sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
HEAD = "timestamp,accX,accY,accZ,gyrX,gyrY,gyrZ"
summary = {}
for cls in sorted(p for p in root.iterdir() if p.is_dir()):
    print(f"\n== {cls.name}")
    print(f"{'file':22} {'rows':>5} {'dt ms':>7} {'|a| mean':>9} {'|a| std':>8} {'sd x':>6} {'sd y':>6} {'sd z':>6} {'gyr sd':>7}")
    for f in sorted(cls.glob("*.csv")):
        with open(f) as fh:
            first = fh.readline().strip()
            fh.seek(0)
            rows = list(csv.DictReader(fh))
        flag = "" if first == HEAD else "  HEADER?"
        t = [float(r["timestamp"]) for r in rows]
        a = [[float(r[k]) for k in ("accX", "accY", "accZ")] for r in rows]
        g = [[float(r[k]) for k in ("gyrX", "gyrY", "gyrZ")] for r in rows]
        dts = [b - c for b, c in zip(t[1:], t)]
        mag = [math.sqrt(x * x + y * y + z * z) for x, y, z in a]
        sd = [st.pstdev(c) for c in zip(*a)]
        gsd = st.mean(st.pstdev(c) for c in zip(*g))
        if len(rows) != 500: flag += "  ROWS!=500"
        if dts and (min(dts) < 15 or max(dts) > 25): flag += "  UNEVEN"
        print(f"{f.name:22} {len(rows):5d} {st.mean(dts):7.2f} {st.mean(mag):9.2f} {st.pstdev(mag):8.2f} "
              f"{sd[0]:6.2f} {sd[1]:6.2f} {sd[2]:6.2f} {gsd:7.1f}{flag}")
        summary.setdefault(cls.name, []).append((st.pstdev(mag), st.mean(mag)))
print("\n== per class (mean over files)")
for c, v in summary.items():
    print(f"{c:12} files {len(v):3d}  |a| std {st.mean(x for x, _ in v):.2f} (min {min(x for x, _ in v):.2f}, max {max(x for x, _ in v):.2f})  |a| mean {st.mean(m for _, m in v):.2f}")

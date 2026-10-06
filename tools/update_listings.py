#!/usr/bin/env python3
"""Refresh the code listings in README.md and part2_tinyml/README.md from the sketch files.

Each listing sits between two marker comments:

    <!-- sketch: part1_bringup/03_imu/03_imu.ino -->
    ...generated link and code block...
    <!-- /sketch -->

Run `python3 tools/update_listings.py` after editing a sketch. The README then
always shows the code that is actually compiled and tested.
"""
import re
import sys

REPO = "https://github.com/Mjrovai/XIAO-IPS-Display-ESP32S3"
BRANCH = "main"

FILES = ["README.md", "part2_tinyml/README.md", "part3_slm/README.md"]
pattern = re.compile(r"(<!-- sketch: (\S+) -->\n).*?(<!-- /sketch -->)", re.S)


def block(match):
    path = match.group(2)
    code = open(path, encoding="utf-8").read().rstrip("\n")
    link = f"{REPO}/blob/{BRANCH}/{path}"
    return (f"{match.group(1)}**Sketch:** [`{path}`]({link})\n\n"
            f"```cpp\n{code}\n```\n{match.group(3)}")


total = 0
for name in FILES:
    try:
        text = open(name, encoding="utf-8").read()
    except FileNotFoundError:
        continue
    new, count = pattern.subn(block, text)
    open(name, "w", encoding="utf-8").write(new)
    print(f"{name}: updated {count} listing(s)")
    total += count
sys.exit(0 if total else 1)

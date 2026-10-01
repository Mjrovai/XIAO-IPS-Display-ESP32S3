#!/usr/bin/env python3
"""Read a serial port for N seconds and print what arrives (macOS and Linux).

Usage: tools/serial_read.py /dev/cu.usbmodem101 10

Needs only the Python standard library. The ESP32-S3 USB serial port ignores
the baud rate, but 115200 is set anyway.
"""
import os
import select
import sys
import termios
import time
import tty

port, seconds = sys.argv[1], float(sys.argv[2])
fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
attrs = termios.tcgetattr(fd)
tty.cfmakeraw(attrs)
attrs[4] = attrs[5] = termios.B115200
termios.tcsetattr(fd, termios.TCSANOW, attrs)

end = time.time() + seconds
while time.time() < end:
    ready, _, _ = select.select([fd], [], [], 0.3)
    if ready:
        try:
            sys.stdout.write(os.read(fd, 4096).decode(errors="replace"))
            sys.stdout.flush()
        except BlockingIOError:
            pass
        except BrokenPipeError:  # e.g. piped into `head`
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            break
os.close(fd)

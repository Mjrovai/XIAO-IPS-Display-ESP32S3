#!/usr/bin/env python3
"""Build an Edge Impulse Arduino library as a program for this computer.

Usage: python3 tools/ei_host_test/build.py [LIBRARY_DIR] [OUT_DIR]

LIBRARY_DIR is the unzipped Arduino library that Edge Impulse generated (the folder that
holds src/ and library.properties). The default is libs/ei-kws/XIAO_IPS_Display_-_KWS_inferencing.
OUT_DIR (default libs/ei-host-test) gets the object files and the program `host_test`.

The program runs the model on WAV clips, on the computer, with the same Edge Impulse SDK
and the same compiled network as on the board. It exists to tell a model problem from a
board problem. Needs clang and clang++ (macOS or Linux). Standard library only.
"""
import concurrent.futures
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LIB = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "libs", "ei-kws", "XIAO_IPS_Display_-_KWS_inferencing"))
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "libs", "ei-host-test"))
SRC = os.path.join(LIB, "src")
os.makedirs(os.path.join(OUT, "obj"), exist_ok=True)

# The model, the signal-processing code, and the TensorFlow Lite kernels. Test helpers and the
# image code are left out; CMSIS and the Espressif folders are not needed on a computer.
found = []
for area in ("tflite-model", "edge-impulse-sdk/dsp", "edge-impulse-sdk/tensorflow"):
    for ext in ("*.cpp", "*.cc", "*.c"):
        found += glob.glob(os.path.join(SRC, area, "**", ext), recursive=True)
skip = ("test", "mock", "kernel_runner", "image" + os.sep + "processing")
srcs = sorted(f for f in found if not any(k in f.lower() for k in skip))
srcs += [os.path.join(HERE, "main.cpp"), os.path.join(HERE, "porting.cpp")]

common = ["-O2", "-w", "-DNDEBUG", "-DEI_PORTING_POSIX=1", "-DEI_PORTING_ARDUINO=0", "-DEI_PORTING_ESPRESSIF=0",
          "-DEIDSP_USE_CMSIS_DSP=0", "-DEIDSP_LOAD_CMSIS_DSP_SOURCES=0", "-DEI_CLASSIFIER_TFLITE_ENABLE_CMSIS_NN=0",
          "-DEI_CLASSIFIER_TFLITE_ENABLE_ESP_NN=0", "-DTF_LITE_STATIC_MEMORY",
          f"-I{SRC}", f"-I{SRC}/edge-impulse-sdk", f"-I{SRC}/edge-impulse-sdk/third_party/flatbuffers/include",
          f"-I{SRC}/edge-impulse-sdk/third_party/gemmlowp", f"-I{SRC}/edge-impulse-sdk/third_party/ruy"]


def compile_one(item):
    i, src = item
    obj = os.path.join(OUT, "obj", f"{i:03d}.o")
    cmd = (["clang"] if src.endswith(".c") else ["clang++", "-std=c++17"]) + common + ["-c", src, "-o", obj]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        open(os.path.join(OUT, "obj", f"{i:03d}.err"), "w").write(src + "\n" + r.stderr[-3000:])
        return src
    return None


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    failed = [f for f in ex.map(compile_one, enumerate(srcs)) if f]
print(f"compiled {len(srcs) - len(failed)} of {len(srcs)} files")
for f in failed:
    print("FAILED:", f.replace(SRC, "src"))
if failed:
    sys.exit(1)
objs = [os.path.join(OUT, "obj", f"{i:03d}.o") for i in range(len(srcs))]
r = subprocess.run(["clang++", "-o", os.path.join(OUT, "host_test")] + objs, capture_output=True, text=True)
print(r.stderr[-2500:] or f"linked {os.path.join(OUT, 'host_test')}")
sys.exit(r.returncode)

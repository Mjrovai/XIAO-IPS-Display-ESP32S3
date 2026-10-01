#!/bin/sh
# Compile every sketch in part1_bringup (and extras) with the patched library.
# Usage: tools/build_all.sh            compile only
#        FQBN=... tools/build_all.sh   use another board string
cd "$(dirname "$0")/.."
FQBN="${FQBN:-esp32:esp32:XIAO_ESP32S3_Plus}"
[ -d libs/Seeed_GFX2_fast ] || ./tools/make_fast_lib.sh
fail=0
log=$(mktemp)
for d in part1_bringup/[0-9]*/ part1_bringup/extras/*/; do
  name=$(basename "$d")
  if arduino-cli compile --fqbn "$FQBN" --library libs/Seeed_GFX2_fast "$d" >"$log" 2>&1; then
    echo "ok    $name   $(grep -o 'Sketch uses [0-9]* bytes (\([0-9]*\)%)' "$log")"
  else
    echo "FAIL  $name"; fail=1
  fi
done
rm -f "$log"
exit $fail

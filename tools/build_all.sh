#!/bin/sh
# Compile every sketch in part1_bringup (and extras) and part2_tinyml with the patched library.
# The Part 2 sketches that need your Edge Impulse model (04 and 05) are built only if the
# unzipped library is in libs/ei-kws/XIAO_IPS_Display_-_KWS_inferencing; otherwise they are skipped.
# Usage: tools/build_all.sh            compile only
#        FQBN=... tools/build_all.sh   use another board string
cd "$(dirname "$0")/.."
FQBN="${FQBN:-esp32:esp32:XIAO_ESP32S3_Plus}"
EI_LIB=libs/ei-kws/XIAO_IPS_Display_-_KWS_inferencing
[ -d libs/Seeed_GFX2_fast ] || ./tools/make_fast_lib.sh
fail=0
log=$(mktemp)
for d in part1_bringup/[0-9]*/ part1_bringup/extras/*/ part2_tinyml/[0-9]*/; do
  name=$(basename "$d")
  extra=""
  case "$name" in
    04_kws_inference|05_kws_replay)
      if [ -d "$EI_LIB" ]; then extra="--library $EI_LIB"; else echo "skip  $name   (no $EI_LIB)"; continue; fi ;;
  esac
  if arduino-cli compile --fqbn "$FQBN" --library libs/Seeed_GFX2_fast $extra "$d" >"$log" 2>&1; then
    echo "ok    $name   $(grep -o 'Sketch uses [0-9]* bytes (\([0-9]*\)%)' "$log")"
  else
    echo "FAIL  $name"; fail=1
  fi
done
rm -f "$log"
exit $fail

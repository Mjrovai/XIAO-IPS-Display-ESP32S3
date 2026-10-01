#!/bin/sh
# Build libs/Seeed_GFX2_fast: a copy of the installed Seeed_GFX2 1.0.0 with
# tools/seeed_gfx2_speedup.patch applied. The installed library is not modified.
set -e
cd "$(dirname "$0")/.."
SRC="${SEEED_GFX2_DIR:-$HOME/Documents/Arduino/libraries/Seeed_GFX2}"
OUT=libs/Seeed_GFX2_fast
[ -d "$SRC" ] || { echo "Seeed_GFX2 not found at $SRC (set SEEED_GFX2_DIR)"; exit 1; }
grep -q '^version=1.0.0' "$SRC/library.properties" || echo "warning: patch was made against Seeed_GFX2 1.0.0"
rm -rf "$OUT" && mkdir -p libs
cp -R "$SRC" "$OUT"
rm -rf "$OUT/examples"
mkdir -p .patchwork && rm -rf .patchwork/a && mkdir .patchwork/a
cp -R "$OUT" .patchwork/a/Seeed_GFX2
(cd .patchwork/a && patch -p1 < ../../tools/seeed_gfx2_speedup.patch)
rm -rf "$OUT" && mv .patchwork/a/Seeed_GFX2 "$OUT" && rm -rf .patchwork
echo "OK: $OUT"

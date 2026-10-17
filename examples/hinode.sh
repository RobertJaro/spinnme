#!/usr/bin/env bash
# Joint Fe I 6301/6302 inversion of a calibrated, WCS-bearing Hinode raster.
set -euo pipefail

DATA="./data/hinode-raster.fits"
RUN="./runs/hinode"
OUTPUT="./products/hinode"

# Assemble slit files with prepare_hinode_raster first; see docs/hinode.md.
spinnme run --config hinode/sp --input "$DATA" --run_path "$RUN"

# Line-dependent parameters use their saved instrument and observing grid.
spinnme export "$RUN/inversion.pme" --all-observations \
  --parameters Brtp Blos Vdop --format fits --output "$OUTPUT"

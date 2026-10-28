#!/usr/bin/env bash
# Joint Fe I 6301/6302 inversion of a folder of calibrated Hinode slit scans.
set -euo pipefail

DATA="./data/hinode-slits"
RUN="./runs/hinode"
OUTPUT="./products/hinode"

# Download a short UTC interval directly into the slit folder.
spinnme download hinode --start 2024-03-23T02:00:35 --end 2024-03-23T02:02:35 \
  --output "$DATA"

# Slit assembly happens inside the Hinode dataset; see docs/hinode.md.
spinnme run --config hinode/sp --input "$DATA" --run_path "$RUN"

# Line-dependent parameters use their saved instrument and observing grid.
spinnme export "$RUN/inversion.pme" --all-observations \
  --parameters Brtp Blos Vdop --format fits --output "$OUTPUT"

#!/usr/bin/env bash
# Download calibrated PHI Stokes data, invert and export explicit parameters.
set -euo pipefail

DATA="./data/phi-hrt"
RUN="./runs/phi-hrt"
OUTPUT="./products/phi-hrt"

# Select an interval with released L2 Stokes data and verify its native WCS.
spinnme download phi --telescope hrt --product stokes \
  --start 2024-03-23T22:00:00 --end 2024-03-24T03:00:00 --output "$DATA"

# The bundled YAML defines wavelengths, normalization and training settings.
spinnme run --config phi/hrt --input "$DATA" --run_path "$RUN"

# One WCS-bearing FITS file per selected scalar parameter.
spinnme export "$RUN/inversion.pme" --all-observations \
  --parameters Brtp Blos Vdop --format fits --output "$OUTPUT"

#!/usr/bin/env bash
# Download HMI, crop a solar subframe, invert its native pixels and export FITS.
set -euo pipefail

DATA="./data/hmi"
SUBFRAME="./data/hmi-subframe"
RUN="./runs/hmi-subframe"
OUTPUT="./products/hmi-subframe"
: "${JSOC_EMAIL:?Set JSOC_EMAIL to your registered JSOC address.}"

# 1. Download one complete acquisition (all 24 I/Q/U/V segments).
spinnme download hmi --product S_720s \
  --start 2024-03-23T01:00:00 --time-scale tai \
  --email "$JSOC_EMAIL" --output "$DATA/stokes"

# 2. Crop each acquisition to the enclosing native-pixel box.
# Bounds are XMIN YMIN XMAX YMAX in helioprojective arcsec.
spinnme crop --input "$DATA/stokes" --output "$SUBFRAME/stokes" \
  --bounds -250 -200 -100 -50

# 3. Crop headers retain the acquisition/camera identity for calibration lookup.
spinnme download hmi-transmission --input "$SUBFRAME/stokes" \
  --output "$SUBFRAME/transmission" --email "$JSOC_EMAIL"

# 4. The subframe preset retains native cropped pixels (no spatial resampling).
spinnme run --config hmi/subframe --data_path "$SUBFRAME" \
  --run_path "$RUN"

# 5. Export with the cropped observing WCS.
spinnme export "$RUN/inversion.pme" --all-observations \
  --parameters Brtp Blos Vdop --format fits --output "$OUTPUT"

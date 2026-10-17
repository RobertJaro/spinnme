#!/usr/bin/env bash
# HMI download → transmission profiles → inversion → FITS export.
set -euo pipefail

DATA="./data/hmi"
RUN="./runs/hmi"
OUTPUT="./products/hmi"
: "${JSOC_EMAIL:?Set JSOC_EMAIL to your registered JSOC address.}"

# 1. Download one complete Stokes acquisition.
spinnme download hmi --product S_720s \
  --start 2024-03-23T01:00:00 --time-scale tai \
  --email "$JSOC_EMAIL" --output "$DATA/stokes"

# 2. Download matching transmission profiles, independently of the run YAML.
spinnme download hmi-transmission --input "$DATA/stokes" \
  --output "$DATA/transmission" --email "$JSOC_EMAIL"

# 3. The bundled YAML defines the grid, validation and training settings.
spinnme run --config hmi/full_disk --data_path "$DATA" --run_path "$RUN"

# 4. Export one FITS per spherical magnetic-field component, with WCS.
spinnme export "$RUN/inversion.pme" --all-observations --format fits --output "$OUTPUT"

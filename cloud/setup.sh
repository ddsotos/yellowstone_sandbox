#!/usr/bin/env bash
set -euo pipefail

# Cloud tasks may run in a network-isolated environment.  Do not contact an
# index when the scientific stack is already provided by the base image.
if ! python -c 'import numpy, torch' >/dev/null 2>&1; then
  echo "ERROR: cloud base image must provide numpy and torch; package-index access is unavailable." >&2
  echo "Choose a PyTorch-enabled environment or provide an internal wheel cache, then retry." >&2
  exit 23
fi

# The project itself has no runtime dependency installation requirement once
# numpy/torch are present; install editable metadata without dependency lookup.
python -m pip install --no-deps --no-index -e .

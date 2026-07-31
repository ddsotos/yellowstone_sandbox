# Codex Cloud collection

Run four independent workers (indices `0` through `3`) in parallel. Each
worker collects 100,000 games with a distinct seed and game-id range. Set
`MAX_GAMES=1000` for a smoke run. Persist the resulting directory or tarball as
the task artifact. After all workers finish, run the validator over the four
replay directories to check manifests, counts, and duplicate game IDs.

The checkpoint is intentionally versioned in `models/`; raw replay artifacts
are not committed to Git.

The base cloud image must provide `numpy`. Checkpoint mode additionally needs
`torch`; `cloud/setup.sh` deliberately avoids PyPI access because some Cloud
tasks deny package-index traffic (403). If checkpoint mode is unavailable,
use the explicit model-free mode below.

For a deliberately model-free collection, set `HEURISTIC_ONLY=1`; this uses
the same random branches but heuristic fallback and records a separate
`model_mode` in the manifest.

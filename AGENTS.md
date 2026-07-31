# Cloud collection

The repository is a self-contained Yellowstone RL bundle. Cloud workers run
`cloud/collect_worker.sh`; each worker writes its replay and status as an
artifact. Do not commit generated `data/`, `logs/`, `results/`, or artifacts.
Workers use disjoint game-id ranges and must be validated before aggregation.

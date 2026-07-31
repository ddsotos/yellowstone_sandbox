# Scripts

Run these from the `rl_bundle` project root, for example:

```powershell
.\scripts\run_6h_data_collection.ps1
.\scripts\run_6h_value_training.ps1
```

All scripts resolve project paths from their own location. Data goes to `data/`,
checkpoints to `models/`, logs to `logs/`, and evaluation output to `results/`.

`run_canonical_training_pipeline.ps1` converts the selected old, six-hour, and
eight-hour increments to `fast_lr_ud_color_v1`, then trains the three one-pass
checkpoints in sequence. It is restartable: converted archives and completed
checkpoints are skipped.

`run_v1_new_collection_50000_vs_88966.ps1` runs the current restartable
conversion, independent one-epoch training, four-seat evaluation, and comparison
pipeline.

`archive_derived_training_data_to_d.ps1` moves the fixed list of inactive,
derived datasets to `D:\codex-backup`. It must not be expanded to include active
pipeline inputs without first updating `docs/storage-layout.md`.

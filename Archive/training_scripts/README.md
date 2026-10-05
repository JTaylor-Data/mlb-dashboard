# Archive: Training Scripts

Training scripts for all models, including both active and retired versions.
These ran offline to produce the model files in `models/` and score parquets in `scores/`.

The `training/` directory was moved here by the 2026-10-05 codebase audit.

## Key scripts (by model)

- `training/train_stuff_plus_ext_lgbm.py` — trained Stuff+ v4 (active)
- `training/train_location_v3.py` — trained Location+ v3 (active)
- `training/train_deception_model.py` — trained Deception+ v1/v2
- `training/train_tunneling_plus_ext.py` — trained Tunneling+ v3 ext

## Retired versions
Scripts for v1/v2 models are also present; corresponding model files are in `Archive/models_retired/`.

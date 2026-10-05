# Archive: Misc Scripts

Utility scripts that are not imported by the live server.

## Contents

- `scripts/rescore_models.py` — manually re-scores all models and writes score parquets.
  Run this after retraining a model. Called by the scheduler indirectly via the
  precompute pipeline, but not imported at server startup.

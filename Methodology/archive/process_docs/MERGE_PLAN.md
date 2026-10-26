# Merge Plan — 2026-10-04

## Summary

**Goal**: Merge `deception-v2` and `closeout-blend` into `master`.

## Branch Lineage

```
master (d509519)
  └── deception-v2 (f18555c)  ← already ancestor of closeout-blend
        └── closeout-blend (4e46e2e)  ← contains all deception-v2 commits + score cache + compatibility fixes
```

`git merge-base --is-ancestor deception-v2 closeout-blend` → TRUE

**Key finding**: `closeout-blend` is a superset of `deception-v2`. Only one merge is needed.

## Branch Contents

### master (d509519)
- Pre-deception-v2 cleanup state
- Stuff+ v4, Location+ v3, Tunneling+ v2 all live
- No deception v2b, no score cache parquets

### deception-v2 (f18555c)
- Deception+ v2b model trained and promoted
- Limited_evidence status
- Arm_angle features, pitcher-season form
- checkpoint files for pre-closeout state

### closeout-blend (4e46e2e)  ← MERGE TARGET
Contains everything in deception-v2 PLUS:
- `feat(closeout-blend) Part 0`: spec_lock_addendum.json
- `Part A-D closeout`: Stuff+ v4 and Location+ v3 validation complete, blend Decision A
- `Add persisted score cache`: parquets + weekly rescore job (api/warmup.py, api/scheduler.py)
- `fix(deception)`: compatibility shims (_raw_predictions_table, _pitcher_breakdown_cached)
  and players.py switch to parquet-based _bracket

## Merge Order

1. Create `integration-merge` from `master`
2. Merge `closeout-blend` into `integration-merge` with `--no-ff`
   - This single merge brings in all deception-v2 AND closeout-blend work

## Expected Conflicts

Low risk — closeout-blend diverged cleanly from master at the pre-deception-v2-cleanup commit.
All model work was done on feature branches; master has no commits beyond the branch point.

Files most likely to have conflicts if any:
- `api/routers/players.py` — multiple branches touched this
- `config/model_policy.py` — might differ
- `frontend/` static files if about-rework merged between

## Pre-merge State

- master: d509519adabbe75951bf2ea0e8cda2463d7e4f0f
- closeout-blend: 4e46e2e90ad8507bd6eee0e27c3e1af546c3655b
- Active model versions: Stuff+ v4, Location+ v3, Deception+ v2b
- DECEPTION_ACTIVE_VERSION = "v2b" in config/model_policy.py

## Tunneling Constraint
Tunneling files are OUT OF SCOPE — byte-identity must be preserved.
Baseline hashes recorded in /tmp/tunneling_baseline_hashes.txt and
Methodology/tunneling_baseline_hashes.txt

"""
Deception+ v1 - shared feature engineering for the delivery-only model.

## What this model is trying to isolate

Deception+ asks a narrower question than Stuff+: independent of how good a
pitch's velocity/movement/spin actually are, does something about a
pitcher's BODY and RELEASE CONSISTENCY predict better outcomes? This is
the "windup and timing mess with a hitter's ability to hit the ball" idea
from the original spec - reduced to what's actually measurable (see
deception_model.py's module docstring for the full story of what was
requested, what was investigated, and what had to be dropped).

Every feature here describes the pitcher's BODY or RELEASE POINT, never
the pitch's velocity, movement, or spin (that's Stuff+'s job, deliberately
kept out so this model can't just re-derive Stuff+ under a new name) and
never pitch type or pitcher identity (same reasoning as stuff_features.py:
a tree that can see pitcher_id would memorize aces instead of learning
what makes a delivery deceptive).

## The features, and why each one

- `arm_angle`, `release_extension`, `release_pos_x_as`, `release_pos_z`:
  raw, per-pitch. The only per-pitch delivery signals Statcast actually
  provides (see module docstring in deception_model.py for what was
  requested but isn't available: leg lift, time held in set, delivery
  length).
- `arm_angle_std`, `release_extension_std`, `release_consistency`:
  per-(pitcher, season) VARIABILITY in the above, joined onto every pitch
  that pitcher threw that season - "variation in the delivery," per the
  original spec, computed as a season-long aggregate since Statcast has no
  cross-pitch grouping finer than that. `release_consistency` (position
  variability, both x and z combined) reuses the exact same computation as
  tunneling_model.py's `_release_consistency` for the whole arsenal - not
  imported directly since it operates on a differently-shaped DataFrame at
  this stage of the pipeline, but the definition is identical.
- `height_in`, `weight_lb`: pitcher biographical data, from
  `pitcher_bio` (pull_pitcher_bio.py) - not in Statcast at all. Current
  listed bio only, not season-specific (same limitation `players.team_id`
  already carries elsewhere on this site).
- `season`: categorical, same era-absorbing reasoning as Stuff+ v2.
"""
import numpy as np
import pandas as pd

RAW_QUERY_COLUMNS = [
    "pitcher_id", "p_throws", "season",
    "release_extension", "release_pos_x", "release_pos_z", "arm_angle",
    "delta_run_exp",
]

NUMERIC_FEATURES = [
    "arm_angle",
    "release_extension",
    "release_pos_x_as",
    "release_pos_z",
    "arm_angle_std",
    "release_extension_std",
    "release_consistency",
    "height_in",
    "weight_lb",
]
CATEGORICAL_FEATURES = ["season"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _arm_side_sign(p_throws):
    """Same convention as stuff_features.py's `_arm_side_sign` - flips
    Statcast's fixed field-relative release_pos_x so positive always means
    "toward the pitcher's arm side," regardless of handedness."""
    return np.where(p_throws == "R", -1, 1)


def load_bio(conn):
    """height_in/weight_lb per pitcher_id, from the pitcher_bio table
    (pull_pitcher_bio.py). Missing for any pitcher not yet pulled or whose
    Stats API record didn't resolve - those pitches get dropped later by
    the dropna in engineer_features, same as any other missing feature."""
    return pd.read_sql("SELECT player_id AS pitcher_id, height_in, weight_lb FROM pitcher_bio", conn)


def engineer_features(df, bio):
    """Full pipeline from a raw `pitches` query result (+ the bio table)
    to a model-ready DataFrame. The ONE function both training and scoring
    call - same train/serve-skew guarantee as stuff_features.py."""
    df = df.copy()
    sign = _arm_side_sign(df["p_throws"])
    df["release_pos_x_as"] = df["release_pos_x"] * sign

    g = df.groupby(["pitcher_id", "season"])
    df["arm_angle_std"] = g["arm_angle"].transform("std")
    df["release_extension_std"] = g["release_extension"].transform("std")
    centers = g[["release_pos_x", "release_pos_z"]].transform("mean")
    sq_dist = (df["release_pos_x"] - centers["release_pos_x"]) ** 2 + (df["release_pos_z"] - centers["release_pos_z"]) ** 2
    df["release_consistency"] = sq_dist.groupby([df["pitcher_id"], df["season"]]).transform("mean") ** 0.5

    df = df.merge(bio, on="pitcher_id", how="left")

    df["season"] = df["season"].astype(str)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["delta_run_exp"])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df

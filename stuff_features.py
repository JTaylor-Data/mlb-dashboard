"""
Stuff+ v2 - shared feature engineering.

## Why this file exists on its own

`train_stuff_model.py` (fits the model, offline, run by hand) and
`stuff_model.py` (scores pitches at request time, inside the API) must
compute features in EXACTLY the same way. If they drifted - e.g. training
used arm-adjusted horizontal break but scoring accidentally used raw
`pfx_x` - the model would silently make garbage predictions in production
while looking fine in training. Centralizing feature computation here in
one place removes that whole class of bug (train/serve skew) by
construction: there is only one code path.

## What "Stuff+" is supposed to isolate

Stuff+ is meant to grade the PHYSICAL SHAPE of a pitch (velo, movement,
spin, release) independent of where it was thrown or the count it was
thrown in. That's the whole reason it's a useful complement to results-based
stats: it strips out command, sequencing, and luck, leaving "how nasty is
this pitch, mechanically." Two things follow from that:

1. Plate location (`plate_x`, `plate_z`) and count (`balls`, `strikes`) are
   deliberately EXCLUDED as features. Those measure command/sequencing, a
   different skill, which is what a future Location+/Command+ model would
   cover. Mixing them in here would make a good FASTBALL DOWN THE MIDDLE
   look bad and a mediocre pitch that happened to catch the corner look
   great - that's not "stuff," that's location.
2. Pitcher identity is NEVER a feature. If the model could see who's
   throwing, a gradient-boosted tree would happily learn "pitcher_id=543037
   -> low run value" (memorizing a specific ace) instead of learning WHY
   his pitches are good. That defeats the entire point of a shape-based
   model and wouldn't generalize to a next pitch that hasn't been thrown
   yet, which is the actual use case for a "stuff" grade.
"""
import numpy as np
import pandas as pd

from metrics import compute_vaa

# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------
# Every one of these describes the PITCH ITSELF - release point, velocity,
# movement, spin - never location, count, or outcome. `vaa_residual` is the
# one non-obvious entry; see `fit_vaa_expectation()` below for why raw VAA
# doesn't belong here unmodified.
NUMERIC_FEATURES = [
    "release_speed",       # raw velo
    "effective_speed",     # "perceived" velo - already adjusts for extension
    "release_spin_rate",
    "spin_axis",           # 0-360 degrees; trees split on ranges fine as-is
    "hb",                  # horizontal break, arm-side-adjusted (inches)
    "ivb",                 # induced vertical break, inches - no handedness flip
                            # needed, "up" means the same thing for every pitcher
    "release_extension",
    "release_pos_x_as",    # release side, arm-side-adjusted
    "release_pos_z",       # release height - no handedness flip needed
    "arm_angle",
    "vaa_residual",        # see fit_vaa_expectation() - VAA with the
                            # plate-height/velocity confound removed
]
CATEGORICAL_FEATURES = ["pitch_type", "p_throws", "stand", "season"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw columns that must be pulled from `pitches` to compute the above.
RAW_QUERY_COLUMNS = [
    "pitcher_id", "pitch_type", "stand", "p_throws", "season",
    "release_speed", "effective_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z", "release_extension", "release_pos_x", "release_pos_z",
    "arm_angle", "plate_z", "vy0", "vz0", "ay", "az", "delta_run_exp",
]


def merge_pitch_type(df):
    """KC (knuckle curve) folded into CU everywhere on this site - same
    convention the rest of the codebase (and Stuff+ v1) already uses."""
    df = df.copy()
    df["pitch_type"] = df["pitch_type"].replace({"KC": "CU"})
    return df


def _arm_side_sign(p_throws):
    """+1/-1 array: flips any field-relative left/right coordinate so
    positive always means "toward the pitcher's arm side," regardless of
    handedness. Same convention as player_page.py's `_add_arm_side_hb` -
    reused here (not imported directly) because it operates on a
    differently-shaped DataFrame at this stage of the pipeline."""
    return np.where(p_throws == "R", -1, 1)


def add_handedness_adjusted_columns(df):
    """
    Statcast's raw `pfx_x` and `release_pos_x` are fixed field-relative
    coordinates - NOT handedness-adjusted. A right-handed pitcher's
    arm-side release point and a left-handed pitcher's arm-side release
    point show up with opposite raw signs even though they describe the
    same thing ("released from my natural arm side"). Without this fix, a
    gradient-boosted tree would have to spend splits rediscovering the
    handedness mirror on every feature separately (or worse, a small
    training sample of one hand could bake in a spurious hand-specific
    pattern). Pre-adjusting once, consistently, removes that burden.

    While here, also converts `pfx_z` from Statcast's raw feet into inches
    (`ivb`) - every other IVB figure on this site (the pitch table,
    movement plot) is already displayed in inches, so scoring in feet
    would be internally consistent for the model itself (tree splits
    don't care about units) but confusing everywhere this feature's value
    gets displayed (a real fastball's ~1.4 feet of rise reading as "1.40"
    next to inch-scaled numbers looks like a tiny, unimpressive value).
    """
    df = df.copy()
    sign = _arm_side_sign(df["p_throws"])
    df["hb"] = df["pfx_x"] * 12 * sign               # feet -> inches, arm-side sign
    df["ivb"] = df["pfx_z"] * 12                     # feet -> inches
    df["release_pos_x_as"] = df["release_pos_x"] * sign
    return df


def fit_vaa_expectation(df):
    """
    Fit "expected VAA given where the pitch crossed the plate and how fast
    it was going" - a plain least-squares regression, deliberately simple
    and transparent (not another gradient-boosted model) since this is a
    correction step, not the thing being evaluated.

    ## Why this exists at all
    Vertical Approach Angle (VAA) and plate height are NOT independent -
    they're two readings of the same trajectory. A pitch that crosses the
    plate higher up is geometrically forced to have a flatter VAA than the
    same pitch crossing lower, because gravity has had less time to act on
    it. So raw VAA partly encodes WHERE the pitch was thrown (location),
    which is exactly what a "stuff-only" model is supposed to exclude (see
    the module docstring). A pitcher who happens to throw a below-average
    fastball at the top of the zone would look like he has a "flat,
    deceptive" fastball by raw VAA alone, when really he just aimed high.

    ## The fix
    Model `vaa ~ plate_z + plate_z^2 + release_speed` across the whole
    league (quadratic in plate_z since VAA doesn't flatten linearly with
    height). The residual (actual VAA minus this expectation) is "flatter
    or steeper than physics alone predicts for that height and velocity" -
    the genuine release/movement contribution, with the location component
    subtracted out. This mirrors the "adjusted VAA" approach used in public
    pitching research for exactly this reason.

    Returns the fitted coefficient array; pass it to `add_vaa_residual()`.
    Fit ONCE on a training set and reused at scoring time (never refit on
    a single season at inference) so train and score apply the identical
    transform.
    """
    clean = df.dropna(subset=["plate_z", "release_speed", "vaa"])
    clean = clean[np.isfinite(clean["vaa"])]
    z = clean["plate_z"].to_numpy(dtype=float)
    v = clean["release_speed"].to_numpy(dtype=float)
    y = clean["vaa"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(clean)), z, z**2, v])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefs


def add_vaa_residual(df, vaa_coefs):
    """Attaches `vaa` (actual) and `vaa_residual` (actual - expected) using
    a previously-fit coefficient array from `fit_vaa_expectation()`."""
    df = df.copy()
    df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
    z = df["plate_z"].to_numpy(dtype=float)
    v = df["release_speed"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(df)), z, z**2, v])
    expected_vaa = design @ vaa_coefs
    df["vaa"] = df["vaa"].astype("float32")
    df["vaa_residual"] = (df["vaa"] - expected_vaa).astype("float32")
    return df


def engineer_features(df, vaa_coefs):
    """
    Full pipeline from a raw `pitches` query result to a model-ready
    DataFrame: merge KC->CU, add handedness-adjusted columns, add the VAA
    residual, coerce categoricals, and drop rows missing any required
    feature. This is the ONE function both training and scoring call -
    that's the train/serve-skew guarantee described in the module
    docstring.
    """
    df = merge_pitch_type(df)
    df = add_handedness_adjusted_columns(df)
    df = add_vaa_residual(df, vaa_coefs)
    df["season"] = df["season"].astype(str)  # categorical, not a numeric trend
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df

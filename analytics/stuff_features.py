"""
Stuff+ - shared feature engineering (v3 architecture, promoted to
production; see `Methodology/STUFF_V3_REPORT.md` for the full methodology
write-up and why it replaced v2). The prior v2 pipeline is preserved,
unused, in `stuff_features_v2_legacy.py`.

## Why this file exists on its own

`training/train_stuff_v3_lgbm.py` (fits the model, offline, run by hand)
and `stuff_model.py` (scores pitches at request time, inside the API) must
compute features in EXACTLY the same way. If they drifted, the model
would silently make garbage predictions in production while looking fine
in training. Centralizing feature computation here in one place removes
that whole class of bug (train/serve skew) by construction: there is only
one code path.

## What changed from v2

Removed (moved to being Deception+-exclusive - delivery/mechanics
properties, not pitch-shape properties; already live in
deception_features.py, untouched by this file): `release_extension`,
`release_pos_x_as`, `release_pos_z`, `arm_angle`.

Added: fastball velo/movement SEPARATION features (this pitch vs. this
pitcher-season's own PRIMARY reference fastball - see Part A/B below) and
`axis_differential` (seam-shifted-wake proxy - see spin_physics.py).

## Primary-only, not a multi-slot scheme

An earlier build supported up to 3 reference fastball slots
(primary/secondary/tertiary) with a 60%-dominant / 10%-qualify threshold
rule. A direct empirical ablation (`training/ablate_stuff_v3_slots.py`)
compared 1-slot vs. 2-slot vs. 3-slot on identical CV + year-over-year
harnesses: per-pitch CV accuracy was flat across all three (differences
were noise), but year-over-year score STABILITY was highest for 1-slot
(0.754) and measurably worse for 2-slot (0.692) and 3-slot (0.720) - adding
secondary/tertiary references didn't just fail to help, it hurt the one
metric that mattered most. Per that result, this module builds exactly
ONE reference per pitcher-season - whichever of FF/SI/FC has the highest
usage, no threshold at all - simpler code and the empirically
best-performing choice at once.

## What "Stuff+" is supposed to isolate

Stuff+ is meant to grade the PHYSICAL SHAPE of a pitch (velo, movement,
spin, release) independent of where it was thrown or the count it was
thrown in. Two things follow from that:

1. Plate location (`plate_x`, `plate_z`) and count (`balls`, `strikes`) are
   deliberately EXCLUDED as features - that measures command/sequencing, a
   different skill.
2. Pitcher identity is NEVER a feature - a tree that could see pitcher_id
   would memorize aces instead of learning WHY their pitches are good,
   which wouldn't generalize to a next pitch that hasn't been thrown yet.

Kept unchanged from v2: `release_speed`, `effective_speed`,
`release_spin_rate`, `spin_axis`, `hb`, `ivb`, `vaa_residual`, and all four
categoricals.
"""
import numpy as np
import pandas as pd

from analytics.metrics import compute_vaa
from analytics.spin_physics import add_axis_differential, fit_spin_axis_calibration

# ---------------------------------------------------------------------------
# Part A - fastball reference classification (pitcher-season level)
# ---------------------------------------------------------------------------

FB_TYPES = ["FF", "SI", "FC"]  # fastball-family SHAPES for reference purposes


def build_fb_reference_table(pitches_df):
    """
    One row per (pitcher_id, season): primary_fb_type (nullable string) -
    whichever of FF/SI/FC has the HIGHEST usage that pitcher-season, no
    threshold at all. A pitcher-season with zero FF/SI/FC pitches gets no
    reference (None) - a real, small edge case (almost entirely position
    players in mop-up innings or genuine slider-primary specialist
    relievers). Also carries that reference type's own profile (mean
    effective_speed, hb, ivb for that pitcher/season/pitch-type) needed by
    Part B.

    `pitches_df` must already have pitch_type KC->CU merged and hb/ivb
    attached (arm-side-adjusted, inches) - same precondition as the rest of
    the Stuff+ feature pipeline.
    """
    df = pitches_df[["pitcher_id", "season", "pitch_type", "effective_speed", "hb", "ivb"]].copy()

    fb = df[df["pitch_type"].isin(FB_TYPES)]
    profile = fb.groupby(["pitcher_id", "season", "pitch_type"]).agg(
        n_pitches=("pitch_type", "size"),
        effective_speed=("effective_speed", "mean"),
        hb=("hb", "mean"),
        ivb=("ivb", "mean"),
    ).reset_index()

    idx = profile.groupby(["pitcher_id", "season"])["n_pitches"].idxmax()
    primary = profile.loc[idx].rename(columns={
        "pitch_type": "primary_fb_type", "effective_speed": "primary_fb_effective_speed",
        "hb": "primary_fb_hb", "ivb": "primary_fb_ivb",
    }).drop(columns=["n_pitches"])
    return primary[["pitcher_id", "season", "primary_fb_type", "primary_fb_effective_speed", "primary_fb_hb", "primary_fb_ivb"]]


def fb_reference_audit_columns(fb_ref_df):
    """Just the persistable audit table - pitcher_id, season,
    primary_fb_type only (the profile numbers are an implementation detail
    Part B needs, not part of the audit table itself)."""
    return fb_ref_df[["pitcher_id", "season", "primary_fb_type"]]


# ---------------------------------------------------------------------------
# Part B - separation features (per-pitch)
# ---------------------------------------------------------------------------

def add_separation_features(df, fb_ref_df):
    """
    Attaches velo_sep_vs_primary_fb, hb_sep_vs_primary_fb,
    ivb_sep_vs_primary_fb. Joined on (pitcher_id, season) - NOT pitcher_id
    alone, so a 2023 pitch is never compared against a 2025 fastball
    profile.

    True NaN (never 0, never imputed) when: (a) this pitcher-season has no
    primary reference at all (no FF/SI/FC thrown), or (b) this pitch's own
    type IS the primary reference type (a pitch cannot be separated from
    itself - e.g. every SI pitch gets NaN for *_sep_vs_primary_fb when SI
    is that pitcher's primary reference, even though other pitchers' SI
    pitches get a real value against a DIFFERENT primary reference).
    """
    df = df.merge(fb_ref_df, on=["pitcher_id", "season"], how="left")

    is_self = df["pitch_type"] == df["primary_fb_type"]
    has_ref = df["primary_fb_type"].notna()
    valid = has_ref & ~is_self

    df["velo_sep_vs_primary_fb"] = np.where(valid, df["effective_speed"] - df["primary_fb_effective_speed"], np.nan).astype("float32")
    df["hb_sep_vs_primary_fb"] = np.where(valid, df["hb"] - df["primary_fb_hb"], np.nan).astype("float32")
    df["ivb_sep_vs_primary_fb"] = np.where(valid, df["ivb"] - df["primary_fb_ivb"], np.nan).astype("float32")

    return df.drop(columns=["primary_fb_effective_speed", "primary_fb_hb", "primary_fb_ivb"])


# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------

SEPARATION_FEATURES = [f"{metric}_sep_vs_primary_fb" for metric in ("velo", "hb", "ivb")]

# Required for every row - a pitch missing any of these is a genuine
# data-quality drop (same as v2). Deliberately does NOT include the
# separation features or axis_differential - those are legitimately absent
# for a real minority of rows (no primary reference that pitcher-season,
# or a self-comparison), and must survive as real NaN into the final table
# for LightGBM's native missing-value handling, not be dropped or imputed.
CORE_NUMERIC_FEATURES = [
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "spin_axis",
    "hb",
    "ivb",
    "vaa_residual",
]
OPTIONAL_NUMERIC_FEATURES = SEPARATION_FEATURES + ["axis_differential"]
NUMERIC_FEATURES = CORE_NUMERIC_FEATURES + OPTIONAL_NUMERIC_FEATURES
CATEGORICAL_FEATURES = ["pitch_type", "p_throws", "stand", "season"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# v3 query columns (no HAA)
RAW_QUERY_COLUMNS_V3 = [
    "pitcher_id", "pitch_type", "stand", "p_throws", "season",
    "release_speed", "effective_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z", "plate_z", "vy0", "vz0", "ay", "az", "delta_run_exp",
]
# v4 query columns — adds vx0, ax, plate_x, release_pos_x for HAA residual
RAW_QUERY_COLUMNS_V4 = RAW_QUERY_COLUMNS_V3 + ["vx0", "ax", "plate_x", "release_pos_x"]

# Default: v4 (the promoted version). Callers that need v3 can import RAW_QUERY_COLUMNS_V3.
RAW_QUERY_COLUMNS = RAW_QUERY_COLUMNS_V4


def merge_pitch_type(df):
    df = df.copy()
    df["pitch_type"] = df["pitch_type"].replace({"KC": "CU"})
    return df


# Pitch types excluded from Stuff+ entirely (training AND scoring
# population, not just display) - these aren't real competitive
# pitch-quality attempts, so letting them into the season mean/std
# population measures "how bad is a pitch that was never trying to be
# good," not stuff. Confirmed empirically before excluding (see
# Methodology/): PO/UN/EP score systematically far above the population
# mean predicted run value, and dropping them shifts the season std down
# ~2-4%, a real if modest standardization effect.
#   PO - pitchout: a deliberate off-plate lob to set up a steal throw.
#   UN - Statcast's own "unclassified" code: a classifier failure, not a
#        real pitch category.
#   EP - eephus: excluded entirely, a deliberate tradeoff. This also drops
#        the rare genuine eephus-as-a-weapon pitcher along with the far
#        more common position-player-mop-up-innings case - accepted, not
#        re-litigated here.
# KN (knuckleball) is deliberately NOT in this set - it's a real,
# competitive pitch type with a known, documented modeling limitation
# (extreme, genuinely hard-to-predict movement most Stuff+-style models
# struggle to grade well), not junk/non-competitive data the way the three
# above are.
EXCLUDED_PITCH_TYPES = {"PO", "UN", "EP"}


def exclude_junk_pitch_types(df):
    return df[~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES)]


def _arm_side_sign(p_throws):
    return np.where(p_throws == "R", -1, 1)


def add_movement_columns(df):
    """Only hb/ivb, not release_pos_x_as/release_pos_z (those two are gone
    - see module docstring) - same formula as v2's
    add_handedness_adjusted_columns for the columns that ARE kept."""
    df = df.copy()
    sign = _arm_side_sign(df["p_throws"])
    df["hb"] = df["pfx_x"] * 12 * sign
    df["ivb"] = df["pfx_z"] * 12
    return df


def add_net_vert_drop(df):
    """Net vertical drop (inches) from release to plate: gravity component
    minus IVB. Captures the velocity × IVB interaction — a 97 mph sinker
    with 3 in IVB still drops ~31 inches total, similar to an 88 mph
    sinker with 9 in IVB (~32 inches), but they feel completely different
    to a batter. By surfacing this combined number the model can learn
    pitch-type-specific patterns (e.g. high-velo + low-IVB sinker) that
    are obscured when release_speed and ivb are only seen independently.

    Formula: 0.5 * g * t² – ivb, where g=386 in/s², t = 726in / (mph×17.6).
    """
    df = df.copy()
    v_in_per_s = df["release_speed"].to_numpy(dtype=float) * 17.6
    t = 726.0 / v_in_per_s
    gravity_drop = 0.5 * 386.0 * t ** 2
    df["net_vert_drop"] = (gravity_drop - df["ivb"].to_numpy(dtype=float)).astype("float32")
    return df


def fit_vaa_expectation(df):
    """Pooled (all-types) VAA-vs-location/velocity regression. Returns a
    4-element coefficient array [intercept, z, z², v]. Used as the fallback
    in per-type scoring and as the sole baseline in per-CV-fold refit during
    training (where per-fold per-type fitting would risk overfitting on small
    fold slices for rare pitch types)."""
    clean = df.dropna(subset=["plate_z", "release_speed", "vaa"])
    clean = clean[np.isfinite(clean["vaa"])]
    z = clean["plate_z"].to_numpy(dtype=float)
    v = clean["release_speed"].to_numpy(dtype=float)
    y = clean["vaa"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(clean)), z, z**2, v])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefs


# Minimum pitch count to fit a stable per-type regression.  Types with fewer
# pitches in the training set fall back to the pooled baseline.
_VAA_PER_TYPE_MIN_SAMPLES = 5000


def fit_vaa_expectations_per_type(df, min_samples=_VAA_PER_TYPE_MIN_SAMPLES):
    """Fit a separate VAA-expectation regression for each pitch type that has
    at least `min_samples` pitches.  Returns a dict keyed by pitch_type whose
    values are lists [intercept, z, z², v].  Types with fewer pitches are
    omitted from the dict; callers must fall back to the pooled baseline for
    those types.

    `df` must already have `vaa` computed (call compute_vaa first) and
    pitch_type merged (KC→CU).

    ## Why per-type matters

    The pooled baseline is learned mostly from the high-volume fastball types
    (FF/SI/FC together are ~56 % of all pitches).  Curveballs arrive at a
    steeper angle than fastballs at comparable or lower speeds, so they always
    fall below a regression line fit on the pooled population — creating a
    systematic ~−1° mean residual for CU and a corresponding ~+0.23° mean
    for FF.  A per-type baseline removes this cross-type offset so that
    vaa_residual measures "steeper/flatter than expected FOR THIS PITCH TYPE"
    rather than "steeper/flatter than the overall-population average."

    ## Important caveat for inference without retraining

    The LightGBM model was *trained* on the pooled-baseline vaa_residual.
    Through that biased feature it learned pitch-type-specific expected VAA as
    an implicit part of the cross-type signal.  Switching to a per-type
    baseline at inference time without retraining changes what the feature
    *means* to the model, which can shift Stuff+ scores in unexpected
    directions (see diagnostic notes in DIAGNOSTIC_REPORT_STUFFPLUS_DECEPTIONPLUS.md).
    The correct long-term fix is to retrain the model with per-type residuals
    so the model can re-learn the within-type VAA effect correctly.
    """
    clean = df.dropna(subset=["plate_z", "release_speed", "vaa"])
    clean = clean[np.isfinite(clean["vaa"])]
    result = {}
    for pt, sub in clean.groupby("pitch_type", observed=True):
        if len(sub) < min_samples:
            continue
        z = sub["plate_z"].to_numpy(dtype=float)
        v = sub["release_speed"].to_numpy(dtype=float)
        y = sub["vaa"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(sub)), z, z**2, v])
        coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
        result[pt] = coefs.tolist()
    return result


def add_vaa_residual(df, vaa_coefs):
    """Compute `vaa` and `vaa_residual` and attach both to `df`.

    `vaa_coefs` may be:
    - a 4-element array/list → pooled baseline applied to all rows (legacy
      behaviour, used during per-fold CV refit in training)
    - a dict with key "pooled" (4-element list) and optionally pitch-type keys
      (4-element lists each) → per-type baseline used where available, pooled
      baseline used for unknown/unseen types.

    The dict form is produced by `fit_v3_calibrations` when `fit_per_type=True`
    and is what gets stored in calibrations.json / meta.json for live scoring.
    """
    df = df.copy()
    df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
    df["vaa"] = df["vaa"].astype("float32")

    z_arr = df["plate_z"].to_numpy(dtype=float)
    v_arr = df["release_speed"].to_numpy(dtype=float)
    ones = np.ones(len(df))

    if isinstance(vaa_coefs, dict):
        # Per-type mode: dispatch each row to its type's own regression line.
        pooled = np.array(vaa_coefs["pooled"])
        per_type = {k: np.array(v) for k, v in vaa_coefs.items() if k != "pooled"}
        expected_vaa = np.empty(len(df), dtype=float)
        pt_series = df["pitch_type"].astype(str)
        for pt, idx in df.groupby(pt_series).groups.items():
            coefs = per_type.get(pt, pooled)
            sub_z = z_arr[df.index.get_indexer(idx)]
            sub_v = v_arr[df.index.get_indexer(idx)]
            design = np.column_stack([np.ones(len(idx)), sub_z, sub_z**2, sub_v])
            expected_vaa[df.index.get_indexer(idx)] = design @ coefs
    else:
        # Legacy pooled mode.
        coefs = np.array(vaa_coefs)
        design = np.column_stack([ones, z_arr, z_arr**2, v_arr])
        expected_vaa = design @ coefs

    df["vaa_residual"] = (df["vaa"].to_numpy(dtype=float) - expected_vaa).astype("float32")
    return df


def fit_v3_calibrations(df, fit_per_type=True):
    """Fits everything that must be fit ONCE on a training set and reused
    at scoring time: the VAA-expectation regression and the spin-axis
    calibration (see spin_physics.py).  `df` should already have pitch_type
    merged and hb/ivb attached.

    When `fit_per_type=True` (default), the returned `vaa_coefs` is a dict
    {"pooled": [...], pitch_type: [...], ...} suitable for per-type residual
    computation.  When False it is the legacy 4-element list.
    """
    df_for_vaa = df.copy()
    df_for_vaa["vaa"] = compute_vaa(
        df_for_vaa["vy0"], df_for_vaa["vz0"], df_for_vaa["ay"], df_for_vaa["az"]
    )
    pooled = fit_vaa_expectation(df_for_vaa).tolist()
    if fit_per_type:
        per_type = fit_vaa_expectations_per_type(df_for_vaa)
        vaa_coefs = {"pooled": pooled, **per_type}
    else:
        vaa_coefs = pooled
    spin_calibration = fit_spin_axis_calibration(df)
    return {"vaa_coefs": vaa_coefs, "spin_calibration": spin_calibration}


def compute_haa(df):
    """Horizontal approach angle (degrees, arm-side mirrored).
    Uses vx0/vy0/ax/ay — same physics as VAA uses vz0/vy0/az/ay.
    Mirrors sign by p_throws so LHP and RHP angles are comparable
    (arm-side positive).  Requires columns: vx0, vy0, ax, ay, p_throws."""
    vy0 = df["vy0"].to_numpy(dtype=float)
    vx0 = df["vx0"].to_numpy(dtype=float)
    ay = df["ay"].to_numpy(dtype=float)
    ax = df["ax"].to_numpy(dtype=float)
    y_front = 17.0 / 12.0
    y_release = -60.5
    delta_y = y_front - y_release
    vy_front_sq = np.maximum(vy0 ** 2 + 2 * ay * delta_y, 0.01)
    vy_front = -np.sqrt(vy_front_sq)
    t = (vy_front - vy0) / ay
    vx_front = vx0 + ax * t
    haa_raw = np.degrees(np.arctan2(vx_front, -vy_front))
    sign = np.where(df["p_throws"].to_numpy() == "R", -1.0, 1.0)
    return (haa_raw * sign).astype("float32")


def add_haa_residual(df, haa_coefs):
    """Compute haa_residual: mirrored HAA minus per-type OLS expectation.
    Requires columns: vx0, vy0, ax, ay, p_throws, plate_x, release_pos_x,
    release_speed, pitch_type.

    `haa_coefs` is a dict {"pooled": [b0,b1,b2,b3,b4], pitch_type: [...], ...}
    from meta["haa_coefs"].  OLS formula: expected_haa = b0 + b1*plate_x_m
    + b2*plate_x_m^2 + b3*release_pos_x_m + b4*release_speed.
    """
    df = df.copy()
    df["haa"] = compute_haa(df)
    sign = np.where(df["p_throws"].to_numpy() == "R", -1.0, 1.0)
    px_m = df["plate_x"].to_numpy(dtype=float) * sign
    rp_m = df["release_pos_x"].to_numpy(dtype=float) * sign
    v = df["release_speed"].to_numpy(dtype=float)
    pooled = np.array(haa_coefs["pooled"])
    per_type = {k: np.array(v_) for k, v_ in haa_coefs.items() if k != "pooled"}
    expected = np.empty(len(df), dtype=float)
    for pt, idx in df.groupby(df["pitch_type"].astype(str)).groups.items():
        coefs = per_type.get(pt, pooled)
        loc = df.index.get_indexer(idx)
        design = np.column_stack([np.ones(len(idx)), px_m[loc], px_m[loc] ** 2, rp_m[loc], v[loc]])
        expected[loc] = design @ coefs
    df["haa_residual"] = (df["haa"].to_numpy(dtype=float) - expected).astype("float32")
    return df


def engineer_features(df, calibrations, fb_ref_df):
    """Full pipeline: merge KC->CU, exclude non-competitive pitch types
    (PO/UN/EP - see EXCLUDED_PITCH_TYPES), attach hb/ivb, VAA residual,
    spin-axis differential, fastball separation features, coerce
    categoricals, drop rows missing a CORE feature only (never the
    optional/reference-dependent ones - those keep real NaN through to the
    returned frame). This is the ONE function both training
    (build_stuff_v3_dataset.py, baked into the saved feature table) and
    live scoring (stuff_model.py) call, so the exclusion applies to both
    automatically - no separate SQL filter to keep in sync in either
    caller.

    v4 addition: if `calibrations` contains "haa_coefs", haa_residual is
    computed and appended.  When scoring with v3 (no haa_coefs in meta),
    the column is absent and the v3 all_features list never asks for it."""
    df = merge_pitch_type(df)
    df = exclude_junk_pitch_types(df)
    df = add_movement_columns(df)
    df = add_vaa_residual(df, calibrations["vaa_coefs"])
    df = add_axis_differential(df, calibrations["spin_calibration"])
    df = add_separation_features(df, fb_ref_df)
    if "haa_coefs" in calibrations and calibrations["haa_coefs"]:
        df = add_haa_residual(df, calibrations["haa_coefs"])
    df["season"] = df["season"].astype(str)
    df[CORE_NUMERIC_FEATURES] = df[CORE_NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=CORE_NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df

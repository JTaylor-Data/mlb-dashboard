"""
Stuff+ v1 - PRESERVED FOR COMPARISON ONLY, not used by the live app.

This is the original model (plain OLS regression, fit separately per pitch
type, per season, no train/test split) that `stuff_model.py` replaced with
a unified gradient-boosted model (see that file's and `train_stuff_model.py`'s
docstrings for the full v2 story). Recreated verbatim here after the fact
specifically to run a head-to-head predictive-validity comparison against
v2 - without this file, there'd be no way to answer "is v2 actually better,"
only "is v2 different." Do not wire this into the API; it is a reference/
benchmark artifact only.
"""
import numpy as np
import pandas as pd

from analytics.metrics import compute_vaa

RAW_FEATURES = [
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
    "release_extension", "release_pos_x", "release_pos_z", "arm_angle",
]
FEATURES = RAW_FEATURES + ["vaa"]
MIN_PITCHES_PER_TYPE = 200  # don't fit a model on a handful of league-wide pitches


def _fit_ols(X, y):
    X_design = np.column_stack([np.ones(len(X)), X])
    coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return coefs


def _score_pitches(conn, season):
    """
    Fits one model per pitch type across the whole league FOR THIS SEASON,
    then returns a per-pitch DataFrame (pitcher_id, pitch_type, stand,
    stuff_plus) - the shared basis both compute_stuff_plus() and
    compute_stuff_plus_by_stand() aggregate from. Pitch types with fewer
    than MIN_PITCHES_PER_TYPE pitches league-wide (within this season) are
    skipped entirely (not enough data to fit anything). The model itself
    isn't refit per batter handedness - what varies by handedness is which
    of a pitcher's pitches (with which shapes) got thrown to that side, not
    the shape-to-quality relationship itself. Fitting per season (rather
    than pooling every backfilled season into one model) matters here since
    league-wide pitch shape/quality baselines drift year to year.
    """
    df = pd.read_sql(f"""
        SELECT pitcher_id, pitch_type, stand, delta_run_exp, vy0, vz0, ay, az,
               {', '.join(RAW_FEATURES)}
        FROM pitches
        WHERE season = ? AND delta_run_exp IS NOT NULL
    """, conn, params=(season,))
    df["pitch_type"] = df["pitch_type"].replace({"KC": "CU"})  # same pitch as a curveball for display
    df["vaa"] = compute_vaa(df["vy0"], df["vz0"], df["ay"], df["az"])
    df = df.dropna(subset=FEATURES + ["delta_run_exp"])

    scored = []
    for pitch_type, group in df.groupby("pitch_type"):
        if len(group) < MIN_PITCHES_PER_TYPE:
            continue
        X = group[FEATURES].to_numpy(dtype=float)
        y = group["delta_run_exp"].to_numpy(dtype=float)
        coefs = _fit_ols(X, y)
        fitted = coefs[0] + X @ coefs[1:]

        std = fitted.std()
        if std == 0:
            continue
        z = (fitted - fitted.mean()) / std

        chunk = group[["pitcher_id", "pitch_type", "stand"]].copy()
        chunk["stuff_plus"] = 100 - 10 * z
        scored.append(chunk)

    if not scored:
        return pd.DataFrame(columns=["pitcher_id", "pitch_type", "stand", "stuff_plus"])
    return pd.concat(scored)


def compute_stuff_plus(conn, season):
    """DataFrame indexed by (pitcher_id, pitch_type) with columns:
    n_pitches, stuff_plus - blended across both batter sides."""
    per_pitch = _score_pitches(conn, season)
    if per_pitch.empty:
        return pd.DataFrame(columns=["n_pitches", "stuff_plus"])
    return per_pitch.groupby(["pitcher_id", "pitch_type"]).agg(
        n_pitches=("stuff_plus", "size"),
        stuff_plus=("stuff_plus", "mean"),
    )


def compute_stuff_plus_by_stand(conn, season):
    """Same as compute_stuff_plus but also split by batter handedness -
    DataFrame indexed by (pitcher_id, pitch_type, stand) with columns:
    n_pitches, stuff_plus."""
    per_pitch = _score_pitches(conn, season)
    if per_pitch.empty:
        return pd.DataFrame(columns=["n_pitches", "stuff_plus"])
    return per_pitch.groupby(["pitcher_id", "pitch_type", "stand"]).agg(
        n_pitches=("stuff_plus", "size"),
        stuff_plus=("stuff_plus", "mean"),
    )


def overall_stuff_plus(stuff_plus_table, pitcher_id):
    """Usage-weighted average Stuff+ across every pitch type this pitcher
    throws, from the table returned by compute_stuff_plus. NaN if the
    pitcher has no pitch type that cleared MIN_PITCHES_PER_TYPE league-wide."""
    if pitcher_id not in stuff_plus_table.index.get_level_values("pitcher_id"):
        return np.nan
    rows = stuff_plus_table.xs(pitcher_id, level="pitcher_id")
    return np.average(rows["stuff_plus"], weights=rows["n_pitches"])

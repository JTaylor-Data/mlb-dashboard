"""
Seam-shifted-wake (SSW) / active-spin proxy for Stuff+ v3.

## The idea

Under a pure-Magnus assumption, a pitch's observed movement (hb, ivb) is
fully explained by its spin axis: the Magnus force is perpendicular to the
spin axis in the plane transverse to the ball's flight, so a "100%
active-spin" pitch's measured `spin_axis` and its movement-vector direction
have a fixed, physics-determined relationship. Real pitches deviate from
this - some of the total spin is "gyroscopic" (bullet-spin, contributes no
movement) and some of the OBSERVED movement comes from seam-shifted-wake
aerodynamic effects that have nothing to do with the measured spin axis at
all. The gap between "the spin axis pure Magnus would require to produce
this pitch's ACTUAL movement" and the REAL measured `spin_axis` is a proxy
for how much of the movement isn't explained by simple backspin/topspin/
side-spin alone - the same idea behind Driveline's/public "active spin"
research, just built from what Statcast exposes rather than lab
instrumentation.

## Why this is calibrated empirically, not hardcoded from a documented convention

Statcast's `spin_axis` clock-face convention (which physical direction
"12:00"/0deg corresponds to, and which rotation direction is positive) is
publicly documented, but trusting a from-memory recollection of the exact
mapping for a feature this consequential is exactly the kind of thing that
produces a silent, hard-to-catch sign/convention bug (the project's own
established lesson - see Tunneling+'s centroid-distance and ratio-coupling
bugs, both caught by checking real numbers, not by reasoning from theory
alone). Instead, the relationship is fit DIRECTLY from real data: four-seam
fastballs are the standard reference pitch in public spin-efficiency
research (typically close to "fully active" backspin for most pitchers), so
the mapping between raw movement angle and measured spin_axis is calibrated
on a large sample of real FF pitches, then applied to every pitch. This is
correct regardless of whether Statcast's own documented clock zero-point
matches what's assumed here - the calibration step absorbs it.

## Handedness

`hb` (horizontal break) is arm-side-adjusted elsewhere in this codebase
(`stuff_features.py`'s `_arm_side_sign`: RHP's raw pfx_x gets negated, LHP's
does not - positive hb always means "toward the pitcher's arm side" for
either hand). Statcast's raw `spin_axis` is NOT adjusted this way - it's a
fixed field-relative measurement, same physical clock for both hands. Since
the movement angle computed here is built from arm-side-adjusted `hb`,
`spin_axis` is mirrored with the EXACT SAME asymmetric rule
(`_arm_side_sign` uses) before comparison - RHP's spin_axis gets mirrored,
LHP's does not - so the final `axis_differential` is not itself a
handedness-coded feature. Getting this backwards (mirroring the wrong hand,
or mirroring both/neither) would silently reintroduce exactly the
handedness leakage `hb`'s own adjustment exists to prevent - flagged here
explicitly, and checked directly in `add_axis_differential`'s own sanity
output (LHP vs RHP axis_differential distributions should look similar
post-mirroring; a large systematic gap means this step is wrong).

## Calibration reference pitch type

Four-seam fastballs (FF) only - the standard "close to fully active spin"
reference pitch in public research. Sinkers/two-seamers are EXCLUDED from
calibration on purpose (they're a primary hypothesis for where real SSW
effects show up - calibrating against them would bake the very effect this
feature is trying to detect into the baseline).
"""
import numpy as np
import pandas as pd

CALIBRATION_PITCH_TYPE = "FF"
MIN_CALIBRATION_PITCHES = 5000


def _arm_side_sign(p_throws):
    """Same convention as stuff_features.py's _arm_side_sign - duplicated
    rather than imported to keep this module's physics self-contained and
    independently readable; same source-of-truth reasoning this project
    already accepts for other small, stable constants (e.g. FIP_CONSTANT
    duplication elsewhere)."""
    return np.where(p_throws == "R", -1, 1)


def _movement_angle_deg(hb, ivb):
    """Angle of the (hb, ivb) movement vector, degrees, 0-360, measured from
    the +ivb axis ("straight rise", 0deg) increasing toward +hb. This is an
    arbitrary but FIXED math convention - it does not need to match
    Statcast's own spin_axis clock convention, because the relationship
    between the two is calibrated empirically (see module docstring), not
    assumed."""
    return (np.degrees(np.arctan2(hb, ivb)) + 360) % 360


def _mirror_spin_axis(spin_axis, p_throws):
    """Mirrors RHP's raw spin_axis into the same arm-side-adjusted frame hb
    already uses (RHP: reflected across the 0/180 axis, matching hb's own
    sign flip for RHP; LHP: unchanged). Reflecting the x-component of a
    vector while holding y fixed maps a "clockwise-from-up" angle theta to
    (360 - theta) mod 360 - the same transform applied here."""
    sign = _arm_side_sign(p_throws)
    mirrored = (360 - spin_axis) % 360
    return np.where(sign == -1, mirrored, spin_axis)


def _circular_diff(a, b):
    """Signed shortest angular difference a-b, in (-180, 180]."""
    d = (a - b + 180) % 360 - 180
    return d


def fit_spin_axis_calibration(df):
    """
    Fits the empirical relationship between raw movement angle (from hb,
    ivb) and mirrored measured spin_axis, using only FF pitches (see module
    docstring for why FF specifically). Two candidate relationships are
    tested - same rotation direction (`mirrored_spin_axis ~ movement_angle
    + offset`) and reversed rotation direction (`mirrored_spin_axis ~
    -movement_angle + offset`) - since it isn't assumed in advance which
    one matches Statcast's real convention. Whichever gives a tighter
    circular fit (smaller mean absolute circular residual) is kept.

    Returns a dict: {"flip": bool, "offset_deg": float, "mean_abs_resid_deg": float,
    "n_calibration_pitches": int} - pass to `add_axis_differential()`. Fit
    ONCE on a training set and reused at scoring time, same pattern as
    `stuff_features.py`'s `fit_vaa_expectation()`/`add_vaa_residual()` split.
    """
    ff = df[df["pitch_type"] == CALIBRATION_PITCH_TYPE].dropna(subset=["hb", "ivb", "spin_axis", "p_throws"])
    if len(ff) < MIN_CALIBRATION_PITCHES:
        raise ValueError(f"Only {len(ff)} FF pitches available for calibration, need >= {MIN_CALIBRATION_PITCHES}")

    movement_angle = _movement_angle_deg(ff["hb"].to_numpy(dtype=float), ff["ivb"].to_numpy(dtype=float))
    mirrored_spin = _mirror_spin_axis(ff["spin_axis"].to_numpy(dtype=float), ff["p_throws"].to_numpy())

    results = {}
    for flip in (False, True):
        candidate = (-movement_angle if flip else movement_angle)
        resid = _circular_diff(mirrored_spin, candidate)
        offset = np.degrees(np.arctan2(np.mean(np.sin(np.radians(resid))), np.mean(np.cos(np.radians(resid)))))
        final_resid = _circular_diff(mirrored_spin, candidate + offset)
        mean_abs_resid = np.mean(np.abs(final_resid))
        results[flip] = {"offset_deg": float(offset), "mean_abs_resid_deg": float(mean_abs_resid)}

    best_flip = min(results, key=lambda k: results[k]["mean_abs_resid_deg"])
    return {
        "flip": best_flip,
        "offset_deg": results[best_flip]["offset_deg"],
        "mean_abs_resid_deg": results[best_flip]["mean_abs_resid_deg"],
        "n_calibration_pitches": int(len(ff)),
        "both_candidates": results,
    }


def add_axis_differential(df, calibration):
    """Attaches `axis_differential` (degrees, 0-180, unsigned magnitude
    only per v1 scope) to every row using a previously-fit calibration from
    `fit_spin_axis_calibration()`. Rows missing hb/ivb/spin_axis/p_throws
    get NaN, not a fabricated value."""
    df = df.copy()
    have = df["hb"].notna() & df["ivb"].notna() & df["spin_axis"].notna() & df["p_throws"].notna()

    movement_angle = _movement_angle_deg(df["hb"].to_numpy(dtype=float), df["ivb"].to_numpy(dtype=float))
    inferred = (-movement_angle if calibration["flip"] else movement_angle) + calibration["offset_deg"]
    inferred = inferred % 360
    mirrored_spin = _mirror_spin_axis(df["spin_axis"].to_numpy(dtype=float), df["p_throws"].to_numpy())

    raw_diff = np.abs(inferred - mirrored_spin) % 360
    angular_diff = np.minimum(raw_diff, 360 - raw_diff)

    df["axis_differential"] = np.where(have, angular_diff, np.nan).astype("float32")
    return df

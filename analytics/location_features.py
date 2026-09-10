"""
Location+ v1 - shared feature engineering. Fourth model in the "+" family
alongside Stuff+, Deception+, Tunneling+ - trained, not hand-built, for the
same reason all three of those are: consistency, and so a future combined
Pitching+ stack can treat all four as commensurable inputs.

## What this model measures, and what it deliberately does not

Location+ measures WHERE a pitch ended up and how valuable that location
was, GIVEN the count and situation - nothing about the pitch's physical
shape (Stuff+), delivery (Deception+), or relationship to other pitches in
the arsenal (Tunneling+). Per FanGraphs' own public Location+ methodology
(their pitching primer), their model deliberately excludes velocity,
movement, spin, and every other physical characteristic - location, pitch
type, and count only. This build follows the same separation principle.

**Deliberate deviation from FanGraphs' public methodology**: FanGraphs
excludes both `stand` (batter handedness) and `p_throws` (pitcher
handedness) from Location+ itself, reserving handedness effects for their
combined Pitching+ stack only. This build keeps `stand` IN Location+ - a
pitch's location value is genuinely relative to which side of the plate the
batter stands on (inside/outside are meaningless without knowing that), so
this is a deliberate choice, not an oversight. `p_throws` stays excluded:
pitcher handedness without the batter-handedness interaction mostly reflects
platoon matchup, a combined-model-level concept, not a location property on
its own - this follows FanGraphs' own reasoning on that specific point even
though `stand` itself is being kept in. See
`training/location_stand_ablation.py` for a direct empirical check of
whether keeping `stand` actually earns its keep.

## The known, accepted limitation - documented, not treated as a gap to close

Statcast doesn't capture catcher target or pitcher intent, only where the
pitch actually crossed. This model measures LOCATION-VALUE-ACHIEVED, which
conflates real command with outcome luck - a badly-missed pitch that
happens to land somewhere good scores well here, same as a well-executed
one. FanGraphs' own public writeup states they tested intent-based
(stringer) data for this exact purpose and found it did NOT add predictive
value over real-location data alone - cited here as precedent for accepting
this limitation rather than trying to work around it with data this project
doesn't have access to either.

## Target and scale

Trained against `delta_run_exp`, the same target Stuff+/Deception+/
Tunneling+ all use - deliberately, so all four models stay commensurable at
the pitch-outcome level for a future Pitching+ stack. Scored on the same
plus-scale convention as the other three (100 - 10*(pred-mean)/std, fit
fresh per season) - FanGraphs' own stated design principle is that 10
points of Stuff+, Location+, or Pitching+ is one standard deviation on the
pitch level, and this build follows that same convention exactly.
"""
import numpy as np
import pandas as pd

from analytics.stuff_features import EXCLUDED_PITCH_TYPES, exclude_junk_pitch_types, merge_pitch_type

NUMERIC_FEATURES = ["plate_x", "plate_z", "balls", "strikes"]
CATEGORICAL_FEATURES = ["pitch_type", "stand", "season"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Explicitly NOT features here (would leak physical-shape/delivery/
# tunneling signal into a model meant to isolate location-and-count value
# alone): release_speed, effective_speed, release_spin_rate, spin_axis, hb,
# ivb, vaa_residual, axis_differential, any *_sep_vs_* separation feature,
# any Deception+ mechanics feature, any Tunneling+ pairwise/sequential
# feature, and p_throws (see module docstring for why p_throws specifically
# is excluded even though stand is kept).

RAW_QUERY_COLUMNS = [
    "pitcher_id", "pitch_type", "stand", "season",
    "plate_x", "plate_z", "balls", "strikes", "delta_run_exp",
]


def engineer_features(df):
    """Full pipeline: merge KC->CU, exclude non-competitive pitch types
    (same EXCLUDED_PITCH_TYPES/exclude_junk_pitch_types Stuff+ uses -
    imported, not redefined, so the two models can't silently drift on
    what counts as junk), coerce categoricals, drop rows missing any
    required feature. `df` must already have RAW_QUERY_COLUMNS."""
    df = merge_pitch_type(df)
    df = exclude_junk_pitch_types(df)
    df = df.copy()
    df["season"] = df["season"].astype(str)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


# --------------------------------------------------------------------- #
# Shared empirical-Bayes shrinkage + plus-scale conversion - same
# methodology as analytics/stuff_model.py's own _shrinkage_params/
# _shrunk_group_means (Location+ is offline-only this session, so this
# lives here rather than in a live-serving module, reused by
# train_location_model.py, validate_location_model.py, and
# location_predictive_validity.py so the math isn't duplicated three
# times). NOT the same numeric K constants as Stuff+ - see module
# docstring and train_location_model.py for why Location+ needs its own,
# estimated fresh from its own predictions.
# --------------------------------------------------------------------- #

DEFAULT_SHRINKAGE_K = 100.0  # fallback for pitch types too rare to estimate their own - a much larger default than Stuff+'s 1.5, since Location+ is expected to stabilize far slower


def to_plus_scale(values, mean, std):
    return 100 - 10 * (values - mean) / std


def shrinkage_params(raw):
    """Empirical-Bayes shrinkage constants (K, in units of pitches),
    estimated fresh from `raw`'s own predictions - identical method-of-
    moments variance decomposition to stuff_model.py's _shrinkage_params,
    just run on Location+'s own prediction distribution rather than
    Stuff+'s. `raw` needs columns: pitcher_id, pitch_type, pred."""
    k_by_type = {}
    for pitch_type, group in raw.groupby("pitch_type", observed=True):
        by_pitcher = group.groupby("pitcher_id")["pred"].agg(["mean", "var", "count"])
        by_pitcher = by_pitcher[by_pitcher["count"] >= 2].dropna()
        if len(by_pitcher) < 10:
            continue
        noise_var = np.average(by_pitcher["var"], weights=by_pitcher["count"])
        weighted_mean = np.average(by_pitcher["mean"], weights=by_pitcher["count"])
        observed_var = np.average((by_pitcher["mean"] - weighted_mean) ** 2, weights=by_pitcher["count"])
        true_var = observed_var - noise_var / by_pitcher["count"].mean()
        if true_var > 0:
            k_by_type[pitch_type] = noise_var / true_var

    fallback_k = float(np.median(list(k_by_type.values()))) if k_by_type else DEFAULT_SHRINKAGE_K
    type_means = raw.groupby("pitch_type", observed=True)["pred"].mean()
    return k_by_type, fallback_k, type_means


def shrunk_group_means(raw, group_cols, k_by_type, fallback_k, type_means):
    """Same shrink-toward-pitch-type-mean logic as stuff_model.py's
    _shrunk_group_means. Returns group_cols + n_pitches + shrunk_mean
    (still in raw delta_run_exp units)."""
    g = raw.groupby(group_cols, observed=True)["pred"].agg(mean="mean", n_pitches="count").reset_index()
    pitch_type_str = g["pitch_type"].astype(str)
    g["type_mean"] = pitch_type_str.map(type_means.to_dict())
    g["k"] = pitch_type_str.map(k_by_type).fillna(fallback_k)
    reliability = g["n_pitches"] / (g["n_pitches"] + g["k"])
    g["shrunk_mean"] = g["type_mean"] + reliability * (g["mean"] - g["type_mean"])
    return g.drop(columns=["mean", "type_mean", "k"])

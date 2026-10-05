"""
Stuff+ - runtime scoring. Loads the model trained offline by
`training/train_stuff_v3_lgbm.py` (v3 architecture - see
`Methodology/STUFF_V3_REPORT.md` for the full methodology write-up) and
scores live pitch data through it. This file has NO fitting logic of its
own - see that training script for how the model was built and why, and
`stuff_features.py` for the feature pipeline both files share (that
sharing is what guarantees scoring computes features identically to how
they were computed at training time). The prior v2 model/pipeline is
preserved, unused, in `stuff_model_v2_legacy.py` / `stuff_features_v2_
legacy.py`.

## Public API (unchanged since v1 - the rest of the codebase doesn't need
to change at all when the underlying model is promoted)
- `compute_stuff_plus(conn, season)` - DataFrame indexed by
  (pitcher_id, pitch_type): n_pitches, stuff_plus.
- `compute_stuff_plus_by_stand(conn, season)` - same, plus split by `stand`.
- `overall_stuff_plus(stuff_plus_table, pitcher_id)` - usage-weighted
  average across a pitcher's arsenal. Pure aggregation, doesn't touch the
  model at all.
- `pitcher_shap_breakdown(...)` - per-pitch-type SHAP attribution for one
  pitcher, i.e. "which specific inputs (velo, movement, VAA shape,
  fastball separation, axis_differential, ...) are driving this guy's
  rating, and by how much" - powers the model internals page
  (`stuff-model.html`) so ratings aren't a black box.

## What changed in v3, and why
v3 removes 4 features that overlapped with Deception+ (`release_extension`,
`release_pos_x_as`, `release_pos_z`, `arm_angle`) and adds fastball
velo/movement SEPARATION features (a pitch vs. that pitcher-season's own
primary reference fastball) plus `axis_differential` (an empirically-
calibrated seam-shifted-wake proxy). Full methodology, ablation results,
and validation battery: `Methodology/STUFF_V3_REPORT.md`.

Standardization is still done WITHIN each requested season (mean/std of
that season's own predictions), matching the original "100 = average
pitcher, that season" convention - a model trained on pooled seasons
doesn't change what "100" means on the page.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from analytics.stuff_features import (
    RAW_QUERY_COLUMNS, add_movement_columns, build_fb_reference_table,
    engineer_features, merge_pitch_type,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ---------------------------------------------------------------------------
# Version switch — change this one constant to roll back to v3.
# Rollback: set STUFF_ACTIVE_VERSION = "v3"
# ---------------------------------------------------------------------------
STUFF_ACTIVE_VERSION = "v4"  # promoted 2026-10-02; rollback: set to "v3"

_MODEL_FILES = {
    "v3": ("stuff_plus_v3_lgbm.txt", "stuff_plus_v3_lgbm_meta.json"),
    "v4": ("stuff_plus_v4_lgbm.txt", "stuff_plus_v4_lgbm_meta.json"),
}

_model = None
_meta = None
_explainer = None
_loaded_version = None  # track which version is currently loaded


def _load():
    """Lazy-loaded module-level singletons - the booster and its metadata
    are read from disk once per process, not once per request."""
    global _model, _meta, _loaded_version
    if _model is None or _loaded_version != STUFF_ACTIVE_VERSION:
        model_file, meta_file = _MODEL_FILES[STUFF_ACTIVE_VERSION]
        _model = lgb.Booster(model_file=str(MODELS_DIR / model_file))
        _meta = json.loads((MODELS_DIR / meta_file).read_text())
        _loaded_version = STUFF_ACTIVE_VERSION
    return _model, _meta


def _get_explainer():
    global _explainer
    if _explainer is None:
        model, _ = _load()
        _explainer = shap.TreeExplainer(model)
    return _explainer


def _prepare_features(raw_df):
    """Runs the shared feature pipeline, then locks each categorical
    column to the EXACT category set/order seen at training time. This
    matters because LightGBM encodes pandas categoricals as integer codes
    based on `.cat.categories` order - if scoring built its own category
    list from just this season's data, a code could point to a different
    pitch type than it did during training. A value not seen in training
    (e.g. a brand-new pitch type after the model was last trained) becomes
    NaN, which LightGBM's native categorical handling treats as its own
    "missing" branch rather than crashing.

    Unlike v2, this also needs a fastball-reference table (Part A of the
    v3 feature pipeline) - built fresh from `raw_df` itself, the same way
    `training/build_stuff_v3_dataset.py` builds it: whichever pitcher-
    seasons are present in `raw_df` get their own primary_fb_type computed
    from THEIR OWN pitches in this exact query, not reused from training
    time. This is the correct behavior, not an approximation - a
    pitcher's primary fastball for a given season is a fact about that
    season's own pitches, and `_raw_predictions` always queries a full
    season at once, so every pitcher-season here has its complete data to
    classify from."""
    _, meta = _load()
    # vaa_coefs may be a plain list (legacy pooled) or a dict
    # {"pooled": [...], pitch_type: [...], ...} (per-type, added in v3.1).
    # add_vaa_residual in stuff_features.py handles both shapes.
    raw_vaa = meta["vaa_coefs"]
    if isinstance(raw_vaa, dict):
        vaa_coefs = raw_vaa  # already the per-type dict; keep as-is
    else:
        vaa_coefs = np.array(raw_vaa)  # legacy: convert to numpy array
    calibrations = {
        "vaa_coefs": vaa_coefs,
        "spin_calibration": meta["spin_calibration"],
        # haa_coefs present in v4 meta; absent in v3 meta (engineer_features
        # skips HAA computation when the key is missing or None).
        "haa_coefs": meta.get("haa_coefs"),
    }
    pre = merge_pitch_type(raw_df)
    pre = add_movement_columns(pre)
    fb_ref = build_fb_reference_table(pre)
    feat = engineer_features(raw_df, calibrations, fb_ref)
    for col, levels in meta["category_levels"].items():
        feat[col] = pd.Categorical(feat[col].astype(str), categories=levels)
    return feat


_raw_predictions_cache = {}  # (version, season) -> (dataframe, computed_at)
RAW_PREDICTIONS_CACHE_TTL_SECONDS = 600


def raw_predictions_if_cached(season):
    """Return cached raw predictions for a season if available and fresh, else None.
    Does NOT trigger computation — safe to call without blocking."""
    import time as _time
    cache_key = (STUFF_ACTIVE_VERSION, season)
    cached = _raw_predictions_cache.get(cache_key)
    if cached is not None and (_time.monotonic() - cached[1]) < RAW_PREDICTIONS_CACHE_TTL_SECONDS:
        return cached[0]
    return None


def _raw_predictions(conn, season):
    """Full-season per-pitch DataFrame: pitcher_id, pitch_type, stand, and
    the model's raw predicted delta_run_exp (`pred`, lower = better pitch
    from the batting team's perspective). This is the shared basis for
    `compute_stuff_plus`, `compute_stuff_plus_by_stand`, and
    `pitcher_shap_breakdown` - all three need the same season-wide
    prediction distribution to convert a raw prediction into a
    standardized Stuff+ number.

    Short-lived in-process memoization (10 min, not the API layer's usual
    12h - this is a narrow fix for one specific redundancy, not general
    request caching, which stays the API layer's job per this project's
    normal architecture): `warm_cache_background()` calls
    compute_stuff_plus() and compute_stuff_plus_by_stand() back-to-back
    on every cache refresh, and without this both would independently
    re-run the full season through the LightGBM model - a real, measurable
    chunk of what made "Refresh data" feel slow, doubling the single most
    expensive step in the whole refresh cycle for no reason."""
    cache_key = (STUFF_ACTIVE_VERSION, season)
    cached = _raw_predictions_cache.get(cache_key)
    if cached is not None and (time.monotonic() - cached[1]) < RAW_PREDICTIONS_CACHE_TTL_SECONDS:
        return cached[0]

    from analytics.stuff_features import RAW_QUERY_COLUMNS
    from config.qualifiers import regular_season_filter
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season = ? AND delta_run_exp IS NOT NULL {regular_season_filter()}
    """, conn, params=(season,))
    if df.empty:
        out = pd.DataFrame(columns=["pitcher_id", "pitch_type", "stand", "pred"])
        _raw_predictions_cache[cache_key] = (out, time.monotonic())
        return out
    feat = _prepare_features(df)
    model, meta = _load()
    preds = model.predict(feat[meta["all_features"]])
    out = feat[["pitcher_id", "pitch_type", "stand"]].copy()
    out["pred"] = preds
    _raw_predictions_cache[cache_key] = (out, time.monotonic())
    return out


def clear_cache():
    """Called after a data pull lands new pitches - without this, a
    refresh could show stale predictions for up to
    RAW_PREDICTIONS_CACHE_TTL_SECONDS even after the underlying data
    changed."""
    _raw_predictions_cache.clear()


DEFAULT_SHRINKAGE_K = 1.5  # fallback for pitch types too rare to estimate their own (see below)


def _shrinkage_params(raw):
    """
    Empirical-Bayes shrinkage parameters, estimated fresh from this
    season's own predictions (never hardcoded - a pitch type's natural
    pitch-to-pitch variability isn't a fixed constant).

    ## Why this is needed
    A pitcher's raw mean predicted value for a pitch type is itself noisy
    when the sample is small - not from contact-outcome luck (that's
    already gone; `pred` is the model's prediction, not the actual
    outcome), but because no pitcher throws a pitch with EXACTLY the same
    velo/movement/release every single time. A 1-2 pitch sample of a rare
    show-me pitch can land on an unrepresentative extreme purely from that
    natural variation - which is exactly what showed up in practice (a
    single-pitch "UN" sample scoring an extreme outlier). Shrinking a
    pitcher's small-sample mean toward the pitch type's league mean, by an
    amount that shrinks less as the sample grows, fixes this without
    touching well-sampled pitch types at all.

    ## The method
    Classic method-of-moments variance decomposition, per pitch type:
    the observed variance of pitchers' mean predictions = true
    between-pitcher variance + sampling noise variance / n. Solving for
    the true variance gives a reliability constant K (in units of
    "pitches") - the pitch count at which a pitcher's own sample and the
    league mean are weighted 50/50. Reliability at sample size n is then
    n / (n + K).

    Measured on 2025 data, K came out tiny (0.4-2.8 pitches) for every
    well-populated pitch type - meaning this metric stabilizes almost
    immediately once a pitcher has more than a handful of pitches, which
    matches an independent split-half reliability check (correlation
    0.98+ even at n=20). The pitch types where this matters in practice
    are the rare ones (a handful of pitchers leaguewide throw a screwball
    or eephus at all) - too few pitchers to even estimate their own K, so
    those fall back to the median K across pitch types that COULD be
    estimated (a reasonable default given how consistent the estimated
    values are with each other).

    Returns (k_by_type: dict, fallback_k: float, type_means: Series).
    """
    k_by_type = {}
    for pitch_type, group in raw.groupby("pitch_type", observed=True):
        by_pitcher = group.groupby("pitcher_id")["pred"].agg(["mean", "var", "count"])
        by_pitcher = by_pitcher[by_pitcher["count"] >= 2].dropna()
        if len(by_pitcher) < 10:
            continue  # not enough pitchers to trust a type-specific estimate
        noise_var = np.average(by_pitcher["var"], weights=by_pitcher["count"])
        weighted_mean = np.average(by_pitcher["mean"], weights=by_pitcher["count"])
        observed_var = np.average((by_pitcher["mean"] - weighted_mean) ** 2, weights=by_pitcher["count"])
        true_var = observed_var - noise_var / by_pitcher["count"].mean()
        if true_var > 0:
            k_by_type[pitch_type] = noise_var / true_var

    fallback_k = float(np.median(list(k_by_type.values()))) if k_by_type else DEFAULT_SHRINKAGE_K
    type_means = raw.groupby("pitch_type", observed=True)["pred"].mean()
    return k_by_type, fallback_k, type_means


def _shrunk_group_means(raw, group_cols, k_by_type, fallback_k, type_means):
    """Groups `raw` by `group_cols` (always including pitch_type) and
    shrinks each group's mean prediction toward its PITCH TYPE's league
    mean (not the grand mean across all types - a slider and a sinker
    have different baseline run values, so shrinking a rare slider sample
    toward the overall league average would bias it toward whatever pitch
    type happens to be most common). Returns group_cols + n_pitches +
    shrunk_mean (still in raw delta_run_exp units - _to_stuff_plus below
    converts to the display scale)."""
    g = raw.groupby(group_cols, observed=True)["pred"].agg(mean="mean", n_pitches="count").reset_index()
    pitch_type_str = g["pitch_type"].astype(str)
    g["type_mean"] = pitch_type_str.map(type_means.to_dict())
    g["k"] = pitch_type_str.map(k_by_type).fillna(fallback_k)
    reliability = g["n_pitches"] / (g["n_pitches"] + g["k"])
    g["shrunk_mean"] = g["type_mean"] + reliability * (g["mean"] - g["type_mean"])
    return g.drop(columns=["mean", "type_mean", "k"])


def _to_stuff_plus(values, norm_mean, norm_std):
    from analytics.location_features import to_plus_scale
    return to_plus_scale(values, norm_mean, norm_std)


def _type_norm_params(raw):
    """Per-pitch-type (mean, std) of raw pitch-level predictions.
    Used for within-type normalization so every pitch type centers at 100 -
    a great curveball grades against other curveballs, not against sliders.
    Falls back to global stats for unseen types."""
    params = {}
    global_mean = float(raw["pred"].mean())
    global_std = float(raw["pred"].std())
    for pt, group in raw.groupby("pitch_type", observed=True):
        s = float(group["pred"].std())
        params[str(pt)] = {
            "mean": float(group["pred"].mean()),
            "std": s if s > 0 else global_std,
        }
    params["_global"] = {"mean": global_mean, "std": global_std}
    return params


def compute_stuff_plus(conn, season):
    """DataFrame indexed by (pitcher_id, pitch_type) with columns:
    n_pitches, stuff_plus - blended across both batter sides. Small
    samples are shrunk toward the pitch type's league mean (see
    `_shrinkage_params`) before converting to the Stuff+ scale.
    Scores are normalized league-wide (all pitch types pooled) so that
    cross-type differences in average value remain visible in the output."""
    raw = _raw_predictions(conn, season)
    if raw.empty or raw["pred"].std() == 0:
        return pd.DataFrame(columns=["n_pitches", "stuff_plus"])
    k_by_type, fallback_k, type_means = _shrinkage_params(raw)
    g = _shrunk_group_means(raw, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
    norm_mean = float(raw["pred"].mean())
    norm_std = float(raw["pred"].std())
    g["stuff_plus"] = _to_stuff_plus(g["shrunk_mean"], norm_mean, norm_std)
    return g.set_index(["pitcher_id", "pitch_type"])[["n_pitches", "stuff_plus"]]


def compute_stuff_plus_if_cached(season):
    """Like compute_stuff_plus but returns None immediately if raw predictions
    aren't already in cache — never triggers model scoring. Pass None as conn."""
    if raw_predictions_if_cached(season) is None:
        return None
    # Cache is warm; _raw_predictions won't touch conn, so None is safe.
    return compute_stuff_plus(None, season)


def compute_stuff_plus_by_stand(conn, season):
    """Same as compute_stuff_plus but also split by batter handedness -
    DataFrame indexed by (pitcher_id, pitch_type, stand) with columns:
    n_pitches, stuff_plus. Shrinkage constants are estimated from the
    ALL-stand pool (splitting by stand first would leave too little data
    per pitch type to estimate K reliably) but applied to each stand
    split's own, smaller sample size - so a pitch type seen mostly vs. one
    side still gets appropriately more shrinkage on its thinner side.
    Scores are normalized league-wide (same as compute_stuff_plus)."""
    raw = _raw_predictions(conn, season)
    if raw.empty or raw["pred"].std() == 0:
        return pd.DataFrame(columns=["n_pitches", "stuff_plus"])
    k_by_type, fallback_k, type_means = _shrinkage_params(raw)
    g = _shrunk_group_means(raw, ["pitcher_id", "pitch_type", "stand"], k_by_type, fallback_k, type_means)
    norm_mean = float(raw["pred"].mean())
    norm_std = float(raw["pred"].std())
    g["stuff_plus"] = _to_stuff_plus(g["shrunk_mean"], norm_mean, norm_std)
    return g.set_index(["pitcher_id", "pitch_type", "stand"])[["n_pitches", "stuff_plus"]]


def overall_stuff_plus(stuff_plus_table, pitcher_id):
    """Usage-weighted average Stuff+ across every pitch type this pitcher
    throws, from the table returned by compute_stuff_plus."""
    if pitcher_id not in stuff_plus_table.index.get_level_values("pitcher_id"):
        return np.nan
    rows = stuff_plus_table.xs(pitcher_id, level="pitcher_id")
    return np.average(rows["stuff_plus"], weights=rows["n_pitches"])


def pitcher_shap_breakdown(conn, season, pitcher_id, league_raw):
    """
    Per-pitch-type SHAP attribution for one pitcher: which inputs are
    pushing their predicted run value down (good) or up (bad), and by how
    much, converted into the same Stuff+-point scale the rest of the site
    uses so it reads on the same terms as everything else.

    `league_raw` (the full season's `_raw_predictions`, cached at the API
    layer same as every other league-wide pool on this site) is the
    CALLER's job to provide - scoring one pitcher's ~1,000-3,000 pitches
    through SHAP is cheap; re-scoring the whole league's ~700K pitches on
    every single pitcher lookup would not be.

    The headline `stuff_plus` per pitch type is shrunk the same way
    `compute_stuff_plus` shrinks it (see `_shrinkage_params`), so this
    page never disagrees with the number shown on the player page. The
    SHAP `contributions`, however, explain the model's RAW (unshrunk)
    prediction - shrinkage is a statistical correction for small-sample
    noise, not something attributable to any input feature, so forcing it
    into the per-feature breakdown would misrepresent what's actually
    driving the model's output. For a low-n pitch type the contributions
    will sum to a slightly different number than (stuff_plus - 100); that
    gap IS the shrinkage adjustment.

    Returns {pitch_type: {n_pitches, stuff_plus, contributions: [(feature,
    stuff_plus_points, mean_feature_value), ...]}}, contributions sorted
    by absolute magnitude, largest first.
    """
    if league_raw.empty or league_raw["pred"].std() == 0:
        return {}
    k_by_type, fallback_k, type_means = _shrinkage_params(league_raw)
    norm_mean = float(league_raw["pred"].mean())
    norm_std = float(league_raw["pred"].std())
    # type_norm is still used to scale SHAP contributions into interpretable
    # Stuff+ point units per pitch type — but the headline stuff_plus number
    # uses the global norm (same as compute_stuff_plus) so the two agree.
    type_norm = _type_norm_params(league_raw)
    _tn_fallback = type_norm["_global"]

    from config.qualifiers import regular_season_filter
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season = ? AND pitcher_id = ? AND delta_run_exp IS NOT NULL {regular_season_filter()}
    """, conn, params=(season, pitcher_id))
    if df.empty:
        return {}

    feat = _prepare_features(df)
    model, meta = _load()
    features = meta["all_features"]
    preds = model.predict(feat[features])
    explainer = _get_explainer()
    shap_values = explainer.shap_values(feat[features])

    result = {}
    for pitch_type in feat["pitch_type"].cat.categories:
        mask = (feat["pitch_type"] == pitch_type).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        tn = type_norm.get(str(pitch_type), _tn_fallback)
        pt_norm_std = tn["std"]
        # SHAP contributions scale by -10/norm_std where norm_std is the
        # global league std (matching the headline Stuff+ normalization).
        # The per-type std is kept only as a fallback if global is missing.
        points_per_unit = -10 / norm_std
        mean_pred = preds[mask].mean()
        k = k_by_type.get(pitch_type, fallback_k)
        type_mean = type_means.get(pitch_type, tn["mean"])
        reliability = n / (n + k)
        shrunk_mean = type_mean + reliability * (mean_pred - type_mean)
        stuff_plus = float(_to_stuff_plus(shrunk_mean, norm_mean, norm_std))

        mean_shap = shap_values[mask].mean(axis=0)
        subset = feat.loc[mask, features]

        def _typical_value(f):
            # v3's separation features are legitimately NaN for every pitch
            # in a slice when this pitch type IS the pitcher's own primary
            # reference (self-comparison exclusion - see
            # stuff_features.py) - subset[f].mean() over an all-NaN column
            # returns NaN, which isn't valid JSON (Starlette's
            # JSONResponse raises rather than serializing a bare
            # float('nan')) and broke this endpoint with a 500 for any
            # pitcher whose primary-fastball pitch type was requested.
            # None (JSON null) is also the honest value here - there IS no
            # typical value for a feature absent from every pitch shown.
            if f not in meta["numeric_features"]:
                return subset[f].mode().iat[0]
            mean_val = subset[f].mean()
            return None if pd.isna(mean_val) else float(mean_val)

        typical_values = [_typical_value(f) for f in features]
        contributions = sorted(
            zip(features, (mean_shap * points_per_unit).tolist(), typical_values),
            key=lambda row: -abs(row[1]),
        )
        result[pitch_type] = dict(n_pitches=n, stuff_plus=stuff_plus, contributions=contributions)
    return result

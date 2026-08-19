"""
Stuff+ v2 - runtime scoring. Loads the model trained offline by
`train_stuff_model.py` and scores live pitch data through it. This file
has NO fitting logic of its own - see `train_stuff_model.py` for how the
model was built and why, and `stuff_features.py` for the feature pipeline
both files share (that sharing is what guarantees scoring computes
features identically to how they were computed at training time).

## Public API (unchanged from v1 - the rest of the codebase doesn't need
to change at all)
- `compute_stuff_plus(conn, season)` - DataFrame indexed by
  (pitcher_id, pitch_type): n_pitches, stuff_plus.
- `compute_stuff_plus_by_stand(conn, season)` - same, plus split by `stand`.
- `overall_stuff_plus(stuff_plus_table, pitcher_id)` - usage-weighted
  average across a pitcher's arsenal. Pure aggregation, doesn't touch the
  model at all - copied verbatim from v1.

## New in v2 (not in v1)
- `pitcher_shap_breakdown(...)` - per-pitch-type SHAP attribution for one
  pitcher, i.e. "which specific inputs (velo, movement, VAA shape, ...)
  are driving this guy's rating, and by how much" - powers the model
  internals page (`stuff-model.html`) so ratings aren't a black box.

## What changed vs. v1, and why
v1 fit a separate plain-OLS regression per pitch type, per season, with no
train/test split. v2 is ONE gradient-boosted model trained across all
pitch types and all seasons at once (see train_stuff_model.py's docstring
for the full reasoning - unified model + pitch_type as a feature is what
lets it learn that e.g. a steep approach angle plays differently on a
curveball than a fastball, without anyone hand-coding that interaction).
Because there's no more "not enough pitches to fit a per-type model"
concern, v2 has no equivalent of v1's `MIN_PITCHES_PER_TYPE` gate - even a
rarely-thrown pitch type gets scored through the shared model, borrowing
statistical strength from every other pitch type it saw in training.
Genuinely tiny samples (a show-me pitch thrown 5 times) are still noisy on
the DISPLAY side (a 5-pitch average is a 5-pitch average, same as any
stat), but that's now visible/inspectable rather than hidden entirely.

The one thing carried over unchanged: standardization is done WITHIN each
requested season (mean/std of that season's own predictions), matching
v1's "100 = average pitcher, that season" convention and keeping the site
consistent - a v2 model trained on six pooled seasons doesn't change what
"100" means on the page.
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from stuff_features import RAW_QUERY_COLUMNS, engineer_features

MODELS_DIR = Path(__file__).parent / "models"

_model = None
_meta = None
_explainer = None


def _load():
    """Lazy-loaded module-level singletons - the booster and its metadata
    are read from disk once per process, not once per request."""
    global _model, _meta
    if _model is None:
        _model = lgb.Booster(model_file=str(MODELS_DIR / "stuff_plus_v2.txt"))
        _meta = json.loads((MODELS_DIR / "stuff_plus_v2_meta.json").read_text())
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
    "missing" branch rather than crashing."""
    _, meta = _load()
    vaa_coefs = np.array(meta["vaa_coefs"])
    feat = engineer_features(raw_df, vaa_coefs)
    for col, levels in meta["category_levels"].items():
        feat[col] = pd.Categorical(feat[col].astype(str), categories=levels)
    return feat


_raw_predictions_cache = {}  # season -> (dataframe, computed_at) - see docstring below
RAW_PREDICTIONS_CACHE_TTL_SECONDS = 600


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
    cached = _raw_predictions_cache.get(season)
    if cached is not None and (time.monotonic() - cached[1]) < RAW_PREDICTIONS_CACHE_TTL_SECONDS:
        return cached[0]

    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season = ? AND delta_run_exp IS NOT NULL
    """, conn, params=(season,))
    if df.empty:
        out = pd.DataFrame(columns=["pitcher_id", "pitch_type", "stand", "pred"])
        _raw_predictions_cache[season] = (out, time.monotonic())
        return out
    feat = _prepare_features(df)
    model, meta = _load()
    preds = model.predict(feat[meta["all_features"]])
    out = feat[["pitcher_id", "pitch_type", "stand"]].copy()
    out["pred"] = preds
    _raw_predictions_cache[season] = (out, time.monotonic())
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


def _to_stuff_plus(values, season_mean, season_std):
    return 100 - 10 * (values - season_mean) / season_std


def compute_stuff_plus(conn, season):
    """DataFrame indexed by (pitcher_id, pitch_type) with columns:
    n_pitches, stuff_plus - blended across both batter sides. Small
    samples are shrunk toward the pitch type's league mean (see
    `_shrinkage_params`) before converting to the Stuff+ scale."""
    raw = _raw_predictions(conn, season)
    if raw.empty or raw["pred"].std() == 0:
        return pd.DataFrame(columns=["n_pitches", "stuff_plus"])
    k_by_type, fallback_k, type_means = _shrinkage_params(raw)
    g = _shrunk_group_means(raw, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
    g["stuff_plus"] = _to_stuff_plus(g["shrunk_mean"], raw["pred"].mean(), raw["pred"].std())
    return g.set_index(["pitcher_id", "pitch_type"])[["n_pitches", "stuff_plus"]]


def compute_stuff_plus_by_stand(conn, season):
    """Same as compute_stuff_plus but also split by batter handedness -
    DataFrame indexed by (pitcher_id, pitch_type, stand) with columns:
    n_pitches, stuff_plus. Shrinkage constants are estimated from the
    ALL-stand pool (splitting by stand first would leave too little data
    per pitch type to estimate K reliably) but applied to each stand
    split's own, smaller sample size - so a pitch type seen mostly vs. one
    side still gets appropriately more shrinkage on its thinner side."""
    raw = _raw_predictions(conn, season)
    if raw.empty or raw["pred"].std() == 0:
        return pd.DataFrame(columns=["n_pitches", "stuff_plus"])
    k_by_type, fallback_k, type_means = _shrinkage_params(raw)
    g = _shrunk_group_means(raw, ["pitcher_id", "pitch_type", "stand"], k_by_type, fallback_k, type_means)
    g["stuff_plus"] = _to_stuff_plus(g["shrunk_mean"], raw["pred"].mean(), raw["pred"].std())
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
    season_pred_mean, season_pred_std = league_raw["pred"].mean(), league_raw["pred"].std()
    k_by_type, fallback_k, type_means = _shrinkage_params(league_raw)

    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)} FROM pitches
        WHERE season = ? AND pitcher_id = ? AND delta_run_exp IS NOT NULL
    """, conn, params=(season, pitcher_id))
    if df.empty:
        return {}

    feat = _prepare_features(df)
    model, meta = _load()
    features = meta["all_features"]
    preds = model.predict(feat[features])
    explainer = _get_explainer()
    shap_values = explainer.shap_values(feat[features])

    # Predicted run value -> Stuff+ points uses the SAME z-score transform
    # as compute_stuff_plus (100 - 10*z). SHAP values are additive in the
    # model's raw output units, so a feature's contribution converts by
    # the same linear scale factor: -10/std per unit of raw prediction.
    # (Negative because LOWER predicted run value is BETTER stuff, i.e.
    # HIGHER Stuff+ - see the module docstring on delta_run_exp's sign.)
    points_per_unit = -10 / season_pred_std

    result = {}
    for pitch_type in feat["pitch_type"].cat.categories:
        mask = (feat["pitch_type"] == pitch_type).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred = preds[mask].mean()
        k = k_by_type.get(pitch_type, fallback_k)
        type_mean = type_means.get(pitch_type, season_pred_mean)
        reliability = n / (n + k)
        shrunk_mean = type_mean + reliability * (mean_pred - type_mean)
        stuff_plus = float(_to_stuff_plus(shrunk_mean, season_pred_mean, season_pred_std))

        mean_shap = shap_values[mask].mean(axis=0)
        subset = feat.loc[mask, features]
        typical_values = [
            float(subset[f].mean()) if f in meta["numeric_features"] else subset[f].mode().iat[0]
            for f in features
        ]
        contributions = sorted(
            zip(features, (mean_shap * points_per_unit).tolist(), typical_values),
            key=lambda row: -abs(row[1]),
        )
        result[pitch_type] = dict(n_pitches=n, stuff_plus=stuff_plus, contributions=contributions)
    return result

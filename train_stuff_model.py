"""
Stuff+ v2 - offline training script. Run by hand, NOT part of the API or
the "Refresh data" button (same category as `pull_prospects.py` - this is
a slow, deliberate job you re-run when you want a new model, not on every
page load):

    python train_stuff_model.py

## What this produces
Three files under `models/`:
  - `stuff_plus_v2.txt`   - the trained LightGBM booster (native text dump,
                            genuinely human-readable - open it and you can
                            read the actual tree splits).
  - `stuff_plus_v2_meta.json` - everything scoring needs to reproduce
                            training's feature pipeline exactly: the fitted
                            VAA-expectation coefficients, the feature list
                            and order, and the category levels LightGBM
                            saw for each categorical column.
  - `stuff_plus_v2_report.json` - the "show your work" artifact: per-season
                            cross-validation metrics, global feature
                            importance (both gain-based and SHAP-based),
                            and a few worked pitcher examples. This is what
                            the model-internals page in the frontend reads.

## Modeling choices, and why (matches the design discussion this is built from)
- Target: `delta_run_exp` (linear-weights run value assigned to EVERY
  pitch outcome - ball, called strike, whiff, foul, or batted ball), not
  raw xwOBA. True xwOBA only exists on pitches that end in a batted ball -
  training on that alone would throw away most of what makes a pitch good
  (whiffs and chases) and train on a biased subsample (pitches that get
  squared up are not a random sample of pitches). This is also what v1
  already used and what public Stuff+ models use.
- One unified model across all pitch types, with `pitch_type` as a
  categorical feature - not a separate regression per type like v1. A
  gradient-boosted tree builds interactions by splitting sequentially
  (e.g. split on pitch_type == curveball, THEN split on vaa_residual), so
  it can learn "steep VAA is good on breaking balls, neutral on
  fastballs" automatically, without anyone specifying that interaction by
  hand - the actual advantage over v1's per-type OLS.
- No pitcher identity as a feature, ever - see stuff_features.py's module
  docstring for why (the model would just memorize aces instead of
  learning what makes their pitches good).
- `season` IS a feature (categorical), deliberately not stripped out. The
  concern discussed was the 2021 sticky-substance crackdown shifting
  league-wide spin rates mid-year - if the model can see season, it can
  absorb any era-level shift into that feature rather than smearing it
  across spin_rate's learned relationship to run value. This does NOT
  suppress real quality differences between eras; a genuinely nastier
  pitch still finds its shape-based edge over its own season's population
  when scored (see "z-score within season" in stuff_model.py).
- Objective: Huber loss, not plain squared error. `delta_run_exp` has a
  heavy right tail (home runs are rare but huge), and squared error would
  let a handful of homers dominate what the model optimizes for. Huber
  behaves like squared error for small residuals and like absolute error
  for large ones, which is the standard fix for a heavy-tailed regression
  target.
- Validation: leave-one-season-out, rotating across the five COMPLETE
  seasons (2021-2025). 2026 is partial (in progress) so it's never used as
  a held-out evaluation season - evaluating against an incomplete season
  would compare against a weird, non-representative slice of the schedule
  - but it's still included in every fold's training data, since more
  real training data is only a good thing here. Each fold also refits the
  VAA-expectation regression (stuff_features.fit_vaa_expectation) using
  ONLY that fold's training data, so nothing about the held-out season
  leaks into feature construction, not just into the target.
"""
import gc
import json
import sqlite3
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import stuff_model as sm_runtime
from metrics import compute_vaa
from stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_QUERY_COLUMNS,
    engineer_features, fit_vaa_expectation,
)

DB_PATH = Path(__file__).parent / "mlb_data.db"
MODELS_DIR = Path(__file__).parent / "models"
COMPLETE_SEASONS = [2021, 2022, 2023, 2024, 2025]  # rotated as CV holdouts
ALL_SEASONS = COMPLETE_SEASONS + [2026]            # 2026 partial, train-only

LGBM_PARAMS = dict(
    objective="huber",       # robust to delta_run_exp's home-run-driven tail
    n_estimators=2000,       # capped in practice by early stopping below
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=200,   # don't let a leaf represent a handful of pitches
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
)


_STRING_COLS = {"pitch_type", "stand", "p_throws"}
_ID_COLS = {"pitcher_id", "season"}


def load_pitches(conn, seasons):
    """
    Loads only the given seasons - deliberately NOT a "load everything
    once, slice in memory per fold" helper. Loading the full multi-season
    dataset once and then also carving out per-fold train/test copies from
    it (the first version of this script did exactly that) means two
    near-full-size copies of a several-million-row DataFrame sit in memory
    at the same time - on a machine that doesn't have RAM to spare, that
    silently killed the process outright (no Python traceback, since it
    wasn't a clean MemoryError - just gone). Querying SQLite fresh per
    fold costs a few extra seconds of I/O; that's a much better trade than
    an unrecoverable, invisible crash.

    Also downcasts to float32 (numeric columns are float64 by default out
    of SQLite/pandas) and `category` dtype for the low-cardinality string
    columns - roughly halves the memory footprint of the resulting frame
    for a precision loss (float32) that's irrelevant at Statcast's actual
    measurement precision.
    """
    print(f"Loading pitches for seasons {seasons}...", flush=True)
    df = pd.read_sql(f"""
        SELECT {', '.join(RAW_QUERY_COLUMNS)}
        FROM pitches
        WHERE delta_run_exp IS NOT NULL AND season IN ({','.join('?' * len(seasons))})
    """, conn, params=seasons)
    for col in df.columns:
        if col in _STRING_COLS:
            df[col] = df[col].astype("category")
        elif col not in _ID_COLS:
            df[col] = df[col].astype("float32")
    print(f"  {len(df):,} rows loaded ({df.memory_usage(deep=True).sum() / 1e6:.0f} MB)", flush=True)
    return df


def _fit_one_model(train_df, feature_cols):
    """
    Fits one LightGBM model on `train_df`, carving off a random 8% of
    training rows purely to decide when to stop adding trees (early
    stopping) - not used for anything else, and never overlapping the
    actual evaluation season in the CV loop that calls this.
    """
    fit_rows, stop_rows = train_test_split(train_df, test_size=0.08, random_state=42)
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        fit_rows[feature_cols], fit_rows["delta_run_exp"],
        eval_set=[(stop_rows[feature_cols], stop_rows["delta_run_exp"])],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def run_cross_validation(conn):
    """
    Leave-one-season-out CV across the five complete seasons. Returns a
    list of per-season metric dicts (so the report can show both the
    average and how much it varies fold to fold - a feature/metric that
    holds up in every season is more trustworthy than one that only
    looked good in one).

    Each fold queries its own train/test data fresh from SQLite (see
    `load_pitches`'s docstring for why) and explicitly deletes+garbage-
    collects its DataFrames before moving to the next fold, so peak memory
    is "one fold's worth of data," not "every fold's worth, accumulated."
    """
    fold_reports = []
    for holdout_season in COMPLETE_SEASONS:
        print(f"\n--- CV fold: holding out {holdout_season} ---", flush=True)
        train_seasons = [s for s in ALL_SEASONS if s != holdout_season]
        train_raw = load_pitches(conn, train_seasons)
        test_raw = load_pitches(conn, [holdout_season])

        vaa_coefs = fit_vaa_expectation(
            train_raw.assign(vaa=lambda d: compute_vaa(d.vy0, d.vz0, d.ay, d.az))
        )
        train_feat = engineer_features(train_raw, vaa_coefs)
        test_feat = engineer_features(test_raw, vaa_coefs)
        del train_raw, test_raw

        # LightGBM needs identical category sets between train and eval;
        # union them explicitly so a value only seen in the holdout season
        # (e.g. a pitch type present in 2025 but rare in earlier years)
        # doesn't crash prediction.
        for col in CATEGORICAL_FEATURES:
            cats = sorted(set(train_feat[col].astype(str)) | set(test_feat[col].astype(str)))
            train_feat[col] = pd.Categorical(train_feat[col].astype(str), categories=cats)
            test_feat[col] = pd.Categorical(test_feat[col].astype(str), categories=cats)

        print(f"  fitting on {len(train_feat):,} rows, evaluating on {len(test_feat):,}...", flush=True)
        model = _fit_one_model(train_feat, ALL_FEATURES)
        preds = model.predict(test_feat[ALL_FEATURES])
        actual = test_feat["delta_run_exp"]

        r2 = r2_score(actual, preds)
        mae = mean_absolute_error(actual, preds)
        print(f"  holdout {holdout_season}: R^2={r2:.4f}  MAE={mae:.4f}  n={len(test_feat):,}  best_iter={model.best_iteration_}", flush=True)

        fold_reports.append(dict(
            holdout_season=holdout_season, r2=r2, mae=mae, n_pitches=len(test_feat),
            best_iteration=int(model.best_iteration_),
            feature_importance_gain=dict(zip(ALL_FEATURES, [float(x) for x in model.booster_.feature_importance("gain")])),
        ))
        del train_feat, test_feat, model
        gc.collect()
    return fold_reports


def train_final_model(conn):
    """Refits the VAA-expectation regression and the LightGBM model on
    EVERY available season (2021 through 2026-partial) - this is the
    model that actually ships and scores live requests. CV above never
    touches this model; it exists purely to estimate how well this
    approach generalizes before trusting it. Loaded fresh here (not reused
    from the CV loop) since the CV loop's per-fold frames are already
    freed by the time this runs - see `load_pitches`'s docstring."""
    print("\n--- Training final production model on all seasons ---", flush=True)
    raw_df = load_pitches(conn, ALL_SEASONS)
    vaa_coefs = fit_vaa_expectation(
        raw_df.assign(vaa=lambda d: compute_vaa(d.vy0, d.vz0, d.ay, d.az))
    )
    feat = engineer_features(raw_df, vaa_coefs)
    del raw_df
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype(str).astype("category")
    print(f"  fitting final model on {len(feat):,} rows...", flush=True)
    model = _fit_one_model(feat, ALL_FEATURES)
    print(f"  final model best_iteration={model.best_iteration_}", flush=True)
    return model, vaa_coefs, feat


def compute_shap_summary(model, feat, sample_size=30_000):
    """
    Mean absolute SHAP value per feature across a random sample of pitches
    - "on average, how much does this feature push a prediction away from
    the model's baseline, in either direction." This is more trustworthy
    than LightGBM's built-in gain importance when features are correlated
    (e.g. effective_speed is partly a function of release_speed and
    extension) - SHAP splits credit fairly using game-theoretic
    attribution rather than just "which feature got picked for a split
    first," which can arbitrarily favor one of two correlated features.
    """
    print(f"\n--- Computing SHAP summary on a sample of {sample_size:,} pitches ---", flush=True)
    sample = feat.sample(n=min(sample_size, len(feat)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample[ALL_FEATURES])
    mean_abs_shap = dict(zip(ALL_FEATURES, np.abs(shap_values).mean(axis=0).tolist()))
    return mean_abs_shap


def worked_examples(model, feat, vaa_coefs):
    """A handful of real, checkable pitches at the extremes (best and
    worst predicted run value) - a quick sanity spot-check that belongs in
    the report next to the raw metrics, same spirit as this project's
    existing habit of verifying against real facts (2021 Rays' record,
    Bobby Witt Jr.'s archetype, etc.)."""
    preds = model.predict(feat[ALL_FEATURES])
    feat = feat.assign(pred=preds)
    best = feat.nsmallest(5, "pred")[["pitcher_id", "pitch_type", "season", "release_speed", "hb", "ivb", "pred"]]
    worst = feat.nlargest(5, "pred")[["pitcher_id", "pitch_type", "season", "release_speed", "hb", "ivb", "pred"]]
    return dict(
        best_predicted_pitches=best.to_dict(orient="records"),
        worst_predicted_pitches=worst.to_dict(orient="records"),
    )


def shrinkage_report(final_feat, preds):
    """Empirical-Bayes shrinkage constants (see stuff_model.py's
    `_shrinkage_params` for the full derivation), reported so the
    model-internals page can show exactly how much correction small
    samples get and why - not just apply it silently."""
    raw = final_feat[["pitcher_id", "pitch_type"]].copy()
    raw["pred"] = preds
    k_by_type, fallback_k, _ = sm_runtime._shrinkage_params(raw)
    return dict(
        k_by_pitch_type={str(k): float(v) for k, v in k_by_type.items()},
        fallback_k=float(fallback_k),
        note=(
            "K = the pitch count at which a pitcher's own sample and the pitch "
            "type's league mean are weighted 50/50 (reliability = n/(n+K)) - "
            "smaller K means the metric stabilizes faster. Pitch types not "
            "listed had too few pitchers league-wide to estimate their own K "
            "and use the fallback instead."
        ),
    )


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    start = time.time()

    fold_reports = run_cross_validation(conn)
    model, vaa_coefs, final_feat = train_final_model(conn)
    shap_summary = compute_shap_summary(model, final_feat)
    examples = worked_examples(model, final_feat, vaa_coefs)
    shrinkage = shrinkage_report(final_feat, model.predict(final_feat[ALL_FEATURES]))

    model.booster_.save_model(str(MODELS_DIR / "stuff_plus_v2.txt"))

    meta = dict(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        all_features=ALL_FEATURES,
        vaa_coefs=vaa_coefs.tolist(),
        category_levels={col: final_feat[col].cat.categories.tolist() for col in CATEGORICAL_FEATURES},
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        seasons_trained_on=ALL_SEASONS,
        lgbm_params=LGBM_PARAMS,
    )
    (MODELS_DIR / "stuff_plus_v2_meta.json").write_text(json.dumps(meta, indent=2))

    report = dict(
        trained_at=meta["trained_at"],
        target="delta_run_exp (linear-weights run value per pitch, not xwOBA - see module docstring)",
        cv_folds=fold_reports,
        cv_r2_mean=float(np.mean([f["r2"] for f in fold_reports])),
        cv_mae_mean=float(np.mean([f["mae"] for f in fold_reports])),
        mean_abs_shap=shap_summary,
        gain_importance_final_model=dict(zip(ALL_FEATURES, [float(x) for x in model.booster_.feature_importance("gain")])),
        worked_examples=examples,
        shrinkage=shrinkage,
    )
    (MODELS_DIR / "stuff_plus_v2_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nDone in {time.time() - start:.0f}s. CV mean R^2={report['cv_r2_mean']:.4f}, mean MAE={report['cv_mae_mean']:.4f}")
    print(f"Saved: {MODELS_DIR / 'stuff_plus_v2.txt'}")
    print(f"Saved: {MODELS_DIR / 'stuff_plus_v2_meta.json'}")
    print(f"Saved: {MODELS_DIR / 'stuff_plus_v2_report.json'}")


if __name__ == "__main__":
    main()

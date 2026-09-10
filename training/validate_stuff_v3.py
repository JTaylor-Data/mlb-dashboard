"""
Stuff+ v3 - remaining validation battery: year-over-year score stability,
by-pitch-type distribution, cross-model correlation (vs Deception+/
Tunneling+), and named-pitcher SHAP collinearity spot-checks. Reuses the
trained models/stuff_plus_v3_lgbm.txt (does not retrain).
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy import stats

from analytics.stuff_features import ALL_FEATURES, CATEGORICAL_FEATURES
from analytics import stuff_model as sm_runtime
from analytics import deception_model

DB_PATH = "mlb_data.db"
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("training/stuff_v3_artifacts")
FEATURES_PATH = ARTIFACTS_DIR / "stuff_v3_features.parquet"
MIN_PITCHES_QUALIFY = 200

SPOT_CHECK_PITCHERS = [
    (605280, "Clay Holmes"),       # known real sinkerballer - hb/ivb-heavy separation profile
    (694819, "Jacob Misiorowski"), # elite FF velo, real cutter usage excluded as a reference (1-ref)
    (543243, "Sonny Gray"),        # sinkerball-primary FB VAA case from earlier session's work
]


def load_final_model():
    booster = lgb.Booster(model_file=str(MODELS_DIR / "stuff_plus_v3_lgbm.txt"))
    return booster


def score_all_pitches(booster, feat):
    """Scores every pitch with the final v3 model, returns per-(pitcher,
    pitch_type,season) shrunk stuff_plus, mirroring compute_stuff_plus's
    own shrinkage+z-score approach (reused directly, model-agnostic)."""
    preds = booster.predict(feat[ALL_FEATURES])
    raw = feat[["pitcher_id", "pitch_type", "season"]].copy()
    raw["pred"] = preds
    per_season = []
    for season, group in raw.groupby("season"):
        k_by_type, fallback_k, type_means = sm_runtime._shrinkage_params(group)
        g = sm_runtime._shrunk_group_means(group, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
        g["stuff_plus"] = sm_runtime._to_stuff_plus(g["shrunk_mean"], group["pred"].mean(), group["pred"].std())
        # _shrunk_group_means only returns group_cols + n_pitches/shrunk_mean/
        # stuff_plus - season isn't one of the group_cols, so it has to be
        # reattached explicitly before concatenating across seasons, or
        # every row silently loses which season it belongs to.
        g["season"] = season
        per_season.append(g)
    by_type = pd.concat(per_season, ignore_index=True)

    totals = by_type.groupby(["pitcher_id", "season"])["n_pitches"].sum().rename("n_pitches")
    weighted = by_type.groupby(["pitcher_id", "season"]).apply(
        lambda d: np.average(d["stuff_plus"], weights=d["n_pitches"]), include_groups=False
    )
    overall = pd.DataFrame({"stuff_plus": weighted}).join(totals).reset_index()
    overall = overall[overall["n_pitches"] >= MIN_PITCHES_QUALIFY]
    return overall, by_type, raw


def year_over_year(overall):
    seasons = sorted(overall["season"].unique())
    results = []
    for i in range(len(seasons) - 1):
        t, t1 = seasons[i], seasons[i + 1]
        dt = overall[overall["season"] == t][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t"})
        dt1 = overall[overall["season"] == t1][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t1"})
        merged = dt.merge(dt1, on="pitcher_id", how="inner")
        if len(merged) < 5:
            continue
        slope, intercept, r, p, se = stats.linregress(merged["t"], merged["t1"])
        results.append({"season_t": t, "season_t1": t1, "n": len(merged), "r": round(float(r), 3), "r2": round(float(r**2), 3)})
    return results


def by_pitch_type(by_type, season="2025"):
    sub = by_type[by_type["season"] == season]
    agg = sub.groupby("pitch_type").apply(
        lambda d: pd.Series({
            "avg_stuff_plus": np.average(d["stuff_plus"], weights=d["n_pitches"]),
            "n_pitches": d["n_pitches"].sum(),
            "n_pitchers": d["pitcher_id"].nunique(),
        }), include_groups=False
    ).sort_values("n_pitches", ascending=False)
    return agg.reset_index().to_dict("records")


def cross_model_correlation(conn, v3_overall, season="2025"):
    v3 = v3_overall[v3_overall["season"] == season][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "stuff_v3"})

    dp_raw = deception_model._raw_predictions(conn, int(season))
    season_mean, season_std = dp_raw["pred"].mean(), dp_raw["pred"].std()
    dp = dp_raw.groupby("pitcher_id")["pred"].agg(mean="mean", n_pitches="count").reset_index()
    dp["deception_plus"] = deception_model._to_plus_scale(dp["mean"], season_mean, season_std)
    dp = dp[dp["n_pitches"] >= MIN_PITCHES_QUALIFY][["pitcher_id", "deception_plus"]]

    try:
        with open(f"models/tunneling_v1_pitchers_{season}.json", encoding="utf-8") as f:
            tj = json.load(f)
        tp = pd.DataFrame([{"pitcher_id": int(pid), "tunneling_plus": v["tunneling_plus"]} for pid, v in tj.items()])
    except FileNotFoundError:
        tp = pd.DataFrame(columns=["pitcher_id", "tunneling_plus"])

    merged = v3.merge(dp, on="pitcher_id", how="inner").merge(tp, on="pitcher_id", how="inner")
    corr = merged[["stuff_v3", "deception_plus", "tunneling_plus"]].corr()
    return corr.round(4).to_dict(), len(merged)


def shap_spot_checks(booster, feat):
    print("\n=== Named-pitcher SHAP spot-checks (collinearity check) ===")
    explainer = shap.TreeExplainer(booster)
    results = {}
    for pid, name in SPOT_CHECK_PITCHERS:
        rows = feat[(feat["pitcher_id"] == pid) & (feat["season"] == "2026")]
        if rows.empty:
            rows = feat[(feat["pitcher_id"] == pid) & (feat["season"] == "2025")]
        if rows.empty:
            print(f"  {name}: not found")
            continue
        shap_values = explainer.shap_values(rows[ALL_FEATURES])
        mean_shap = dict(zip(ALL_FEATURES, shap_values.mean(axis=0).tolist()))
        top = sorted(mean_shap.items(), key=lambda x: -abs(x[1]))[:8]
        print(f"\n{name} ({len(rows)} pitches, season {rows['season'].iloc[0]}):")
        for feat_name, val in top:
            print(f"  {feat_name}: {val:+.4f} (raw delta_run_exp units)")
        results[name] = {"n_pitches": int(len(rows)), "top_shap": top}
    return results


def main():
    conn = sqlite3.connect(DB_PATH)
    booster = load_final_model()
    feat = pd.read_parquet(FEATURES_PATH)
    # A raw lgb.Booster reloaded from its native text dump (not the sklearn
    # wrapper) doesn't carry the pandas_categorical metadata .fit() attaches
    # automatically - predict() needs each categorical column to be a real
    # `category` dtype with the EXACT SAME category levels seen at training
    # time, or it raises "train and valid dataset categorical_feature do
    # not match." Reconstructed here from the meta JSON saved alongside the
    # model, not just cast to str (which was the actual bug on first run).
    with open(MODELS_DIR / "stuff_plus_v3_lgbm_meta.json") as f:
        meta = json.load(f)
    for col in CATEGORICAL_FEATURES:
        feat[col] = pd.Categorical(feat[col].astype(str), categories=meta["category_levels"][col])

    print("Scoring all pitches with final v3 model...", flush=True)
    overall, by_type, raw = score_all_pitches(booster, feat)
    print(f"  {len(overall):,} qualified pitcher-seasons")

    print("\n=== Year-over-year (v3 score t vs t+1) ===")
    yoy = year_over_year(overall)
    for r in yoy:
        print(f"  {r['season_t']}->{r['season_t1']}: n={r['n']}, r={r['r']}, r2={r['r2']}")

    print("\n=== By pitch type, 2025 ===")
    bpt = by_pitch_type(by_type)
    for r in bpt:
        print(f"  {r['pitch_type']}: {r['avg_stuff_plus']:.1f} (n_pitches={r['n_pitches']:.0f}, n_pitchers={r['n_pitchers']:.0f})")

    print("\n=== Cross-model correlation, 2025 ===")
    corr, n = cross_model_correlation(conn, overall)
    print(f"  n={n} pitchers in all 3 models")
    print(pd.DataFrame(corr).round(3))

    shap_results = shap_spot_checks(booster, feat)

    out = {
        "year_over_year": yoy,
        "by_pitch_type_2025": bpt,
        "cross_model_correlation_2025": {"corr": corr, "n": n},
        "shap_spot_checks": shap_results,
    }
    with open(ARTIFACTS_DIR / "v3_validation_battery.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {ARTIFACTS_DIR / 'v3_validation_battery.json'}")


if __name__ == "__main__":
    main()

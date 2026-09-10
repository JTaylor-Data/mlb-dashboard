"""
Validation battery for the PO/UN/EP pitch-type exclusion (Stuff+ v3,
retrained with EXCLUDED_PITCH_TYPES applied in analytics/stuff_features.py's
engineer_features). Does NOT retrain the production model (that's already
done by training/train_stuff_v3_lgbm.py against the rebuilt, exclusion-
applied feature table) - this script only validates it, plus trains ONE
extra one-off out-of-sample model (mirrors validate_predictive_power.py's
methodology) for the predictive-validity check.

Covers Part C items 2-8 of the exclusion task. Item 1 (season mean/std
before/after) was computed directly against the live pre-retrain v3 model
and is documented in the write-up, not recomputed here - it doesn't depend
on anything this script builds.

Reuses validate_stuff_v3.py's already-tested functions (score_all_pitches,
year_over_year, by_pitch_type, cross_model_correlation) rather than
reimplementing them, so results are directly comparable to what's already
on record.
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
from scipy import stats

from analytics.percentiles import pitcher_percentile_metrics
from analytics.stuff_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, EXCLUDED_PITCH_TYPES,
)
import analytics.stuff_model as sm_runtime
from training.train_stuff_v3_lgbm import _fit_one_model, refit_vaa_per_fold
from training.validate_stuff_v3 import (
    by_pitch_type, cross_model_correlation, load_final_model, score_all_pitches,
)

DB_PATH = "mlb_data.db"
ARTIFACTS_DIR = Path("training/stuff_v3_artifacts")
FEATURES_PATH = ARTIFACTS_DIR / "stuff_v3_features.parquet"
OUT_PATH = ARTIFACTS_DIR / "pitch_type_exclusion_validation.json"

COMPLETE_SEASONS = [2021, 2022, 2023, 2024, 2025]
MIN_PITCHES_QUALIFY = 200
STATS_URL = "https://statsapi.mlb.com/api/v1/stats"

OUT_OF_SAMPLE_SCORE_SEASON = 2024
OUT_OF_SAMPLE_OUTCOME_SEASON = 2025
OUT_OF_SAMPLE_TRAIN_SEASONS = [2021, 2022, 2023, 2025, 2026]
PREDICTIVE_OUTCOMES = ["xwoba", "whiff_pct", "swstr_pct", "k_minus_bb_pct", "chase_pct"]


# --------------------------------------------------------------- C.2: YoY --

def pooled_and_per_pair_yoy(overall):
    """Same per-pair, individual-pitcher-matched r^2 as validate_stuff_v3's
    year_over_year (merge on pitcher_id within each consecutive season
    pair), PLUS a genuine pooled r^2: every (pitcher, season_t, season_t1)
    matched pair from every consecutive season, stacked into one sample and
    regressed once - not an average of the 5 per-pair r^2 values, an
    actual single regression on the union. These can differ (pooling
    weights each pair by its own qualified-pitcher count, an average of
    per-pair r^2 doesn't)."""
    seasons = sorted(overall["season"].unique())
    per_pair = []
    pooled_t, pooled_t1 = [], []
    for i in range(len(seasons) - 1):
        t, t1 = seasons[i], seasons[i + 1]
        dt = overall[overall["season"] == t][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t"})
        dt1 = overall[overall["season"] == t1][["pitcher_id", "stuff_plus"]].rename(columns={"stuff_plus": "t1"})
        merged = dt.merge(dt1, on="pitcher_id", how="inner")
        if len(merged) < 5:
            continue
        slope, intercept, r, p, se = stats.linregress(merged["t"], merged["t1"])
        per_pair.append({"season_t": t, "season_t1": t1, "n": len(merged), "r": round(float(r), 3), "r2": round(float(r ** 2), 3)})
        pooled_t.extend(merged["t"].tolist())
        pooled_t1.extend(merged["t1"].tolist())

    slope, intercept, r, p, se = stats.linregress(pooled_t, pooled_t1)
    pooled = {"n_pooled_pairs": len(pooled_t), "r": round(float(r), 3), "r2": round(float(r ** 2), 3)}
    mean_of_per_pair_r2 = float(np.mean([p_["r2"] for p_ in per_pair])) if per_pair else float("nan")
    return {"per_pair": per_pair, "pooled": pooled, "mean_of_per_pair_r2": round(mean_of_per_pair_r2, 3)}


# ------------------------------------------------------- C.3/C.4: real stats --

def fetch_real_era_bulk(season):
    """Every pitcher who threw a pitch that season, with their real season
    ERA - MLB Stats API's own leaderboard endpoint (playerPool=ALL, paged),
    same trusted source season_era() already uses per-player elsewhere in
    this app, just batched here since we need every qualified pitcher, not
    one at a time (~870 pitchers/season in one paged loop instead of one
    HTTP call per pitcher)."""
    all_stats, offset, limit = [], 0, 200
    while True:
        resp = requests.get(STATS_URL, params={
            "stats": "season", "group": "pitching", "season": season, "sportId": 1,
            "limit": limit, "offset": offset, "playerPool": "ALL",
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        splits = data["stats"][0]["splits"]
        all_stats.extend(splits)
        total = data["stats"][0]["totalSplits"]
        offset += limit
        if offset >= total:
            break
    rows = [
        {"pitcher_id": s["player"]["id"], "era": s["stat"].get("era")}
        for s in all_stats if s["stat"].get("era") is not None
    ]
    df = pd.DataFrame(rows)
    df["era"] = pd.to_numeric(df["era"], errors="coerce")
    return df.dropna(subset=["era"])


def overall_stuff_plus_by_pitcher_season(overall):
    """overall (from score_all_pitches) is already usage-weighted per
    (pitcher_id, season) - just the columns this section needs, renamed
    for clarity when merged against real stats below."""
    return overall[["pitcher_id", "season", "stuff_plus", "n_pitches"]].copy()


def regress_stuff_on_real_stat(stuff_by_season, real_stat_df, stat_col, seasons):
    """Pools qualified pitcher-seasons across every season in `seasons`
    into ONE regression of stuff_plus -> stat_col (not per-season - a
    single r^2/coefficient/n, matching the task's ask for one number)."""
    pooled = []
    for season in seasons:
        real = real_stat_df.get(season)
        if real is None or real.empty:
            continue
        s = stuff_by_season[(stuff_by_season["season"] == str(season)) & (stuff_by_season["n_pitches"] >= MIN_PITCHES_QUALIFY)]
        merged = s.merge(real, on="pitcher_id", how="inner")
        pooled.append(merged)
    if not pooled:
        return {"n": 0, "r2": None, "coef": None, "note": "no seasons with real data available"}
    all_merged = pd.concat(pooled, ignore_index=True)
    slope, intercept, r, p, se = stats.linregress(all_merged["stuff_plus"], all_merged[stat_col])
    return {
        "n": int(len(all_merged)), "r2": round(float(r ** 2), 4), "r": round(float(r), 4),
        "coef": round(float(slope), 5), "coef_direction_negative": bool(slope < 0),
        "seasons_included": sorted(all_merged["season"].unique().tolist()) if "season" in all_merged else None,
    }


# ------------------------------------------------- C.5: out-of-sample validity --

def train_out_of_sample_excl_model(full_df):
    """One-off model trained on every COMPLETE_SEASONS season except
    OUT_OF_SAMPLE_SCORE_SEASON (plus 2026, same convention as the CV
    folds/production model - 2026 is partial/train-only, never a holdout).
    Mirrors validate_predictive_power.py's out-of-sample methodology
    exactly, applied to the exclusion-filtered v3 architecture instead of
    v1/v2, so this is a single-model check, not a 3-way comparison - the
    numbers it produces are meant to be read next to the already-recorded
    v1/v2 predictive-validity report, not inside the same run."""
    train_df = full_df[full_df["season"] != str(OUT_OF_SAMPLE_SCORE_SEASON)].copy()
    score_df = full_df[full_df["season"] == str(OUT_OF_SAMPLE_SCORE_SEASON)].copy()
    train_df, score_df, _ = refit_vaa_per_fold(train_df, score_df)

    for col in CATEGORICAL_FEATURES:
        cats = sorted(set(train_df[col].astype(str)) | set(score_df[col].astype(str)))
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
        score_df[col] = pd.Categorical(score_df[col].astype(str), categories=cats)

    print(f"  fitting out-of-sample model on {len(train_df):,} rows (excludes {OUT_OF_SAMPLE_SCORE_SEASON})...", flush=True)
    model = _fit_one_model(train_df, ALL_FEATURES)
    preds = model.predict(score_df[ALL_FEATURES])
    out = score_df[["pitcher_id", "pitch_type", "stand"]].copy()
    out["pred"] = preds
    return out


def out_of_sample_predictive_validity(conn, raw_2024_preds):
    k_by_type, fallback_k, type_means = sm_runtime._shrinkage_params(raw_2024_preds)
    g = sm_runtime._shrunk_group_means(raw_2024_preds, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
    g["stuff_plus"] = sm_runtime._to_stuff_plus(g["shrunk_mean"], raw_2024_preds["pred"].mean(), raw_2024_preds["pred"].std())

    flat = g.reset_index() if g.index.name else g
    overall = flat.groupby("pitcher_id").apply(
        lambda d: pd.Series({
            "stuff_plus_2024": np.average(d["stuff_plus"], weights=d["n_pitches"]),
            "n_2024_pitches": d["n_pitches"].sum(),
        }), include_groups=False,
    ).reset_index()

    print("  loading real 2025 outcomes...", flush=True)
    pool = pitcher_percentile_metrics(conn, OUT_OF_SAMPLE_OUTCOME_SEASON, stand_filter=None).reset_index()
    pool = pool.rename(columns={pool.columns[0]: "pitcher_id"})
    merged = overall.merge(pool, on="pitcher_id", how="inner")

    results_by_min_pitches = {}
    for min_n in [100, 200, 300]:
        sub = merged[merged["n_2024_pitches"] >= min_n]
        correlations = {outcome: round(float(sub["stuff_plus_2024"].corr(sub[outcome])), 3) for outcome in PREDICTIVE_OUTCOMES}
        results_by_min_pitches[str(min_n)] = {"n_qualified": int(len(sub)), "correlations": correlations}
    return results_by_min_pitches


# ------------------------------------------------------------- C.8: sanity --

def sample_size_sanity(conn):
    raw = pd.read_sql(f"""
        SELECT season, pitcher_id, pitch_type, COUNT(*) AS n
        FROM pitches
        WHERE season IN ({','.join('?' * len(COMPLETE_SEASONS))}) AND pitch_type IN ('PO','UN','EP')
        GROUP BY season, pitcher_id, pitch_type
    """, conn, params=COMPLETE_SEASONS)
    by_type = raw.groupby("pitch_type")["n"].sum().to_dict()
    total_pitches_excluded = int(raw["n"].sum())
    total_pitcher_seasons_touched = raw[["season", "pitcher_id"]].drop_duplicates().shape[0]
    return {
        "seasons_checked": COMPLETE_SEASONS,
        "total_pitches_excluded": total_pitches_excluded,
        "by_pitch_type": {k: int(v) for k, v in by_type.items()},
        "distinct_pitcher_seasons_with_at_least_one_excluded_pitch": int(total_pitcher_seasons_touched),
    }


def main():
    conn = sqlite3.connect(DB_PATH)
    booster = load_final_model()
    feat = pd.read_parquet(FEATURES_PATH)
    with open("models/stuff_plus_v3_lgbm_meta.json") as f:
        meta = json.load(f)
    for col in CATEGORICAL_FEATURES:
        feat[col] = pd.Categorical(feat[col].astype(str), categories=meta["category_levels"][col])

    assert not (set(feat["pitch_type"].astype(str).unique()) & EXCLUDED_PITCH_TYPES), \
        "Excluded pitch types still present in the feature table - exclusion did not take effect."

    print("Scoring all pitches with final (post-exclusion) v3 model...", flush=True)
    overall, by_type, raw = score_all_pitches(booster, feat)
    print(f"  {len(overall):,} qualified pitcher-seasons")

    print("\n=== C.2: Year-over-year (individual-pitcher-matched, pooled + per-pair) ===")
    yoy = pooled_and_per_pair_yoy(overall)
    for r in yoy["per_pair"]:
        print(f"  {r['season_t']}->{r['season_t1']}: n={r['n']}, r={r['r']}, r2={r['r2']}")
    print(f"  POOLED across all pairs: n={yoy['pooled']['n_pooled_pairs']}, r={yoy['pooled']['r']}, r2={yoy['pooled']['r2']}")
    print(f"  (mean of per-pair r2s: {yoy['mean_of_per_pair_r2']})")

    print("\n=== C.3/C.4: Stuff+ vs real xFIP / real ERA (pooled across seasons) ===")
    stuff_by_season = overall_stuff_plus_by_pitcher_season(overall)
    real_xfip_by_season = {}
    real_era_by_season = {}
    for season in COMPLETE_SEASONS:
        xfip_df = pd.read_sql(
            "SELECT player_id AS pitcher_id, xfip FROM player_advanced_stats WHERE season = ? AND xfip IS NOT NULL",
            conn, params=(season,),
        )
        real_xfip_by_season[season] = xfip_df
        print(f"  season {season}: {len(xfip_df)} real xFIP rows available")
        era_df = fetch_real_era_bulk(season)
        real_era_by_season[season] = era_df
        print(f"  season {season}: {len(era_df)} real ERA rows fetched")

    xfip_result = regress_stuff_on_real_stat(stuff_by_season, real_xfip_by_season, "xfip", COMPLETE_SEASONS)
    era_result = regress_stuff_on_real_stat(stuff_by_season, real_era_by_season, "era", COMPLETE_SEASONS)
    print(f"  Stuff+ vs real xFIP: n={xfip_result['n']}, r2={xfip_result['r2']}, coef={xfip_result['coef']} (negative expected: {xfip_result.get('coef_direction_negative')})")
    print(f"  Stuff+ vs real ERA:  n={era_result['n']}, r2={era_result['r2']}, coef={era_result['coef']} (negative expected: {era_result.get('coef_direction_negative')})")

    print(f"\n=== C.5: Out-of-sample predictive validity ({OUT_OF_SAMPLE_SCORE_SEASON} Stuff+ vs real {OUT_OF_SAMPLE_OUTCOME_SEASON} outcomes) ===")
    raw_2024 = train_out_of_sample_excl_model(feat)
    predictive_validity = out_of_sample_predictive_validity(conn, raw_2024)
    for min_n, r in predictive_validity.items():
        print(f"  min {min_n} pitches (n={r['n_qualified']}): {r['correlations']}")

    print(f"\n=== C.6: By-pitch-type distribution, post-exclusion, {2025} ===")
    bpt = by_pitch_type(by_type, season="2025")
    for r in bpt:
        print(f"  {r['pitch_type']}: {r['avg_stuff_plus']:.1f} (n_pitches={r['n_pitches']:.0f}, n_pitchers={r['n_pitchers']:.0f})")
    excluded_still_present = [r["pitch_type"] for r in bpt if r["pitch_type"] in EXCLUDED_PITCH_TYPES]
    print(f"  excluded types still appearing: {excluded_still_present} (should be empty)")

    print(f"\n=== C.7: Cross-model correlation, {2025} ===")
    corr, n = cross_model_correlation(conn, overall)
    print(f"  n={n} pitchers in all 3 models")
    print(pd.DataFrame(corr).round(3))

    print("\n=== C.8: Sample-size sanity check ===")
    sanity = sample_size_sanity(conn)
    print(json.dumps(sanity, indent=2))

    out = {
        "yoy": yoy,
        "stuff_vs_real_xfip": xfip_result,
        "stuff_vs_real_era": era_result,
        "out_of_sample_predictive_validity": predictive_validity,
        "by_pitch_type_2025": bpt,
        "cross_model_correlation_2025": {"corr": corr, "n": n},
        "sample_size_sanity": sanity,
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()

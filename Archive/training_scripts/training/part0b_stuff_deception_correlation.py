"""
Part 0b — Stuff+/Deception+ cross-model correlation.

Answers the one thing left open from the prior Part 0 checkpoint: what is the
actual Pearson correlation between live Stuff+ and live Deception+ pitcher-season
scores? The feature-list reconciliation (mechanics features removed from Stuff+,
confirmed present only in Deception+) was already confirmed by reading the source
code. This script measures whether the resulting scorer outputs are empirically
independent.

Procedure:
  1. Load Stuff+ pitcher-season scores from pitcher_model_scores table.
  2. Score Deception+ v1 live from the booster file on the same pitcher-seasons.
  3. Compute Pearson r and report.

Run from mlb_dashboard root:
    python training/part0b_stuff_deception_correlation.py
"""
import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from analytics.deception_features import (
    ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    RAW_QUERY_COLUMNS, engineer_features,
)
from analytics.deception_features import load_bio as _load_bio_sql
from analytics.location_features import to_plus_scale

DB_PATH = "mlb_data.db"
DECEPTION_MODEL_PATH = ROOT / "models" / "deception_plus_v1.txt"
OUT_PATH = ROOT / "training" / "v2_artifacts" / "part0b_stuff_deception_correlation.json"

SEASONS = list(range(2021, 2026))
MIN_PITCHES = 300


def main():
    print("Part 0b — Stuff+/Deception+ cross-model correlation")
    conn = sqlite3.connect(DB_PATH)

    # ---- Stuff+ scores from pre-cached table ----
    print("Loading Stuff+ scores from pitcher_model_scores...")
    stuff_df = pd.read_sql("""
        SELECT pitcher_id, season, stuff_plus
        FROM pitcher_model_scores
        WHERE season BETWEEN 2021 AND 2025
          AND stuff_plus IS NOT NULL
    """, conn)
    print(f"  {len(stuff_df):,} pitcher-season Stuff+ scores")

    # ---- Deception+ scores — score live from v1 booster ----
    print("Loading pitches for Deception+ live scoring (2021–2025)...")
    # RAW_QUERY_COLUMNS = pitcher_id, p_throws, season, release_extension,
    #                      release_pos_x, release_pos_z, arm_angle, delta_run_exp
    df_pitches = pd.read_sql(f"""
        SELECT {", ".join(RAW_QUERY_COLUMNS)}
        FROM pitches
        WHERE season BETWEEN 2021 AND 2025
          AND game_type = 'R'
    """, conn)
    conn.close()
    print(f"  {len(df_pitches):,} raw pitches loaded")

    conn2 = sqlite3.connect(DB_PATH)
    bio_df = _load_bio_sql(conn2)
    conn2.close()
    print(f"  Bio: {len(bio_df):,} rows")

    # engineer features (bio merged inside engineer_features)
    print("Engineering Deception+ features...")
    df_feat = engineer_features(df_pitches, bio_df)
    print(f"  {len(df_feat):,} engineered rows")

    # load booster
    booster = lgb.Booster(model_file=str(DECEPTION_MODEL_PATH))
    preds = booster.predict(df_feat[ALL_FEATURES])

    # aggregate to pitcher-season plus-scale (same as live model)
    df_feat["_raw_pred"] = preds
    pitcher_seasons = (
        df_feat.groupby(["pitcher_id", "season"])
        .agg(n_pitches=("_raw_pred", "size"), mean_pred=("_raw_pred", "mean"))
        .reset_index()
    )
    pitcher_seasons = pitcher_seasons[pitcher_seasons["n_pitches"] >= MIN_PITCHES]
    print(f"  {len(pitcher_seasons):,} pitcher-seasons with >= {MIN_PITCHES} pitches")

    # plus-scale per season: 100 - 10*(pred - season_mean)/season_std
    records = []
    for season, grp in pitcher_seasons.groupby("season"):
        vals = grp["mean_pred"].values
        s_mean = vals.mean()
        s_std  = vals.std()
        plus = to_plus_scale(vals, s_mean, s_std)
        for i, (_, row) in enumerate(grp.iterrows()):
            records.append({"pitcher_id": row["pitcher_id"], "season": row["season"], "deception_plus": plus[i]})
    dec_df = pd.DataFrame(records)
    print(f"  {len(dec_df):,} pitcher-seasons scored for Deception+")

    # ---- Merge and correlate ----
    stuff_df["season"] = stuff_df["season"].astype(int)
    dec_df["season"] = dec_df["season"].astype(int)
    merged = stuff_df.merge(dec_df, on=["pitcher_id", "season"], how="inner")
    print(f"  {len(merged):,} pitcher-seasons in common for correlation")

    r, p = stats.pearsonr(merged["stuff_plus"], merged["deception_plus"])
    print(f"\nStuff+ x Deception+ Pearson r = {r:.4f}  (p = {p:.4g}, n = {len(merged)})")

    # Also report by season
    by_season = {}
    for season, grp in merged.groupby("season"):
        rs, _ = stats.pearsonr(grp["stuff_plus"], grp["deception_plus"])
        by_season[int(season)] = round(float(rs), 4)
        print(f"  {season}: r = {rs:.4f}  (n={len(grp)})")

    result = {
        "pearson_r": round(float(r), 4),
        "p_value": float(p),
        "n_pitcher_seasons": int(len(merged)),
        "seasons": SEASONS,
        "by_season": by_season,
        "stuff_plus_mean": float(merged["stuff_plus"].mean()),
        "deception_plus_mean": float(merged["deception_plus"].mean()),
        "note": "Stuff+ from pitcher_model_scores; Deception+ scored live from deception_plus_v1.txt"
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()

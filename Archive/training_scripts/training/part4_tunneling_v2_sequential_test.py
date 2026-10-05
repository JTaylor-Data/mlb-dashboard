"""
Part 4 — Tunneling+ v2 sequential same-at-bat check.

Uses the live v2 booster files (models/tunneling_plus_v2_*.txt) to score
every non-1st-pitch-of-PA in the training seasons, then buckets by
predicted xRV into terciles and measures real whiff rate per tercile.

Sign convention: lower predicted xRV = better/more-tunneled pitch (harder
to hit) → should yield higher whiff rate.

Compare against v3 result: z ≈ 135.5 (per tunneling_v3_validation.json).
Architecture note: v2 uses the same lag-feature inputs as v3, so the test
is directly comparable in methodology.

Run from mlb_dashboard root:
    python training/part4_tunneling_v2_sequential_test.py
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import lightgbm as lgb
import numpy as np
import pandas as pd

from analytics.tunneling_features import (
    lag_feature_columns, load_pitches, sequential_lag_features, MAX_LAG,
    build_commit_point_table,
)

MODELS_DIR = ROOT / "models"
V2_ARTIFACTS_DIR = ROOT / "training" / "v2_artifacts"
RV_PATH = V2_ARTIFACTS_DIR / "run_value_table.json"
COMMIT_TABLE_PATH = (
    ROOT / "project_artifacts" / "training_artifacts"
    / "tunneling_v2_artifacts" / "200ms" / "pitcher_season_commit_point.csv"
)
OUT_PATH = ROOT / "training" / "v2_artifacts" / "tunneling_v2_sequential_test.json"
LOG_PATH = ROOT / "training" / "part4_tunneling_v2_sequential.log"
DB_PATH = "mlb_data.db"

COMPLETE_SEASONS = list(range(2015, 2026))
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "foul_tip",
                      "bunt_foul_tip", "missed_bunt", "swinging_pitchout"}

LAG_COLS = lag_feature_columns()
NUMERIC_LAG_COLS = [c for c in LAG_COLS if not c.endswith("_pitch_type")]
CATEGORICAL_LAG_COLS = [c for c in LAG_COLS if c.endswith("_pitch_type")]
CURRENT_CAT = ["pitch_type", "p_throws", "stand", "season"]
ALL_CAT = CATEGORICAL_LAG_COLS + CURRENT_CAT

_log_fh = None


def log(msg):
    from datetime import datetime
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def load_models():
    return {
        n: lgb.Booster(model_file=str(MODELS_DIR / f"tunneling_plus_v2_{n}.txt"))
        for n in ("xswing", "xcs", "xwhiff", "xfoul", "bip")
    }


def load_rv():
    d = json.loads(RV_PATH.read_text())
    return d["rv_table"], d["bip_classes"]


def stringify_keep_na(series):
    is_na = series.isna()
    return pd.Series(np.where(is_na, np.nan, series.astype(str)), index=series.index)


def combine_xrv(X, models, rv, bip_classes, hbp_rate):
    p_sw = models["xswing"].predict(X)
    p_cs = models["xcs"].predict(X)
    p_wh = models["xwhiff"].predict(X)
    p_fo = models["xfoul"].predict(X)
    p_bip = models["bip"].predict(X)
    bip_rvs = np.array([rv.get(c, 0.0) for c in bip_classes])
    xrv_bip = p_bip @ bip_rvs
    xrv_contact = p_fo * rv["foul"] + (1.0 - p_fo) * xrv_bip
    xrv_swing = p_wh * rv["whiff"] + (1.0 - p_wh) * xrv_contact
    p_ball = np.clip(1.0 - p_cs - hbp_rate, 0.0, 1.0)
    xrv_no_swing = p_cs * rv["called_strike"] + hbp_rate * rv["hbp"] + p_ball * rv["ball"]
    return p_sw * xrv_swing + (1.0 - p_sw) * xrv_no_swing


def main():
    global _log_fh
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8")

    log("Part 4 — Tunneling+ v2 sequential same-at-bat check")
    log(f"Commit table: {COMMIT_TABLE_PATH}")

    commit_table = pd.read_csv(COMMIT_TABLE_PATH)
    log(f"  Commit table: {len(commit_table):,} rows")

    models = load_models()
    rv, bip_classes = load_rv()
    hbp_rate = 0.0051  # from meta

    # Derive category levels from all training seasons
    conn = sqlite3.connect(DB_PATH)
    log("Deriving category levels from DB (2015-2026)...")
    raw_cats = pd.read_sql("""
        SELECT DISTINCT pitch_type, p_throws, stand, season FROM pitches
        WHERE season BETWEEN 2015 AND 2026
    """, conn)
    raw_cats["pitch_type"] = raw_cats["pitch_type"].replace({"KC": "CU"})
    raw_cats = raw_cats[~raw_cats["pitch_type"].isin({"PO", "UN", "EP"})]
    raw_cats["season"] = raw_cats["season"].astype(str)
    pt_levels = sorted(raw_cats["pitch_type"].dropna().unique().tolist())
    cat_levels = {
        "pitch_type": pt_levels,
        "p_throws": sorted(raw_cats["p_throws"].dropna().unique().tolist()),
        "stand": sorted(raw_cats["stand"].dropna().unique().tolist()),
        "season": sorted(raw_cats["season"].dropna().unique().tolist()),
    }
    for col in CATEGORICAL_LAG_COLS:
        cat_levels[col] = pt_levels

    # Read v2 meta for feat_cols
    from analytics.tunneling_v2_model import META_PATH
    meta = json.loads(META_PATH.read_text())
    feat_cols = meta["feat_cols"]

    log("\nBuilding lag features per season and scoring...")
    all_rows = []
    t_total = time.time()

    for season in COMPLETE_SEASONS:
        t0 = time.time()
        log(f"  Season {season}...")
        df_s = load_pitches(conn, season)
        if len(df_s) == 0:
            log(f"    No pitches, skipping"); continue

        feat = sequential_lag_features(df_s, commit_table, max_lag=MAX_LAG)
        # Keep only rows with at least lag1 (i.e., not the 1st pitch of PA)
        feat = feat.dropna(subset=["lag1_commit_distance", "lag1_plate_distance"])
        if len(feat) == 0:
            log(f"    No lag rows, skipping"); continue

        feat["season"] = str(season)
        for col in ALL_CAT:
            feat[col] = stringify_keep_na(feat[col])
            levels = cat_levels.get(col)
            if levels:
                feat[col] = pd.Categorical(feat[col], categories=levels)
            else:
                feat[col] = pd.Categorical(feat[col])

        X = feat[feat_cols]
        xrv = combine_xrv(X, models, rv, bip_classes, hbp_rate)

        # sequential_lag_features preserves all original columns including description
        desc_col = feat["description"] if "description" in feat.columns else pd.Series(np.nan, index=feat.index)
        chunk = pd.DataFrame({"xrv": xrv, "description": desc_col.values}, index=feat.index)
        all_rows.append(chunk)
        log(f"    {len(feat):,} lag rows scored ({time.time()-t0:.0f}s)")

    conn.close()

    log(f"\nTotal scoring time: {(time.time()-t_total)/60:.1f} min")
    df = pd.concat(all_rows, ignore_index=True)
    log(f"Total scored rows: {len(df):,}")

    # Get descriptions if we have them
    if df["description"].isna().all():
        log("WARNING: description column missing — cannot compute whiff rates. Re-querying...")
        # Fallback: re-read from DB for whiff computation
        conn2 = sqlite3.connect(DB_PATH)
        desc_df = pd.read_sql("""
            SELECT description FROM pitches
            WHERE season BETWEEN 2015 AND 2025
              AND game_type = 'R'
        """, conn2)
        conn2.close()
        log(f"  Description rows from DB: {len(desc_df):,} (df has {len(df):,})")
        log("  NOTE: row alignment not guaranteed — using xrv distribution only")
        df["is_whiff"] = np.nan
    else:
        df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)

    # Bucket by xrv into terciles
    # Lower xrv = model says this pitch was better for pitcher = harder to hit = should have higher whiff
    log("\nBucketing into terciles...")
    df["tercile"] = pd.qcut(df["xrv"], 3,
                            labels=["low_xrv", "mid_xrv", "high_xrv"],
                            duplicates="drop")

    log(f"  xrv percentiles: p10={np.percentile(df['xrv'],10):.4f}  p50={np.percentile(df['xrv'],50):.4f}  p90={np.percentile(df['xrv'],90):.4f}")
    log(f"  xrv range: [{df['xrv'].min():.4f}, {df['xrv'].max():.4f}]")

    if df["is_whiff"].notna().sum() > 0:
        by_tercile = df.groupby("tercile", observed=True)["is_whiff"].agg(["mean", "count"])
        log("\nWhiff rate by predicted xrv tercile:")
        for t, row in by_tercile.iterrows():
            log(f"  {t:12s}: whiff_rate={row['mean']:.4f}  n={row['count']:,}")

        p_good = by_tercile.loc["low_xrv", "mean"]
        n_good = by_tercile.loc["low_xrv", "count"]
        p_bad  = by_tercile.loc["high_xrv", "mean"]
        n_bad  = by_tercile.loc["high_xrv", "count"]
        se = np.sqrt(p_good*(1-p_good)/n_good + p_bad*(1-p_bad)/n_bad)
        z = (p_good - p_bad) / se
        log(f"\n  z-score (low_xrv - high_xrv whiff rate): {z:.3f}")
        log(f"  Compare: v3 z = 135.516 (tunneling_plus_ext_artifacts/tunneling_v3_validation.json)")

        result = dict(
            n_total=int(len(df)),
            xrv_percentiles={
                "p10": float(np.percentile(df["xrv"], 10)),
                "p50": float(np.percentile(df["xrv"], 50)),
                "p90": float(np.percentile(df["xrv"], 90)),
            },
            whiff_rate_by_predicted_tercile={
                t: dict(whiff_rate=float(r["mean"]), n=int(r["count"]))
                for t, r in by_tercile.iterrows()
            },
            good_minus_bad_z_score=float(z),
            v3_comparison_z=135.516,
            methodology_note=(
                "Same methodology as v3's sequential test: bucket ALL non-1st-PA-pitch rows "
                "by predicted xrv into terciles; check real whiff rate by tercile; "
                "z = (low_xrv_whiff_rate - high_xrv_whiff_rate) / pooled_se. "
                "Lower xrv = model predicts this pitch is better for the pitcher (harder to hit)."
            ),
        )
    else:
        log("\nWARNING: could not compute whiff rates (no description column).")
        log("Reporting xrv distribution statistics only.")
        counts = df["tercile"].value_counts().sort_index()
        result = dict(
            n_total=int(len(df)),
            xrv_percentiles={
                "p10": float(np.percentile(df["xrv"], 10)),
                "p50": float(np.percentile(df["xrv"], 50)),
                "p90": float(np.percentile(df["xrv"], 90)),
            },
            tercile_counts=counts.to_dict(),
            good_minus_bad_z_score=None,
            error="description column not available for whiff rate computation",
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    log(f"\nResult saved to {OUT_PATH}")
    _log_fh.close()


if __name__ == "__main__":
    main()

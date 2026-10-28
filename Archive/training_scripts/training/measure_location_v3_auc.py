"""
Location+ v3 sub-classifier AUC measurement.

Computes held-out AUC for xSwing, xCS, xWhiff, xFoul by joining the
existing Part-3 OOF predictions (V0a variant) back to true labels —
recreating the same deterministic subsample/fold split.

For BIP multiclass accuracy: trains BIP on 2015-2024 (excl 2020),
evaluates on 2025, which is a clean one-year time-based holdout.

Output: diagnostics/location_v3/sub_classifier_auc.json
"""
import datetime
import json
import sqlite3
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB_PATH = ROOT / "mlb_data.db"
DIAG_DIR = ROOT / "diagnostics" / "location_v3"

sys.path.insert(0, str(ROOT))
from analytics.stuff_features import merge_pitch_type, exclude_junk_pitch_types

# --- Constants matching train_location_v3.py exactly ---
EXCL_SEASONS = {2020}
SUBSAMPLE_FRAC = 0.30
SUBSAMPLE_SEED = 42
SEED_A = 42

FEAT_COLS_BASE = ["plate_x", "plate_z", "balls", "strikes", "pitch_type", "stand", "season"]
CAT_COLS = ["pitch_type", "stand", "season"]

SWING_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip", "bunt_foul_tip",
    "missed_bunt", "foul", "foul_bunt", "foul_pitchout",
    "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    "swinging_pitchout",
}
WHIFF_DESCS = {
    "swinging_strike", "swinging_strike_blocked", "foul_tip",
    "bunt_foul_tip", "missed_bunt", "swinging_pitchout",
}
FOUL_DESCS = {"foul", "foul_bunt", "foul_pitchout"}
BIP_DESCS = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}
CALLED_STRIKE_DESCS = {"called_strike", "automatic_strike"}

EV_BINS = [0, 90, 95, 100, 105, 999]
EV_LABELS = ["ev_lt90", "ev_90_95", "ev_95_100", "ev_100_105", "ev_105plus"]

MULTI_PARAMS = dict(
    objective="multiclass", n_estimators=400, learning_rate=0.05,
    num_leaves=31, min_child_samples=100, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    metric="multi_logloss",
)


def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_pitches(conn):
    log("Loading pitches from DB...")
    df = pd.concat(pd.read_sql("""
        SELECT pitcher_id, plate_x, plate_z, balls, strikes,
               pitch_type, stand, season, description
        FROM pitches
        WHERE season BETWEEN 2015 AND 2026
          AND plate_x IS NOT NULL AND plate_z IS NOT NULL
    """, conn, chunksize=200_000), ignore_index=True)
    df = merge_pitch_type(df)
    df = exclude_junk_pitch_types(df)
    for col in ["plate_x", "plate_z", "balls", "strikes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["plate_x", "plate_z", "balls", "strikes", "pitch_type", "stand"])
    df["season"] = df["season"].astype(int)
    log(f"  {len(df):,} pitches loaded")
    return df


def assign_labels(df):
    desc = df["description"].fillna("")
    df = df.copy()
    df["is_swing"] = desc.isin(SWING_DESCS).astype(np.int8)
    df["is_whiff"] = desc.isin(WHIFF_DESCS).astype(np.int8)
    df["is_foul"] = desc.isin(FOUL_DESCS).astype(np.int8)
    df["is_called_strike"] = desc.isin(CALLED_STRIKE_DESCS).astype(np.int8)
    df["is_contact"] = ((df["is_swing"] == 1) & (df["is_whiff"] == 0)).astype(np.int8)
    return df


def make_subsample_folds(df, subsample_frac, subsample_seed, seed_a):
    """Recreate the same pitcher subsample and fold assignments as Part 3."""
    df_train = df[~df["season"].isin(EXCL_SEASONS)].copy()

    counts_by_pitcher = df_train.groupby("pitcher_id").size()
    terciles = pd.qcut(counts_by_pitcher, q=3, labels=["low", "mid", "high"])
    rng = np.random.default_rng(subsample_seed)
    subsample_ids = set()
    for t in ["low", "mid", "high"]:
        pool = counts_by_pitcher[terciles == t].index.tolist()
        n_take = max(1, int(len(pool) * subsample_frac))
        chosen = rng.choice(pool, size=n_take, replace=False)
        subsample_ids.update(chosen)

    df_sub = df_train[df_train["pitcher_id"].isin(subsample_ids)].copy()

    pitchers = df_sub["pitcher_id"].unique()
    rng2 = np.random.default_rng(seed_a)
    perm = rng2.permutation(len(pitchers))
    fold_assignments = np.array_split(perm, 5)
    fold_map = {}
    for fold_i, indices in enumerate(fold_assignments):
        for idx in indices:
            fold_map[pitchers[idx]] = fold_i
    df_sub["fold"] = df_sub["pitcher_id"].map(fold_map)
    return df_sub


# ─── Part 1: OOF AUC for binary classifiers ──────────────────────────────────

def compute_binary_aucs():
    log("\n=== Binary classifier OOF AUC (V0a variant) ===")
    oof = pd.read_parquet(DIAG_DIR / "oof_V0a.parquet")
    log(f"  OOF rows: {len(oof):,}")

    conn = sqlite3.connect(str(DB_PATH))
    df = load_pitches(conn)
    conn.close()

    df = assign_labels(df)
    df_sub = make_subsample_folds(df, SUBSAMPLE_FRAC, SUBSAMPLE_SEED, SEED_A)

    log(f"  df_sub rows: {len(df_sub):,}  (OOF rows: {len(oof):,})")

    # The OOF rows were generated in fold order (fold 0, 1, 2, 3, 4).
    # df_sub rows within each fold should be in the same order as df_sub.
    # Reset index on df_sub, sort by fold then original row order.
    df_sub = df_sub.reset_index(drop=True)
    df_sub["_orig_order"] = df_sub.index

    # Recreate per-fold order: fold 0 rows in original df_sub order, then fold 1, etc.
    parts = []
    for fold_i in range(5):
        parts.append(df_sub[df_sub["fold"] == fold_i])
    df_ordered = pd.concat(parts, ignore_index=True)

    if len(df_ordered) != len(oof):
        log(f"  WARNING: row count mismatch ({len(df_ordered)} vs {len(oof)})")
        log("  Falling back to pitcher_id/season/fold merge with rank-order alignment")
        # Fallback: assign a within-group rank and merge on (pitcher_id, season, fold, rank)
        oof["_rank"] = oof.groupby(["pitcher_id", "season", "fold"]).cumcount()
        df_ordered["_rank"] = df_ordered.groupby(["pitcher_id", "season", "fold"]).cumcount()
        merged = df_ordered.merge(oof, on=["pitcher_id", "season", "fold", "_rank"], how="inner")
    else:
        # Check pitcher_id alignment
        match_pct = (df_ordered["pitcher_id"].values == oof["pitcher_id"].values).mean()
        log(f"  Row alignment check: {match_pct:.4f} pitcher_id match rate")
        if match_pct < 0.95:
            log("  WARNING: low alignment, using merge fallback")
            oof["_rank"] = oof.groupby(["pitcher_id", "season", "fold"]).cumcount()
            df_ordered["_rank"] = df_ordered.groupby(["pitcher_id", "season", "fold"]).cumcount()
            merged = df_ordered.merge(oof, on=["pitcher_id", "season", "fold", "_rank"], how="inner")
        else:
            merged = df_ordered.copy()
            for col in ["xrv", "p_swing", "p_cs", "p_whiff", "p_foul"]:
                merged[col] = oof[col].values

    log(f"  Merged rows for AUC: {len(merged):,}")

    results = {}

    # xSwing: all pitches
    auc_sw = roc_auc_score(merged["is_swing"], merged["p_swing"])
    results["xswing_auc"] = round(auc_sw, 4)
    log(f"  xSwing AUC (all pitches, n={len(merged):,}): {auc_sw:.4f}")

    # xCS: no-swing pitches only
    no_sw = merged[merged["is_swing"] == 0]
    auc_cs = roc_auc_score(no_sw["is_called_strike"], no_sw["p_cs"])
    results["xcs_auc"] = round(auc_cs, 4)
    log(f"  xCS AUC (no-swing, n={len(no_sw):,}): {auc_cs:.4f}")

    # xWhiff: swing pitches only
    sw = merged[merged["is_swing"] == 1]
    auc_wh = roc_auc_score(sw["is_whiff"], sw["p_whiff"])
    results["xwhiff_auc"] = round(auc_wh, 4)
    log(f"  xWhiff AUC (swing, n={len(sw):,}): {auc_wh:.4f}")

    # xFoul: contact pitches only
    ct = merged[(merged["is_swing"] == 1) & (merged["is_whiff"] == 0)]
    auc_fo = roc_auc_score(ct["is_foul"], ct["p_foul"])
    results["xfoul_auc"] = round(auc_fo, 4)
    log(f"  xFoul AUC (contact, n={len(ct):,}): {auc_fo:.4f}")

    return results


# ─── Part 2: BIP multiclass accuracy (2025 time-based holdout) ───────────────

def compute_bip_accuracy():
    log("\n=== BIP multiclass accuracy (train 2015-2024 excl 2020, holdout 2025) ===")

    conn = sqlite3.connect(str(DB_PATH))
    bip_df = pd.concat(pd.read_sql("""
        SELECT pitcher_id, season, pitch_type, stand, plate_x, plate_z,
               balls, strikes, bb_type, launch_speed
        FROM pitches
        WHERE season BETWEEN 2015 AND 2025
          AND bb_type IS NOT NULL AND launch_speed IS NOT NULL
    """, conn, chunksize=100_000), ignore_index=True)
    conn.close()

    from analytics.stuff_features import merge_pitch_type, exclude_junk_pitch_types
    bip_df = merge_pitch_type(bip_df)
    bip_df = exclude_junk_pitch_types(bip_df)
    bip_df["season"] = bip_df["season"].astype(int)
    bip_df["ev_bucket"] = pd.cut(
        bip_df["launch_speed"].astype(float),
        bins=EV_BINS, labels=EV_LABELS, right=False
    ).astype(str)
    bip_df["bip_class"] = bip_df["bb_type"].fillna("unknown") + "__" + bip_df["ev_bucket"]

    valid_classes = sorted(bip_df["bip_class"].unique())
    log(f"  BIP classes: {len(valid_classes)}, total rows: {len(bip_df):,}")

    train_mask = ~bip_df["season"].isin(EXCL_SEASONS) & (bip_df["season"] <= 2024)
    test_mask = bip_df["season"] == 2025
    df_tr = bip_df[train_mask].copy()
    df_te = bip_df[test_mask].copy()
    log(f"  Train: {len(df_tr):,}, Holdout 2025: {len(df_te):,}")

    # Coerce categoricals
    df_tr["season"] = df_tr["season"].astype(str)
    df_te["season"] = df_te["season"].astype(str)
    cat_cols_bip = [c for c in CAT_COLS if c in df_tr.columns]
    feat_cols_bip = [c for c in FEAT_COLS_BASE if c in df_tr.columns]
    all_cats = {}
    for col in cat_cols_bip:
        cats = sorted(set(df_tr[col].dropna().astype(str)) | set(df_te[col].dropna().astype(str)))
        df_tr[col] = pd.Categorical(df_tr[col].astype(str), categories=cats)
        df_te[col] = pd.Categorical(df_te[col].astype(str), categories=cats)
        all_cats[col] = cats

    # Encode BIP class
    df_tr["bip_class_code"] = pd.Categorical(df_tr["bip_class"], categories=valid_classes).codes
    df_te["bip_class_code"] = pd.Categorical(df_te["bip_class"], categories=valid_classes).codes

    n_cls = len(valid_classes)
    n_val = max(500, int(0.05 * len(df_tr)))
    X_tr = df_tr[feat_cols_bip]
    y_tr = df_tr["bip_class_code"]

    params_bip = {**MULTI_PARAMS, "num_class": n_cls, "random_state": SEED_A}
    m_bip = lgb.LGBMClassifier(**params_bip)
    log(f"  Fitting BIP model ({len(df_tr):,} rows, {n_cls} classes)...")
    m_bip.fit(
        X_tr.iloc[:-n_val], y_tr.iloc[:-n_val],
        eval_set=[(X_tr.iloc[-n_val:], y_tr.iloc[-n_val:])],
        categorical_feature=cat_cols_bip,
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    X_te = df_te[feat_cols_bip]
    y_te = df_te["bip_class_code"]
    preds = m_bip.predict(X_te)
    accuracy = float((preds == y_te.values).mean())
    log(f"  BIP holdout accuracy (2025): {accuracy:.4f}  (n={len(df_te):,})")

    return {"bip_holdout_accuracy_2025": round(accuracy, 4), "bip_n_holdout": int(len(df_te)),
            "bip_n_classes": n_cls, "bip_train_seasons": "2015-2024 excl 2020",
            "bip_holdout_season": 2025}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log("Location+ v3 sub-classifier AUC measurement")

    binary_results = compute_binary_aucs()
    bip_results = compute_bip_accuracy()

    out = {
        "measured_at": datetime.datetime.now().isoformat(),
        "note": (
            "Binary AUCs from Part-3 OOF (V0a, 30% pitcher subsample, 5-fold grouped CV). "
            "BIP accuracy from time-based holdout: train 2015-2024 excl 2020, eval 2025."
        ),
        **binary_results,
        **bip_results,
    }

    out_path = DIAG_DIR / "sub_classifier_auc.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"\nResults saved to {out_path}")
    log(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

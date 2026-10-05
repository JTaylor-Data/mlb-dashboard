"""
Deception+ v2 Part 3 -- Cross-fitted scoring.

Trains the final ensemble using chosen_config.json from Part 2.
Scores every pitcher-season, applies reliability rule, to_plus_scale.

Outputs:
  models/deception_plus_v2_fold{k}_rep{r}.txt
  models/deception_plus_v2_meta.json
  diagnostics/deception_v2/deception_v2_scores.parquet
"""
import gc
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT / "diagnostics" / "deception_v2"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "mlb_data.db"
LOG_PATH = DIAG_DIR / "progress.log"

DIAG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
SAMPLE_FRAC = 0.30
MIN_PITCHES = 300
RELIABILITY_THRESHOLD = 0.70
EB_K_DEFAULT = 1.0  # empirical Bayes shrinkage default

# Feature sets
F0 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle",
      "height_in", "weight_lb", "arm_angle_std", "release_extension_std", "release_consistency", "season"]
F1 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle",
      "arm_angle_std", "release_extension_std", "release_consistency", "season"]
F2 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle", "season"]

FEATURE_SETS = {"F0": F0, "F1": F1, "F2": F2}


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(obj, path):
    def default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def get_conn():
    return sqlite3.connect(DB_PATH)


def to_plus_scale(raw_mean, season_mean, season_std):
    """Same convention as location_features.to_plus_scale: 100 - 10*z (higher = worse run value prevented)."""
    if season_std == 0:
        return 100.0
    return 100.0 - 10.0 * (raw_mean - season_mean) / season_std


def model_hash(path):
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def load_data(conn):
    log("Loading pitch data")
    df = pd.read_sql("""
        SELECT p.pitcher_id, p.season, p.p_throws,
               p.release_pos_x, p.release_pos_z, p.release_extension,
               p.arm_angle, p.delta_run_exp
        FROM pitches p
        WHERE p.season >= 2021
          AND p.pitch_type NOT IN ('PO','UN','EP')
          AND p.release_pos_x IS NOT NULL AND p.release_pos_z IS NOT NULL
          AND p.release_extension IS NOT NULL AND p.arm_angle IS NOT NULL
          AND p.delta_run_exp IS NOT NULL
    """, conn)
    bio = pd.read_sql("SELECT player_id AS pitcher_id, height_in, weight_lb FROM pitcher_bio", conn)
    sign = np.where(df['p_throws'] == 'R', -1, 1)
    df['release_pos_x_as'] = df['release_pos_x'] * sign
    g = df.groupby(['pitcher_id', 'season'])
    df['arm_angle_std'] = g['arm_angle'].transform('std')
    df['release_extension_std'] = g['release_extension'].transform('std')
    cx = g['release_pos_x'].transform('mean')
    cz = g['release_pos_z'].transform('mean')
    sq = (df['release_pos_x'] - cx)**2 + (df['release_pos_z'] - cz)**2
    df['release_consistency'] = sq.groupby([df['pitcher_id'], df['season']]).transform('mean')**0.5
    df = df.merge(bio, on='pitcher_id', how='left')
    df['season'] = df['season'].astype(str)
    df['fold'] = df['pitcher_id'] % N_FOLDS
    return df


def subsample_pitches(df, frac, seed):
    rng = np.random.default_rng(seed)
    sampled = []
    for (pid, s), grp in df.groupby(['pitcher_id', 'season']):
        n = max(1, int(len(grp) * frac))
        idx = rng.choice(len(grp), size=n, replace=False)
        sampled.append(grp.iloc[idx])
    return pd.concat(sampled, ignore_index=True)


def lgb_params_per_pitch(cfg, seed):
    return {
        "objective": "huber", "alpha": 0.9,
        "num_leaves": cfg['num_leaves'], "min_child_samples": cfg['min_child_samples'],
        "n_estimators": cfg['n_estimators'], "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed,
        "verbose": -1, "n_jobs": -1,
    }


def lgb_params_pitcher_season(cfg, seed):
    return {
        "objective": "regression",
        "num_leaves": cfg['num_leaves'], "min_child_samples": cfg['min_child_samples'],
        "n_estimators": cfg['n_estimators'], "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed,
        "verbose": -1, "n_jobs": -1,
    }


def train_fold(df, feat_cols, cfg, fold, rep_seed, form):
    train_df = df[df['fold'] != fold].copy()
    test_df = df[df['fold'] == fold].copy()

    if form == 'per_pitch':
        train_sub = subsample_pitches(train_df, SAMPLE_FRAC, rep_seed + fold)
        for d in [train_sub, test_df]:
            d['season'] = d['season'].astype('category')
        # Drop rows missing features for this feature set
        num_feats = [f for f in feat_cols if f != 'season']
        train_sub = train_sub.dropna(subset=num_feats + ['delta_run_exp'])
        test_df = test_df.dropna(subset=num_feats + ['delta_run_exp'])
        X_tr = train_sub[feat_cols]
        y_tr = train_sub['delta_run_exp']
        X_te = test_df[feat_cols]
        params = lgb_params_per_pitch(cfg, rep_seed + fold * 17)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        oof = test_df[['pitcher_id', 'season', 'delta_run_exp']].copy()
        oof['pred'] = preds
    else:  # pitcher_season
        num_feats = [f for f in feat_cols if f != 'season']
        ps_df = df.groupby(['pitcher_id', 'season'])[num_feats + ['delta_run_exp']].mean().reset_index()
        ps_counts = df.groupby(['pitcher_id', 'season']).size().reset_index(name='n_pitches')
        ps_df = ps_df.merge(ps_counts, on=['pitcher_id', 'season'])
        ps_df = ps_df[ps_df['n_pitches'] >= MIN_PITCHES].copy()
        ps_df['fold'] = ps_df['pitcher_id'].astype(int) % N_FOLDS
        train_ps = ps_df[ps_df['fold'] != fold].copy()
        test_ps = ps_df[ps_df['fold'] == fold].copy()
        for d in [train_ps, test_ps]:
            d['season'] = d['season'].astype('category')
        params = lgb_params_pitcher_season(cfg, rep_seed + fold * 17)
        model = lgb.LGBMRegressor(**params)
        model.fit(train_ps[feat_cols], train_ps['delta_run_exp'])
        preds = model.predict(test_ps[feat_cols])
        oof = test_ps[['pitcher_id', 'season', 'delta_run_exp', 'n_pitches']].copy()
        oof['pred'] = preds

    return model, oof


def main():
    log("=== PART 3 START ===")

    # Load chosen config
    chosen_path = DIAG_DIR / "chosen_config.json"
    if not chosen_path.exists():
        log("ERROR: chosen_config.json not found -- run Part 2 first")
        sys.exit(1)

    with open(chosen_path) as f:
        chosen = json.load(f)

    cfg_key = chosen['key']
    cfg = {k: chosen[k] for k in ['num_leaves', 'min_child_samples', 'n_estimators', 'form', 'features']}
    feat_cols = FEATURE_SETS.get(cfg['features'], F2)
    form = cfg['form']
    log(f"Chosen config: {cfg_key}, form={form}, features={cfg['features']}, feat_cols={feat_cols}")

    # Number of repeats
    n_repeats = 3 if form == 'pitcher_season' else 2
    log(f"Training {n_repeats} repeats × {N_FOLDS} folds")

    conn = get_conn()
    df = load_data(conn)
    conn.close()

    fold_models = {}  # (rep, fold) -> model path
    all_oof = []
    rep_seeds = [SEED, SEED + 100, SEED + 200]

    for rep in range(n_repeats):
        rep_seed = rep_seeds[rep]
        log(f"--- Repeat {rep} (seed={rep_seed}) ---")
        for fold in range(N_FOLDS):
            t0 = time.time()
            model, oof = train_fold(df, feat_cols, cfg, fold, rep_seed, form)
            elapsed = time.time() - t0

            # Save model
            model_path = MODELS_DIR / f"deception_plus_v2_fold{fold}_rep{rep}.txt"
            model.booster_.save_model(str(model_path))
            fold_models[(rep, fold)] = str(model_path)
            log(f"  rep={rep} fold={fold}: {elapsed:.1f}s -> {model_path.name}")

            oof['rep'] = rep
            oof['fold'] = fold
            all_oof.append(oof)
            del model; gc.collect()

    oof_all = pd.concat(all_oof, ignore_index=True)

    # ---- Score every pitcher-season ----
    log("Scoring all pitcher-seasons from OOF predictions")
    # For each pitcher-season, average across all reps/folds that held them out
    ps_preds = oof_all.groupby(['pitcher_id', 'season'])['pred'].mean().reset_index()
    ps_counts = oof_all.groupby(['pitcher_id', 'season'])['pred'].count().reset_index(name='rep_fold_count')
    ps_preds = ps_preds.merge(ps_counts, on=['pitcher_id', 'season'])

    # Pitch counts per pitcher-season (for qualification and reliability)
    conn = get_conn()
    pc_df = pd.read_sql("""
        SELECT pitcher_id, CAST(season AS TEXT) as season, COUNT(*) as n_pitches
        FROM pitches
        WHERE season >= 2021
          AND pitch_type NOT IN ('PO','UN','EP')
          AND release_pos_x IS NOT NULL AND release_pos_z IS NOT NULL
          AND release_extension IS NOT NULL AND arm_angle IS NOT NULL
          AND delta_run_exp IS NOT NULL
        GROUP BY pitcher_id, season
    """, conn)
    conn.close()

    pc_df['pitcher_id'] = pc_df['pitcher_id'].astype(int)
    ps_preds['pitcher_id'] = ps_preds['pitcher_id'].astype(int)
    ps_preds = ps_preds.merge(pc_df, on=['pitcher_id', 'season'], how='left')

    # Qualified flag
    ps_preds['qualified'] = ps_preds['n_pitches'] >= MIN_PITCHES

    # Reliability check: split-half correlation at 300 pitches
    # For pitcher-season form: reliability applies at pitcher-season level
    # For per-pitch form: apply empirical Bayes shrinkage if below threshold
    # We'll compute a simplified split-half on the OOF predictions if we have enough data
    # For now, flag pitchers below 300 pitches
    ps_preds['flags'] = ''
    ps_preds.loc[ps_preds['n_pitches'] < MIN_PITCHES, 'flags'] = 'below_min_pitches'

    # Apply empirical Bayes shrinkage for below-threshold samples
    # Reliability threshold = 0.70 -> at 300 pitches per spec
    # Simple EB: shrink towards seasonal mean by 1 - (n / (n + K)) where K is a shrinkage constant
    EB_K = 300  # pitches to half-shrink (rough estimate; proper EB would estimate K from data)
    # For qualified pitchers, no shrinkage; for unqualified, shrink towards 0 (raw score)
    # (In plus-scale, 0 = league mean)
    # We'll apply shrinkage in the plus-scale step

    # Plus scale: standardize raw predictions within each season
    seasons = sorted(ps_preds['season'].unique())
    ps_preds['deception_plus_v2'] = np.nan
    ps_preds['league_mean_raw'] = np.nan
    ps_preds['league_std_raw'] = np.nan

    for s in seasons:
        mask_qual = (ps_preds['season'] == s) & ps_preds['qualified']
        mask_all = ps_preds['season'] == s
        if mask_qual.sum() < 10:
            log(f"  Season {s}: insufficient qualified pitchers ({mask_qual.sum()})")
            continue
        season_mean = ps_preds.loc[mask_qual, 'pred'].mean()
        season_std = ps_preds.loc[mask_qual, 'pred'].std()
        ps_preds.loc[mask_all, 'league_mean_raw'] = season_mean
        ps_preds.loc[mask_all, 'league_std_raw'] = season_std
        if season_std > 0:
            raw_plus = 100.0 - 10.0 * (ps_preds.loc[mask_all, 'pred'] - season_mean) / season_std
            # EB shrinkage for unqualified
            n = ps_preds.loc[mask_all, 'n_pitches'].fillna(0)
            shrink = n / (n + EB_K)
            shrink[mask_qual] = 1.0  # no shrinkage for qualified
            ps_preds.loc[mask_all, 'deception_plus_v2'] = 100.0 + shrink * (raw_plus - 100.0)
        log(f"  Season {s}: mean_raw={season_mean:.4f} std_raw={season_std:.4f} n_qual={mask_qual.sum()}")

    # Add league percentile within season (qualified only)
    ps_preds['percentile'] = np.nan
    for s in seasons:
        mask_qual = (ps_preds['season'] == s) & ps_preds['qualified']
        if mask_qual.sum() < 2:
            continue
        vals = ps_preds.loc[mask_qual, 'deception_plus_v2'].rank(pct=True)
        ps_preds.loc[mask_qual, 'percentile'] = vals * 100

    # Save scores parquet
    scores_path = DIAG_DIR / "deception_v2_scores.parquet"
    ps_preds.to_parquet(scores_path, index=False)
    log(f"Saved scores to {scores_path}")
    log(f"Total pitcher-seasons scored: {len(ps_preds)}")
    log(f"Qualified pitcher-seasons: {ps_preds['qualified'].sum()}")

    # ---- Save meta ----
    meta = {
        "version": "v2",
        "created": datetime.now().isoformat(),
        "chosen_config_key": cfg_key,
        "features": feat_cols,
        "form": form,
        "num_leaves": cfg['num_leaves'],
        "min_child_samples": cfg['min_child_samples'],
        "n_estimators": cfg['n_estimators'],
        "n_repeats": n_repeats,
        "n_folds": N_FOLDS,
        "seeds": rep_seeds[:n_repeats],
        "fold_assignment": "pitcher_id_mod_5",
        "min_pitches_qualified": MIN_PITCHES,
        "model_paths": {f"rep{r}_fold{f}": fold_models[(r, f)] for r in range(n_repeats) for f in range(N_FOLDS)},
        "model_hashes": {f"rep{r}_fold{f}": model_hash(fold_models[(r, f)]) for r in range(n_repeats) for f in range(N_FOLDS)},
        "eb_shrinkage_K": EB_K,
        "scored_seasons": seasons,
    }
    meta_path = MODELS_DIR / "deception_plus_v2_meta.json"
    save_json(meta, meta_path)
    log(f"Saved meta to {meta_path}")

    # ---- In-sample model (for Part 4 comparison only) ----
    log("Fitting in-sample model (for Part 4 only, never displayed)")
    all_df = df.copy()
    if form == 'per_pitch':
        all_sub = subsample_pitches(all_df, SAMPLE_FRAC, SEED + 999)
        num_feats = [f for f in feat_cols if f != 'season']
        all_sub = all_sub.dropna(subset=num_feats + ['delta_run_exp'])
        all_sub['season'] = all_sub['season'].astype('category')
        params_is = lgb_params_per_pitch(cfg, SEED + 999)
        is_model = lgb.LGBMRegressor(**params_is)
        is_model.fit(all_sub[feat_cols], all_sub['delta_run_exp'])
    else:
        num_feats = [f for f in feat_cols if f != 'season']
        ps_all = df.groupby(['pitcher_id', 'season'])[num_feats + ['delta_run_exp']].mean().reset_index()
        ps_all['season'] = ps_all['season'].astype('category')
        params_is = lgb_params_pitcher_season(cfg, SEED + 999)
        is_model = lgb.LGBMRegressor(**params_is)
        is_model.fit(ps_all[feat_cols], ps_all['delta_run_exp'])

    is_model_path = MODELS_DIR / "deception_plus_v2_insample.txt"
    is_model.booster_.save_model(str(is_model_path))
    log(f"Saved in-sample model to {is_model_path}")

    log("=== PART 3 COMPLETE ===")


if __name__ == "__main__":
    main()

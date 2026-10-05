"""
Deception+ v2 Part 2 -- Two core tests (A0, A1, A2, B) with pitcher-grouped CV.

Outputs:
  diagnostics/deception_v2/chosen_config.json
  diagnostics/deception_v2/part2_results.json
"""
import gc
import json
import sys
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT / "diagnostics" / "deception_v2"
DB_PATH = ROOT / "mlb_data.db"
LOG_PATH = DIAG_DIR / "progress.log"

DIAG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
SAMPLE_FRAC = 0.30
MIN_PITCHES = 300
BOOTSTRAP_N = 1000
EQUIVALENCE_MARGIN = 0.03

np.random.seed(SEED)


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
        if isinstance(o, pd.Series): return o.to_list()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    return conn


# ---- FEATURE SETS ----
F0 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle",
      "height_in", "weight_lb", "arm_angle_std", "release_extension_std", "release_consistency", "season"]
F1 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle",
      "arm_angle_std", "release_extension_std", "release_consistency", "season"]
F2 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle", "season"]

FEATURE_SETS = {"F0": F0, "F1": F1, "F2": F2}

CONFIGS = {
    "A0": {"features": "F0", "form": "per_pitch", "num_leaves": 31, "min_child_samples": 100, "n_estimators": 500},
    "A1": {"features": "F0", "form": "per_pitch", "num_leaves": 15, "min_child_samples": 2000, "n_estimators": 300},
    "A2": {"features": "F2", "form": "per_pitch", "num_leaves": 15, "min_child_samples": 2000, "n_estimators": 300},
    "B":  {"features": "F2", "form": "pitcher_season", "num_leaves": 4, "min_child_samples": 100, "n_estimators": 200},
}


# ---- DATA LOADING ----

def load_data(conn):
    log("Loading pitch data from DB")
    df = pd.read_sql("""
        SELECT p.pitcher_id, p.season, p.p_throws,
               p.release_pos_x, p.release_pos_z, p.release_extension,
               p.arm_angle, p.delta_run_exp
        FROM pitches p
        WHERE p.season >= 2021
          AND p.pitch_type NOT IN ('PO','UN','EP')
          AND p.release_pos_x IS NOT NULL
          AND p.release_pos_z IS NOT NULL
          AND p.release_extension IS NOT NULL
          AND p.arm_angle IS NOT NULL
          AND p.delta_run_exp IS NOT NULL
    """, conn)
    log(f"Loaded {len(df):,} pitches")

    # Bio
    bio = pd.read_sql("SELECT player_id AS pitcher_id, height_in, weight_lb FROM pitcher_bio", conn)

    # Arm-side sign
    sign = np.where(df['p_throws'] == 'R', -1, 1)
    df['release_pos_x_as'] = df['release_pos_x'] * sign

    # Pitcher-season consistency features (computed on FULL season, then subsampled)
    log("Computing pitcher-season aggregate features")
    g = df.groupby(['pitcher_id', 'season'])
    df['arm_angle_std'] = g['arm_angle'].transform('std')
    df['release_extension_std'] = g['release_extension'].transform('std')
    centers_x = g['release_pos_x'].transform('mean')
    centers_z = g['release_pos_z'].transform('mean')
    sq_dist = (df['release_pos_x'] - centers_x)**2 + (df['release_pos_z'] - centers_z)**2
    df['release_consistency'] = sq_dist.groupby([df['pitcher_id'], df['season']]).transform('mean')**0.5

    # Merge bio
    df = df.merge(bio, on='pitcher_id', how='left')
    df['season'] = df['season'].astype(str)

    # Drop rows missing any F0 feature
    required = list(set(F0) - {'season'})
    df = df.dropna(subset=required + ['delta_run_exp'])
    log(f"After dropna: {len(df):,} pitches")

    # Pitcher-season pitch counts
    ps_counts = df.groupby(['pitcher_id', 'season']).size().reset_index(name='n_pitches')
    qualified_ps = ps_counts[ps_counts['n_pitches'] >= MIN_PITCHES].copy()
    log(f"Qualified pitcher-seasons (>={MIN_PITCHES} pitches): {len(qualified_ps)}")

    # Fold assignment: pitcher_id % 5
    df['fold'] = df['pitcher_id'] % N_FOLDS

    return df, qualified_ps


def subsample_pitches(df, frac, seed):
    """Sample frac% of each pitcher-season's pitches."""
    rng = np.random.default_rng(seed)
    sampled = []
    for (pid, s), grp in df.groupby(['pitcher_id', 'season']):
        n = max(1, int(len(grp) * frac))
        idx = rng.choice(len(grp), size=n, replace=False)
        sampled.append(grp.iloc[idx])
    return pd.concat(sampled, ignore_index=True)


# ---- LGB PARAMS ----

def lgb_params(cfg, seed):
    return {
        "objective": "huber",
        "alpha": 0.9,
        "num_leaves": cfg['num_leaves'],
        "min_child_samples": cfg['min_child_samples'],
        "n_estimators": cfg['n_estimators'],
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbose": -1,
        "n_jobs": -1,
    }


# ---- CROSS-VALIDATED TRAINING (per-pitch form) ----

def train_cv_per_pitch(df, features, cfg_key, cfg, seed=SEED, sample_frac=SAMPLE_FRAC):
    log(f"Training {cfg_key} (per-pitch) -- {N_FOLDS}-fold pitcher-grouped CV")
    feat_cols = FEATURE_SETS[cfg['features']]
    params = lgb_params(cfg, seed)

    all_oof = []
    fold_times = []

    for fold in range(N_FOLDS):
        t0 = time.time()
        train_df = df[df['fold'] != fold].copy()
        test_df = df[df['fold'] == fold].copy()

        # Subsample training pitches
        train_sub = subsample_pitches(train_df, sample_frac, seed + fold)

        # Categorical season
        for d in [train_sub, test_df]:
            d['season'] = d['season'].astype('category')

        X_tr = train_sub[feat_cols]
        y_tr = train_sub['delta_run_exp']
        X_te = test_df[feat_cols]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        oof = test_df[['pitcher_id', 'season', 'delta_run_exp']].copy()
        oof['pred'] = preds
        oof['fold'] = fold
        all_oof.append(oof)

        elapsed = time.time() - t0
        fold_times.append(elapsed)
        log(f"  fold {fold}: {len(train_sub):,} train pitches, {len(test_df):,} test pitches, {elapsed:.1f}s")

        gc.collect()

    oof_df = pd.concat(all_oof, ignore_index=True)
    total_time = sum(fold_times)
    log(f"  Total CV time: {total_time:.1f}s")
    return oof_df, fold_times


# ---- CROSS-VALIDATED TRAINING (pitcher-season form) ----

def train_cv_pitcher_season(df, cfg_key, cfg, qualified_ps, seed=SEED):
    log(f"Training {cfg_key} (pitcher-season) -- {N_FOLDS}-fold pitcher-grouped CV")
    feat_cols = FEATURE_SETS[cfg['features']]
    numeric_feats = [f for f in feat_cols if f != 'season']
    params = lgb_params(cfg, seed)
    # Override for regression
    params['objective'] = 'regression'
    params['alpha'] = None

    # Aggregate to pitcher-season means
    ps_df = df.groupby(['pitcher_id', 'season'])[numeric_feats + ['delta_run_exp']].mean().reset_index()
    ps_counts = df.groupby(['pitcher_id', 'season']).size().reset_index(name='n_pitches')
    ps_df = ps_df.merge(ps_counts, on=['pitcher_id', 'season'])
    ps_df['fold'] = ps_df['pitcher_id'].astype(int) % N_FOLDS

    # Qualified only
    ps_df = ps_df[ps_df['n_pitches'] >= MIN_PITCHES].copy()
    log(f"  Pitcher-season rows: {len(ps_df)}")

    all_oof = []
    for fold in range(N_FOLDS):
        train_ps = ps_df[ps_df['fold'] != fold].copy()
        test_ps = ps_df[ps_df['fold'] == fold].copy()

        # Add season back as categorical
        train_ps['season'] = train_ps['season'].astype('category')
        test_ps['season'] = test_ps['season'].astype('category')

        X_tr = train_ps[feat_cols]
        y_tr = train_ps['delta_run_exp']
        X_te = test_ps[feat_cols]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        oof = test_ps[['pitcher_id', 'season', 'delta_run_exp', 'n_pitches']].copy()
        oof['pred'] = preds
        oof['fold'] = fold
        all_oof.append(oof)

    oof_df = pd.concat(all_oof, ignore_index=True)
    return oof_df, []


# ---- METRICS ----

def load_xfip(conn):
    df = pd.read_sql("""
        SELECT player_id AS pitcher_id, season, xfip, era
        FROM player_advanced_stats
        WHERE xfip IS NOT NULL
    """, conn)
    df['season'] = df['season'].astype(str)
    return df


def pitcher_season_corr(oof_df, xfip_df):
    """Aggregate OOF predictions to pitcher-season means, merge xFIP, compute Pearson r."""
    ps = oof_df.groupby(['pitcher_id', 'season'])['pred'].mean().reset_index()
    ps['pitcher_id'] = ps['pitcher_id'].astype(int)
    ps['season'] = ps['season'].astype(str)
    xfip_df['pitcher_id'] = xfip_df['pitcher_id'].astype(int)
    xfip_df['season'] = xfip_df['season'].astype(str)
    merged = ps.merge(xfip_df, on=['pitcher_id', 'season'], how='inner')
    if len(merged) < 20:
        return None, None, None
    r_xfip, p_xfip = stats.pearsonr(merged['pred'], merged['xfip'])
    r_era, p_era = stats.pearsonr(merged['pred'], merged['era'])
    return r_xfip, r_era, len(merged)


def bootstrap_corr_ci(oof_df, xfip_df, n_boot=BOOTSTRAP_N, seed=SEED):
    """Pitcher-clustered bootstrap CI for same-season xFIP correlation."""
    ps = oof_df.groupby(['pitcher_id', 'season'])['pred'].mean().reset_index()
    ps['pitcher_id'] = ps['pitcher_id'].astype(int)
    merged = ps.merge(xfip_df, on=['pitcher_id', 'season'], how='inner')
    if len(merged) < 20:
        return None, None, None

    rng = np.random.default_rng(seed)
    pitchers = merged['pitcher_id'].unique()
    boot_rs = []
    for _ in range(n_boot):
        sample_pitchers = rng.choice(pitchers, size=len(pitchers), replace=True)
        rows = []
        for p in sample_pitchers:
            rows.append(merged[merged['pitcher_id'] == p])
        boot_df = pd.concat(rows)
        if boot_df['xfip'].std() < 1e-9:
            continue
        r, _ = stats.pearsonr(boot_df['pred'], boot_df['xfip'])
        boot_rs.append(r)

    boot_rs = np.array(boot_rs)
    point = float(stats.pearsonr(merged['pred'], merged['xfip'])[0])
    ci_lo = float(np.percentile(boot_rs, 2.5))
    ci_hi = float(np.percentile(boot_rs, 97.5))
    return point, ci_lo, ci_hi


def paired_bootstrap_diff(oof_a, oof_b, xfip_df, n_boot=BOOTSTRAP_N, seed=SEED):
    """Paired bootstrap CI for difference in xFIP correlation between two configs."""
    ps_a = oof_a.groupby(['pitcher_id', 'season'])['pred'].mean().reset_index().rename(columns={'pred': 'pred_a'})
    ps_b = oof_b.groupby(['pitcher_id', 'season'])['pred'].mean().reset_index().rename(columns={'pred': 'pred_b'})
    ps_a['pitcher_id'] = ps_a['pitcher_id'].astype(int)
    ps_b['pitcher_id'] = ps_b['pitcher_id'].astype(int)
    xfip_df['pitcher_id'] = xfip_df['pitcher_id'].astype(int)
    merged = ps_a.merge(ps_b, on=['pitcher_id', 'season']).merge(xfip_df, on=['pitcher_id', 'season'])
    if len(merged) < 20:
        return None, None, None

    rng = np.random.default_rng(seed)
    pitchers = merged['pitcher_id'].unique()
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(pitchers, size=len(pitchers), replace=True)
        rows = pd.concat([merged[merged['pitcher_id'] == p] for p in samp])
        if rows['xfip'].std() < 1e-9:
            continue
        ra, _ = stats.pearsonr(rows['pred_a'], rows['xfip'])
        rb, _ = stats.pearsonr(rows['pred_b'], rows['xfip'])
        diffs.append(ra - rb)

    diffs = np.array(diffs)
    point_a, _ = stats.pearsonr(merged['pred_a'], merged['xfip'])
    point_b, _ = stats.pearsonr(merged['pred_b'], merged['xfip'])
    diff_point = float(point_a - point_b)
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))
    return diff_point, ci_lo, ci_hi


def apply_equivalence_rule(diff_point, ci_lo, ci_hi, margin=EQUIVALENCE_MARGIN):
    """
    'choose_M2_unless_M1_significantly_better'
    A2 vs A1: A2 (smaller feature set) is preferred unless A1 is significantly better.
    A1 is significantly better if diff (A1 - A2) > margin with CI lower > 0.
    """
    if diff_point is None:
        return "A2_chosen_insufficient_data"
    # diff_point = corr(A1) - corr(A2); positive means A1 is better
    # Significantly better if point > margin AND CI lower > 0
    if diff_point > margin and ci_lo > 0:
        return "A1_significantly_better_choose_A1"
    return "A2_chosen_equivalent_or_better"


def screening_test(df, base_cfg_key, base_cfg, drop_feature_or_features, xfip_df, qualified_ps, seed=SEED):
    """Quick 5-fold test of base_cfg minus specified feature(s). Returns corr diff."""
    cfg = dict(base_cfg)
    feats_base = list(FEATURE_SETS[cfg['features']])

    if isinstance(drop_feature_or_features, str):
        drop = [drop_feature_or_features]
    else:
        drop = list(drop_feature_or_features)

    feats_reduced = [f for f in feats_base if f not in drop]
    if 'season' not in feats_reduced and 'season' in feats_base:
        # season was dropped intentionally
        pass

    # Create a temporary feature set
    FEATURE_SETS['_tmp'] = feats_reduced
    cfg['features'] = '_tmp'

    if cfg.get('form') == 'pitcher_season':
        oof, _ = train_cv_pitcher_season(df, 'B_screen', cfg, qualified_ps, seed=seed)
    else:
        oof, _ = train_cv_per_pitch(df, feats_reduced, 'B_screen', cfg, seed=seed, sample_frac=SAMPLE_FRAC)

    r_xfip, _, _ = pitcher_season_corr(oof, xfip_df.copy())

    # Clean up
    del FEATURE_SETS['_tmp']
    return r_xfip, oof


def main():
    log("=== PART 2 START ===")
    conn = get_conn()

    df, qualified_ps = load_data(conn)
    xfip_df = load_xfip(conn)
    conn.close()

    results = {}
    oof_store = {}

    # ---- Time check: fit fold 0 of A1 ----
    log("Timing check: fitting fold 0 of A1")
    t0 = time.time()
    train_sub0 = subsample_pitches(df[df['fold'] != 0], SAMPLE_FRAC, SEED)
    train_sub0['season'] = train_sub0['season'].astype('category')
    X0 = train_sub0[F0]
    y0 = train_sub0['delta_run_exp']
    params_check = lgb_params(CONFIGS['A1'], SEED)
    model0 = lgb.LGBMRegressor(**params_check)
    model0.fit(X0, y0)
    fold0_time = time.time() - t0
    projected_total = fold0_time * N_FOLDS
    log(f"Fold 0 took {fold0_time:.1f}s; projected total = {projected_total:.1f}s ({projected_total/60:.1f} min)")
    del model0, X0, y0, train_sub0; gc.collect()

    effective_frac = SAMPLE_FRAC
    if projected_total > 9000:  # > 2.5 hours
        effective_frac = 0.15
        log(f"DECISION: switching to 15% sample (projected > 2.5h). Logged.")

    # ---- Train A0 ----
    log("--- Training A0 (v1 honest baseline) ---")
    oof_A0, times_A0 = train_cv_per_pitch(df, F0, 'A0', CONFIGS['A0'], seed=SEED, sample_frac=effective_frac)
    r_A0_xfip, r_A0_era, n_A0 = pitcher_season_corr(oof_A0, xfip_df.copy())
    ci_A0 = bootstrap_corr_ci(oof_A0, xfip_df.copy())
    log(f"A0 xFIP r={r_A0_xfip:.4f} ERA r={r_A0_era:.4f} n={n_A0} CI=[{ci_A0[1]:.4f},{ci_A0[2]:.4f}]")
    oof_store['A0'] = oof_A0

    # ---- Train A1 ----
    log("--- Training A1 ---")
    oof_A1, times_A1 = train_cv_per_pitch(df, F0, 'A1', CONFIGS['A1'], seed=SEED, sample_frac=effective_frac)
    r_A1_xfip, r_A1_era, n_A1 = pitcher_season_corr(oof_A1, xfip_df.copy())
    ci_A1 = bootstrap_corr_ci(oof_A1, xfip_df.copy())
    log(f"A1 xFIP r={r_A1_xfip:.4f} ERA r={r_A1_era:.4f} n={n_A1} CI=[{ci_A1[1]:.4f},{ci_A1[2]:.4f}]")
    oof_store['A1'] = oof_A1

    # ---- Train A2 ----
    log("--- Training A2 ---")
    oof_A2, times_A2 = train_cv_per_pitch(df, F2, 'A2', CONFIGS['A2'], seed=SEED, sample_frac=effective_frac)
    r_A2_xfip, r_A2_era, n_A2 = pitcher_season_corr(oof_A2, xfip_df.copy())
    ci_A2 = bootstrap_corr_ci(oof_A2, xfip_df.copy())
    log(f"A2 xFIP r={r_A2_xfip:.4f} ERA r={r_A2_era:.4f} n={n_A2} CI=[{ci_A2[1]:.4f},{ci_A2[2]:.4f}]")
    oof_store['A2'] = oof_A2

    # ---- Test 1: A2 vs A1 equivalence ----
    log("--- Test 1: A2 vs A1 equivalence ---")
    diff_A1_A2, ci_lo_diff, ci_hi_diff = paired_bootstrap_diff(oof_A1, oof_A2, xfip_df.copy())
    decision_A2_A1 = apply_equivalence_rule(diff_A1_A2, ci_lo_diff, ci_hi_diff)
    log(f"Test 1: diff(A1-A2)={diff_A1_A2:.4f} CI=[{ci_lo_diff:.4f},{ci_hi_diff:.4f}] -> {decision_A2_A1}")

    # Run A3 only if A2 significantly worse
    run_A3 = decision_A2_A1 == "A1_significantly_better_choose_A1"
    if run_A3:
        log("--- Training A3 (F1 features) ---")
        oof_A3, _ = train_cv_per_pitch(df, F1, 'A3', CONFIGS['A3'], seed=SEED, sample_frac=effective_frac)
        r_A3_xfip, r_A3_era, n_A3 = pitcher_season_corr(oof_A3, xfip_df.copy())
        ci_A3 = bootstrap_corr_ci(oof_A3, xfip_df.copy())
        log(f"A3 xFIP r={r_A3_xfip:.4f} CI=[{ci_A3[1]:.4f},{ci_A3[2]:.4f}]")
        oof_store['A3'] = oof_A3
    else:
        r_A3_xfip = None; ci_A3 = (None, None, None); n_A3 = None

    # Choose the per-pitch winner
    if decision_A2_A1 == "A1_significantly_better_choose_A1":
        per_pitch_winner = "A1"
        oof_winner = oof_A1
    else:
        per_pitch_winner = "A2"
        oof_winner = oof_A2

    # ---- Train B ----
    log("--- Training B (pitcher-season form) ---")
    oof_B, _ = train_cv_pitcher_season(df, 'B', CONFIGS['B'], qualified_ps, seed=SEED)
    r_B_xfip, r_B_era, n_B = pitcher_season_corr(oof_B, xfip_df.copy())
    ci_B = bootstrap_corr_ci(oof_B, xfip_df.copy())
    log(f"B xFIP r={r_B_xfip:.4f} ERA r={r_B_era:.4f} n={n_B} CI=[{ci_B[1]:.4f},{ci_B[2]:.4f}]")
    oof_store['B'] = oof_B

    # ---- Test 2: B vs per-pitch winner (form rule) ----
    # form_rule: choose_M2_unless_M1_significantly_better
    # M1=per_pitch, M2=pitcher_season
    # So choose B unless per_pitch_winner is significantly better
    log("--- Test 2: B vs per-pitch winner (form rule) ---")
    diff_PP_B, ci_lo_ppb, ci_hi_ppb = paired_bootstrap_diff(oof_winner, oof_B, xfip_df.copy())
    # "significantly better" means per_pitch winner is better by > margin and CI lower > 0
    if diff_PP_B is not None and diff_PP_B > EQUIVALENCE_MARGIN and ci_lo_ppb > 0:
        decision_form = f"{per_pitch_winner}_significantly_better_choose_{per_pitch_winner}"
        chosen_config = per_pitch_winner
    else:
        decision_form = "B_chosen_form_rule"
        chosen_config = "B"
    log(f"Test 2: diff({per_pitch_winner}-B)={diff_PP_B:.4f} CI=[{ci_lo_ppb:.4f},{ci_hi_ppb:.4f}] -> {decision_form}")

    # ---- Screening with B: drop tests ----
    log("--- Screening: B minus arm_angle ---")
    r_B_no_aa, oof_B_no_aa = screening_test(df, 'B', CONFIGS['B'], 'arm_angle', xfip_df.copy(), qualified_ps)
    log(f"B minus arm_angle: xFIP r={r_B_no_aa:.4f} (vs B={r_B_xfip:.4f})")

    log("--- Screening: B minus extension features ---")
    r_B_no_ext, oof_B_no_ext = screening_test(df, 'B', CONFIGS['B'],
        ['release_extension'], xfip_df.copy(), qualified_ps)
    log(f"B minus release_extension: xFIP r={r_B_no_ext:.4f} (vs B={r_B_xfip:.4f})")

    log("--- Screening: B minus season ---")
    r_B_no_season, _ = screening_test(df, 'B', CONFIGS['B'], 'season', xfip_df.copy(), qualified_ps)
    log(f"B minus season: xFIP r={r_B_no_season:.4f} (vs B={r_B_xfip:.4f})")

    # Apply drop rules
    # arm_angle: drop if R²>=0.90 AND equivalent (from Part 1: R²=0.789 < 0.90 -> keep)
    aa_drop_decision = "KEEP: arm_angle R2=0.789 < 0.90 threshold from Part 1"

    # extension: drop if removal_shown_equivalent
    ext_diff = (r_B_xfip or 0) - (r_B_no_ext or 0)
    if r_B_no_ext is not None and abs(ext_diff) <= EQUIVALENCE_MARGIN:
        ext_drop_decision = f"KEEP in F2 (already dropped extension_std); marginal diff={ext_diff:.4f}"
    else:
        ext_drop_decision = f"KEEP release_extension: diff={ext_diff:.4f} > {EQUIVALENCE_MARGIN} margin"

    # season: drop if inconclusive
    season_diff = (r_B_xfip or 0) - (r_B_no_season or 0)
    if r_B_no_season is not None and abs(season_diff) <= EQUIVALENCE_MARGIN:
        season_drop_decision = f"DECISION: inconclusive (diff={season_diff:.4f} <= margin) -> DROP per pre-registered rule"
        if chosen_config == 'B':
            log("DECISION: season inconclusive -> applying drop rule, season removed from chosen config B")
    else:
        season_drop_decision = f"KEEP season: diff={season_diff:.4f} (meaningful contribution)"

    log(f"arm_angle drop decision: {aa_drop_decision}")
    log(f"extension drop decision: {ext_drop_decision}")
    log(f"season drop decision: {season_drop_decision}")

    # ---- Compile results ----
    all_results = {
        "A0": {
            "features": "F0", "form": "per_pitch",
            "xfip_corr": r_A0_xfip, "era_corr": r_A0_era, "n": n_A0,
            "ci_lo": ci_A0[1], "ci_hi": ci_A0[2],
        },
        "A1": {
            "features": "F0", "form": "per_pitch",
            "xfip_corr": r_A1_xfip, "era_corr": r_A1_era, "n": n_A1,
            "ci_lo": ci_A1[1], "ci_hi": ci_A1[2],
        },
        "A2": {
            "features": "F2", "form": "per_pitch",
            "xfip_corr": r_A2_xfip, "era_corr": r_A2_era, "n": n_A2,
            "ci_lo": ci_A2[1], "ci_hi": ci_A2[2],
        },
        "B": {
            "features": "F2", "form": "pitcher_season",
            "xfip_corr": r_B_xfip, "era_corr": r_B_era, "n": n_B,
            "ci_lo": ci_B[1], "ci_hi": ci_B[2],
        },
        "B_no_arm_angle": {"xfip_corr": r_B_no_aa},
        "B_no_extension": {"xfip_corr": r_B_no_ext},
        "B_no_season": {"xfip_corr": r_B_no_season},
        "A3": {
            "features": "F1", "form": "per_pitch",
            "xfip_corr": r_A3_xfip, "ci_lo": ci_A3[1], "ci_hi": ci_A3[2], "n": n_A3,
            "note": "only run if A2 significantly worse than A1"
        },
    }

    tests = {
        "test1_A2_vs_A1": {
            "diff_A1_minus_A2": diff_A1_A2,
            "ci": [ci_lo_diff, ci_hi_diff],
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "decision": decision_A2_A1,
        },
        "test2_form_rule": {
            "diff_per_pitch_minus_B": diff_PP_B,
            "ci": [ci_lo_ppb, ci_hi_ppb],
            "per_pitch_contender": per_pitch_winner,
            "decision": decision_form,
        },
        "screening_arm_angle": {"decision": aa_drop_decision},
        "screening_extension": {"decision": ext_drop_decision},
        "screening_season": {"decision": season_drop_decision},
    }

    save_json({"configs": all_results, "tests": tests}, DIAG_DIR / "part2_results.json")
    log("Wrote part2_results.json")

    # Chosen config
    chosen_features = FEATURE_SETS.get(CONFIGS[chosen_config]['features'], F2)
    # If season drop rule triggered on B, note it
    chosen_cfg_doc = dict(CONFIGS[chosen_config])
    chosen_cfg_doc['key'] = chosen_config
    chosen_cfg_doc['feature_list'] = chosen_features
    chosen_cfg_doc['xfip_corr'] = all_results[chosen_config]['xfip_corr']
    chosen_cfg_doc['ci'] = [all_results[chosen_config]['ci_lo'], all_results[chosen_config]['ci_hi']]
    chosen_cfg_doc['season_drop_note'] = season_drop_decision

    save_json(chosen_cfg_doc, DIAG_DIR / "chosen_config.json")
    log(f"Wrote chosen_config.json -> chosen: {chosen_config}")
    log(f"=== PART 2 COMPLETE: chosen config = {chosen_config} ===")


if __name__ == "__main__":
    main()

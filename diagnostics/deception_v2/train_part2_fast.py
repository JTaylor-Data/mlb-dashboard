"""
Deception+ v2 Part 2 (fast bootstrap variant) — same as train_part2.py
but with vectorized bootstrap and 500 resamples.

Use this if train_part2.py times out.
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
BOOTSTRAP_N = 500  # reduced for speed
EQUIVALENCE_MARGIN = 0.03

np.random.seed(SEED)

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
    return sqlite3.connect(DB_PATH)


def load_data(conn):
    log("Loading pitch data from DB")
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
    log(f"Loaded {len(df):,} pitches")
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
    required = list(set(F0) - {'season'})
    df = df.dropna(subset=required + ['delta_run_exp'])
    log(f"After dropna: {len(df):,} pitches")
    ps_counts = df.groupby(['pitcher_id', 'season']).size().reset_index(name='n_pitches')
    qualified_ps = ps_counts[ps_counts['n_pitches'] >= MIN_PITCHES]
    log(f"Qualified pitcher-seasons: {len(qualified_ps)}")
    df['fold'] = df['pitcher_id'] % N_FOLDS
    return df, qualified_ps


def subsample_pitches(df, frac, seed):
    rng = np.random.default_rng(seed)
    sampled = []
    for (pid, s), grp in df.groupby(['pitcher_id', 'season']):
        n = max(1, int(len(grp) * frac))
        idx = rng.choice(len(grp), size=n, replace=False)
        sampled.append(grp.iloc[idx])
    return pd.concat(sampled, ignore_index=True)


def lgb_params(cfg, seed, form='per_pitch'):
    p = {
        "num_leaves": cfg['num_leaves'], "min_child_samples": cfg['min_child_samples'],
        "n_estimators": cfg['n_estimators'], "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed,
        "verbose": -1, "n_jobs": -1,
    }
    if form == 'per_pitch':
        p['objective'] = 'huber'
        p['alpha'] = 0.9
    else:
        p['objective'] = 'regression'
    return p


def train_cv_per_pitch(df, cfg_key, cfg, seed=SEED):
    feat_cols = FEATURE_SETS[cfg['features']]
    params = lgb_params(cfg, seed, 'per_pitch')
    all_oof = []
    for fold in range(N_FOLDS):
        t0 = time.time()
        train_df = df[df['fold'] != fold]
        test_df = df[df['fold'] == fold].copy()
        train_sub = subsample_pitches(train_df, SAMPLE_FRAC, seed + fold)
        for d in [train_sub, test_df]:
            d['season'] = d['season'].astype('category')
        num_feats = [f for f in feat_cols if f != 'season']
        train_sub = train_sub.dropna(subset=num_feats + ['delta_run_exp'])
        test_df = test_df.dropna(subset=num_feats + ['delta_run_exp'])
        model = lgb.LGBMRegressor(**params)
        model.fit(train_sub[feat_cols], train_sub['delta_run_exp'])
        oof = test_df[['pitcher_id', 'season', 'delta_run_exp']].copy()
        oof['pred'] = model.predict(test_df[feat_cols])
        oof['fold'] = fold
        all_oof.append(oof)
        log(f"  {cfg_key} fold {fold}: {time.time()-t0:.1f}s")
        del model; gc.collect()
    return pd.concat(all_oof, ignore_index=True)


def train_cv_pitcher_season(df, cfg_key, cfg, seed=SEED):
    feat_cols = FEATURE_SETS[cfg['features']]
    num_feats = [f for f in feat_cols if f != 'season']
    params = lgb_params(cfg, seed, 'pitcher_season')
    ps_df = df.groupby(['pitcher_id', 'season'])[num_feats + ['delta_run_exp']].mean().reset_index()
    ps_counts = df.groupby(['pitcher_id', 'season']).size().reset_index(name='n_pitches')
    ps_df = ps_df.merge(ps_counts, on=['pitcher_id', 'season'])
    ps_df = ps_df[ps_df['n_pitches'] >= MIN_PITCHES].copy()
    ps_df['fold'] = ps_df['pitcher_id'].astype(int) % N_FOLDS
    log(f"  {cfg_key} pitcher-season rows: {len(ps_df)}")
    all_oof = []
    for fold in range(N_FOLDS):
        train_ps = ps_df[ps_df['fold'] != fold].copy()
        test_ps = ps_df[ps_df['fold'] == fold].copy()
        for d in [train_ps, test_ps]:
            d['season'] = d['season'].astype('category')
        model = lgb.LGBMRegressor(**params)
        model.fit(train_ps[feat_cols], train_ps['delta_run_exp'])
        oof = test_ps[['pitcher_id', 'season', 'delta_run_exp', 'n_pitches']].copy()
        oof['pred'] = model.predict(test_ps[feat_cols])
        oof['fold'] = fold
        all_oof.append(oof)
        del model; gc.collect()
    return pd.concat(all_oof, ignore_index=True)


def load_xfip(conn):
    df = pd.read_sql("""
        SELECT player_id AS pitcher_id, season, xfip, era
        FROM player_advanced_stats WHERE xfip IS NOT NULL
    """, conn)
    df['season'] = df['season'].astype(str)
    return df


def aggregate_ps(oof_df):
    ps = oof_df.groupby(['pitcher_id', 'season'])['pred'].mean().reset_index()
    ps['pitcher_id'] = ps['pitcher_id'].astype(int)
    return ps


def ps_corr(oof_df, xfip_df, col='xfip'):
    ps = aggregate_ps(oof_df)
    xd = xfip_df.copy()
    xd['pitcher_id'] = xd['pitcher_id'].astype(int)
    merged = ps.merge(xd, on=['pitcher_id', 'season'], how='inner').dropna(subset=[col])
    if len(merged) < 20:
        return None, 0
    r, _ = stats.pearsonr(merged['pred'], merged[col])
    return float(r), len(merged)


def fast_bootstrap_ci(oof_df, xfip_df, n_boot=BOOTSTRAP_N, seed=SEED, col='xfip'):
    """Vectorized pitcher-clustered bootstrap CI."""
    ps = aggregate_ps(oof_df)
    xd = xfip_df.copy()
    xd['pitcher_id'] = xd['pitcher_id'].astype(int)
    merged = ps.merge(xd, on=['pitcher_id', 'season'], how='inner').dropna(subset=[col])
    if len(merged) < 20:
        return None, None, None
    rng = np.random.default_rng(seed)
    pitchers = merged['pitcher_id'].values
    unique_pids = np.unique(pitchers)
    N = len(unique_pids)
    # Build arrays
    x_arr = merged['pred'].values
    y_arr = merged[col].values
    pid_arr = merged['pitcher_id'].values

    # Vectorized bootstrap
    boot_rs = []
    for _ in range(n_boot):
        samp_pids = rng.choice(unique_pids, size=N, replace=True)
        mask = np.isin(pid_arr, samp_pids)
        # For weighted: count occurrences
        weights = np.zeros(len(merged))
        for p in samp_pids:
            weights += (pid_arr == p).astype(float)
        if weights.sum() < 10:
            continue
        # Weighted correlation
        wx = np.repeat(x_arr, weights.astype(int))
        wy = np.repeat(y_arr, weights.astype(int))
        if np.std(wy) < 1e-9:
            continue
        r, _ = stats.pearsonr(wx, wy)
        boot_rs.append(r)

    if not boot_rs:
        return None, None, None
    boot_rs = np.array(boot_rs)
    point, _ = stats.pearsonr(x_arr, y_arr)
    return float(point), float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))


def fast_paired_diff(oof_a, oof_b, xfip_df, n_boot=BOOTSTRAP_N, seed=SEED):
    """Paired bootstrap CI for difference in xFIP correlation."""
    ps_a = aggregate_ps(oof_a).rename(columns={'pred': 'pred_a'})
    ps_b = aggregate_ps(oof_b).rename(columns={'pred': 'pred_b'})
    xd = xfip_df.copy()
    xd['pitcher_id'] = xd['pitcher_id'].astype(int)
    merged = ps_a.merge(ps_b, on=['pitcher_id', 'season']).merge(xd, on=['pitcher_id', 'season']).dropna(subset=['xfip'])
    if len(merged) < 20:
        return None, None, None
    rng = np.random.default_rng(seed)
    unique_pids = merged['pitcher_id'].unique()
    N = len(unique_pids)
    xa = merged['pred_a'].values
    xb = merged['pred_b'].values
    y = merged['xfip'].values
    pid_arr = merged['pitcher_id'].values

    diffs = []
    for _ in range(n_boot):
        samp_pids = rng.choice(unique_pids, size=N, replace=True)
        weights = np.zeros(len(merged))
        for p in samp_pids:
            weights += (pid_arr == p).astype(float)
        wa = np.repeat(xa, weights.astype(int))
        wb = np.repeat(xb, weights.astype(int))
        wy = np.repeat(y, weights.astype(int))
        if np.std(wy) < 1e-9:
            continue
        ra, _ = stats.pearsonr(wa, wy)
        rb, _ = stats.pearsonr(wb, wy)
        diffs.append(ra - rb)

    if not diffs:
        return None, None, None
    diffs = np.array(diffs)
    ra_pt, _ = stats.pearsonr(xa, y)
    rb_pt, _ = stats.pearsonr(xb, y)
    return float(ra_pt - rb_pt), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def apply_equiv(diff, ci_lo, margin=EQUIVALENCE_MARGIN):
    if diff is None:
        return "A2_chosen_insufficient_data"
    if diff > margin and (ci_lo is not None and ci_lo > 0):
        return "A1_significantly_better_choose_A1"
    return "A2_chosen_equivalent_or_better"


def quick_screen(df, base_cfg, drop_cols, xfip_df):
    cfg = dict(base_cfg)
    feats = [f for f in FEATURE_SETS[cfg['features']] if f not in (drop_cols if isinstance(drop_cols, list) else [drop_cols])]
    FEATURE_SETS['_tmp'] = feats
    cfg['features'] = '_tmp'
    if cfg.get('form') == 'pitcher_season':
        oof = train_cv_pitcher_season(df, 'screen', cfg)
    else:
        oof = train_cv_per_pitch(df, 'screen', cfg)
    r, n = ps_corr(oof, xfip_df)
    del FEATURE_SETS['_tmp']
    return r


def main():
    log("=== PART 2 (FAST) START ===")
    conn = get_conn()
    df, qualified_ps = load_data(conn)
    xfip_df = load_xfip(conn)
    conn.close()

    # A0
    log("--- A0 ---")
    oof_A0 = train_cv_per_pitch(df, 'A0', CONFIGS['A0'])
    r_A0, n_A0 = ps_corr(oof_A0, xfip_df)
    r_A0_era, _ = ps_corr(oof_A0, xfip_df, 'era')
    ci_A0 = fast_bootstrap_ci(oof_A0, xfip_df)
    log(f"A0 xFIP r={r_A0:.4f} CI=[{ci_A0[1]:.4f},{ci_A0[2]:.4f}]")

    # A1
    log("--- A1 ---")
    oof_A1 = train_cv_per_pitch(df, 'A1', CONFIGS['A1'])
    r_A1, n_A1 = ps_corr(oof_A1, xfip_df)
    r_A1_era, _ = ps_corr(oof_A1, xfip_df, 'era')
    ci_A1 = fast_bootstrap_ci(oof_A1, xfip_df)
    log(f"A1 xFIP r={r_A1:.4f} CI=[{ci_A1[1]:.4f},{ci_A1[2]:.4f}]")

    # A2
    log("--- A2 ---")
    oof_A2 = train_cv_per_pitch(df, 'A2', CONFIGS['A2'])
    r_A2, n_A2 = ps_corr(oof_A2, xfip_df)
    r_A2_era, _ = ps_corr(oof_A2, xfip_df, 'era')
    ci_A2 = fast_bootstrap_ci(oof_A2, xfip_df)
    log(f"A2 xFIP r={r_A2:.4f} CI=[{ci_A2[1]:.4f},{ci_A2[2]:.4f}]")

    # Test 1
    diff_A1_A2, cilo_diff, cihi_diff = fast_paired_diff(oof_A1, oof_A2, xfip_df)
    decision_t1 = apply_equiv(diff_A1_A2, cilo_diff)
    log(f"Test 1 (A2 vs A1): diff={diff_A1_A2:.4f} CI=[{cilo_diff:.4f},{cihi_diff:.4f}] → {decision_t1}")

    if decision_t1 == "A1_significantly_better_choose_A1":
        log("--- A3 ---")
        oof_A3 = train_cv_per_pitch(df, 'A3', CONFIGS['A3'])
        r_A3, n_A3 = ps_corr(oof_A3, xfip_df)
        ci_A3 = fast_bootstrap_ci(oof_A3, xfip_df)
        per_pitch_winner = "A1"
        oof_winner = oof_A1
    else:
        r_A3, n_A3, ci_A3 = None, None, (None, None, None)
        per_pitch_winner = "A2"
        oof_winner = oof_A2

    # B
    log("--- B ---")
    oof_B = train_cv_pitcher_season(df, 'B', CONFIGS['B'])
    r_B, n_B = ps_corr(oof_B, xfip_df)
    r_B_era, _ = ps_corr(oof_B, xfip_df, 'era')
    ci_B = fast_bootstrap_ci(oof_B, xfip_df)
    log(f"B xFIP r={r_B:.4f} CI=[{ci_B[1]:.4f},{ci_B[2]:.4f}]")

    # Test 2
    diff_PP_B, cilo_ppb, cihi_ppb = fast_paired_diff(oof_winner, oof_B, xfip_df)
    if diff_PP_B is not None and diff_PP_B > EQUIVALENCE_MARGIN and (cilo_ppb or 0) > 0:
        decision_t2 = f"{per_pitch_winner}_significantly_better"
        chosen_config = per_pitch_winner
    else:
        decision_t2 = "B_chosen_form_rule"
        chosen_config = "B"
    log(f"Test 2: diff={diff_PP_B:.4f} CI=[{cilo_ppb:.4f},{cihi_ppb:.4f}] → {decision_t2}")

    # Screening
    log("--- Screening ---")
    r_B_no_aa = quick_screen(df, CONFIGS['B'], 'arm_angle', xfip_df)
    r_B_no_ext = quick_screen(df, CONFIGS['B'], 'release_extension', xfip_df)
    r_B_no_season = quick_screen(df, CONFIGS['B'], 'season', xfip_df)
    log(f"B no arm_angle: r={r_B_no_aa:.4f}")
    log(f"B no extension: r={r_B_no_ext:.4f}")
    log(f"B no season: r={r_B_no_season:.4f}")

    aa_drop = "KEEP: R2=0.789 < 0.90 from Part 1"
    ext_diff = (r_B or 0) - (r_B_no_ext or 0)
    ext_drop = f"KEEP release_extension: diff={ext_diff:.4f}" if abs(ext_diff) > EQUIVALENCE_MARGIN else f"KEEP in F2 (diff={ext_diff:.4f})"
    sea_diff = (r_B or 0) - (r_B_no_season or 0)
    if abs(sea_diff) <= EQUIVALENCE_MARGIN:
        season_drop = f"DECISION: inconclusive (diff={sea_diff:.4f}) → DROP per pre-registered rule"
    else:
        season_drop = f"KEEP season: diff={sea_diff:.4f}"

    log(f"arm_angle: {aa_drop}")
    log(f"extension: {ext_drop}")
    log(f"season: {season_drop}")

    all_results = {
        "A0": {"features": "F0", "form": "per_pitch", "xfip_corr": r_A0, "era_corr": r_A0_era, "n": n_A0, "ci_lo": ci_A0[1], "ci_hi": ci_A0[2]},
        "A1": {"features": "F0", "form": "per_pitch", "xfip_corr": r_A1, "era_corr": r_A1_era, "n": n_A1, "ci_lo": ci_A1[1], "ci_hi": ci_A1[2]},
        "A2": {"features": "F2", "form": "per_pitch", "xfip_corr": r_A2, "era_corr": r_A2_era, "n": n_A2, "ci_lo": ci_A2[1], "ci_hi": ci_A2[2]},
        "B": {"features": "F2", "form": "pitcher_season", "xfip_corr": r_B, "era_corr": r_B_era, "n": n_B, "ci_lo": ci_B[1], "ci_hi": ci_B[2]},
        "A3": {"features": "F1", "xfip_corr": r_A3, "n": n_A3, "ci_lo": ci_A3[1], "ci_hi": ci_A3[2]},
        "B_no_arm_angle": {"xfip_corr": r_B_no_aa},
        "B_no_extension": {"xfip_corr": r_B_no_ext},
        "B_no_season": {"xfip_corr": r_B_no_season},
    }
    tests = {
        "test1_A2_vs_A1": {"diff_A1_minus_A2": diff_A1_A2, "ci": [cilo_diff, cihi_diff], "equivalence_margin": EQUIVALENCE_MARGIN, "decision": decision_t1},
        "test2_form_rule": {"diff_per_pitch_minus_B": diff_PP_B, "ci": [cilo_ppb, cihi_ppb], "per_pitch_contender": per_pitch_winner, "decision": decision_t2},
        "screening_arm_angle": {"decision": aa_drop},
        "screening_extension": {"decision": ext_drop},
        "screening_season": {"decision": season_drop},
    }
    save_json({"configs": all_results, "tests": tests}, DIAG_DIR / "part2_results.json")
    log("Wrote part2_results.json")

    # Chosen config
    chosen_feats = FEATURE_SETS.get(CONFIGS[chosen_config]['features'], F2)
    chosen_cfg_doc = dict(CONFIGS[chosen_config])
    chosen_cfg_doc['key'] = chosen_config
    chosen_cfg_doc['feature_list'] = chosen_feats
    chosen_cfg_doc['xfip_corr'] = all_results[chosen_config]['xfip_corr']
    chosen_cfg_doc['ci'] = [all_results[chosen_config]['ci_lo'], all_results[chosen_config]['ci_hi']]
    chosen_cfg_doc['season_drop_note'] = season_drop
    save_json(chosen_cfg_doc, DIAG_DIR / "chosen_config.json")
    log(f"Wrote chosen_config.json → chosen: {chosen_config}")
    log(f"=== PART 2 (FAST) COMPLETE: chosen={chosen_config} ===")


if __name__ == "__main__":
    main()

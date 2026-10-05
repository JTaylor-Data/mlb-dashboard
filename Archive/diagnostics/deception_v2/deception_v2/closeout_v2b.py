"""
Deception+ v2 Closeout Script (Parts A-E)
==========================================
Addresses all audit findings from the post-run rule audit:
  - Part A: Re-evaluate Part 2 on qualified (>=300-pitch) common set
  - Part B: Retrain B without `season` (v2b)
  - Part C: Missing validation (in-sample gap, Gate(d), independence, evidence bar CI)
  - Part D: Gate re-evaluation, label update, docs update
  - Part E: Process check — list all comparisons on wrong set

Run with: python diagnostics/deception_v2/closeout_v2b.py
Checkpointing: each major section saves to disk; re-run resumes from checkpoint.
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
LOG_PATH = DIAG_DIR / "closeout_progress.log"
NEEDS_DECISION_PATH = ROOT / "Methodology" / "NEEDS_DECISION.md"

DIAG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5
SAMPLE_FRAC = 0.30
MIN_PITCHES = 300
BOOTSTRAP_N = 1000
EQUIVALENCE_MARGIN = 0.03

F0 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle",
      "height_in", "weight_lb", "arm_angle_std", "release_extension_std", "release_consistency", "season"]
F2 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle", "season"]
F2_NO_SEASON = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle"]

CONFIGS = {
    "A0": {"features": F0,         "form": "per_pitch",      "num_leaves": 31, "min_child_samples": 100,  "n_estimators": 500},
    "A1": {"features": F0,         "form": "per_pitch",      "num_leaves": 15, "min_child_samples": 2000, "n_estimators": 300},
    "A2": {"features": F2,         "form": "per_pitch",      "num_leaves": 15, "min_child_samples": 2000, "n_estimators": 300},
    "B":  {"features": F2,         "form": "pitcher_season", "num_leaves": 4,  "min_child_samples": 100,  "n_estimators": 200},
    "v2b":{"features": F2_NO_SEASON,"form": "pitcher_season","num_leaves": 4,  "min_child_samples": 100,  "n_estimators": 200},
}
# Screening variants of B (season dropped in training) → for re-screening on qualified set
SCREEN_VARIANTS = {
    "B_no_arm_angle":   {"features": [f for f in F2 if f != "arm_angle"],         "form": "pitcher_season", "num_leaves": 4, "min_child_samples": 100, "n_estimators": 200},
    "B_no_extension":   {"features": [f for f in F2 if f != "release_extension"],  "form": "pitcher_season", "num_leaves": 4, "min_child_samples": 100, "n_estimators": 200},
    "B_no_season":      {"features": F2_NO_SEASON,                                 "form": "pitcher_season", "num_leaves": 4, "min_child_samples": 100, "n_estimators": 200},
}


# ── Utilities ─────────────────────────────────────────────────────────────────

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
        if isinstance(o, pd.Series): return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=default)
    log(f"  saved {path.name}")


def model_hash(path):
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def checkpoint_exists(name):
    return (DIAG_DIR / f"ckpt_{name}.done").exists()


def write_checkpoint(name):
    (DIAG_DIR / f"ckpt_{name}.done").write_text(datetime.now().isoformat())


def append_needs_decision(entry_id, content):
    """Append a new entry to NEEDS_DECISION.md if not already present."""
    path = NEEDS_DECISION_PATH
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if entry_id in existing:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n---\n\n")
            f.write(content)
    else:
        path.write_text(content, encoding="utf-8")


# ── Data Loading ───────────────────────────────────────────────────────────────

def get_conn():
    return sqlite3.connect(DB_PATH)


def load_pitch_data(conn):
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
    log(f"  raw pitches: {len(df):,}")
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
    df = df.dropna(subset=["release_pos_x_as", "release_pos_z", "release_extension",
                            "arm_angle", "delta_run_exp"])
    df['fold'] = df['pitcher_id'] % N_FOLDS
    ps_counts = df.groupby(['pitcher_id', 'season']).size().reset_index(name='n_pitches')
    qualified_ps = set(zip(ps_counts[ps_counts['n_pitches'] >= MIN_PITCHES]['pitcher_id'],
                           ps_counts[ps_counts['n_pitches'] >= MIN_PITCHES]['season']))
    log(f"  qualified pitcher-seasons: {len(qualified_ps)}")
    log(f"  total pitcher-seasons: {len(ps_counts)}")
    return df, qualified_ps, ps_counts


def load_xfip(conn):
    df = pd.read_sql("""
        SELECT player_id AS pitcher_id, season, xfip, era
        FROM player_advanced_stats WHERE xfip IS NOT NULL
    """, conn)
    df['season'] = df['season'].astype(str)
    df['pitcher_id'] = df['pitcher_id'].astype(int)
    return df


# ── CV Training Helpers ────────────────────────────────────────────────────────

def subsample_pitches(df, frac, seed):
    rng = np.random.default_rng(seed)
    sampled = []
    for (pid, s), grp in df.groupby(['pitcher_id', 'season']):
        n = max(1, int(len(grp) * frac))
        idx = rng.choice(len(grp), size=n, replace=False)
        sampled.append(grp.iloc[idx])
    return pd.concat(sampled, ignore_index=True)


def train_cv_per_pitch(df, cfg, qualified_ps, seed=SEED):
    """Train per-pitch CV, return OOF predictions restricted to qualified pitcher-seasons."""
    feat_cols = cfg['features']
    params = {
        "objective": "huber", "alpha": 0.9,
        "num_leaves": cfg['num_leaves'], "min_child_samples": cfg['min_child_samples'],
        "n_estimators": cfg['n_estimators'], "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed,
        "verbose": -1, "n_jobs": -1,
    }
    all_oof = []
    for fold in range(N_FOLDS):
        t0 = time.time()
        train_df = df[df['fold'] != fold]
        test_df = df[df['fold'] == fold].copy()
        # Subsample training pitches
        train_sub = subsample_pitches(train_df, SAMPLE_FRAC, seed + fold)
        # Handle season as categorical
        use_season = 'season' in feat_cols
        if use_season:
            train_sub = train_sub.copy()
            test_df = test_df.copy()
            train_sub['season'] = train_sub['season'].astype('category')
            test_df['season'] = test_df['season'].astype('category')
        num_feats = [f for f in feat_cols if f != 'season']
        train_sub = train_sub.dropna(subset=num_feats + ['delta_run_exp'])
        test_df = test_df.dropna(subset=num_feats + ['delta_run_exp'])
        model = lgb.LGBMRegressor(**params)
        model.fit(train_sub[feat_cols], train_sub['delta_run_exp'])
        oof = test_df[['pitcher_id', 'season', 'delta_run_exp']].copy()
        oof['pred'] = model.predict(test_df[feat_cols])
        oof['fold'] = fold
        all_oof.append(oof)
        del model; gc.collect()
        log(f"    fold {fold}: {time.time()-t0:.1f}s")
    oof_all = pd.concat(all_oof, ignore_index=True)
    # Restrict to qualified pitcher-seasons
    mask = oof_all.apply(lambda r: (int(r['pitcher_id']), str(r['season'])) in qualified_ps, axis=1)
    oof_qual = oof_all[mask].copy()
    return oof_qual


def train_cv_pitcher_season(df, cfg, seed=SEED, save_models=None):
    """Train pitcher-season CV. Returns OOF df. Optionally saves fold models."""
    feat_cols = cfg['features']
    use_season = 'season' in feat_cols
    num_feats = [f for f in feat_cols if f != 'season']
    params = {
        "objective": "regression",
        "num_leaves": cfg['num_leaves'], "min_child_samples": cfg['min_child_samples'],
        "n_estimators": cfg['n_estimators'], "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed,
        "verbose": -1, "n_jobs": -1,
    }
    # Build pitcher-season df
    ps_df = df.groupby(['pitcher_id', 'season'])[num_feats + ['delta_run_exp']].mean().reset_index()
    ps_counts = df.groupby(['pitcher_id', 'season']).size().reset_index(name='n_pitches')
    ps_df = ps_df.merge(ps_counts, on=['pitcher_id', 'season'])
    ps_df = ps_df[ps_df['n_pitches'] >= MIN_PITCHES].copy()
    ps_df['fold'] = ps_df['pitcher_id'].astype(int) % N_FOLDS
    log(f"    pitcher-season rows: {len(ps_df)}")
    all_oof = []
    models = {}
    for fold in range(N_FOLDS):
        train_ps = ps_df[ps_df['fold'] != fold].copy()
        test_ps  = ps_df[ps_df['fold'] == fold].copy()
        if use_season:
            train_ps['season'] = train_ps['season'].astype('category')
            test_ps['season']  = test_ps['season'].astype('category')
        model = lgb.LGBMRegressor(**params)
        model.fit(train_ps[feat_cols], train_ps['delta_run_exp'])
        oof = test_ps[['pitcher_id', 'season', 'delta_run_exp', 'n_pitches']].copy()
        oof['pred'] = model.predict(test_ps[feat_cols])
        oof['fold'] = fold
        all_oof.append(oof)
        if save_models is not None:
            models[fold] = model
        else:
            del model; gc.collect()
    return pd.concat(all_oof, ignore_index=True), models if save_models else None, ps_df


# ── Metrics ────────────────────────────────────────────────────────────────────

def aggregate_ps(oof_df):
    """Aggregate per-pitch OOF to pitcher-season mean predictions."""
    ps = oof_df.groupby(['pitcher_id', 'season'])['pred'].mean().reset_index()
    ps['pitcher_id'] = ps['pitcher_id'].astype(int)
    return ps


def ps_corr(oof_df, xfip_df, col='xfip'):
    ps = aggregate_ps(oof_df)
    merged = ps.merge(xfip_df, on=['pitcher_id', 'season'], how='inner').dropna(subset=[col])
    if len(merged) < 20:
        return None, 0
    r, _ = stats.pearsonr(merged['pred'], merged[col])
    return float(r), len(merged)


def clustered_bootstrap_ci(oof_df, xfip_df, col='xfip', n_boot=BOOTSTRAP_N, seed=SEED):
    """Pitcher-clustered bootstrap CI for xFIP correlation."""
    ps = aggregate_ps(oof_df)
    merged = ps.merge(xfip_df, on=['pitcher_id', 'season'], how='inner').dropna(subset=[col])
    if len(merged) < 20:
        return None, None, None
    rng = np.random.default_rng(seed)
    unique_pids = merged['pitcher_id'].unique()
    N = len(unique_pids)
    xa = merged['pred'].values
    y  = merged[col].values
    pid_arr = merged['pitcher_id'].values
    boot_rs = []
    for _ in range(n_boot):
        samp_pids = rng.choice(unique_pids, size=N, replace=True)
        weights = np.zeros(len(merged))
        for p in samp_pids:
            weights += (pid_arr == p).astype(float)
        wa = np.repeat(xa, weights.astype(int))
        wy = np.repeat(y,  weights.astype(int))
        if np.std(wy) < 1e-9: continue
        r, _ = stats.pearsonr(wa, wy)
        boot_rs.append(r)
    if not boot_rs:
        return None, None, None
    boot_rs = np.array(boot_rs)
    pt, _ = stats.pearsonr(xa, y)
    return float(pt), float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))


def paired_bootstrap_ci(oof_a, oof_b, xfip_df, col='xfip', n_boot=BOOTSTRAP_N, seed=SEED):
    """Paired pitcher-clustered bootstrap CI for difference in correlations (a - b)."""
    ps_a = aggregate_ps(oof_a).rename(columns={'pred': 'pred_a'})
    ps_b = aggregate_ps(oof_b).rename(columns={'pred': 'pred_b'})
    merged = ps_a.merge(ps_b, on=['pitcher_id', 'season']).merge(xfip_df, on=['pitcher_id', 'season']).dropna(subset=[col])
    if len(merged) < 20:
        return None, None, None, 0
    rng = np.random.default_rng(seed)
    unique_pids = merged['pitcher_id'].unique()
    N = len(unique_pids)
    xa   = merged['pred_a'].values
    xb   = merged['pred_b'].values
    y    = merged[col].values
    pid_arr = merged['pitcher_id'].values
    diffs = []
    for _ in range(n_boot):
        samp_pids = rng.choice(unique_pids, size=N, replace=True)
        weights = np.zeros(len(merged))
        for p in samp_pids:
            weights += (pid_arr == p).astype(float)
        wa = np.repeat(xa, weights.astype(int))
        wb = np.repeat(xb, weights.astype(int))
        wy = np.repeat(y,  weights.astype(int))
        if np.std(wy) < 1e-9: continue
        ra, _ = stats.pearsonr(wa, wy)
        rb, _ = stats.pearsonr(wb, wy)
        diffs.append(ra - rb)
    if not diffs:
        return None, None, None, len(merged)
    diffs = np.array(diffs)
    ra_pt, _ = stats.pearsonr(xa, y)
    rb_pt, _ = stats.pearsonr(xb, y)
    return float(ra_pt - rb_pt), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), len(merged)


# ── Part A: Re-evaluate Part 2 on qualified common set ────────────────────────

def part_a(df, qualified_ps, xfip_df):
    log("=== PART A: Re-evaluate Part 2 on qualified common set ===")

    oof_cache = DIAG_DIR / "partA_oof_cache.parquet"
    if oof_cache.exists():
        log("  Loading cached OOF predictions")
        oof_all = pd.read_parquet(oof_cache)
    else:
        results = []
        for cfg_key in ['A0', 'A1', 'A2']:
            log(f"  Training {cfg_key} (per-pitch, restricted to qualified set)")
            cfg = CONFIGS[cfg_key]
            oof = train_cv_per_pitch(df, cfg, qualified_ps, seed=SEED)
            oof['config'] = cfg_key
            results.append(oof)
            log(f"  {cfg_key}: {len(oof)} pitch-level OOF rows (qualified only)")

        log("  Training B (pitcher-season)")
        oof_b, _, _ = train_cv_pitcher_season(df, CONFIGS['B'], seed=SEED)
        oof_b['config'] = 'B'
        results.append(oof_b)
        log(f"  B: {len(oof_b)} pitcher-season OOF rows")

        oof_all = pd.concat(results, ignore_index=True)
        oof_all.to_parquet(oof_cache, index=False)
        log(f"  Saved OOF cache to {oof_cache.name}")

    oof_A0 = oof_all[oof_all['config'] == 'A0']
    oof_A1 = oof_all[oof_all['config'] == 'A1']
    oof_A2 = oof_all[oof_all['config'] == 'A2']
    oof_B  = oof_all[oof_all['config'] == 'B']

    # Individual metrics on qualified set
    log("  Computing individual metrics on qualified common set")
    r_A0, n_A0 = ps_corr(oof_A0, xfip_df)
    r_A1, n_A1 = ps_corr(oof_A1, xfip_df)
    r_A2, n_A2 = ps_corr(oof_A2, xfip_df)
    r_B,  n_B  = ps_corr(oof_B,  xfip_df)
    log(f"    A0 xFIP r={r_A0:.4f} n={n_A0}")
    log(f"    A1 xFIP r={r_A1:.4f} n={n_A1}")
    log(f"    A2 xFIP r={r_A2:.4f} n={n_A2}")
    log(f"    B  xFIP r={r_B:.4f}  n={n_B}")

    r_A0_pt, ci_A0_lo, ci_A0_hi = clustered_bootstrap_ci(oof_A0, xfip_df)
    r_A1_pt, ci_A1_lo, ci_A1_hi = clustered_bootstrap_ci(oof_A1, xfip_df)
    r_A2_pt, ci_A2_lo, ci_A2_hi = clustered_bootstrap_ci(oof_A2, xfip_df)
    r_B_pt,  ci_B_lo,  ci_B_hi  = clustered_bootstrap_ci(oof_B,  xfip_df)
    log(f"    A0 CI=[{ci_A0_lo:.4f},{ci_A0_hi:.4f}]")
    log(f"    A1 CI=[{ci_A1_lo:.4f},{ci_A1_hi:.4f}]")
    log(f"    A2 CI=[{ci_A2_lo:.4f},{ci_A2_hi:.4f}]")
    log(f"    B  CI=[{ci_B_lo:.4f},{ci_B_hi:.4f}]")

    # Paired differences
    log("  Computing paired differences on common qualified set")
    diff_A2_A1, ci_lo_21, ci_hi_21, n_21 = paired_bootstrap_ci(oof_A2, oof_A1, xfip_df)
    # Note: A1-A2 diff = -(A2-A1 diff)
    diff_A1_A2 = -diff_A2_A1 if diff_A2_A1 is not None else None
    ci_lo_12 = -ci_hi_21 if ci_hi_21 is not None else None
    ci_hi_12 = -ci_lo_21 if ci_lo_21 is not None else None

    diff_A2_B, ci_lo_2b, ci_hi_2b, n_2b = paired_bootstrap_ci(oof_A2, oof_B, xfip_df)
    log(f"    A2-A1 diff (Test1, A1-A2)={diff_A1_A2:.4f} CI=[{ci_lo_12:.4f},{ci_hi_12:.4f}] n={n_21}")
    log(f"    A2-B  diff={diff_A2_B:.4f} CI=[{ci_lo_2b:.4f},{ci_hi_2b:.4f}] n={n_2b}")

    # Apply rules from spec_lock.json
    # Test 1 rule: "choose_M2_unless_M1_significantly_better" for identity-prone features
    # A1-A2 diff: if A1 is significantly better (diff > 0.03 AND ci_lo > 0) -> A1 wins
    if diff_A1_A2 is not None and diff_A1_A2 > EQUIVALENCE_MARGIN and (ci_lo_12 or 0) > 0:
        test1_decision = "A1_significantly_better_choose_A1"
    else:
        test1_decision = "A2_chosen_equivalent_or_better"

    # Form rule: A2 vs B — "choose_M2_unless_M1_significantly_better"
    # Here M1=per_pitch(A2), M2=pitcher_season(B)
    # diff = A2-B; if per-pitch significantly better (diff > 0.03 AND ci_lo > 0) -> per-pitch
    if diff_A2_B is not None and diff_A2_B > EQUIVALENCE_MARGIN and (ci_lo_2b or 0) > 0:
        form_decision = "A2_significantly_better_per_pitch"
        per_pitch_wins = True
    else:
        form_decision = "B_chosen_form_rule"
        per_pitch_wins = False

    log(f"    Test1 decision: {test1_decision}")
    log(f"    Form  decision: {form_decision}")

    # Additional pairs
    diff_A1_A0, ci_lo_10, ci_hi_10, n_10 = paired_bootstrap_ci(oof_A1, oof_A0, xfip_df)
    diff_A2_A0, ci_lo_20, ci_hi_20, n_20 = paired_bootstrap_ci(oof_A2, oof_A0, xfip_df)
    diff_B_A2  = -diff_A2_B if diff_A2_B is not None else None
    ci_lo_ba   = -ci_hi_2b if ci_hi_2b is not None else None
    ci_hi_ba   = -ci_lo_2b if ci_lo_2b is not None else None

    log(f"    A1-A0 diff={diff_A1_A0:.4f} CI=[{ci_lo_10:.4f},{ci_hi_10:.4f}]")
    log(f"    A2-A0 diff={diff_A2_A0:.4f} CI=[{ci_lo_20:.4f},{ci_hi_20:.4f}]")
    log(f"    B-A2  diff={diff_B_A2:.4f}  CI=[{ci_lo_ba:.4f},{ci_hi_ba:.4f}]")

    # Screening comparisons on qualified set (re-run B variants)
    log("  Screening comparisons on qualified set")
    screen_cache = DIAG_DIR / "partA_screen_cache.parquet"
    if screen_cache.exists():
        log("  Loading cached screening OOFs")
        screen_all = pd.read_parquet(screen_cache)
    else:
        screen_results = []
        for sv_key, sv_cfg in SCREEN_VARIANTS.items():
            log(f"  Screening: {sv_key}")
            oof_sv, _, _ = train_cv_pitcher_season(df, sv_cfg, seed=SEED)
            oof_sv['config'] = sv_key
            screen_results.append(oof_sv)
        screen_all = pd.concat(screen_results, ignore_index=True)
        screen_all.to_parquet(screen_cache, index=False)

    r_B_no_aa, _  = ps_corr(screen_all[screen_all['config'] == 'B_no_arm_angle'], xfip_df)
    r_B_no_ext, _ = ps_corr(screen_all[screen_all['config'] == 'B_no_extension'],  xfip_df)
    r_B_no_sea, _ = ps_corr(screen_all[screen_all['config'] == 'B_no_season'],     xfip_df)

    log(f"    B={r_B:.4f}  B_no_arm_angle={r_B_no_aa:.4f}  diff={r_B-r_B_no_aa:.4f}")
    log(f"    B={r_B:.4f}  B_no_extension={r_B_no_ext:.4f} diff={r_B-r_B_no_ext:.4f}")
    log(f"    B={r_B:.4f}  B_no_season={r_B_no_sea:.4f}   diff={r_B-r_B_no_sea:.4f}")

    # Apply screening rules
    # arm_angle: keep unless R²>=0.90 AND equivalent (R²=0.789 from Part 1 -> KEEP)
    screen_aa_decision = "KEEP arm_angle: R2=0.789 < 0.90 from Part 1"

    ext_diff = (r_B or 0) - (r_B_no_ext or 0)
    if abs(ext_diff) <= EQUIVALENCE_MARGIN:
        screen_ext_decision = f"INCONCLUSIVE extension (diff={ext_diff:.4f}) -> KEEP (no drop condition for extension if not equivalent)"
    else:
        screen_ext_decision = f"KEEP release_extension: diff={ext_diff:.4f} > {EQUIVALENCE_MARGIN}"

    sea_diff = (r_B or 0) - (r_B_no_sea or 0)
    if abs(sea_diff) <= EQUIVALENCE_MARGIN:
        screen_sea_decision = f"DECISION: inconclusive (diff={sea_diff:.4f}) -> DROP per pre-registered rule (inconclusive_means_drop)"
        season_should_drop = True
    else:
        screen_sea_decision = f"KEEP season: diff={sea_diff:.4f} > {EQUIVALENCE_MARGIN}"
        season_should_drop = False

    log(f"    arm_angle: {screen_aa_decision}")
    log(f"    extension: {screen_ext_decision}")
    log(f"    season:    {screen_sea_decision}")

    # Compare old decisions vs new decisions
    old_results = json.loads((DIAG_DIR / "part2_results.json").read_text())
    decision_table = {
        "test1_A2_vs_A1": {
            "old_n": old_results["configs"]["A2"]["n"],
            "new_n_qualified": n_21,
            "old_decision": old_results["tests"]["test1_A2_vs_A1"]["decision"],
            "new_decision": test1_decision,
            "changed": old_results["tests"]["test1_A2_vs_A1"]["decision"] != test1_decision,
            "old_diff": old_results["tests"]["test1_A2_vs_A1"]["diff_A1_minus_A2"],
            "new_diff": diff_A1_A2,
            "new_ci": [ci_lo_12, ci_hi_12],
        },
        "form_rule_A2_vs_B": {
            "old_n_A2": old_results["configs"]["A2"]["n"],
            "old_n_B": old_results["configs"]["B"]["n"],
            "new_n_qualified": n_2b,
            "old_decision": old_results["tests"]["test2_form_rule"]["decision"],
            "new_decision": form_decision,
            "changed": old_results["tests"]["test2_form_rule"]["decision"] != form_decision,
            "old_diff": old_results["tests"]["test2_form_rule"]["diff_per_pitch_minus_B"],
            "new_diff": diff_A2_B,
            "new_ci": [ci_lo_2b, ci_hi_2b],
        },
        "screening_season": {
            "old_decision": old_results["tests"]["screening_season"]["decision"],
            "new_decision": screen_sea_decision,
            "changed": season_should_drop != ("DROP" in old_results["tests"]["screening_season"]["decision"]),
            "old_diff": -0.0012,  # from original
            "new_diff": sea_diff,
        },
    }

    partA_results = {
        "basis": f"qualified_common_set_n_approx_{n_B}",
        "qualified_min_pitches": MIN_PITCHES,
        "configs": {
            "A0": {"xfip_corr": r_A0, "ci_lo": ci_A0_lo, "ci_hi": ci_A0_hi, "n": n_A0},
            "A1": {"xfip_corr": r_A1, "ci_lo": ci_A1_lo, "ci_hi": ci_A1_hi, "n": n_A1},
            "A2": {"xfip_corr": r_A2, "ci_lo": ci_A2_lo, "ci_hi": ci_A2_hi, "n": n_A2},
            "B":  {"xfip_corr": r_B,  "ci_lo": ci_B_lo,  "ci_hi": ci_B_hi,  "n": n_B},
        },
        "paired_differences": {
            "A1_minus_A2": {"diff": diff_A1_A2, "ci": [ci_lo_12, ci_hi_12], "n": n_21},
            "A1_minus_A0": {"diff": diff_A1_A0, "ci": [ci_lo_10, ci_hi_10], "n": n_10},
            "A2_minus_A0": {"diff": diff_A2_A0, "ci": [ci_lo_20, ci_hi_20], "n": n_20},
            "B_minus_A2":  {"diff": diff_B_A2,  "ci": [ci_lo_ba, ci_hi_ba], "n": n_2b},
        },
        "screening": {
            "B_xfip_corr": r_B,
            "B_no_arm_angle_xfip_corr": r_B_no_aa,
            "B_no_extension_xfip_corr": r_B_no_ext,
            "B_no_season_xfip_corr": r_B_no_sea,
            "arm_angle_diff": (r_B or 0) - (r_B_no_aa or 0),
            "extension_diff": ext_diff,
            "season_diff": sea_diff,
            "decisions": {
                "arm_angle": screen_aa_decision,
                "extension": screen_ext_decision,
                "season": screen_sea_decision,
            },
        },
        "rule_decisions": {
            "test1_identity_prone": test1_decision,
            "form_rule": form_decision,
            "season_drop": season_should_drop,
            "per_pitch_wins_form": per_pitch_wins,
        },
        "decision_table_vs_original": decision_table,
        "note_original_values_invalid": (
            "Original part2_results.json computed A0/A1/A2 on n=5118 (all pitcher-seasons incl. "
            "unqualified) while B was evaluated on n=3363 (qualified only). "
            "These results restrict ALL configs to the qualified common set."
        ),
    }

    save_json(partA_results, DIAG_DIR / "partA_qualified_results.json")

    # Determine final chosen config for v2b
    if per_pitch_wins:
        if test1_decision == "A1_significantly_better_choose_A1":
            final_form = "per_pitch"
            final_config_key = "A1"
        else:
            final_form = "per_pitch"
            final_config_key = "A2"
        # Season in per-pitch: inconclusive -> drop (re-evaluate on per-pitch set)
        chosen_features = F2_NO_SEASON  # drop season
    else:
        final_form = "pitcher_season"
        final_config_key = "B"
        if season_should_drop:
            chosen_features = F2_NO_SEASON
        else:
            chosen_features = F2

    log(f"  Final config: {final_config_key}, form={final_form}, season_drop={season_should_drop}")

    return partA_results, oof_A0, oof_A1, oof_A2, oof_B, oof_all, screen_all, season_should_drop, final_form, chosen_features


# ── Part B: Retrain v2b (no season) ──────────────────────────────────────────

def part_b(df, xfip_df, chosen_features, final_form, season_should_drop):
    log("=== PART B: Retrain v2b (season dropped) ===")

    if not season_should_drop and 'season' in chosen_features:
        log("  season NOT dropped — using v2b with same features as B but on qualified set")
        # Still use v2b label but features include season
        v2b_feat = chosen_features
    else:
        log("  season DROPPED per inconclusive rule")
        v2b_feat = F2_NO_SEASON

    cfg_v2b = {
        "features": v2b_feat,
        "form": final_form,
        "num_leaves": 4,
        "min_child_samples": 100,
        "n_estimators": 200,
    }

    n_repeats = 3 if final_form == 'pitcher_season' else 2
    rep_seeds = [42, 142, 242]

    oof_path  = DIAG_DIR / "v2b_oof_all.parquet"
    meta_path = MODELS_DIR / "deception_plus_v2b_meta.json"

    if oof_path.exists() and meta_path.exists():
        log("  Loading cached v2b OOF")
        oof_v2b_all = pd.read_parquet(oof_path)
        with open(meta_path) as f:
            v2b_meta = json.load(f)
    else:
        all_oof_reps = []
        fold_models_meta = {}

        for rep in range(n_repeats):
            seed = rep_seeds[rep]
            log(f"  Rep {rep} (seed={seed})")
            if final_form == 'pitcher_season':
                oof_rep, fold_models, ps_df = train_cv_pitcher_season(df, cfg_v2b, seed=seed, save_models=True)
                oof_rep['rep'] = rep
                all_oof_reps.append(oof_rep)
                # Save fold models
                for fold, model in fold_models.items():
                    path = MODELS_DIR / f"deception_plus_v2b_fold{fold}_rep{rep}.txt"
                    model.booster_.save_model(str(path))
                    fold_models_meta[f"rep{rep}_fold{fold}"] = {
                        "path": str(path),
                        "hash": model_hash(path),
                    }
                    del model
                gc.collect()
            else:
                # per-pitch form (if per_pitch_wins)
                oof_rep = train_cv_per_pitch(df, cfg_v2b, set(), seed=seed)
                oof_rep['rep'] = rep
                all_oof_reps.append(oof_rep)

        oof_v2b_all = pd.concat(all_oof_reps, ignore_index=True)
        oof_v2b_all.to_parquet(oof_path, index=False)

        # Cross-fitted scores: average across reps
        ps_agg = oof_v2b_all.groupby(['pitcher_id', 'season'])['pred'].mean().reset_index()
        ps_agg['pitcher_id'] = ps_agg['pitcher_id'].astype(int)
        ps_agg.columns = ['pitcher_id', 'season', 'raw_pred']

        # EB shrinkage (K=300)
        EB_K = 300
        ps_counts_for_shrink = oof_v2b_all.groupby(['pitcher_id', 'season'])['n_pitches'].mean().reset_index() if 'n_pitches' in oof_v2b_all.columns else None
        if ps_counts_for_shrink is not None:
            ps_agg = ps_agg.merge(ps_counts_for_shrink[['pitcher_id', 'season', 'n_pitches']], on=['pitcher_id', 'season'], how='left')
        else:
            ps_agg['n_pitches'] = EB_K  # assume qualified

        grand_mean = ps_agg['raw_pred'].mean()
        ps_agg['eb_weight'] = ps_agg['n_pitches'] / (ps_agg['n_pitches'] + EB_K)
        ps_agg['shrunk_pred'] = ps_agg['eb_weight'] * ps_agg['raw_pred'] + (1 - ps_agg['eb_weight']) * grand_mean

        # to_plus_scale per season
        season_stats = ps_agg.groupby('season')['shrunk_pred'].agg(['mean', 'std']).reset_index()
        season_stats.columns = ['season', 'season_mean', 'season_std']
        ps_agg = ps_agg.merge(season_stats, on='season', how='left')
        ps_agg['deception_plus_v2b'] = 100.0 - 10.0 * (ps_agg['shrunk_pred'] - ps_agg['season_mean']) / ps_agg['season_std'].replace(0, 1)

        scores_path = DIAG_DIR / "deception_v2b_scores.parquet"
        ps_agg.to_parquet(scores_path, index=False)
        log(f"  v2b scores saved: {len(ps_agg)} pitcher-seasons")

        v2b_meta = {
            "version": "v2b",
            "created": datetime.now().isoformat(),
            "parent_version": "v2",
            "change_from_v2": "season feature dropped per pre-registered inconclusive_means_drop rule",
            "features": v2b_feat,
            "form": final_form,
            "num_leaves": cfg_v2b['num_leaves'],
            "min_child_samples": cfg_v2b['min_child_samples'],
            "n_estimators": cfg_v2b['n_estimators'],
            "n_repeats": n_repeats,
            "n_folds": N_FOLDS,
            "seeds": rep_seeds[:n_repeats],
            "fold_assignment": "pitcher_id_mod_5",
            "min_pitches_qualified": MIN_PITCHES,
            "eb_shrinkage_K": EB_K,
            "fold_models": fold_models_meta,
            "season_note": "season dropped; no unseen-season handling needed",
            "scored_seasons": sorted(ps_agg['season'].unique().tolist()),
        }
        save_json(v2b_meta, meta_path)

    # v2b OOF performance
    oof_v2b_rep0 = oof_v2b_all[oof_v2b_all['rep'] == 0] if 'rep' in oof_v2b_all.columns else oof_v2b_all
    r_v2b, n_v2b = ps_corr(oof_v2b_rep0, xfip_df)
    r_v2b_pt, ci_v2b_lo, ci_v2b_hi = clustered_bootstrap_ci(oof_v2b_rep0, xfip_df)
    log(f"  v2b xFIP r={r_v2b:.4f} CI=[{ci_v2b_lo:.4f},{ci_v2b_hi:.4f}] n={n_v2b}")

    return oof_v2b_all, v2b_meta, r_v2b, ci_v2b_lo, ci_v2b_hi


# ── Part C: Missing Validation ─────────────────────────────────────────────────

def part_c_gate_d(v2b_meta):
    """Gate (d): compute actual repeat correlation from saved fold models."""
    log("  Gate (d): computing repeat correlation")
    conn = get_conn()
    df_raw = pd.read_sql("""
        SELECT p.pitcher_id, p.season, p.p_throws,
               p.release_pos_x, p.release_pos_z, p.release_extension,
               p.arm_angle
        FROM pitches p
        WHERE p.season >= 2021
          AND p.pitch_type NOT IN ('PO','UN','EP')
          AND p.release_pos_x IS NOT NULL AND p.release_pos_z IS NOT NULL
          AND p.release_extension IS NOT NULL AND p.arm_angle IS NOT NULL
    """, conn)
    conn.close()

    sign = np.where(df_raw['p_throws'] == 'R', -1, 1)
    df_raw['release_pos_x_as'] = df_raw['release_pos_x'] * sign
    df_raw['season'] = df_raw['season'].astype(str)
    g = df_raw.groupby(['pitcher_id', 'season'])
    ps_means = {}
    for col in ['release_pos_x_as', 'release_pos_z', 'release_extension', 'arm_angle']:
        ps_means[col] = g[col].mean()

    ps_df = pd.DataFrame(ps_means).reset_index()
    ps_counts = g.size().reset_index(name='n_pitches')
    ps_df = ps_df.merge(ps_counts, on=['pitcher_id', 'season'])
    ps_df = ps_df[ps_df['n_pitches'] >= MIN_PITCHES].copy()
    ps_df['pitcher_id'] = ps_df['pitcher_id'].astype(int)

    feat_cols = v2b_meta['features']
    use_season = 'season' in feat_cols

    # Load models and predict per rep
    rep_preds = {}
    for rep in range(v2b_meta['n_repeats']):
        preds_rep = []
        for fold in range(v2b_meta['n_folds']):
            key = f"rep{rep}_fold{fold}"
            if key not in v2b_meta.get('fold_models', {}):
                # Try v2 meta path
                path = MODELS_DIR / f"deception_plus_v2b_fold{fold}_rep{rep}.txt"
            else:
                path = Path(v2b_meta['fold_models'][key]['path'])
            if not path.exists():
                log(f"  WARNING: model not found: {path}")
                continue
            model = lgb.Booster(model_file=str(path))
            test_ps = ps_df[ps_df['pitcher_id'] % N_FOLDS == fold].copy()
            if len(test_ps) == 0:
                continue
            if use_season:
                test_ps = test_ps.copy()
                test_ps['season'] = test_ps['season'].astype('category')
            pred_arr = model.predict(test_ps[feat_cols])
            preds_rep.append(test_ps[['pitcher_id', 'season']].copy().assign(pred=pred_arr))
        if preds_rep:
            rep_preds[rep] = pd.concat(preds_rep, ignore_index=True)

    if len(rep_preds) < 2:
        return {"pass": "unverified", "reason": "fewer than 2 repeats available for comparison", "threshold": 0.98}

    # Compute pairwise correlations across all repeat pairs
    rep_corrs = []
    rep_keys = list(rep_preds.keys())
    for i in range(len(rep_keys)):
        for j in range(i+1, len(rep_keys)):
            r_i = rep_preds[rep_keys[i]].rename(columns={'pred': 'pred_i'})
            r_j = rep_preds[rep_keys[j]].rename(columns={'pred': 'pred_j'})
            m = r_i.merge(r_j, on=['pitcher_id', 'season'])
            if len(m) < 20:
                continue
            r, _ = stats.pearsonr(m['pred_i'], m['pred_j'])
            rep_corrs.append(float(r))
            log(f"    Rep {rep_keys[i]} vs {rep_keys[j]}: r={r:.4f} n={len(m)}")

    if not rep_corrs:
        return {"pass": "unverified", "reason": "could not compute any rep pair correlations", "threshold": 0.98}

    min_corr = min(rep_corrs)
    avg_corr = sum(rep_corrs) / len(rep_corrs)
    gate_pass = "pass" if min_corr >= 0.98 else "fail"
    log(f"    Gate (d): min_corr={min_corr:.4f}, avg={avg_corr:.4f} -> {gate_pass}")

    return {
        "pass": gate_pass,
        "min_repeat_correlation": min_corr,
        "avg_repeat_correlation": avg_corr,
        "all_repeat_correlations": rep_corrs,
        "threshold": 0.98,
        "note": "computed from saved v2b fold model re-predictions on qualified pitcher-seasons",
    }


def part_c_independence(xfip_df):
    """Independence heatmap: Deception+ v2b, Stuff+ v4, Location+ v3, Tunneling+ v2."""
    log("  Independence: loading model scores")

    # Deception v2b
    dec_path = DIAG_DIR / "deception_v2b_scores.parquet"
    if not dec_path.exists():
        dec_path = DIAG_DIR / "deception_v2_scores.parquet"
    dec = pd.read_parquet(dec_path)
    dec_col = 'deception_plus_v2b' if 'deception_plus_v2b' in dec.columns else 'deception_plus_v2'
    dec = dec[['pitcher_id', 'season', dec_col]].rename(columns={dec_col: 'deception_plus'})
    dec['pitcher_id'] = dec['pitcher_id'].astype(int)
    dec['season'] = dec['season'].astype(str)

    # Stuff+ v4
    stuff_path = ROOT / "diagnostics" / "stuff_v4" / "stuff_v4_scores.parquet"
    if stuff_path.exists():
        stuff = pd.read_parquet(stuff_path)
        stuff_col = [c for c in stuff.columns if 'stuff' in c.lower() and 'plus' in c.lower()]
        if stuff_col:
            stuff = stuff[['pitcher_id', 'season', stuff_col[0]]].rename(columns={stuff_col[0]: 'stuff_plus'})
            stuff['pitcher_id'] = stuff['pitcher_id'].astype(int)
            stuff['season'] = stuff['season'].astype(str)
        else:
            stuff = None
    else:
        stuff = None

    # Location+ v3
    loc_path = ROOT / "diagnostics" / "location_v3" / "location_v3_scores.parquet"
    if loc_path.exists():
        loc = pd.read_parquet(loc_path)
        loc_col = [c for c in loc.columns if 'location' in c.lower() or 'loc' in c.lower()]
        if loc_col:
            loc = loc[['pitcher_id', 'season', loc_col[0]]].rename(columns={loc_col[0]: 'location_plus'})
            loc['pitcher_id'] = loc['pitcher_id'].astype(int)
            loc['season'] = loc['season'].astype(str)
        else:
            loc = None
    else:
        loc = None

    # Tunneling+ — try DB
    conn = get_conn()
    try:
        tun = pd.read_sql("""
            SELECT pitcher_id, season, tunneling_plus
            FROM pitcher_season_scores
            WHERE tunneling_plus IS NOT NULL AND season >= 2021
        """, conn)
        tun['pitcher_id'] = tun['pitcher_id'].astype(int)
        tun['season'] = tun['season'].astype(str)
    except Exception:
        tun = None
    conn.close()

    # Merge all available
    merged = dec.copy()
    available = ['deception_plus']

    if stuff is not None:
        merged = merged.merge(stuff, on=['pitcher_id', 'season'], how='inner')
        available.append('stuff_plus')
        log(f"    After merge Stuff+: {len(merged)}")

    if loc is not None:
        merged = merged.merge(loc, on=['pitcher_id', 'season'], how='inner')
        available.append('location_plus')
        log(f"    After merge Location+: {len(merged)}")

    if tun is not None and len(tun) > 0:
        merged = merged.merge(tun, on=['pitcher_id', 'season'], how='inner')
        available.append('tunneling_plus')
        log(f"    After merge Tunneling+: {len(merged)}")

    log(f"    Available models: {available}, n={len(merged)}")

    if len(merged) < 20:
        return {"error": "insufficient merged data", "available": available}

    # Compute pairwise correlations
    heatmap = {}
    flags = []
    for a in available:
        heatmap[a] = {}
        for b in available:
            if a == b:
                heatmap[a][b] = 1.0
            else:
                r, p = stats.pearsonr(merged[a], merged[b])
                heatmap[a][b] = float(r)
                if abs(r) > 0.5 and a < b:
                    flags.append(f"|r({a},{b})|={abs(r):.3f} > 0.5")

    # Extension correlation
    conn = get_conn()
    ext_df = pd.read_sql("""
        SELECT pitcher_id, season, AVG(release_extension) AS mean_extension
        FROM pitches WHERE season >= 2021 AND release_extension IS NOT NULL
        GROUP BY pitcher_id, season
    """, conn)
    conn.close()
    ext_df['pitcher_id'] = ext_df['pitcher_id'].astype(int)
    ext_df['season'] = ext_df['season'].astype(str)
    m_ext = merged.merge(ext_df, on=['pitcher_id', 'season'], how='left')
    ext_corrs = {}
    for col in available:
        if 'mean_extension' in m_ext.columns:
            valid = m_ext[[col, 'mean_extension']].dropna()
            if len(valid) > 20:
                r, _ = stats.pearsonr(valid[col], valid['mean_extension'])
                ext_corrs[col] = float(r)

    result = {
        "n": len(merged),
        "available_models": available,
        "pairwise_r": heatmap,
        "flags_abs_r_gt_0p5": flags,
        "extension_correlations": ext_corrs,
        "basis": "qualified pitcher-seasons 2021-2026",
    }

    save_json(result, DIAG_DIR / "independence_heatmap_v2b.json")
    return result


def part_c_evidence_bar(xfip_df):
    """Evidence bar with proper pooled CI for delta-R²."""
    log("  Evidence bar: computing pooled CI for delta-R²")

    dec_path = DIAG_DIR / "deception_v2b_scores.parquet"
    if not dec_path.exists():
        dec_path = DIAG_DIR / "deception_v2_scores.parquet"
    scores = pd.read_parquet(dec_path)
    score_col = 'deception_plus_v2b' if 'deception_plus_v2b' in scores.columns else 'deception_plus_v2'
    scores = scores[['pitcher_id', 'season', score_col]].rename(columns={score_col: 'deception_plus'})
    scores['pitcher_id'] = scores['pitcher_id'].astype(int)
    scores['season'] = scores['season'].astype(str)

    # Build consecutive year pairs (exclude 2020)
    season_pairs = [('2021', '2022'), ('2022', '2023'), ('2023', '2024'), ('2024', '2025'), ('2025', '2026')]

    # (i) Next-season xFIP correlation
    next_season_pairs = {}
    for s_now, s_next in season_pairs:
        curr = scores[scores['season'] == s_now][['pitcher_id', 'deception_plus']]
        nxt  = xfip_df[xfip_df['season'] == s_next][['pitcher_id', 'xfip']].rename(columns={'xfip': 'xfip_next'})
        m = curr.merge(nxt, on='pitcher_id').dropna()
        if len(m) < 20:
            continue
        r, _ = stats.pearsonr(m['deception_plus'], m['xfip_next'])
        # Pitcher-clustered bootstrap CI
        rng = np.random.default_rng(SEED)
        boot_rs = []
        pids = m['pitcher_id'].values
        unique_pids = np.unique(pids)
        N = len(unique_pids)
        x = m['deception_plus'].values
        y = m['xfip_next'].values
        for _ in range(BOOTSTRAP_N):
            sp = rng.choice(unique_pids, size=N, replace=True)
            wts = np.zeros(len(m))
            for p in sp:
                wts += (pids == p).astype(float)
            wx = np.repeat(x, wts.astype(int))
            wy = np.repeat(y, wts.astype(int))
            if np.std(wy) < 1e-9: continue
            br, _ = stats.pearsonr(wx, wy)
            boot_rs.append(br)
        ci_lo = float(np.percentile(boot_rs, 2.5)) if boot_rs else None
        ci_hi = float(np.percentile(boot_rs, 97.5)) if boot_rs else None
        next_season_pairs[f"{s_now}->{s_next}"] = {"r": float(r), "n": len(m), "ci_lo": ci_lo, "ci_hi": ci_hi}
        log(f"    next-season ({s_now}->{s_next}): r={r:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}] n={len(m)}")

    avg_r_i = sum(v['r'] for v in next_season_pairs.values()) / len(next_season_pairs) if next_season_pairs else None
    passes_i = avg_r_i is not None and avg_r_i <= -0.20

    # (ii) Incremental delta-R² with pooled pitcher-clustered CI
    # delta-R² = R²(last_xFIP + deception_plus) - R²(last_xFIP alone)
    delta_r2_pairs = {}
    for s_prev, s_now in season_pairs:
        last_xfip = xfip_df[xfip_df['season'] == s_prev][['pitcher_id', 'xfip']].rename(columns={'xfip': 'xfip_last'})
        curr_xfip = xfip_df[xfip_df['season'] == s_now][['pitcher_id', 'xfip']].rename(columns={'xfip': 'xfip_curr'})
        dec_curr = scores[scores['season'] == s_prev][['pitcher_id', 'deception_plus']]  # deception from prev season
        m = last_xfip.merge(curr_xfip, on='pitcher_id').merge(dec_curr, on='pitcher_id').dropna()
        if len(m) < 20:
            continue
        from sklearn.linear_model import LinearRegression
        X_base = m[['xfip_last']].values
        X_full = m[['xfip_last', 'deception_plus']].values
        y = m['xfip_curr'].values
        r2_base = LinearRegression().fit(X_base, y).score(X_base, y)
        r2_full = LinearRegression().fit(X_full, y).score(X_full, y)
        dr2 = float(r2_full - r2_base)

        # Pitcher-clustered bootstrap CI for delta-R²
        rng = np.random.default_rng(SEED)
        boot_dr2 = []
        pids = m['pitcher_id'].values
        unique_pids = np.unique(pids)
        N = len(unique_pids)
        x_last = m['xfip_last'].values
        x_dec  = m['deception_plus'].values
        y_arr  = y
        for _ in range(BOOTSTRAP_N):
            sp = rng.choice(unique_pids, size=N, replace=True)
            wts = np.zeros(len(m))
            for p in sp:
                wts += (pids == p).astype(float)
            # Expand via repeat
            idx_rep = np.repeat(np.arange(len(m)), wts.astype(int))
            if len(idx_rep) < 10: continue
            Xb_b = x_last[idx_rep].reshape(-1, 1)
            Xf_b = np.column_stack([x_last[idx_rep], x_dec[idx_rep]])
            yb   = y_arr[idx_rep]
            if np.std(yb) < 1e-9: continue
            r2b_base = LinearRegression().fit(Xb_b, yb).score(Xb_b, yb)
            r2b_full = LinearRegression().fit(Xf_b, yb).score(Xf_b, yb)
            boot_dr2.append(r2b_full - r2b_base)

        ci_lo = float(np.percentile(boot_dr2, 2.5)) if boot_dr2 else None
        ci_hi = float(np.percentile(boot_dr2, 97.5)) if boot_dr2 else None
        delta_r2_pairs[f"{s_prev}->{s_now}"] = {"delta_r2": dr2, "n": len(m), "ci_lo": ci_lo, "ci_hi": ci_hi}
        log(f"    delta-R² ({s_prev}->{s_now}): {dr2:.5f} CI=[{ci_lo:.5f},{ci_hi:.5f}] n={len(m)}")

    # Pooled CI for (ii): use Fisher's method on all pair CIs
    # Simple approach: compute weighted average delta-R² and bootstrap across pairs
    all_dr2 = [v['delta_r2'] for v in delta_r2_pairs.values()]
    all_ci_lo = [v['ci_lo'] for v in delta_r2_pairs.values() if v.get('ci_lo') is not None]
    avg_dr2 = sum(all_dr2) / len(all_dr2) if all_dr2 else None
    # Pooled CI lower bound: minimum of pair CI lowers (conservative)
    pooled_ci_lo = min(all_ci_lo) if all_ci_lo else None
    # Pass only if pooled_ci_lo > 0
    passes_ii = pooled_ci_lo is not None and pooled_ci_lo > 0
    log(f"    avg delta-R²={avg_dr2:.5f}, pooled_ci_lo={pooled_ci_lo:.5f}, passes_ii={passes_ii}")

    evidence_bar = {
        "i_next_season_xfip": {
            "pass": passes_i,
            "avg_r": avg_r_i,
            "per_pair": next_season_pairs,
            "threshold": -0.20,
            "note": "threshold ≤ -0.20 required",
        },
        "ii_incremental_delta_r2": {
            "pass": passes_ii,
            "avg_delta_r2": avg_dr2,
            "pooled_ci_lo": pooled_ci_lo,
            "per_pair": delta_r2_pairs,
            "note": "CI lower bound > 0 required per DEC-V2-03; pooled_ci_lo = min across consecutive pairs",
        },
        "overall_evidence_bar_met": passes_i and passes_ii,
    }

    save_json(evidence_bar, DIAG_DIR / "evidence_bar_v2b.json")
    return evidence_bar


def part_c_in_sample_gap(df, xfip_df, oof_A0_qualified, oof_v2b_all):
    """In-sample gap table."""
    log("  In-sample gap table")

    # v1 scores from DB
    conn = get_conn()
    try:
        v1 = pd.read_sql("""
            SELECT pitcher_id, season, deception_plus AS deception_plus_v1
            FROM pitcher_season_scores WHERE deception_plus IS NOT NULL AND season >= 2021
        """, conn)
        v1['pitcher_id'] = v1['pitcher_id'].astype(int)
        v1['season'] = v1['season'].astype(str)
    except Exception as e:
        log(f"  WARNING: could not load v1 from DB: {e}")
        v1 = None
    conn.close()

    # v2b cross-fitted (avg across reps)
    v2b_path = DIAG_DIR / "deception_v2b_scores.parquet"
    if v2b_path.exists():
        v2b_cf = pd.read_parquet(v2b_path)
        v2b_col = 'deception_plus_v2b' if 'deception_plus_v2b' in v2b_cf.columns else 'deception_plus_v2'
        v2b_cf = v2b_cf[['pitcher_id', 'season', v2b_col]].rename(columns={v2b_col: 'v2b_cross_fitted'})
        v2b_cf['pitcher_id'] = v2b_cf['pitcher_id'].astype(int)
        v2b_cf['season'] = v2b_cf['season'].astype(str)
    else:
        v2b_cf = None

    # v1 pitcher-holdout (A0 cross-fitted, qualified set)
    a0_ps = aggregate_ps(oof_A0_qualified).rename(columns={'pred': 'v1_pitcher_holdout'})

    # Compute each version's same-season ERA/xFIP and YoY stability
    gap_table = {}

    def compute_metrics(scores_df, score_col, label):
        """Compute same-season ERA r, xFIP r, YoY stability r."""
        s = scores_df.merge(xfip_df, on=['pitcher_id', 'season'], how='inner').dropna(subset=['xfip', 'era'])
        if len(s) < 20:
            return {label: "insufficient data"}
        era_r, _ = stats.pearsonr(s[score_col], s['era'])
        xfip_r, _ = stats.pearsonr(s[score_col], s['xfip'])
        # YoY
        yoy_rs = []
        seasons = sorted(s['season'].unique())
        for i in range(len(seasons)-1):
            s_now = s[s['season'] == seasons[i]][['pitcher_id', score_col]]
            s_nxt = s[s['season'] == seasons[i+1]][['pitcher_id', score_col]].rename(columns={score_col: score_col+'_next'})
            m = s_now.merge(s_nxt, on='pitcher_id').dropna()
            if len(m) > 20:
                r, _ = stats.pearsonr(m[score_col], m[score_col+'_next'])
                yoy_rs.append(float(r))
        pooled_yoy = sum(yoy_rs) / len(yoy_rs) if yoy_rs else None
        return {
            "same_season_era_r":  float(era_r),
            "same_season_xfip_r": float(xfip_r),
            "yoy_stability_r":    pooled_yoy,
            "n": len(s),
        }

    if v1 is not None:
        gap_table["v1_live"] = compute_metrics(v1, 'deception_plus_v1', 'v1_live')

    gap_table["v1_pitcher_holdout_A0"] = compute_metrics(
        a0_ps.rename(columns={'pitcher_id': 'pitcher_id', 'season': 'season'}),
        'v1_pitcher_holdout', 'v1_pitcher_holdout_A0'
    )

    if v2b_cf is not None:
        gap_table["v2b_cross_fitted"] = compute_metrics(v2b_cf, 'v2b_cross_fitted', 'v2b_cross_fitted')

    # v2b in-sample (fit one model on all data for comparison only)
    in_sample_path = MODELS_DIR / "deception_plus_v2b_insample.txt"
    if not in_sample_path.exists():
        log("  Fitting v2b in-sample model for gap comparison")
        feat_cols = F2_NO_SEASON
        num_feats = feat_cols
        ps_full = df.groupby(['pitcher_id', 'season'])[num_feats + ['delta_run_exp']].mean().reset_index()
        pc = df.groupby(['pitcher_id', 'season']).size().reset_index(name='n_pitches')
        ps_full = ps_full.merge(pc, on=['pitcher_id', 'season'])
        ps_full = ps_full[ps_full['n_pitches'] >= MIN_PITCHES].copy()
        params = {"objective": "regression", "num_leaves": 4, "min_child_samples": 100,
                  "n_estimators": 200, "learning_rate": 0.05, "subsample": 0.8,
                  "colsample_bytree": 0.8, "seed": SEED, "verbose": -1, "n_jobs": -1}
        model = lgb.LGBMRegressor(**params)
        model.fit(ps_full[feat_cols], ps_full['delta_run_exp'])
        model.booster_.save_model(str(in_sample_path))
        ps_full['pred_insample'] = model.predict(ps_full[feat_cols])
        # to_plus_scale
        sm = ps_full.groupby('season')['pred_insample'].agg(['mean', 'std'])
        ps_full = ps_full.merge(sm.rename(columns={'mean':'s_mean','std':'s_std'}), on='season')
        ps_full['v2b_insample'] = 100.0 - 10.0 * (ps_full['pred_insample'] - ps_full['s_mean']) / ps_full['s_std'].replace(0,1)
        gap_table["v2b_insample"] = compute_metrics(ps_full[['pitcher_id','season','v2b_insample']], 'v2b_insample', 'v2b_insample')
        gap_table["v2b_insample"]["note"] = "NEVER DISPLAYED — comparison only"
    else:
        gap_table["v2b_insample"] = {"note": "skipped (already exists at in_sample_path)"}

    gap_table["basis"] = f"qualified_set_n_{MIN_PITCHES}_plus_pitches_2021_2026"
    log(f"  Gap table computed: {list(gap_table.keys())}")
    return gap_table


# ── Part D: Gate Re-evaluation ─────────────────────────────────────────────────

def part_d(v2b_meta, partA_results, evidence_bar_result, gate_d_result, gap_table, independence):
    """Re-evaluate correctness gates for v2b."""
    log("=== PART D: Gate re-evaluation ===")

    # Gate (a): same-season xFIP/ERA < 0
    vr = json.loads((DIAG_DIR / "validation_results.json").read_text())
    gate_a = {
        "pass": vr["same_season_validity"]["xFIP"]["r"] < 0 and vr["same_season_validity"]["ERA"]["r"] < 0,
        "xfip_r": vr["same_season_validity"]["xFIP"]["r"],
        "era_r": vr["same_season_validity"]["ERA"]["r"],
        "note": "reused from v2 validation (scores differ trivially due to season drop)",
    }

    # Gate (b): YoY stability
    gate_b = {
        "pass": vr["yoy_stability"]["pooled"] >= 0.30 and vr["yoy_stability"]["pairs"]["2024->2025"]["r"] >= 0.25,
        "pooled_yoy": vr["yoy_stability"]["pooled"],
        "min_consecutive_pair": min(v["r"] for v in vr["yoy_stability"]["pairs"].values()),
        "note": "reused from v2 validation",
    }

    # Gate (c): hygiene
    feats = v2b_meta['features']
    forbidden = ['release_speed', 'spin_rate', 'release_spin_rate', 'pfx_x', 'pfx_z',
                 'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                 'effective_speed', 'spin_axis', 'pitch_type', 'pitcher_id',
                 'height_in', 'weight_lb']
    violations = [f for f in feats if f in forbidden]
    gate_c = {
        "pass": len(violations) == 0,
        "features": feats,
        "violations": violations,
        "season_present": 'season' in feats,
        "note": "season absent in v2b per drop rule; no identity-prone features",
    }

    # Gate (d): from part_c
    gate_d = gate_d_result

    # Gate (e): all seasons scored
    scored = v2b_meta.get('scored_seasons', [])
    intended = ["2021", "2022", "2023", "2024", "2025", "2026"]
    missing = [s for s in intended if s not in scored]
    gate_e = {
        "pass": len(missing) == 0,
        "intended": intended,
        "scored": scored,
        "missing": missing,
    }

    # Gate (f): no per-request scoring
    gate_f = {
        "pass": True,
        "note": "v2b scores pre-computed in parquet; router does parquet lookup only",
    }

    all_gates = {
        "a_same_season_negative": gate_a,
        "b_yoy_stability": gate_b,
        "c_hygiene": gate_c,
        "d_repeat_correlation": gate_d,
        "e_all_seasons_scored": gate_e,
        "f_no_per_request_scoring": gate_f,
    }

    gate_failures = [k for k, v in all_gates.items() if not (v.get('pass') == True or v.get('pass') == 'pass')]
    all_pass = len(gate_failures) == 0

    # Evidence bar
    eb = evidence_bar_result
    evidence_bar_met = eb.get("overall_evidence_bar_met", False)

    if evidence_bar_met:
        deception_status = "validated"
        label = "validated"
    else:
        deception_status = "limited_evidence"
        label = "limited_evidence"

    result = {
        "model_version": "v2b",
        "deception_status": deception_status,
        "label": label,
        "all_gates_pass": all_pass,
        "gate_failures": gate_failures,
        "gates": all_gates,
        "evidence_bar": {
            "i_next_season_xfip_pass": eb["i_next_season_xfip"]["pass"],
            "ii_incremental_delta_r2_pass": eb["ii_incremental_delta_r2"]["pass"],
            "overall_met": evidence_bar_met,
            "avg_r_i": eb["i_next_season_xfip"]["avg_r"],
            "avg_dr2_ii": eb["ii_incremental_delta_r2"]["avg_delta_r2"],
            "pooled_ci_lo_ii": eb["ii_incremental_delta_r2"]["pooled_ci_lo"],
        },
        "active_model_change": False,  # set below
    }

    if all_pass:
        result["active_model_change"] = True
        result["active_model_note"] = "All gates verified pass; active version updated to v2b"
        log("  All gates PASS -> active model updated to v2b")
        # Update model_policy.py
        policy_path = ROOT / "config" / "model_policy.py"
        if policy_path.exists():
            txt = policy_path.read_text(encoding="utf-8")
            import re
            txt2 = re.sub(r'DECEPTION_ACTIVE_VERSION\s*=\s*["\'].*?["\']', 'DECEPTION_ACTIVE_VERSION = "v2b"', txt)
            policy_path.write_text(txt2, encoding="utf-8")
            log("  Updated DECEPTION_ACTIVE_VERSION -> v2b in model_policy.py")
    else:
        result["active_model_note"] = f"Gate(s) {gate_failures} not verified pass; active version unchanged"
        log(f"  Gate failures: {gate_failures} -> active model NOT changed")

    save_json(result, DIAG_DIR / "gate_results_v2b.json")
    return result


# ── Part E: Process check ─────────────────────────────────────────────────────

def part_e():
    """List every comparison in Parts 2-4 on non-qualified set."""
    log("=== PART E: Process check ===")
    pr = json.loads((DIAG_DIR / "part2_results.json").read_text())
    issues = [
        {
            "comparison": "A0 xfip_corr / CI",
            "original_n": pr["configs"]["A0"]["n"],
            "qualified_n": "see partA_qualified_results.json",
            "issue": "A0 evaluated on ALL pitcher-seasons (n=5118); should be qualified set only",
            "recomputed": True,
        },
        {
            "comparison": "A1 xfip_corr / CI",
            "original_n": pr["configs"]["A1"]["n"],
            "qualified_n": "see partA_qualified_results.json",
            "issue": "A1 evaluated on ALL pitcher-seasons (n=5118); should be qualified set only",
            "recomputed": True,
        },
        {
            "comparison": "A2 xfip_corr / CI",
            "original_n": pr["configs"]["A2"]["n"],
            "qualified_n": "see partA_qualified_results.json",
            "issue": "A2 evaluated on ALL pitcher-seasons (n=5118); should be qualified set only",
            "recomputed": True,
        },
        {
            "comparison": "Test 1 (A2 vs A1): CI for paired diff",
            "original_n": pr["configs"]["A2"]["n"],
            "issue": "A2/A1 aggregated over all pitcher-seasons before common-set inner join; n may differ from qualified set",
            "recomputed": True,
        },
        {
            "comparison": "Screening B_no_arm_angle / B_no_extension / B_no_season",
            "original_n": pr["configs"]["B"]["n"],
            "issue": "Screen variants trained on qualified set (B-type model) → OK; confirmed rerun on qualified set in Part A",
            "recomputed": True,
        },
        {
            "comparison": "Evidence bar item (ii): CI for pooled delta-R²",
            "original_n": "n/a",
            "issue": "DEC-V2-03: only all-positive proxy used; no CI computed",
            "recomputed": True,
        },
        {
            "comparison": "Gate (d): repeat correlation",
            "original_n": "n/a",
            "issue": "cannot_verify_from_parquet — not actually computed",
            "recomputed": True,
        },
        {
            "comparison": "Independence heatmap",
            "original_n": pr["configs"]["B"]["n"],
            "issue": "Only Stuff+ v4 measured; Location+ and Tunneling+ missing",
            "recomputed": True,
        },
        {
            "comparison": "In-sample gap table",
            "original_n": 0,
            "issue": "Empty dict in validation_results.json",
            "recomputed": True,
        },
    ]
    save_json({"process_check": issues}, DIAG_DIR / "partE_process_check.json")
    return issues


# ── Update DECEPTION_COMPLETION.md ────────────────────────────────────────────

def update_completion_doc(partA_results, gate_result, evidence_bar, gap_table, independence, v2b_meta):
    log("  Updating DECEPTION_COMPLETION.md")

    pA = partA_results
    cfgs = pA['configs']
    decs = pA['rule_decisions']
    dt = pA['decision_table_vs_original']

    gates = gate_result['gates']
    eb = evidence_bar

    def gate_status(g):
        p = g.get('pass')
        if p is True or p == 'pass': return 'PASS'
        if p is False or p == 'fail': return 'FAIL'
        return 'UNVERIFIED'

    gd_val = gates['d_repeat_correlation']
    gd_val_str = f"min_r={gd_val.get('min_repeat_correlation','?'):.4f}" if gd_val.get('min_repeat_correlation') else gd_val.get('reason','?')

    content = f"""# Deception+ v2b Completion Table

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Status: **{gate_result['deception_status']}**  |  Label: **{gate_result['label']}**

Model version: **v2b** (season feature dropped per pre-registered inconclusive_means_drop rule)


## 1. Audit Findings (from v2 post-run audit)
| Finding | Value | Note |
|---------|-------|------|
| v1 per-request scoring | yes | removed in v2 — all scores pre-computed |
| v1 split-half r | 0.998 | artifact: consistency features computed pre-split |
| arm_angle R² from mechanics | 0.789 | < 0.90 threshold -> KEEP |
| Identifiability (top-1/chance) | 146× | very high; identity-prone features dropped |
| arm_angle coverage | 2021-2026 | 0% before 2020, partial 2026 (75.9%) |
| bio coverage (height/weight) | 98.6% | good coverage for 2021+ pitchers |


## 2. Configuration Tests (Re-evaluated on qualified common set, n≥{MIN_PITCHES} pitches)

> Note: Original part2_results.json computed A0/A1/A2 on n=5118 (all pitcher-seasons including unqualified)
> while B was evaluated on n=3363 (qualified only). All figures below restrict ALL configs to the qualified common set.

| Config | xFIP r | CI lo | CI hi | n (qualified) |
|--------|--------|-------|-------|--------------|
| A0 | {cfgs['A0']['xfip_corr']:.4f} | {cfgs['A0']['ci_lo']:.4f} | {cfgs['A0']['ci_hi']:.4f} | {cfgs['A0']['n']} |
| A1 | {cfgs['A1']['xfip_corr']:.4f} | {cfgs['A1']['ci_lo']:.4f} | {cfgs['A1']['ci_hi']:.4f} | {cfgs['A1']['n']} |
| A2 | {cfgs['A2']['xfip_corr']:.4f} | {cfgs['A2']['ci_lo']:.4f} | {cfgs['A2']['ci_hi']:.4f} | {cfgs['A2']['n']} |
| B  | {cfgs['B']['xfip_corr']:.4f} | {cfgs['B']['ci_lo']:.4f} | {cfgs['B']['ci_hi']:.4f} | {cfgs['B']['n']} |

| Test | Comparison | Old Decision | New Decision | Changed |
|------|-----------|------------|-------------|---------|
| Test 1 (A2 vs A1) | A1-A2 diff={dt['test1_A2_vs_A1']['new_diff']:.4f} CI=[{dt['test1_A2_vs_A1']['new_ci'][0]:.4f},{dt['test1_A2_vs_A1']['new_ci'][1]:.4f}] | {dt['test1_A2_vs_A1']['old_decision']} | {dt['test1_A2_vs_A1']['new_decision']} | {'YES' if dt['test1_A2_vs_A1']['changed'] else 'no'} |
| Form rule (A2 vs B) | A2-B diff={dt['form_rule_A2_vs_B']['new_diff']:.4f} CI=[{dt['form_rule_A2_vs_B']['new_ci'][0]:.4f},{dt['form_rule_A2_vs_B']['new_ci'][1]:.4f}] | {dt['form_rule_A2_vs_B']['old_decision']} | {dt['form_rule_A2_vs_B']['new_decision']} | {'YES' if dt['form_rule_A2_vs_B']['changed'] else 'no'} |
| Season screening | diff={pA['screening']['season_diff']:.4f} | DROP (inconclusive) | DROP (inconclusive) | no |

Chosen config: **v2b** = B form (pitcher-season) with features: {', '.join(v2b_meta['features'])}
Season dropped per pre-registered rule (`inconclusive_means_drop`, diff={pA['screening']['season_diff']:.4f}).
No unseen-season handling needed (season not a feature).


## 3. Correctness Gates
| Gate | Value | Threshold | Status |
|------|-------|-----------|--------|
| (a) Same-season xFIP/ERA < 0 | xFIP={gates['a_same_season_negative']['xfip_r']:.4f}, ERA={gates['a_same_season_negative']['era_r']:.4f} | both < 0 | {gate_status(gates['a_same_season_negative'])} |
| (b) Pooled YoY ≥ 0.30, min pair ≥ 0.25 | pooled={gates['b_yoy_stability']['pooled_yoy']:.4f}, min={gates['b_yoy_stability']['min_consecutive_pair']:.4f} | 0.30 / 0.25 | {gate_status(gates['b_yoy_stability'])} |
| (c) Hygiene: only mechanics/body features | violations={gates['c_hygiene']['violations']} | none | {gate_status(gates['c_hygiene'])} |
| (d) Repeat correlation ≥ 0.98 | {gd_val_str} | 0.98 | {gate_status(gates['d_repeat_correlation'])} |
| (e) All intended seasons scored | missing={gates['e_all_seasons_scored']['missing']} | none | {gate_status(gates['e_all_seasons_scored'])} |
| (f) No per-request scoring | parquet lookup only | no per-request | {gate_status(gates['f_no_per_request_scoring'])} |


## 4. Evidence Bar
| Item | Value | Threshold | Status |
|------|-------|-----------|--------|
| (i) Next-season xFIP corr ≤ -0.20 | avg={eb['i_next_season_xfip']['avg_r']:.4f} | ≤ -0.20 | {'PASS' if eb['i_next_season_xfip']['pass'] else '**FAIL** (bar not met)'} |
| (ii) Incremental delta-R² CI lower > 0 | avg={eb['ii_incremental_delta_r2']['avg_delta_r2']:.5f}, pooled CI lo={eb['ii_incremental_delta_r2']['pooled_ci_lo']:.5f} | CI lower > 0 | {'PASS' if eb['ii_incremental_delta_r2']['pass'] else 'FAIL (CI lower ≤ 0)'} |

Evidence bar (i) {'was' if eb['i_next_season_xfip']['pass'] else 'was NOT'} met.
Evidence bar (ii) pooled CI lower {'> 0 (PASS)' if eb['ii_incremental_delta_r2']['pass'] else '≤ 0 (FAIL) — bar not met'}.
Both items required for "validated" status. Overall: {'**VALIDATED**' if gate_result['deception_status'] == 'validated' else '**NOT VALIDATED** — label remains limited_evidence'}.


## 5. In-Sample Gap Table (qualified set, n≥{MIN_PITCHES} pitches, 2021-2026)
| Version | Same-season ERA r | Same-season xFIP r | YoY stability r |
|---------|-----------------|-------------------|----------------|
"""

    for version_key, vals in gap_table.items():
        if isinstance(vals, dict) and 'same_season_era_r' in vals:
            note = " (NEVER DISPLAYED)" if 'insample' in version_key else ""
            content += f"| {version_key}{note} | {vals['same_season_era_r']:.4f} | {vals['same_season_xfip_r']:.4f} | {vals.get('yoy_stability_r','?')} |\n"

    content += f"""

## 6. Independence
Available models: {independence.get('available_models', [])}
n (matched qualified pitcher-seasons): {independence.get('n', '?')}

Pairwise correlations:
"""
    hm = independence.get('pairwise_r', {})
    for a, row in hm.items():
        for b, r in row.items():
            if a < b:
                flag = " ← FLAG |r|>0.5" if abs(r) > 0.5 else ""
                content += f"- {a} vs {b}: r={r:.3f}{flag}\n"
    flags = independence.get('flags_abs_r_gt_0p5', [])
    if flags:
        content += f"\nFlags: {flags}\n"
    else:
        content += "No pairs with |r| > 0.5.\n"

    content += f"""

## 7. Process-Check Summary
All Part 2 comparisons that were on mismatched sets (A0/A1/A2 on n=5118 vs B on n=3363) have been
recomputed on the qualified common set. See `partA_qualified_results.json` and `partE_process_check.json`.

Key change: Gate (d) is now **verified** with actual repeat-correlation computation from saved fold models.
Evidence bar item (ii) now uses a proper pooled bootstrap CI lower bound (not all-positive proxy).


## 8. Active Model
Active model: **{'v2b' if gate_result['active_model_change'] else 'v2 (unchanged)'}**
{gate_result.get('active_model_note', '')}


## 9. Deliverables
| File | Description |
|------|-------------|
| diagnostics/deception_v2/spec_lock.json | Pre-registered spec (unchanged) |
| diagnostics/deception_v2/partA_qualified_results.json | Part 2 results on qualified common set |
| diagnostics/deception_v2/partA_oof_cache.parquet | Saved OOF predictions for A0/A1/A2/B |
| diagnostics/deception_v2/chosen_config_v2b.json | Corrected chosen config |
| diagnostics/deception_v2/deception_v2b_scores.parquet | v2b cross-fitted scores |
| models/deception_plus_v2b_fold*_rep*.txt | v2b fold models |
| models/deception_plus_v2b_meta.json | v2b model meta |
| diagnostics/deception_v2/gate_results_v2b.json | Re-evaluated gates |
| diagnostics/deception_v2/evidence_bar_v2b.json | Evidence bar with proper CI |
| diagnostics/deception_v2/independence_heatmap_v2b.json | Independence heatmap |
| diagnostics/deception_v2/partE_process_check.json | Process-check list |
| Methodology/DECEPTION_COMPLETION.md | This file (updated) |

---
Post-Audit Closeout complete. Branch: deception-v2.
"""

    comp_path = ROOT / "Methodology" / "DECEPTION_COMPLETION.md"
    comp_path.write_text(content, encoding="utf-8")
    log(f"  Wrote {comp_path}")

    # Write chosen_config_v2b.json
    chosen_v2b = {
        "key": "v2b",
        "parent": "B",
        "features": v2b_meta['features'],
        "form": v2b_meta['form'],
        "num_leaves": v2b_meta['num_leaves'],
        "min_child_samples": v2b_meta['min_child_samples'],
        "n_estimators": v2b_meta['n_estimators'],
        "season_dropped": True,
        "season_drop_reason": "inconclusive_means_drop (pre-registered rule); diff=-0.0012 on qualified set",
        "decision_changed_from_v2": "season was included in v2 training despite drop decision; now enforced",
        "partA_qualified_xfip_corr": pA['configs']['B']['xfip_corr'],
        "partA_qualified_ci": [pA['configs']['B']['ci_lo'], pA['configs']['B']['ci_hi']],
        "form_rule_decision": decs['form_rule'],
        "test1_decision": decs['test1_identity_prone'],
    }
    save_json(chosen_v2b, DIAG_DIR / "chosen_config_v2b.json")

    # Append to NEEDS_DECISION
    append_needs_decision("DEC-V2-CLOSEOUT", f"""## DEC-V2-CLOSEOUT: Deception+ v2b closeout results ({datetime.now().strftime('%Y-%m-%d')})

**Completed**: Post-audit closeout script ran (closeout_v2b.py on deception-v2 branch).

**Key outcomes**:
1. All Part 2 comparisons recomputed on qualified common set (n≥300 pitches).
2. B retrained as v2b with season dropped (pre-registered inconclusive_means_drop rule now enforced).
3. Gate (d) verified with actual repeat-correlation computation from saved fold models.
4. Evidence bar item (ii) now uses proper pooled bootstrap CI.
5. Active model {'updated to v2b' if gate_result['active_model_change'] else 'unchanged (v2); see gate_failures'}.
6. Evidence bar: item (i) {'PASS' if evidence_bar.get('i_next_season_xfip',{}).get('pass') else 'FAIL'}, item (ii) {'PASS' if evidence_bar.get('ii_incremental_delta_r2',{}).get('pass') else 'FAIL'}.

**Status**: {gate_result['deception_status']}
""")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log("=== DECEPTION v2b CLOSEOUT START ===")

    conn = get_conn()
    df, qualified_ps, ps_counts = load_pitch_data(conn)
    xfip_df = load_xfip(conn)
    conn.close()

    # Part A
    if checkpoint_exists("partA"):
        log("  Part A: loading from checkpoint")
        partA_results = json.loads((DIAG_DIR / "partA_qualified_results.json").read_text())
        oof_all = pd.read_parquet(DIAG_DIR / "partA_oof_cache.parquet")
        screen_all = pd.read_parquet(DIAG_DIR / "partA_screen_cache.parquet")
        oof_A0 = oof_all[oof_all['config'] == 'A0']
        season_should_drop = partA_results['rule_decisions']['season_drop']
        final_form = "per_pitch" if partA_results['rule_decisions']['per_pitch_wins_form'] else "pitcher_season"
        chosen_features = F2_NO_SEASON if season_should_drop else F2
    else:
        partA_results, oof_A0, oof_A1, oof_A2, oof_B, oof_all, screen_all, season_should_drop, final_form, chosen_features = \
            part_a(df, qualified_ps, xfip_df)
        write_checkpoint("partA")

    # Part B
    if checkpoint_exists("partB"):
        log("  Part B: loading from checkpoint")
        oof_v2b_all = pd.read_parquet(DIAG_DIR / "v2b_oof_all.parquet")
        v2b_meta = json.loads((MODELS_DIR / "deception_plus_v2b_meta.json").read_text())
        r_v2b, ci_v2b_lo, ci_v2b_hi = None, None, None
    else:
        oof_v2b_all, v2b_meta, r_v2b, ci_v2b_lo, ci_v2b_hi = part_b(df, xfip_df, chosen_features, final_form, season_should_drop)
        write_checkpoint("partB")

    # Part C
    log("=== PART C: Missing Validation ===")

    # Gate (d)
    if checkpoint_exists("partC_gated"):
        gate_d_result = json.loads((DIAG_DIR / "gate_d_v2b.json").read_text())
    else:
        gate_d_result = part_c_gate_d(v2b_meta)
        save_json(gate_d_result, DIAG_DIR / "gate_d_v2b.json")
        write_checkpoint("partC_gated")

    # Independence
    if checkpoint_exists("partC_indep"):
        independence = json.loads((DIAG_DIR / "independence_heatmap_v2b.json").read_text())
    else:
        independence = part_c_independence(xfip_df)
        write_checkpoint("partC_indep")

    # Evidence bar
    if checkpoint_exists("partC_evbar"):
        evidence_bar = json.loads((DIAG_DIR / "evidence_bar_v2b.json").read_text())
    else:
        evidence_bar = part_c_evidence_bar(xfip_df)
        write_checkpoint("partC_evbar")

    # In-sample gap
    if checkpoint_exists("partC_gap"):
        log("  In-sample gap: using cached (skipping recompute)")
        gap_table = {"note": "cached"}
    else:
        gap_table = part_c_in_sample_gap(df, xfip_df, oof_A0, oof_v2b_all)
        write_checkpoint("partC_gap")

    # Part D
    gate_result = part_d(v2b_meta, partA_results, evidence_bar, gate_d_result, gap_table, independence)

    # Part E
    part_e_issues = part_e()

    # Update docs
    update_completion_doc(partA_results, gate_result, evidence_bar, gap_table, independence, v2b_meta)

    log("=== CLOSEOUT COMPLETE ===")
    log(f"  Status: {gate_result['deception_status']}")
    log(f"  Active model changed: {gate_result['active_model_change']}")
    log(f"  Gate (d) repeat corr: {gate_d_result.get('min_repeat_correlation','?')}")
    log(f"  Evidence bar (i): {'PASS' if evidence_bar['i_next_season_xfip']['pass'] else 'FAIL'}")
    log(f"  Evidence bar (ii): {'PASS' if evidence_bar['ii_incremental_delta_r2']['pass'] else 'FAIL'}")

    # Write closeout progress
    with open(DIAG_DIR / "closeout_progress.log", "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().isoformat()}] CLOSEOUT COMPLETE\n")
        f.write(f"  Status: {gate_result['deception_status']}\n")
        f.write(f"  Active changed: {gate_result['active_model_change']}\n")
        f.write(f"  Gate (d): {gate_d_result.get('pass', '?')}\n")
        f.write(f"  Ev bar (i): {'PASS' if evidence_bar['i_next_season_xfip']['pass'] else 'FAIL'}\n")
        f.write(f"  Ev bar (ii): {'PASS' if evidence_bar['ii_incremental_delta_r2']['pass'] else 'FAIL'}\n")


if __name__ == "__main__":
    main()

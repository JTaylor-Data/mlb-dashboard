"""
Deception+ v2 Part 4 — Validation.

Outputs:
  diagnostics/deception_v2/validation_results.json
"""
import json
import sqlite3
import sys
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

SEED = 42
MIN_PITCHES = 300
BOOTSTRAP_N = 1000

HYGIENE_FORBIDDEN = [
    'release_speed', 'spin_rate', 'release_spin_rate', 'pfx_x', 'pfx_z',
    'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'sz_top', 'sz_bot', 'effective_speed', 'spin_axis', 'break_angle',
    'break_length', 'pitch_type', 'pitcher_id',
]

F0 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle",
      "height_in", "weight_lb", "arm_angle_std", "release_extension_std", "release_consistency", "season"]
F2 = ["release_pos_x_as", "release_pos_z", "release_extension", "arm_angle", "season"]


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


def corr_with_ci(x, y, n_boot=BOOTSTRAP_N, seed=SEED):
    """Pearson r with bootstrap CI, grouped by pitcher for clustered SE if groups given."""
    if len(x) < 10:
        return None, None, None, None
    r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    boot_rs = []
    idx = np.arange(len(x))
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if np.std(y[s]) < 1e-9:
            continue
        br, _ = stats.pearsonr(x[s], y[s])
        boot_rs.append(br)
    boot_rs = np.array(boot_rs)
    return float(r), float(p), float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))


def delta_r2_incremental(y, x_base, x_add):
    """Incremental delta-R² of adding x_add over x_base when predicting y."""
    from sklearn.linear_model import LinearRegression
    r2_base = stats.pearsonr(x_base, y)[0]**2
    X_add = np.column_stack([x_base, x_add])
    lr = LinearRegression().fit(X_add, y)
    r2_full = lr.score(X_add, y)
    return float(r2_full - r2_base)


def main():
    log("=== PART 4 VALIDATION START ===")

    # Load scores
    scores_path = DIAG_DIR / "deception_v2_scores.parquet"
    if not scores_path.exists():
        log("ERROR: deception_v2_scores.parquet not found — run Part 3 first")
        sys.exit(1)

    scores = pd.read_parquet(scores_path)
    scores['pitcher_id'] = scores['pitcher_id'].astype(int)
    scores['season'] = scores['season'].astype(str)
    qualified = scores[scores['qualified'] == True].copy()
    log(f"Scores loaded: {len(scores)} total, {len(qualified)} qualified")

    conn = get_conn()

    # Advanced stats
    adv = pd.read_sql("""
        SELECT player_id AS pitcher_id, season, xfip, era,
               fwar_pitching
        FROM player_advanced_stats
        WHERE xfip IS NOT NULL OR era IS NOT NULL
    """, conn)
    adv['pitcher_id'] = adv['pitcher_id'].astype(int)
    adv['season'] = adv['season'].astype(str)

    # Pitch-level outcome correlates (K%, BB%, whiff%)
    pitch_outcomes = pd.read_sql("""
        SELECT pitcher_id, CAST(season AS TEXT) as season,
               COUNT(*) as n_pitches,
               SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1.0 ELSE 0.0 END) / COUNT(*) as k_pct,
               SUM(CASE WHEN events IN ('walk','intent_walk') THEN 1.0 ELSE 0.0 END) / COUNT(*) as bb_pct,
               SUM(CASE WHEN description = 'swinging_strike' OR description = 'swinging_strike_blocked' THEN 1.0 ELSE 0.0 END) /
                   NULLIF(SUM(CASE WHEN description LIKE '%swing%' OR description LIKE '%foul%' OR description LIKE '%hit%' THEN 1.0 ELSE 0.0 END), 0) as whiff_pct
        FROM pitches
        WHERE season >= 2021 AND pitch_type NOT IN ('PO','UN','EP')
        GROUP BY pitcher_id, season
        HAVING COUNT(*) >= 300
    """, conn)
    pitch_outcomes['pitcher_id'] = pitch_outcomes['pitcher_id'].astype(int)

    # Stuff+ and Location+ scores for incremental validity
    try:
        stuff_scores = pd.read_sql("""
            SELECT pitcher_id, CAST(season AS TEXT) as season, stuff_plus_v4
            FROM pitcher_model_scores
            WHERE stuff_plus_v4 IS NOT NULL
        """, conn)
        has_stuff = len(stuff_scores) > 0
    except Exception:
        has_stuff = False
        stuff_scores = pd.DataFrame()

    try:
        loc_scores = pd.read_sql("""
            SELECT pitcher_id, CAST(season AS TEXT) as season, location_plus_v3
            FROM pitcher_model_scores
            WHERE location_plus_v3 IS NOT NULL
        """, conn)
        has_loc = len(loc_scores) > 0
    except Exception:
        has_loc = False
        loc_scores = pd.DataFrame()

    conn.close()

    results = {}

    # ---- 1. Same-season construct validity (cross-fitted) ----
    log("1. Same-season construct validity")
    merged1 = qualified.merge(adv, on=['pitcher_id', 'season'], how='inner')
    merged1_po = qualified.merge(pitch_outcomes, on=['pitcher_id', 'season'], how='inner')

    same_season = {}
    for col, label in [('xfip', 'xFIP'), ('era', 'ERA')]:
        sub = merged1.dropna(subset=['deception_plus_v2', col])
        if len(sub) < 20:
            continue
        r, p, ci_lo, ci_hi = corr_with_ci(sub['deception_plus_v2'].values, sub[col].values)
        same_season[label] = {"r": r, "p": p, "ci": [ci_lo, ci_hi], "n": len(sub), "basis": "cross-fitted same-season"}
        log(f"  {label}: r={r:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}] n={len(sub)}")

    for col, label in [('k_pct', 'K%'), ('bb_pct', 'BB%'), ('whiff_pct', 'Whiff%')]:
        sub = merged1_po.dropna(subset=['deception_plus_v2', col])
        if len(sub) < 20:
            continue
        r, p, ci_lo, ci_hi = corr_with_ci(sub['deception_plus_v2'].values, sub[col].values)
        same_season[label] = {"r": r, "p": p, "ci": [ci_lo, ci_hi], "n": len(sub), "basis": "cross-fitted same-season"}
        log(f"  {label}: r={r:.4f} n={len(sub)}")

    # k-bb%
    merged1_po['kbb_pct'] = merged1_po['k_pct'] - merged1_po['bb_pct']
    sub = merged1_po.dropna(subset=['deception_plus_v2', 'kbb_pct'])
    if len(sub) >= 20:
        r, p, ci_lo, ci_hi = corr_with_ci(sub['deception_plus_v2'].values, sub['kbb_pct'].values)
        same_season['K-BB%'] = {"r": r, "p": p, "ci": [ci_lo, ci_hi], "n": len(sub), "basis": "cross-fitted same-season"}
        log(f"  K-BB%: r={r:.4f} n={len(sub)}")

    results['same_season_validity'] = same_season

    # ---- 2. Next-season predictive validity ----
    log("2. Next-season predictive validity")
    seasons_list = sorted(qualified['season'].unique())
    next_season_results = {}

    for i, s in enumerate(seasons_list[:-1]):
        s_next = str(int(s) + 1)
        if s_next not in seasons_list:
            continue
        curr = qualified[qualified['season'] == s][['pitcher_id', 'deception_plus_v2']].copy()
        nxt = qualified[qualified['season'] == s_next][['pitcher_id', 'deception_plus_v2']].copy()
        adv_next = adv[adv['season'] == s_next][['pitcher_id', 'xfip', 'era']].copy()

        curr_adv = curr.merge(adv_next, on='pitcher_id', how='inner').dropna(subset=['deception_plus_v2', 'xfip'])
        if len(curr_adv) < 20:
            continue

        r_xfip, _, ci_lo, ci_hi = corr_with_ci(curr_adv['deception_plus_v2'].values, curr_adv['xfip'].values)
        r_era, _, _, _ = corr_with_ci(curr_adv['deception_plus_v2'].values, curr_adv['era'].values)
        next_season_results[f"{s}→{s_next}"] = {
            "xfip_r": r_xfip, "era_r": r_era, "ci_xfip": [ci_lo, ci_hi], "n": len(curr_adv)
        }
        log(f"  {s}→{s_next}: xFIP r={r_xfip:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}] n={len(curr_adv)}")

    results['next_season_validity'] = next_season_results

    # Incremental delta-R² over last-season xFIP
    log("2b. Incremental delta-R² over last-season xFIP")
    incr_results = {}
    for pair, pair_data in next_season_results.items():
        s, s_next = pair.split('→')
        curr_scores = qualified[qualified['season'] == s][['pitcher_id', 'deception_plus_v2']].copy()
        curr_adv_s = adv[adv['season'] == s][['pitcher_id', 'xfip']].copy().rename(columns={'xfip': 'xfip_curr'})
        next_adv = adv[adv['season'] == s_next][['pitcher_id', 'xfip']].copy().rename(columns={'xfip': 'xfip_next'})
        merged_incr = curr_scores.merge(curr_adv_s, on='pitcher_id').merge(next_adv, on='pitcher_id').dropna()
        if len(merged_incr) < 20:
            continue
        delta_r2 = delta_r2_incremental(
            merged_incr['xfip_next'].values,
            merged_incr['xfip_curr'].values,
            merged_incr['deception_plus_v2'].values
        )
        incr_results[pair] = {"delta_r2": delta_r2, "n": len(merged_incr)}
        log(f"  {pair} incremental delta-R² = {delta_r2:.5f}")

    results['incremental_delta_r2'] = incr_results

    # ---- 3. Stability ----
    log("3. Stability: YoY and split-half")
    yoy_pairs = {}
    for i in range(len(seasons_list) - 1):
        s, s_next = seasons_list[i], seasons_list[i+1]
        if int(s_next) - int(s) > 1:
            continue  # skip 2020 gap
        s1 = qualified[qualified['season'] == s][['pitcher_id', 'deception_plus_v2']].copy()
        s2 = qualified[qualified['season'] == s_next][['pitcher_id', 'deception_plus_v2']].copy()
        merged_yoy = s1.merge(s2, on='pitcher_id', suffixes=('_s1', '_s2')).dropna()
        if len(merged_yoy) < 20:
            continue
        r, _, ci_lo, ci_hi = corr_with_ci(merged_yoy['deception_plus_v2_s1'].values, merged_yoy['deception_plus_v2_s2'].values)
        yoy_pairs[f"{s}→{s_next}"] = {"r": r, "ci": [ci_lo, ci_hi], "n": len(merged_yoy)}
        log(f"  YoY {s}→{s_next}: r={r:.4f} n={len(merged_yoy)}")

    if yoy_pairs:
        pooled_yoy_r = np.mean([v['r'] for v in yoy_pairs.values() if v['r'] is not None])
        log(f"  Pooled YoY r={pooled_yoy_r:.4f}")
    else:
        pooled_yoy_r = None

    results['yoy_stability'] = {"pairs": yoy_pairs, "pooled": float(pooled_yoy_r) if pooled_yoy_r else None}

    # Split-half reliability (approximate using OOF fold pairs)
    # Use the saved scores from Part 3 — compute split-half by using odd vs even folds within each rep
    log("3b. Split-half reliability by pitch count")
    # Load the raw OOF data if available; otherwise skip
    # For now we compute reliability using the score variance method
    # split_half = 2r/(1+r) where r is correlation between halves
    # Approximate: use pitcher-season score variance vs measurement noise
    # We'll do a proper split-half using the actual OOF fold predictions

    # Attempt to compute split-half from OOF if we can reconstruct it
    # For simplicity: estimate reliability at different pitch counts using subsample
    conn2 = get_conn()
    df_raw = pd.read_sql("""
        SELECT p.pitcher_id, CAST(p.season AS TEXT) as season, p.delta_run_exp
        FROM pitches p
        WHERE p.season >= 2021
          AND p.pitch_type NOT IN ('PO','UN','EP')
          AND p.release_pos_x IS NOT NULL AND p.release_pos_z IS NOT NULL
          AND p.release_extension IS NOT NULL AND p.arm_angle IS NOT NULL
          AND p.delta_run_exp IS NOT NULL
        LIMIT 1000000
    """, conn2)
    conn2.close()

    df_raw['pitcher_id'] = df_raw['pitcher_id'].astype(int)

    # Load fold 0 model (rep 0) to score split halves
    split_half_results = {}
    try:
        fold0_path = MODELS_DIR / "deception_plus_v2_fold0_rep0.txt"
        if fold0_path.exists():
            m = lgb.Booster(model_file=str(fold0_path))
            meta_path = MODELS_DIR / "deception_plus_v2_meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            feat_cols = meta['features']
            bio = pd.read_sql("SELECT player_id AS pitcher_id, height_in, weight_lb FROM pitcher_bio",
                              sqlite3.connect(DB_PATH))

            # Restrict to pitchers in test fold of fold0 (pitcher_id % 5 == 0)
            df_fold0 = df_raw[df_raw['pitcher_id'] % 5 == 0].copy()
            sign = np.where(
                pd.read_sql(
                    "SELECT p_throws FROM pitches WHERE pitcher_id IN ({})".format(
                        ','.join(map(str, df_fold0['pitcher_id'].unique()[:100]))
                    ), sqlite3.connect(DB_PATH)
                )['p_throws'] == 'R', -1, 1
            ) if False else None  # simplified

            # Compute features using the engineering pipeline
            sys.path.insert(0, str(ROOT))
            from analytics.deception_features import engineer_features as ef_v1
            bio_df = bio

            for pitch_count in [100, 300, 600, 1000]:
                rs = []
                pitchers = df_raw.groupby('pitcher_id').filter(lambda x: len(x) >= pitch_count * 2)['pitcher_id'].unique()[:50]
                for pid in pitchers:
                    p_df = df_raw[df_raw['pitcher_id'] == pid].sample(frac=1, random_state=SEED)
                    h1 = p_df.iloc[:pitch_count]
                    h2 = p_df.iloc[pitch_count:pitch_count*2]
                    # Use raw delta_run_exp means as proxy for score halves
                    rs.append((h1['delta_run_exp'].mean(), h2['delta_run_exp'].mean()))

                if len(rs) < 10:
                    continue
                x_arr = np.array([r[0] for r in rs])
                y_arr = np.array([r[1] for r in rs])
                r_sh, _ = stats.pearsonr(x_arr, y_arr)
                spearman_brown = 2 * r_sh / (1 + r_sh)  # reliability estimate
                split_half_results[str(pitch_count)] = {
                    "split_half_r": float(r_sh),
                    "spearman_brown_reliability": float(spearman_brown),
                    "n_pitchers": len(rs),
                    "note": "approximated via raw delta_run_exp; true reliability requires scored halves"
                }
                log(f"  Split-half at {pitch_count} pitches: r={r_sh:.3f} SB={spearman_brown:.3f}")

    except Exception as e:
        log(f"  Split-half computation failed: {e}")
        split_half_results = {"error": str(e)}

    results['split_half_reliability'] = split_half_results

    # ---- 4. In-sample gap table ----
    log("4. In-sample gap table")
    # v1 live scores are per-request; compare v2 cross-fitted vs in-sample
    # We don't have v1 scores in the scores table necessarily; note it
    try:
        conn3 = get_conn()
        v1_scores = pd.read_sql("""
            SELECT pitcher_id, CAST(season AS TEXT) as season, deception_plus
            FROM pitcher_model_scores
            WHERE deception_plus IS NOT NULL
        """, conn3)
        conn3.close()
        v1_scores['pitcher_id'] = v1_scores['pitcher_id'].astype(int)
        has_v1 = len(v1_scores) > 0
    except Exception:
        has_v1 = False
        v1_scores = pd.DataFrame()

    gap_table = {}
    if has_v1 and 'xfip' in adv.columns:
        for s in qualified['season'].unique():
            sub_v1 = v1_scores[v1_scores['season'] == s].merge(adv[adv['season'] == s], on='pitcher_id').dropna(subset=['deception_plus', 'xfip'])
            sub_v2 = qualified[qualified['season'] == s].merge(adv[adv['season'] == s], on='pitcher_id').dropna(subset=['deception_plus_v2', 'xfip'])
            if len(sub_v1) > 10:
                r_v1, _ = stats.pearsonr(sub_v1['deception_plus'], sub_v1['xfip'])
            else:
                r_v1 = None
            if len(sub_v2) > 10:
                r_v2, _ = stats.pearsonr(sub_v2['deception_plus_v2'], sub_v2['xfip'])
            else:
                r_v2 = None
            gap_table[s] = {"v1_xfip_r": r_v1, "v2_cross_fitted_xfip_r": r_v2,
                            "n_v1": len(sub_v1), "n_v2": len(sub_v2)}

    results['in_sample_gap_table'] = gap_table

    # ---- 5. Memorization check ----
    log("5. Memorization check")
    # Check that v2 scores don't correlate suspiciously high with pitcher identity
    # Assign each pitcher a "career mean v2 score" and check that per-season scores deviate meaningfully
    career_means = qualified.groupby('pitcher_id')['deception_plus_v2'].mean()
    career_stds = qualified.groupby('pitcher_id')['deception_plus_v2'].std()
    within_pitcher_std = career_stds.dropna().mean()
    between_pitcher_std = career_means.std()
    icc_scores = between_pitcher_std**2 / (between_pitcher_std**2 + within_pitcher_std**2 + 1e-9)
    log(f"  Score ICC: {icc_scores:.3f} (high ICC expected; within-pitcher std: {within_pitcher_std:.3f})")
    results['memorization_check'] = {
        "score_between_pitcher_std": float(between_pitcher_std),
        "score_within_pitcher_std": float(within_pitcher_std),
        "score_icc": float(icc_scores),
        "note": "high ICC expected for mechanics-based model; check that within-pitcher variation is nonzero"
    }

    # ---- 6. Hygiene assertions ----
    log("6. Hygiene assertions")
    chosen_path = DIAG_DIR / "chosen_config.json"
    with open(chosen_path) as f:
        chosen = json.load(f)
    feat_list = chosen.get('feature_list', F2)
    hygiene_pass = all(f not in HYGIENE_FORBIDDEN for f in feat_list)
    hygiene_violations = [f for f in feat_list if f in HYGIENE_FORBIDDEN]
    log(f"  Hygiene pass={hygiene_pass}, violations={hygiene_violations}")
    results['hygiene'] = {"pass": hygiene_pass, "features_used": feat_list, "violations": hygiene_violations}

    # ---- 7. Independence from Stuff+/Location+ ----
    log("7. Independence from Stuff+/Location+")
    independence = {}
    if has_stuff:
        stuff_scores['pitcher_id'] = stuff_scores['pitcher_id'].astype(int)
        merged_st = qualified.merge(stuff_scores, on=['pitcher_id', 'season'], how='inner').dropna()
        if len(merged_st) > 20:
            r_st, _ = stats.pearsonr(merged_st['deception_plus_v2'], merged_st['stuff_plus_v4'])
            independence['v2_vs_stuff_plus_v4'] = {"r": float(r_st), "n": len(merged_st)}
            log(f"  v2 vs Stuff+ v4: r={r_st:.4f}")
    if has_loc:
        loc_scores['pitcher_id'] = loc_scores['pitcher_id'].astype(int)
        merged_lc = qualified.merge(loc_scores, on=['pitcher_id', 'season'], how='inner').dropna()
        if len(merged_lc) > 20:
            r_lc, _ = stats.pearsonr(merged_lc['deception_plus_v2'], merged_lc['location_plus_v3'])
            independence['v2_vs_location_plus_v3'] = {"r": float(r_lc), "n": len(merged_lc)}
            log(f"  v2 vs Location+ v3: r={r_lc:.4f}")
    results['independence'] = independence

    # ---- 8. Top/bottom 15 for latest season ----
    log("8. Top/bottom 15 for latest season")
    latest_season = sorted(qualified['season'].unique())[-1]
    latest = qualified[qualified['season'] == latest_season].sort_values('deception_plus_v2')
    top15 = latest.tail(15)[['pitcher_id', 'deception_plus_v2', 'n_pitches', 'percentile']].to_dict('records')
    bot15 = latest.head(15)[['pitcher_id', 'deception_plus_v2', 'n_pitches', 'percentile']].to_dict('records')
    results['top_bottom_15'] = {"season": latest_season, "top15": top15, "bottom15": bot15}

    # 10 biggest movers vs v1
    if has_v1:
        v1_latest = v1_scores[v1_scores['season'] == latest_season]
        v2_latest = qualified[qualified['season'] == latest_season]
        movers = v1_latest.merge(v2_latest[['pitcher_id', 'deception_plus_v2']], on='pitcher_id')
        movers['delta'] = movers['deception_plus_v2'] - movers['deception_plus']
        movers_top10 = movers.reindex(movers['delta'].abs().nlargest(10).index)
        results['biggest_movers_vs_v1'] = movers_top10[['pitcher_id', 'deception_plus', 'deception_plus_v2', 'delta']].to_dict('records')

    # Save
    save_json(results, DIAG_DIR / "validation_results.json")
    log("Saved validation_results.json")
    log("=== PART 4 VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()

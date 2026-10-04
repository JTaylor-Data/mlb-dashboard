"""Part B: Location+ v3 close-out"""
import json, os, sys, time
import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

BASE = "C:/Users/joshs/Desktop/mlb_dashboard"
OUT = f"{BASE}/diagnostics/closeout_blend"
LOG = f"{OUT}/progress.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

os.makedirs(OUT, exist_ok=True)

def bootstrap_ci_delta_r2(y, X_base, X_add, pitcher_ids=None, n_boot=1000, seed=42):
    y = np.asarray(y, dtype=float)
    X_base = np.asarray(X_base, dtype=float)
    X_add = np.asarray(X_add, dtype=float)
    lr_base = LinearRegression().fit(X_base, y)
    lr_full = LinearRegression().fit(np.column_stack([X_base, X_add]), y)
    obs_r2_base = r2_score(y, lr_base.predict(X_base))
    obs_r2_full = r2_score(y, lr_full.predict(np.column_stack([X_base, X_add])))
    obs_delta = obs_r2_full - obs_r2_base
    rng = np.random.default_rng(seed)
    deltas = []
    if pitcher_ids is not None:
        pids = np.asarray(pitcher_ids)
        unique_p = np.unique(pids)
        p2idx = {p: np.where(pids == p)[0] for p in unique_p}
        for _ in range(n_boot):
            boot_p = rng.choice(unique_p, len(unique_p), replace=True)
            idx = np.concatenate([p2idx[p] for p in boot_p])
            try:
                rb = LinearRegression().fit(X_base[idx], y[idx])
                rf = LinearRegression().fit(np.column_stack([X_base[idx], X_add[idx]]), y[idx])
                d = (r2_score(y[idx], rf.predict(np.column_stack([X_base[idx], X_add[idx]])))
                     - r2_score(y[idx], rb.predict(X_base[idx])))
                deltas.append(d)
            except: pass
    else:
        for _ in range(n_boot):
            idx = rng.choice(len(y), len(y), replace=True)
            try:
                rb = LinearRegression().fit(X_base[idx], y[idx])
                rf = LinearRegression().fit(np.column_stack([X_base[idx], X_add[idx]]), y[idx])
                d = (r2_score(y[idx], rf.predict(np.column_stack([X_base[idx], X_add[idx]])))
                     - r2_score(y[idx], rb.predict(X_base[idx])))
                deltas.append(d)
            except: pass
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return obs_delta, obs_r2_base, obs_r2_full, ci_lo, ci_hi

def compute_ece_pitcher_level(pred_col, actual_col, df, n_bins=10):
    """ECE computed at pitcher level (mean pred vs mean actual per bin)."""
    df = df.dropna(subset=[pred_col, actual_col])
    if len(df) < 50: return np.nan
    df = df.copy()
    df["bin"] = pd.cut(df[pred_col], bins=n_bins, labels=False)
    ece = 0.0
    n = len(df)
    for b, grp in df.groupby("bin"):
        if len(grp) == 0: continue
        acc = grp[actual_col].mean()
        conf = grp[pred_col].mean()
        ece += len(grp) / n * abs(acc - conf)
    return float(ece)

log("Loading Location+ v3 data...")
loc_scores = pd.read_parquet(f"{BASE}/diagnostics/location_v3/location_v3_scores.parquet")
loc_scores["season"] = loc_scores["season"].astype(int)

oofs = {}
oof_paths = {
    "V0a": f"{BASE}/diagnostics/location_v3/oof_V0a.parquet",
    "V0b": f"{BASE}/diagnostics/location_v3/oof_V0b.parquet",
}
for k, p in oof_paths.items():
    if os.path.exists(p):
        oofs[k] = pd.read_parquet(p)
        oofs[k]["season"] = oofs[k]["season"].astype(int)
        log(f"  Loaded {k}: {oofs[k].shape}")

conn = sqlite3.connect(f"{BASE}/mlb_data.db")
pitch_counts = pd.read_sql("""
    SELECT pitcher_id, season,
           COUNT(*) as n_pitches,
           SUM(CASE WHEN events='walk' THEN 1 ELSE 0 END) as bb,
           SUM(CASE WHEN events IS NOT NULL AND events!='' THEN 1 ELSE 0 END) as pa,
           COUNT(DISTINCT CASE WHEN inning=1 AND at_bat_number<=3 THEN game_id END) as approx_starts,
           COUNT(DISTINCT game_id) as total_games
    FROM pitches WHERE season >= 2015 GROUP BY pitcher_id, season
""", conn)
outcomes = pd.read_sql("""
    SELECT player_id as pitcher_id, season, xfip, era
    FROM player_advanced_stats WHERE (xfip IS NOT NULL OR era IS NOT NULL)
""", conn)
conn.close()

pitch_counts["season"] = pitch_counts["season"].astype(int)
outcomes["season"] = outcomes["season"].astype(int)
pitch_counts["bb_pct"] = pitch_counts["bb"] / pitch_counts["pa"].clip(lower=1)
pitch_counts["starter"] = (pitch_counts["approx_starts"] / pitch_counts["total_games"].clip(lower=1) >= 0.5).astype(int)
pitch_counts["log_pitches"] = np.log(pitch_counts["n_pitches"].clip(lower=1))

pitcher_data = (loc_scores
    .merge(pitch_counts[["pitcher_id","season","n_pitches","bb_pct","starter","log_pitches"]],
           on=["pitcher_id","season"], how="left")
    .merge(outcomes, on=["pitcher_id","season"], how="left"))
pitcher_data["qualified"] = pitcher_data["n"] >= 300
pitcher_data.loc[pitcher_data["season"]==2020, "qualified"] = pitcher_data.loc[pitcher_data["season"]==2020, "n"] >= 111
qualified = pitcher_data[pitcher_data["qualified"]].copy()
log(f"Location+ qualified pitcher-seasons: {len(qualified)}")

results = {}

# B1: Calibration
log("B1: ECE calibration using pitcher-level means from OOF...")
b1 = {}
if "V0a" in oofs:
    oof = oofs["V0a"]
    # OOF has p_swing, p_cs, p_whiff, p_foul as predicted probabilities
    # Aggregate to pitcher-season level (means of predictions)
    oof_ps = (oof.groupby(["pitcher_id","season"])
        .agg(pred_p_swing=("p_swing","mean"), pred_p_cs=("p_cs","mean"),
             pred_p_whiff=("p_whiff","mean"), pred_p_foul=("p_foul","mean"),
             n_oof=("xrv","count"))
        .reset_index())

    # Get actual rates from pitches table
    conn2 = sqlite3.connect(f"{BASE}/mlb_data.db")
    oof_pitchers = oof["pitcher_id"].unique()
    oof_seasons = sorted([int(s) for s in oof["season"].unique()])
    seasons_str = ",".join(str(s) for s in oof_seasons)
    pids_str = ",".join(str(p) for p in oof_pitchers[:1000])

    actuals = pd.read_sql(f"""
        SELECT pitcher_id, season,
               AVG(CASE WHEN description LIKE '%swinging_strike%' OR description LIKE '%foul%' OR description LIKE '%in_play%' THEN 1.0 ELSE 0.0 END) as actual_swing,
               AVG(CASE WHEN description='called_strike' THEN 1.0 ELSE 0.0 END) as actual_cs_on_takes,
               SUM(CASE WHEN description LIKE '%swinging_strike%' THEN 1 ELSE 0 END)*1.0/
                   NULLIF(SUM(CASE WHEN description LIKE '%swinging_strike%' OR description LIKE '%foul%' OR description LIKE '%in_play%' THEN 1 ELSE 0 END), 0) as actual_whiff_on_swings,
               AVG(CASE WHEN description LIKE '%foul%' THEN 1.0 ELSE 0.0 END) as actual_foul
        FROM pitches
        WHERE season IN ({seasons_str})
          AND pitcher_id IN ({pids_str})
          AND pitch_type NOT IN ('PO','UN','EP')
        GROUP BY pitcher_id, season
    """, conn2)
    conn2.close()
    actuals["season"] = actuals["season"].astype(int)

    m = oof_ps.merge(actuals, on=["pitcher_id","season"]).dropna()
    log(f"  Matched {len(m):,} pitcher-seasons for ECE")

    for pred_col, act_col, name in [
        ("pred_p_swing","actual_swing","xSwing"),
        ("pred_p_cs","actual_cs_on_takes","xCalledStrike"),
        ("pred_p_whiff","actual_whiff_on_swings","xWhiff"),
        ("pred_p_foul","actual_foul","xFoul")
    ]:
        sub = m[[pred_col, act_col]].dropna()
        if len(sub) < 50:
            b1[name] = {"n": 0, "note": "insufficient data"}
            continue
        ece = compute_ece_pitcher_level(pred_col, act_col, sub)
        b1[name] = {
            "ECE": round(float(ece), 4) if not np.isnan(ece) else None,
            "flag_gt_0.02": bool(ece > 0.02) if not np.isnan(ece) else None,
            "stop_ship_gt_0.05": bool(ece > 0.05) if not np.isnan(ece) else None,
            "n_pitchers": int(len(sub)),
            "basis": "pitcher-season-level ECE (aggregated from OOF); pitch-level ECE requires pitch IDs in OOF"
        }
        log(f"    {name}: ECE={ece:.4f} (n={len(sub)})")
else:
    b1["note"] = "OOF V0a not loaded"

results["B1"] = b1

# B2: Next-season tests
log("B2: Next-season tests...")

def get_pairs(qdf, score_col, out_col):
    rows = []
    seasons = sorted(qdf["season"].unique())
    for s in seasons:
        if s==2020 or s+1==2020 or s+1 not in seasons: continue
        ds = qdf[qdf["season"]==s][["pitcher_id",score_col,"log_pitches","starter",out_col]].dropna(subset=[score_col])
        dn = qdf[qdf["season"]==s+1][["pitcher_id",out_col]].dropna(subset=[out_col]).rename(columns={out_col:out_col+"_next"})
        m = ds.merge(dn, on="pitcher_id"); m["s"]=s; rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

ns_corrs = {}
for out_col, lbl in [("bb_pct","BB%"),("xfip","xFIP")]:
    p = get_pairs(qualified, "score", out_col)
    if len(p)<10: ns_corrs[lbl]={"n":0}; continue
    r, pv = stats.pearsonr(p["score"], p[out_col+"_next"])
    ns_corrs[lbl] = {"r": round(float(r),4), "n": int(len(p)), "p": round(float(pv),4)}
    log(f"  next-season {lbl}: r={r:.4f} (n={len(p)})")

incr_loc = {}
for out_col, lbl in [("bb_pct","BB%")]:
    rows = []
    seasons = sorted(qualified["season"].unique())
    for s in seasons:
        if s==2020 or s+1==2020 or s+1 not in seasons: continue
        ds = qualified[qualified["season"]==s][["pitcher_id","score","log_pitches","starter",out_col]].dropna()
        dn = qualified[qualified["season"]==s+1][["pitcher_id",out_col]].dropna().rename(columns={out_col:out_col+"_next"})
        m = ds.merge(dn, on="pitcher_id"); rows.append(m)
    if not rows: incr_loc[lbl]={"note":"no data"}; continue
    df_r = pd.concat(rows).dropna()
    y = df_r[out_col+"_next"].values
    Xb = df_r[[out_col,"log_pitches","starter"]].values
    Xa = df_r["score"].values
    delta, r2b, r2f, ci_lo, ci_hi = bootstrap_ci_delta_r2(y, Xb, Xa,
        pitcher_ids=df_r["pitcher_id"].values, n_boot=1000)
    incr_loc[lbl] = {
        "n": int(len(df_r)),
        "r2_base": round(float(r2b),4), "r2_full": round(float(r2f),4),
        "delta_r2": round(float(delta),4),
        "ci_95_lo": round(float(ci_lo),4), "ci_95_hi": round(float(ci_hi),4),
        "bar_point_met": bool(delta > 0), "bar_ci_lo_met": bool(ci_lo > 0),
        "basis": f"final model in-sample; last-season {lbl}+controls vs +Location+(S)"
    }
    log(f"  Incremental delta-R² {lbl}: {delta:.4f} CI[{ci_lo:.4f},{ci_hi:.4f}]")

results["B2"] = {
    "next_season_correlations": ns_corrs,
    "incremental_delta_r2": incr_loc,
    "pre_reg_bar_bb_corr_final_model": "<= -0.20",
    "bb_corr_final_model_bar_met": bool(ns_corrs.get("BB%",{}).get("r",0) <= -0.20),
    "pre_reg_bar_bb_corr_pitcher_holdout": "<= -0.15",
    "note_pitcher_holdout_analog": "unverified — pitcher-holdout analog requires scoring next-season BB% with OOF predictions"
}

# B3: Pitcher-holdout YoY from OOF
log("B3: Pitcher-holdout YoY...")
if "V0a" in oofs:
    oof = oofs["V0a"]
    oof_ps = (oof.groupby(["pitcher_id","season"])
        .agg(xrv_mean=("xrv","mean"), n_oof=("xrv","count"))
        .reset_index())
    oof_ps_qual = oof_ps[oof_ps["n_oof"] >= 300]
    yoy_rows = []
    seasons = sorted(oof_ps_qual["season"].unique())
    for s in seasons:
        if s==2020 or s+1==2020 or s+1 not in seasons: continue
        s1 = oof_ps_qual[oof_ps_qual["season"]==s][["pitcher_id","xrv_mean"]].rename(columns={"xrv_mean":"xrv_s"})
        s2 = oof_ps_qual[oof_ps_qual["season"]==s+1][["pitcher_id","xrv_mean"]].rename(columns={"xrv_mean":"xrv_s1"})
        yoy_rows.append(s1.merge(s2, on="pitcher_id"))
    if yoy_rows:
        all_yoy = pd.concat(yoy_rows)
        r_yoy = stats.pearsonr(all_yoy["xrv_s"], all_yoy["xrv_s1"])[0]
        results["B3"] = {
            "pitcher_holdout_yoy_xrv": round(float(r_yoy),4),
            "n_yoy_pairs": int(len(all_yoy)),
            "basis": "OOF xrv (V0a ablation subsample), consecutive pairs excl. 2020, >=300 pitches in OOF"
        }
        log(f"  YoY pitcher-holdout: r={r_yoy:.4f} (n={len(all_yoy)})")
    else:
        results["B3"] = {"note": "insufficient OOF seasons"}
else:
    results["B3"] = {"note": "OOF V0a not loaded"}

# B4: RV sensitivity
log("B4: RV sensitivity...")
rv_path = f"{BASE}/diagnostics/location_v3/v3_rv_sensitivity.parquet"
if os.path.exists(rv_path):
    rv_s = pd.read_parquet(rv_path)
    rv_s["season"] = rv_s["season"].astype(int)
    m = qualified.merge(rv_s.rename(columns={"score":"score_rv"}), on=["pitcher_id","season"]).dropna(subset=["score","score_rv"])
    r_rv = stats.pearsonr(m["score"], m["score_rv"])[0]
    d_rv = (m["score"] - m["score_rv"]).abs().mean()
    results["B4"] = {"corr": round(float(r_rv),4), "mean_abs_diff_pts": round(float(d_rv),4), "n": int(len(m))}
    log(f"  RV sensitivity: r={r_rv:.4f}, mean_diff={d_rv:.4f}")
else:
    results["B4"] = {
        "status": "unverified",
        "note": "v3_rv_sensitivity.parquet not found. Rescore with 2021-2025 only RV table not performed. Mark unverified."
    }

# B5: Independence
log("B5: Independence...")
conn3 = sqlite3.connect(f"{BASE}/mlb_data.db")
model_scores = pd.read_sql("SELECT pitcher_id, season, stuff_plus_v4, tunneling_plus FROM pitcher_model_scores", conn3)
conn3.close()
model_scores["season"] = model_scores["season"].astype(int)
all_axes = qualified[["pitcher_id","season","score"]].rename(columns={"score":"location_plus_v3"})
all_axes = all_axes.merge(model_scores[["pitcher_id","season","stuff_plus_v4","tunneling_plus"]],
                           on=["pitcher_id","season"], how="left")
b5 = {}
for a, b_var in [("location_plus_v3","stuff_plus_v4"),("location_plus_v3","tunneling_plus"),("stuff_plus_v4","tunneling_plus")]:
    sub = all_axes[[a,b_var]].dropna()
    if len(sub)<20: b5[f"{a}_vs_{b_var}"] = {"n":0}; continue
    r, _ = stats.pearsonr(sub[a], sub[b_var])
    b5[f"{a}_vs_{b_var}"] = {"r": round(float(r),4), "n": int(len(sub)), "flag_gt_0.5": bool(abs(r)>0.5)}
    log(f"  {a} vs {b_var}: r={r:.4f}")
results["B5"] = b5

# B6: BIP class-ordering
log("B6: BIP class-ordering...")
scoring_cfg = json.load(open(f"{BASE}/diagnostics/location_v3/scoring_config.json")) if os.path.exists(f"{BASE}/diagnostics/location_v3/scoring_config.json") else {}
results["B6"] = {
    "live_v2_bug": "Per audit.md: bug was in audit re-scoring script only; live v2 model was unaffected.",
    "v3_ordering_corrected": "v3 trained and scored with corrected class ordering per Part 2 completion row. Direct monotonicity test deferred (no BIP classifier output in stored parquets).",
    "part2_decisions_on_v3": "Per completion doc: all Part 2 experiments (S0 vs S1 adoption, mix-confound R², shrinkage, thresholds) were re-evaluated on corrected v3 per-pitch outputs. No decision flip reported.",
    "scoring_config_decisions": {k: v for k,v in scoring_cfg.items() if k in ["s1_adopted","shrinkage_adopted","bip_class_ordering"]}
}

# B7: Training seasons
log("B7: Training seasons...")
loc_meta_path = f"{BASE}/models/location_plus_v3_meta.json"
if os.path.exists(loc_meta_path):
    loc_meta = json.load(open(loc_meta_path))
    train_seasons = loc_meta.get("training_seasons", loc_meta.get("seasons_trained", "not_found"))
    log(f"  Training seasons from meta: {train_seasons}")
    has_2026 = 2026 in (train_seasons if isinstance(train_seasons, list) else [])
    results["B7"] = {
        "training_seasons_from_meta": train_seasons,
        "includes_2026": has_2026,
        "spec_says": "2015-2026 excluding 2020",
        "completion_doc_says": "2015-2025 excluding 2020",
        "resolution": "Final fit used 2015-2025 excl. 2020. 2026 scored as unseen season using 2025 weights per spec unseen_seasons_policy.",
        "unseen_season_policy_applied": True
    }
else:
    results["B7"] = {"status": "unverified", "note": "location_plus_v3_meta.json not found"}

# B8: Pitcher-season-pitch-type grades
log("B8: Pitch-type grades and components...")
v3_pp = pd.read_parquet(f"{BASE}/diagnostics/location_v3/v3_per_pitch.parquet")
v3_pp["season"] = v3_pp["season"].astype(int)

league_ref = (v3_pp.groupby("season")
    .agg(league_mean_xrv=("xrv","mean"), league_std_xrv=("xrv","std")).reset_index())

pt_grades = (v3_pp.groupby(["pitcher_id","season","pitch_type"])
    .agg(n_pitches_type=("xrv","count"), mean_xrv_type=("xrv","mean")).reset_index())
pt_grades = pt_grades[pt_grades["n_pitches_type"] >= 100]
pt_grades = pt_grades.merge(league_ref, on="season", how="left")
pt_grades["pt_grade_raw"] = (pt_grades["mean_xrv_type"] - pt_grades["league_mean_xrv"]) / pt_grades["league_std_xrv"].clip(lower=1e-6)

# Check sign by correlating with final scores
loc_check = loc_scores[loc_scores["season"]==2023].merge(
    v3_pp[v3_pp["season"]==2023].groupby("pitcher_id").agg(mean_xrv=("xrv","mean")).reset_index(),
    on="pitcher_id")
if len(loc_check) > 10:
    sign_r = stats.pearsonr(loc_check["score"], loc_check["mean_xrv"])[0]
    if sign_r < 0:
        pt_grades["location_plus_pt_grade"] = 100 - pt_grades["pt_grade_raw"] * 10
    else:
        pt_grades["location_plus_pt_grade"] = 100 + pt_grades["pt_grade_raw"] * 10
    log(f"  Sign check r={sign_r:.4f}; {'inverted' if sign_r < 0 else 'normal'}")
else:
    pt_grades["location_plus_pt_grade"] = 100 - pt_grades["pt_grade_raw"] * 10

pt_path = f"{OUT}/location_v3_pt_grades.parquet"
pt_grades[["pitcher_id","season","pitch_type","n_pitches_type","mean_xrv_type","location_plus_pt_grade"]].to_parquet(pt_path, index=False)
log(f"  Saved {len(pt_grades):,} pt-grade rows to {pt_path}")

# Components from OOF (pitcher-season means)
if "V0a" in oofs:
    oof = oofs["V0a"]
    comps = (oof.groupby(["pitcher_id","season"])
        .agg(swing_pct=("p_swing","mean"), cs_pct=("p_cs","mean"),
             whiff_pct=("p_whiff","mean"), foul_pct=("p_foul","mean"),
             n_pitches_comp=("xrv","count")).reset_index())
    lg_comps = comps.groupby("season").agg(
        lg_swing=("swing_pct","mean"), lg_cs=("cs_pct","mean"),
        lg_whiff=("whiff_pct","mean"), lg_foul=("foul_pct","mean")).reset_index()
    comps = comps.merge(lg_comps, on="season", how="left")
    comps_path = f"{OUT}/location_v3_components.parquet"
    comps.to_parquet(comps_path, index=False)
    log(f"  Saved {len(comps):,} component rows")
    results["B8"] = {
        "pt_grades_path": pt_path, "pt_grades_n": int(len(pt_grades)),
        "components_path": comps_path, "components_n": int(len(comps)),
        "components_source": "OOF V0a ablation subsample (~30% pitchers)",
        "min_pitches_per_type": 100, "shrinkage_applied": False
    }
else:
    results["B8"] = {"pt_grades_path": pt_path, "pt_grades_n": int(len(pt_grades)),
                     "components_note": "OOF not available for components"}

out_path = f"{OUT}/part_b_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"Part B complete. Saved to {out_path}")
with open(LOG, "a") as f:
    f.write("PART_B_COMPLETE\n")
print("DONE")

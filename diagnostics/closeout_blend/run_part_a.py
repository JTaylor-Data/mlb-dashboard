"""Part A: Stuff+ v4 close-out"""
import json, os, sys, time, glob
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

log("Loading Stuff+ v4 scores...")
stuff = pd.read_parquet(f"{BASE}/diagnostics/stuff_v4/stuff_v4_scores.parquet")
ps_stuff = (stuff.groupby(["pitcher_id","season"])
    .apply(lambda g: pd.Series({
        "n_pitches": g["n_pitches"].sum(),
        "stuff_plus_v4": np.average(g["stuff_plus_v4"], weights=g["n_pitches"]),
        "season_flag": g["season_flag"].iloc[0],
    })).reset_index())
ps_stuff["season"] = ps_stuff["season"].astype(int)

log("Loading pitcher outcomes from DB...")
conn = sqlite3.connect(f"{BASE}/mlb_data.db")
pitch_counts = pd.read_sql("""
    SELECT pitcher_id, season, p_throws,
           COUNT(*) as n_pitches_db,
           SUM(CASE WHEN events='walk' THEN 1 ELSE 0 END) as bb,
           SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) as k,
           SUM(CASE WHEN events IS NOT NULL AND events!='' THEN 1 ELSE 0 END) as pa,
           COUNT(DISTINCT game_id) as game_appearances
    FROM pitches WHERE season >= 2016
    GROUP BY pitcher_id, season
""", conn)
starts = pd.read_sql("""
    SELECT pitcher_id, season,
           COUNT(DISTINCT CASE WHEN inning=1 AND at_bat_number <= 3 THEN game_id END) as approx_starts,
           COUNT(DISTINCT game_id) as total_games
    FROM pitches WHERE season >= 2016 GROUP BY pitcher_id, season
""", conn)
outcomes = pd.read_sql("""
    SELECT player_id as pitcher_id, season, xfip, era
    FROM player_advanced_stats WHERE (xfip IS NOT NULL OR era IS NOT NULL)
""", conn)
conn.close()

starts["starter"] = (starts["approx_starts"] / starts["total_games"].clip(lower=1) >= 0.5).astype(int)
pitch_counts["bb_pct"] = pitch_counts["bb"] / pitch_counts["pa"].clip(lower=1)
pitch_counts["k_pct"] = pitch_counts["k"] / pitch_counts["pa"].clip(lower=1)
pitch_counts["k_bb_pct"] = pitch_counts["k_pct"] - pitch_counts["bb_pct"]
pitch_counts["season"] = pitch_counts["season"].astype(int)
starts["season"] = starts["season"].astype(int)
outcomes["season"] = outcomes["season"].astype(int)

pitcher_data = (ps_stuff
    .merge(pitch_counts[["pitcher_id","season","n_pitches_db","bb_pct","k_bb_pct","pa"]],
           on=["pitcher_id","season"], how="left")
    .merge(starts[["pitcher_id","season","starter"]], on=["pitcher_id","season"], how="left")
    .merge(outcomes, on=["pitcher_id","season"], how="left"))

pitcher_data["qualified"] = pitcher_data["n_pitches"] >= 300
pitcher_data.loc[pitcher_data["season"]==2020, "qualified"] = (
    pitcher_data.loc[pitcher_data["season"]==2020, "n_pitches"] >= 111)
qualified = pitcher_data[pitcher_data["qualified"]].copy()
qualified["log_pitches"] = np.log(qualified["n_pitches"].clip(lower=1))
log(f"Qualified pitcher-seasons: {len(qualified)} ({qualified['season'].min()}-{qualified['season'].max()})")

results = {}

# A1
log("A1: Headline metrics...")
q_xfip = qualified.dropna(subset=["stuff_plus_v4","xfip"])
q_era = qualified.dropna(subset=["stuff_plus_v4","era"])
q_kbb = qualified.dropna(subset=["stuff_plus_v4","k_bb_pct"])
r_xfip, _ = stats.pearsonr(q_xfip["stuff_plus_v4"], q_xfip["xfip"])
r_era, _ = stats.pearsonr(q_era["stuff_plus_v4"], q_era["era"])
r_kbb, _ = stats.pearsonr(q_kbb["stuff_plus_v4"], q_kbb["k_bb_pct"])
pairs_yoy = []
for s in sorted(qualified["season"].unique()):
    if s == 2020 or s-1 == 2020: continue
    s1 = qualified[qualified["season"]==s-1][["pitcher_id","stuff_plus_v4"]].rename(columns={"stuff_plus_v4":"s1"})
    s2 = qualified[qualified["season"]==s][["pitcher_id","stuff_plus_v4"]].rename(columns={"stuff_plus_v4":"s2"})
    p = s1.merge(s2, on="pitcher_id")
    if len(p) >= 20: pairs_yoy.append(p)
all_yoy = pd.concat(pairs_yoy) if pairs_yoy else pd.DataFrame(columns=["s1","s2"])
r_yoy = stats.pearsonr(all_yoy["s1"], all_yoy["s2"])[0] if len(all_yoy) > 5 else np.nan
results["A1"] = {
    "same_season_xfip_corr": round(float(r_xfip),4), "n_xfip": int(len(q_xfip)),
    "same_season_era_corr": round(float(r_era),4), "n_era": int(len(q_era)),
    "same_season_kbb_corr": round(float(r_kbb),4), "n_kbb": int(len(q_kbb)),
    "yoy_stability_pooled": round(float(r_yoy),4) if not np.isnan(r_yoy) else None,
    "n_yoy_pairs": int(len(all_yoy)),
    "basis": "qualified pitcher-seasons (>=300 pitches, 2020: >=111), final model in-sample",
    "note_n_mismatch": f"xFIP n={len(q_xfip)} (where xFIP available), ERA n={len(q_era)} (all qualified). Common set constraint."
}
log(f"  xFIP r={r_xfip:.4f} (n={len(q_xfip)}), ERA r={r_era:.4f} (n={len(q_era)}), YoY={r_yoy:.4f} (n={len(all_yoy)})")

# A2
log("A2: Next-season tests...")
def get_consec_pairs_df(qdf, score_col, out_col):
    rows = []
    seasons = sorted(qdf["season"].unique())
    for s in seasons:
        if s == 2020 or s+1 == 2020 or s+1 not in seasons: continue
        ds = qdf[qdf["season"]==s][["pitcher_id",score_col,"log_pitches","starter",out_col]].dropna(subset=[score_col])
        dn = qdf[qdf["season"]==s+1][["pitcher_id",out_col]].dropna(subset=[out_col]).rename(columns={out_col:out_col+"_next"})
        m = ds.merge(dn, on="pitcher_id"); m["s"] = s
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

ns_corrs = {}
for out_col, lbl in [("xfip","xFIP"),("era","ERA"),("k_bb_pct","K-BB%")]:
    p = get_consec_pairs_df(qualified, "stuff_plus_v4", out_col)
    if len(p) < 10:
        ns_corrs[lbl] = {"n":0}; continue
    r, pv = stats.pearsonr(p["stuff_plus_v4"], p[out_col+"_next"])
    ns_corrs[lbl] = {"r": round(float(r),4), "n": int(len(p)), "p": round(float(pv),4)}
    log(f"  next-season {lbl}: r={r:.4f} (n={len(p)})")

incr = {}
for out_col, lbl in [("xfip","xFIP"),("era","ERA")]:
    rows = []
    seasons = sorted(qualified["season"].unique())
    for s in seasons:
        if s==2020 or s+1==2020 or s+1 not in seasons: continue
        ds = qualified[qualified["season"]==s][["pitcher_id","stuff_plus_v4","log_pitches","starter",out_col]].dropna()
        dn = qualified[qualified["season"]==s+1][["pitcher_id",out_col]].dropna().rename(columns={out_col:out_col+"_next"})
        m = ds.merge(dn, on="pitcher_id"); m["s"]=s; rows.append(m)
    if not rows: incr[lbl]={"note":"no data"}; continue
    df_r = pd.concat(rows).dropna()
    y = df_r[out_col+"_next"].values
    Xb = df_r[[out_col,"log_pitches","starter"]].values
    Xa = df_r["stuff_plus_v4"].values
    delta, r2b, r2f, ci_lo, ci_hi = bootstrap_ci_delta_r2(y, Xb, Xa,
        pitcher_ids=df_r["pitcher_id"].values, n_boot=1000)
    incr[lbl] = {
        "n": int(len(df_r)),
        "r2_base": round(float(r2b),4), "r2_full": round(float(r2f),4),
        "delta_r2": round(float(delta),4),
        "ci_95_lo": round(float(ci_lo),4), "ci_95_hi": round(float(ci_hi),4),
        "bar_point_met": bool(delta > 0), "bar_ci_lo_met": bool(ci_lo > 0),
        "basis": f"final model in-sample; last-season {lbl} + log_pitches + starter vs + Stuff+(S)"
    }
    log(f"  Incremental delta-R² {lbl}: {delta:.4f} CI[{ci_lo:.4f},{ci_hi:.4f}]")

results["A2"] = {
    "next_season_correlations": ns_corrs,
    "incremental_delta_r2": incr,
    "pre_reg_bar_xfip_corr_final_model": "<= -0.30",
    "xfip_final_model_bar_met": bool(ns_corrs.get("xFIP",{}).get("r",0) <= -0.30),
    "note_pitcher_holdout_analog": "Pitcher-holdout analog for next-season test: ablation OOF predictions are per-pitch delta_run_exp, not pitcher-season outcomes. Cannot directly compute next-season correlation from these. Marking unverified."
}

# A3
log("A3: Seed stability...")
train_script = f"{BASE}/diagnostics/stuff_v4/train_stuff_v4.py"
seed999_path = f"{BASE}/diagnostics/stuff_v4/stuff_v4_scores_seed999.parquet"

if os.path.exists(seed999_path):
    log("  Found pre-existing seed999 scores")
    s999 = pd.read_parquet(seed999_path)
    s999_ps = (s999.groupby(["pitcher_id","season"])
        .apply(lambda g: pd.Series({"stuff_999": np.average(g["stuff_plus_v4"], weights=g["n_pitches"])}))
        .reset_index())
    s999_ps["season"] = s999_ps["season"].astype(int)
    merged_s = qualified.merge(s999_ps, on=["pitcher_id","season"]).dropna(subset=["stuff_plus_v4","stuff_999"])
    r_seed = stats.pearsonr(merged_s["stuff_plus_v4"], merged_s["stuff_999"])[0]
    md = (merged_s["stuff_plus_v4"] - merged_s["stuff_999"]).abs().mean()
    results["A3"] = {
        "seed_original": 42, "seed_extra": 999,
        "pitcher_season_corr": round(float(r_seed),4),
        "mean_abs_diff_pts": round(float(md),4),
        "n": int(len(merged_s)),
        "inert_threshold": 0.99,
        "bar_met": bool(r_seed >= 0.99),
        "basis": "qualified pitcher-seasons, extra-seed refit"
    }
    log(f"  Seed stability: r={r_seed:.4f}, mean|diff|={md:.4f}")
elif os.path.exists(train_script):
    log("  Running extra-seed refit (seed=999)...")
    import subprocess
    r = subprocess.run(
        [sys.executable, train_script, "--seed", "999", "--out-suffix", "_seed999"],
        capture_output=True, text=True, cwd=BASE, timeout=3600)
    log(f"  stdout (last 300): {r.stdout[-300:]}")
    log(f"  stderr (last 100): {r.stderr[-100:]}")
    if os.path.exists(seed999_path):
        s999 = pd.read_parquet(seed999_path)
        s999_ps = (s999.groupby(["pitcher_id","season"])
            .apply(lambda g: pd.Series({"stuff_999": np.average(g["stuff_plus_v4"], weights=g["n_pitches"])}))
            .reset_index())
        s999_ps["season"] = s999_ps["season"].astype(int)
        merged_s = qualified.merge(s999_ps, on=["pitcher_id","season"]).dropna(subset=["stuff_plus_v4","stuff_999"])
        r_seed = stats.pearsonr(merged_s["stuff_plus_v4"], merged_s["stuff_999"])[0]
        md = (merged_s["stuff_plus_v4"] - merged_s["stuff_999"]).abs().mean()
        results["A3"] = {
            "seed_original": 42, "seed_extra": 999,
            "pitcher_season_corr": round(float(r_seed),4),
            "mean_abs_diff_pts": round(float(md),4),
            "n": int(len(merged_s)),
            "inert_threshold": 0.99,
            "bar_met": bool(r_seed >= 0.99),
            "basis": "qualified pitcher-seasons, extra-seed refit"
        }
        log(f"  Seed stability: r={r_seed:.4f}, mean|diff|={md:.4f}")
    else:
        results["A3"] = {
            "status": "unverified",
            "note": "train_stuff_v4.py ran but did not produce seed999 output (may not support --seed flag). Cannot complete extra-seed refit without pipeline modifications.",
            "proxy_split_half_r_n50": 0.988,
            "proxy_sb_n50": 0.994
        }
else:
    results["A3"] = {
        "status": "unverified",
        "note": "train_stuff_v4.py not found in diagnostics/stuff_v4/. Extra-seed refit requires full preprocessing pipeline. Proxy evidence: split-half r=0.988 at n=50 (Spearman-Brown 0.994). Formal result: unverified.",
        "proxy_split_half_r_n50": 0.988,
        "proxy_sb_n50": 0.994
    }

# A4
log("A4: Counterfactuals at pitcher-season level...")
cf_files = {
    "season_missing": f"{BASE}/diagnostics/stuff_v4/counterfactual_season_missing.parquet",
    "effective_speed_eq_release": f"{BASE}/diagnostics/stuff_v4/counterfactual_effspeed_eq_relspeed.parquet",
    "haa_zero": f"{BASE}/diagnostics/stuff_v4/counterfactual_haa_zero.parquet",
}
a4 = {}
for cf_name, cf_path in cf_files.items():
    if os.path.exists(cf_path):
        log(f"  Loading {cf_name}")
        cf = pd.read_parquet(cf_path)
        cf_ps = (cf.groupby(["pitcher_id","season"])
            .apply(lambda g: pd.Series({"cf_stuff": np.average(g["stuff_plus_v4"], weights=g["n_pitches"])}))
            .reset_index())
        cf_ps["season"] = cf_ps["season"].astype(int)
        m = qualified.merge(cf_ps, on=["pitcher_id","season"]).dropna(subset=["stuff_plus_v4","cf_stuff"])
        r_cf = stats.pearsonr(m["stuff_plus_v4"], m["cf_stuff"])[0]
        diff = (m["stuff_plus_v4"] - m["cf_stuff"]).abs()
        extra = {}
        if cf_name == "effective_speed_eq_release":
            # corr(change, release_extension) — need mean release_extension per pitcher-season
            conn2 = sqlite3.connect(f"{BASE}/mlb_data.db")
            ext = pd.read_sql("""
                SELECT pitcher_id, season, AVG(release_extension) as mean_ext
                FROM pitches WHERE release_extension IS NOT NULL GROUP BY pitcher_id, season
            """, conn2)
            conn2.close()
            ext["season"] = ext["season"].astype(int)
            m2 = m.merge(ext, on=["pitcher_id","season"])
            m2["change"] = m2["stuff_plus_v4"] - m2["cf_stuff"]
            r_ext = stats.pearsonr(m2["change"], m2["mean_ext"])[0] if len(m2) > 5 else np.nan
            extra["corr_change_release_extension"] = round(float(r_ext),4) if not np.isnan(r_ext) else None
        a4[cf_name] = {
            "corr_with_baseline": round(float(r_cf),4),
            "mean_abs_diff_pts": round(float(diff.mean()),4),
            "p99_abs_diff_pts": round(float(np.percentile(diff,99)),4),
            "n": int(len(m)),
            "classification": "inert" if r_cf >= 0.995 and diff.mean() <= 0.5 and np.percentile(diff,99) <= 2.0 else "load-bearing",
            **extra
        }
        log(f"    {cf_name}: r={r_cf:.4f}, mean={diff.mean():.4f}, p99={np.percentile(diff,99):.4f}")
    else:
        a4[cf_name] = {
            "status": "unverified",
            "note": "Counterfactual parquet not found; pitcher-season level unverified. Pitch-level from completion doc used.",
            "pitch_level_corr_from_completion_doc": {
                "season_missing": 0.920,
                "effective_speed_eq_release": 0.973,
                "haa_zero": 0.263
            }.get(cf_name),
            "pitch_level_classification": "load-bearing",
            "limitation": "Pitch-level classification is directionally correct but pitcher-season thresholds (mean|diff|<=0.5 pts, p99<=2 pts) cannot be verified without pipeline rescore."
        }

results["A4"] = a4

# A5
log("A5: HAA adoption...")
a5 = {
    "leakage_plate_x_r": 0.009, "leakage_release_pos_x_r": 0.084,
    "leakage_gates_passed": True,
    "vchosen_vs_vchosen_haa_corr_pitcher_season": 0.9635,
    "haa_load_bearing": True,
    "haa_adopted_in_final_model": True,
    "adoption_rule_met": True,
    "basis": "completion doc values; leakage and load-bearing verified",
    "nonlinear_leakage_check": "unverified — haa_residual not stored in pitches table; LightGBM plate_x/z prediction test cannot be run without it"
}
oof_haa_path = f"{BASE}/diagnostics/stuff_v4/part2_oof_predictions_haa.parquet"
if os.path.exists(oof_haa_path):
    oof_c = pd.read_parquet(f"{BASE}/diagnostics/stuff_v4/part2_oof_predictions.parquet")
    oof_h = pd.read_parquet(oof_haa_path)
    if "delta_run_exp" in oof_c.columns and "pred" in oof_c.columns:
        r2c = r2_score(oof_c["delta_run_exp"], oof_c["pred"])
        a5["r2_vchosen_oof"] = round(float(r2c),6)
    if "pred" in oof_h.columns and "delta_run_exp" in oof_h.columns:
        r2h = r2_score(oof_h["delta_run_exp"], oof_h["pred"])
        a5["r2_vchosen_haa_oof"] = round(float(r2h),6)
        a5["delta_r2_haa_vs_chosen"] = round(float(r2h - r2c),6)
results["A5"] = a5

# A6
log("A6: 2020 sensitivity...")
s2020 = qualified[qualified["season"]==2020][["pitcher_id","stuff_plus_v4"]].rename(columns={"stuff_plus_v4":"s2020"})
s2019 = qualified[qualified["season"]==2019][["pitcher_id","stuff_plus_v4"]].rename(columns={"stuff_plus_v4":"s2019"})
s2021 = qualified[qualified["season"]==2021][["pitcher_id","stuff_plus_v4"]].rename(columns={"stuff_plus_v4":"s2021"})
m1920 = s2020.merge(s2019, on="pitcher_id").dropna()
m2021 = s2020.merge(s2021, on="pitcher_id").dropna()
r_1920 = stats.pearsonr(m1920["s2020"],m1920["s2019"])[0] if len(m1920)>5 else np.nan
r_2021 = stats.pearsonr(m2021["s2020"],m2021["s2021"])[0] if len(m2021)>5 else np.nan
d2019 = (m1920["s2020"]-m1920["s2019"]).abs().mean() if len(m1920)>0 else np.nan
d2021 = (m2021["s2020"]-m2021["s2021"]).abs().mean() if len(m2021)>0 else np.nan
results["A6"] = {
    "r_2020_vs_2019": round(float(r_1920),4) if not np.isnan(r_1920) else None,
    "mean_abs_diff_2020_vs_2019_pts": round(float(d2019),4) if not np.isnan(d2019) else None,
    "n_2020_2019": int(len(m1920)),
    "r_2020_vs_2021": round(float(r_2021),4) if not np.isnan(r_2021) else None,
    "mean_abs_diff_2020_vs_2021_pts": round(float(d2021),4) if not np.isnan(d2021) else None,
    "n_2020_2021": int(len(m2021)),
    "basis": "qualified pitcher-seasons (2020:>=111 pitches)"
}
log(f"  2020 vs 2019: r={r_1920:.4f}, diff={d2019:.4f} pts (n={len(m1920)})")
log(f"  2020 vs 2021: r={r_2021:.4f}, diff={d2021:.4f} pts (n={len(m2021)})")

# A7
log("A7: Pitch-type checks...")
pt_stats = (stuff.merge(qualified[["pitcher_id","season"]].drop_duplicates(), on=["pitcher_id","season"])
    .groupby("pitch_type")
    .agg(n_pitchers=("pitcher_id","nunique"), n_pitches_sum=("n_pitches","sum"), mean_stuff=("stuff_plus_v4","mean"))
    .sort_values("n_pitches_sum", ascending=False).reset_index())
cu_row = pt_stats[pt_stats["pitch_type"]=="CU"]
cu_mean = float(cu_row["mean_stuff"].iloc[0]) if len(cu_row)>0 else None
loc_codes = set(pd.read_parquet(f"{BASE}/diagnostics/location_v3/v3_per_pitch.parquet")["pitch_type"].unique())
stuff_codes = set(stuff["pitch_type"].unique())
non_tracked = stuff_codes - loc_codes - {"PO","UN","EP"}
results["A7"] = {
    "pitch_type_mean_stuff": [{k: (round(float(v),2) if isinstance(v,float) else v) for k,v in row.items()}
                               for row in pt_stats[["pitch_type","mean_stuff","n_pitches_sum"]].to_dict(orient="records")],
    "cu_mean_stuff": round(cu_mean,2) if cu_mean else None,
    "cu_note": "below 100 average" if cu_mean and cu_mean < 100 else "near average",
    "stuff_pitch_codes": sorted(stuff_codes),
    "location_pitch_codes": sorted(loc_codes),
    "non_tracked_in_location": sorted(non_tracked),
    "note": "Tunneling+/Deception+ code populations: report only; full populations require live model re-scoring"
}
log(f"  CU mean: {cu_mean}, non-tracked in Location+: {non_tracked}")

out_path = f"{OUT}/part_a_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"Part A complete. Saved to {out_path}")
with open(LOG, "a") as f:
    f.write("PART_A_COMPLETE\n")
print("DONE")

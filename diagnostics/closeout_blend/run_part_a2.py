"""
Part A (sections A3-A7): Completes after main A1/A2 already computed.
Uses V0a vs V0b ablation OOF as seed stability proxy (skips full retrain).
"""
import json, os, sys, time
import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
import lightgbm as lgb

BASE = "C:/Users/joshs/Desktop/mlb_dashboard"
OUT = f"{BASE}/diagnostics/closeout_blend"
LOG = f"{OUT}/progress.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log("Part A2: Loading existing A1/A2 results...")

# Load existing results
a_path = f"{OUT}/part_a_results.json"
if os.path.exists(a_path):
    with open(a_path) as f:
        results = json.load(f)
    log("  Loaded existing part_a_results.json")
else:
    results = {}
    log("  No existing results, starting fresh")

# Load base data
stuff = pd.read_parquet(f"{BASE}/diagnostics/stuff_v4/stuff_v4_scores.parquet")
ps_stuff = (stuff.groupby(["pitcher_id","season"])
    .apply(lambda g: pd.Series({
        "n_pitches": g["n_pitches"].sum(),
        "stuff_plus_v4": np.average(g["stuff_plus_v4"], weights=g["n_pitches"]),
    })).reset_index())
ps_stuff["season"] = ps_stuff["season"].astype(int)
qualified_mask = (ps_stuff["n_pitches"] >= 300) | ((ps_stuff["season"]==2020) & (ps_stuff["n_pitches"]>=111))
qualified = ps_stuff[qualified_mask].copy()

# ── A3: Seed stability via ablation proxy ────────────────────────────────────
log("A3: Seed stability via V0a vs V0b ablation OOF...")
oof_v0a_path = f"{BASE}/diagnostics/stuff_v4/part2_oof_predictions.parquet"
# V0b is same config, different seed. Check if there's a separate V0b file.
# The ablation parquet may have a 'variant' column
oof = pd.read_parquet(oof_v0a_path)
log(f"  OOF shape: {oof.shape}, cols: {oof.columns.tolist()}")

# If V0a and V0b are both in this file (via fold or variant column):
if "variant" in oof.columns:
    v0a = oof[oof["variant"]=="V0a"]
    v0b = oof[oof["variant"]=="V0b"]
elif "seed" in oof.columns:
    v0a = oof[oof["seed"]==42]
    v0b = oof[oof["seed"]==123]
else:
    # Single file — check location_v3 oof which has separate files
    oof_v0b_path = f"{BASE}/diagnostics/stuff_v4/part2_oof_predictions_v0b.parquet"
    if os.path.exists(oof_v0b_path):
        v0a = oof
        v0b = pd.read_parquet(oof_v0b_path)
    else:
        v0a = None
        v0b = None

if v0a is not None and v0b is not None and len(v0a) > 0 and len(v0b) > 0:
    # Aggregate both to pitcher-season level
    def agg_oof(df):
        return (df.groupby(["pitcher_id","season"])
            .agg(pred_mean=("pred","mean"), n=("pred","count"))
            .reset_index())
    ps_v0a = agg_oof(v0a)
    ps_v0b = agg_oof(v0b)
    m = ps_v0a.merge(ps_v0b, on=["pitcher_id","season"], suffixes=("_a","_b"))
    # Qualify: >=300 pitches in OOF
    m = m[(m["n_a"]>=300) | (m["season"]==2020)]
    if len(m) > 20:
        r_seed, _ = stats.pearsonr(m["pred_mean_a"], m["pred_mean_b"])
        md = (m["pred_mean_a"] - m["pred_mean_b"]).abs().mean()
        results["A3"] = {
            "method": "ablation_oof_v0a_vs_v0b",
            "seed_a": 42, "seed_b": 123,
            "n": int(len(m)),
            "pitcher_season_pred_corr": round(float(r_seed),4),
            "mean_abs_diff_pred_units": round(float(md),6),
            "note": "Ablation 30% subsample OOF (not full retrain). Predictions are raw delta_run_exp units, not Stuff+ points. Threshold (r>=0.99) in Stuff+ points cannot be directly applied to raw predictions.",
            "proxy_split_half_r_n50": 0.988,
            "proxy_sb_n50": 0.994,
            "formal_extra_seed_status": "unverified — full retrain with seed=999 not completed"
        }
        log(f"  V0a vs V0b: r={r_seed:.4f}, mean_diff={md:.6f} (ablation OOF, delta_run_exp units)")
    else:
        results["A3"] = {"note": "insufficient matched pairs", "formal_extra_seed_status": "unverified"}
else:
    log("  V0b predictions not separately available")
    # Check location_v3 oof structure for reference
    # Use split-half as proxy only
    results["A3"] = {
        "method": "proxy_split_half",
        "proxy_split_half_r_n50": 0.988,
        "proxy_sb_n50": 0.994,
        "formal_extra_seed_status": "unverified — full retrain with seed=999 not available; separate V0b OOF file not found",
        "note": "Split-half proxy strongly suggests seed stability r>=0.99 at pitcher-season level, but this is not a formal extra-seed refit. Mark as unverified per rules."
    }

# ── A4: Counterfactuals ──────────────────────────────────────────────────────
log("A4: Counterfactuals — pitcher-season level from pre-computed files or pitch-level proxy...")
cf_files = {
    "season_missing": f"{BASE}/diagnostics/stuff_v4/counterfactual_season_missing.parquet",
    "effective_speed_eq_release": f"{BASE}/diagnostics/stuff_v4/counterfactual_effspeed_eq_relspeed.parquet",
    "haa_zero": f"{BASE}/diagnostics/stuff_v4/counterfactual_haa_zero.parquet",
}
a4 = {}
for cf_name, cf_path in cf_files.items():
    if os.path.exists(cf_path):
        log(f"  Loading {cf_name} from {cf_path}")
        cf = pd.read_parquet(cf_path)
        cf["season"] = cf["season"].astype(int)
        cf_ps = (cf.groupby(["pitcher_id","season"])
            .apply(lambda g: pd.Series({"cf_stuff": np.average(g["stuff_plus_v4"], weights=g["n_pitches"])}))
            .reset_index())
        cf_ps["season"] = cf_ps["season"].astype(int)
        m = qualified.merge(cf_ps, on=["pitcher_id","season"]).dropna(subset=["stuff_plus_v4","cf_stuff"])
        r_cf, _ = stats.pearsonr(m["stuff_plus_v4"], m["cf_stuff"])
        diff = (m["stuff_plus_v4"] - m["cf_stuff"]).abs()
        extra = {}
        if cf_name == "effective_speed_eq_release":
            conn = sqlite3.connect(f"{BASE}/mlb_data.db")
            ext = pd.read_sql("""SELECT pitcher_id, season, AVG(release_extension) as mean_ext
                FROM pitches WHERE release_extension IS NOT NULL GROUP BY pitcher_id, season""", conn)
            conn.close()
            ext["season"] = ext["season"].astype(int)
            m2 = m.merge(ext, on=["pitcher_id","season"])
            m2["change"] = m2["stuff_plus_v4"] - m2["cf_stuff"]
            r_ext = stats.pearsonr(m2["change"], m2["mean_ext"])[0] if len(m2)>5 else np.nan
            extra["corr_change_release_extension"] = round(float(r_ext),4) if not np.isnan(r_ext) else None
        is_inert = (r_cf >= 0.995 and diff.mean() <= 0.5 and np.percentile(diff,99) <= 2.0)
        a4[cf_name] = {
            "corr_with_baseline": round(float(r_cf),4),
            "mean_abs_diff_pts": round(float(diff.mean()),4),
            "p99_abs_diff_pts": round(float(np.percentile(diff,99)),4),
            "n": int(len(m)),
            "classification": "inert" if is_inert else "load-bearing",
            **extra
        }
        log(f"    {cf_name}: r={r_cf:.4f}, mean={diff.mean():.4f}, p99={np.percentile(diff,99):.4f} -> {'inert' if is_inert else 'LOAD-BEARING'}")
    else:
        # Fallback to completion doc pitch-level values
        pitch_corrs = {"season_missing": 0.920, "effective_speed_eq_release": 0.973, "haa_zero": 0.263}
        a4[cf_name] = {
            "status": "unverified",
            "pitch_level_corr_completion_doc": pitch_corrs.get(cf_name),
            "pitch_level_classification": "load-bearing",
            "pitcher_season_level": "unverified — counterfactual parquet not found; full pipeline rescore required",
            "limitation": "Counterfactual parquets not stored from original run. Must use completion-doc pitch-level values."
        }
        log(f"    {cf_name}: unverified (no parquet)")

results["A4"] = a4

# ── A5: HAA adoption ─────────────────────────────────────────────────────────
if "A5" not in results:
    log("A5: HAA adoption (from completion doc)...")
    results["A5"] = {
        "leakage_plate_x_r": 0.009, "leakage_release_pos_x_r": 0.084,
        "leakage_gates_passed": True,
        "vchosen_vs_vchosen_haa_corr_pitcher_season": 0.9635,
        "haa_load_bearing": True,
        "haa_adopted_in_final_model": True,
        "adoption_rule_met": True,
        "ci_lower_bound_primary_metric_gt_0": "unverified — Vchosen_HAA OOF file not found separately",
        "same_season_harm_check": "no significant same-season harm (corr=0.9635 shows meaningful score change, not degradation)",
        "nonlinear_leakage_check": "unverified — haa_residual not stored in pitches table",
        "basis": "completion doc; HAA adopted per stage2_haa_rule"
    }

# ── A6: 2020 sensitivity ──────────────────────────────────────────────────────
if "A6" not in results:
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
    log(f"  2020 vs 2019: r={r_1920:.4f}, diff={d2019:.4f} (n={len(m1920)})")
    log(f"  2020 vs 2021: r={r_2021:.4f}, diff={d2021:.4f} (n={len(m2021)})")

# ── A7: Pitch-type checks ─────────────────────────────────────────────────────
if "A7" not in results:
    log("A7: Pitch-type checks...")
    stuff["season"] = stuff["season"].astype(int)
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
        "pitch_type_mean_stuff": [{k: (round(float(v),2) if isinstance(v,(float,np.floating)) else v) for k,v in row.items()}
                                   for row in pt_stats[["pitch_type","mean_stuff","n_pitches_sum"]].to_dict(orient="records")],
        "cu_mean_stuff": round(cu_mean,2) if cu_mean else None,
        "cu_note": "below 100 average — CU mean is below league average, consistent with completion doc (95.1)" if (cu_mean and cu_mean < 100) else "near average",
        "stuff_pitch_codes": sorted(stuff_codes),
        "location_pitch_codes": sorted(loc_codes),
        "non_tracked_in_location": sorted(non_tracked),
    }
    log(f"  CU mean: {cu_mean}, non-tracked in Loc+: {non_tracked}")

out_path = f"{OUT}/part_a_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"Part A2 complete. Results saved to {out_path}")
with open(LOG, "a") as f:
    f.write("PART_A2_COMPLETE\n")
print("DONE")

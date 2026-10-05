"""Part B: fix B8 and complete"""
import json, os, time
import numpy as np
import pandas as pd
import sqlite3
from scipy import stats

BASE = "C:/Users/joshs/Desktop/mlb_dashboard"
OUT = f"{BASE}/diagnostics/closeout_blend"
LOG = f"{OUT}/progress.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Load partial results from previous run (B1-B7 mostly done)
b_path = f"{OUT}/part_b_results.json"
if os.path.exists(b_path):
    with open(b_path) as f:
        results = json.load(f)
    log("Loaded partial Part B results")
else:
    results = {}

# Add B1 ECE correction note
if "B1" not in results:
    results["B1"] = {}
results["B1"]["methodology_note"] = ("Pitcher-level ECE computed from OOF mean predictions vs actual rates. "
    "xCS and xFoul ECE values are inflated by comparison mismatch: model predicts P(CS) over all pitches, "
    "actual rate computed on takes only. Pitch-level ECE cannot be computed without pitch IDs in OOF. "
    "True pitch-level ECE is unverified. Values below are pitcher-level proxies, not formal pitch-level ECE.")
results["B1"]["ECE_xSwing_pitcher_level"] = 0.1761
results["B1"]["ECE_xCS_pitcher_level_inflated"] = 0.3096
results["B1"]["ECE_xWhiff_pitcher_level"] = 0.0126
results["B1"]["ECE_xFoul_pitcher_level_inflated"] = 0.3735
results["B1"]["formal_pitch_level_ECE"] = "unverified — OOF parquets lack pitch IDs for exact match"
results["B1"]["gate_status"] = ("CANNOT VERIFY — pitcher-level proxy shows xSwing ECE=0.176 and xWhiff ECE=0.013 "
    "but xCS and xFoul values are methodologically contaminated. Completion doc: 'ASSUMED PASS'. "
    "Formal computation requires pitch-level OOF with pitch IDs.")

# Fix B5 flagging
if "B5" in results:
    results["B5"]["flag_loc_vs_tunneling"] = (
        "FLAGGED: |r| = 0.58 > 0.5 threshold between Location+ and Tunneling+. "
        "Indicates moderate overlap in what these two axes measure. "
        "Partial correlation analysis needed to understand conditional independence.")

# B8: Complete with fix
log("B8: Pitch-type grades (fix)...")
v3_pp = pd.read_parquet(f"{BASE}/diagnostics/location_v3/v3_per_pitch.parquet")
v3_pp["season"] = v3_pp["season"].astype(int)

league_ref = (v3_pp.groupby("season")
    .agg(league_mean_xrv=("xrv","mean"), league_std_xrv=("xrv","std")).reset_index())

pt_grades = (v3_pp.groupby(["pitcher_id","season","pitch_type"])
    .agg(n_pitches_type=("xrv","count"), mean_xrv_type=("xrv","mean")).reset_index())
pt_grades = pt_grades[pt_grades["n_pitches_type"] >= 100]
pt_grades = pt_grades.merge(league_ref, on="season", how="left")
pt_grades["pt_grade_raw"] = (pt_grades["mean_xrv_type"] - pt_grades["league_mean_xrv"]) / pt_grades["league_std_xrv"].clip(lower=1e-6)

# Sign check: use loc_scores vs per-pitcher xrv aggregates
loc_scores = pd.read_parquet(f"{BASE}/diagnostics/location_v3/location_v3_scores.parquet")
loc_scores["season"] = loc_scores["season"].astype(int)

xrv_ps = v3_pp.groupby(["pitcher_id","season"]).agg(mean_xrv_ps=("xrv","mean")).reset_index()
loc_check = loc_scores[loc_scores["season"]==2023].merge(
    xrv_ps[xrv_ps["season"]==2023], on=["pitcher_id","season"])
if len(loc_check) > 10 and "mean_xrv_ps" in loc_check.columns:
    sign_r = stats.pearsonr(loc_check["score"], loc_check["mean_xrv_ps"])[0]
    log(f"  Sign check r={sign_r:.4f}")
    if sign_r < 0:
        pt_grades["location_plus_pt_grade"] = 100 - pt_grades["pt_grade_raw"] * 10
    else:
        pt_grades["location_plus_pt_grade"] = 100 + pt_grades["pt_grade_raw"] * 10
else:
    log("  Sign check failed, using inverted (lower xrv = better location)")
    pt_grades["location_plus_pt_grade"] = 100 - pt_grades["pt_grade_raw"] * 10

pt_path = f"{OUT}/location_v3_pt_grades.parquet"
pt_grades[["pitcher_id","season","pitch_type","n_pitches_type","mean_xrv_type","location_plus_pt_grade"]].to_parquet(pt_path, index=False)
log(f"  Saved {len(pt_grades):,} pt-grade rows")

# Components from V0a OOF
oof_path = f"{BASE}/diagnostics/location_v3/oof_V0a.parquet"
if os.path.exists(oof_path):
    oof = pd.read_parquet(oof_path)
    oof["season"] = oof["season"].astype(int)
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
        "components_source": "OOF V0a (30% pitcher subsample)",
        "min_pitches_per_type": 100, "shrinkage_applied": False,
        "note": "Pitch-type grades use same z-score standardization as overall score. No shrinkage (not adopted per spec)."
    }
else:
    results["B8"] = {"pt_grades_path": pt_path, "pt_grades_n": int(len(pt_grades))}

# Add B2/B3 from progress log (already computed)
if "B2" not in results:
    results["B2"] = {
        "next_season_correlations": {
            "BB%": {"r": -0.4335, "n": 3527},
            "xFIP": {"r": -0.1053, "n": 1956}
        },
        "incremental_delta_r2": {
            "BB%": {"delta_r2": 0.0188, "ci_95_lo": 0.0117, "ci_95_hi": 0.0278,
                    "bar_point_met": True, "bar_ci_lo_met": True, "n": 3527}
        },
        "pre_reg_bar_bb_corr_final_model": "<= -0.20",
        "bb_corr_final_model_bar_met": True,
        "note_pitcher_holdout_analog": "unverified"
    }
if "B3" not in results:
    results["B3"] = {"pitcher_holdout_yoy_xrv": 0.5859, "n_yoy_pairs": 987}
if "B5" not in results:
    results["B5"] = {
        "location_plus_v3_vs_stuff_plus_v4": {"r": -0.054, "flag_gt_0.5": False},
        "location_plus_v3_vs_tunneling_plus": {"r": -0.5798, "flag_gt_0.5": True},
        "stuff_plus_v4_vs_tunneling_plus": {"r": -0.108, "flag_gt_0.5": False}
    }

with open(b_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
log("Part B complete")
with open(LOG, "a") as f:
    f.write("PART_B_COMPLETE\n")
print("DONE")

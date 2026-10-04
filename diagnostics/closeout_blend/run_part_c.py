"""Part C: The blend test"""
import json, os, sys, time
import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

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

def pitcher_clustered_bootstrap_ci(y_true, y_pred, y_pred_base, pitcher_ids, n_boot=1000, seed=42):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    y_pred_base = np.asarray(y_pred_base, float)
    pids = np.asarray(pitcher_ids)
    obs_r2 = r2_score(y_true, y_pred)
    obs_r2_base = r2_score(y_true, y_pred_base)
    obs_delta = obs_r2 - obs_r2_base
    rng = np.random.default_rng(seed)
    unique_p = np.unique(pids)
    p2idx = {p: np.where(pids == p)[0] for p in unique_p}
    deltas = []
    for _ in range(n_boot):
        boot_p = rng.choice(unique_p, len(unique_p), replace=True)
        idx = np.concatenate([p2idx[p] for p in boot_p])
        try:
            d = r2_score(y_true[idx], y_pred[idx]) - r2_score(y_true[idx], y_pred_base[idx])
            deltas.append(d)
        except: pass
    deltas = np.array(deltas)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return obs_delta, obs_r2_base, obs_r2, ci_lo, ci_hi

log("Loading all axis scores...")
stuff = pd.read_parquet(f"{BASE}/diagnostics/stuff_v4/stuff_v4_scores.parquet")
ps_stuff = (stuff.groupby(["pitcher_id","season"])
    .apply(lambda g: pd.Series({
        "n_pitches": g["n_pitches"].sum(),
        "stuff_plus_v4": np.average(g["stuff_plus_v4"], weights=g["n_pitches"]),
    })).reset_index())
ps_stuff["season"] = ps_stuff["season"].astype(int)

loc_scores = pd.read_parquet(f"{BASE}/diagnostics/location_v3/location_v3_scores.parquet")
loc_scores["season"] = loc_scores["season"].astype(int)

conn = sqlite3.connect(f"{BASE}/mlb_data.db")
model_scores = pd.read_sql("SELECT pitcher_id, season, tunneling_plus, location_plus FROM pitcher_model_scores WHERE stuff_plus_v4 IS NOT NULL OR tunneling_plus IS NOT NULL", conn)
outcomes = pd.read_sql("SELECT player_id as pitcher_id, season, xfip, era FROM player_advanced_stats WHERE (xfip IS NOT NULL OR era IS NOT NULL)", conn)
pitch_counts = pd.read_sql("""
    SELECT pitcher_id, season, COUNT(*) as n_pitches_db,
           SUM(CASE WHEN events='walk' THEN 1 ELSE 0 END) as bb,
           SUM(CASE WHEN events IS NOT NULL AND events!='' THEN 1 ELSE 0 END) as pa,
           SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) as k,
           COUNT(DISTINCT CASE WHEN inning=1 AND at_bat_number<=3 THEN game_id END) as approx_starts,
           COUNT(DISTINCT game_id) as total_games
    FROM pitches WHERE season >= 2016 GROUP BY pitcher_id, season
""", conn)
conn.close()

model_scores["season"] = model_scores["season"].astype(int)
outcomes["season"] = outcomes["season"].astype(int)
pitch_counts["season"] = pitch_counts["season"].astype(int)
pitch_counts["bb_pct"] = pitch_counts["bb"] / pitch_counts["pa"].clip(lower=1)
pitch_counts["k_bb_pct"] = (pitch_counts["k"] - pitch_counts["bb"]) / pitch_counts["pa"].clip(lower=1)
pitch_counts["starter"] = (pitch_counts["approx_starts"] / pitch_counts["total_games"].clip(lower=1) >= 0.5).astype(int)
pitch_counts["log_pitches"] = np.log(pitch_counts["n_pitches_db"].clip(lower=1))

main_df = (ps_stuff
    .merge(loc_scores[["pitcher_id","season","score"]].rename(columns={"score":"location_plus_v3"}),
           on=["pitcher_id","season"], how="inner")
    .merge(model_scores[["pitcher_id","season","tunneling_plus"]], on=["pitcher_id","season"], how="left")
    .merge(pitch_counts[["pitcher_id","season","n_pitches_db","bb_pct","k_bb_pct","starter","log_pitches"]],
           on=["pitcher_id","season"], how="left")
    .merge(outcomes, on=["pitcher_id","season"], how="left"))

main_df["qualified"] = main_df["n_pitches"] >= 300
main_df.loc[main_df["season"]==2020, "qualified"] = main_df.loc[main_df["season"]==2020, "n_pitches"] >= 111
qdf = main_df[main_df["qualified"]].copy()
log(f"Qualified: {len(qdf)}, seasons: {sorted(qdf['season'].unique())}")

tun_meta = json.load(open(f"{BASE}/models/tunneling_v3_meta.json")) if os.path.exists(f"{BASE}/models/tunneling_v3_meta.json") else {}
dec_meta = json.load(open(f"{BASE}/models/deception_plus_v2_meta.json")) if os.path.exists(f"{BASE}/models/deception_plus_v2_meta.json") else {}
log(f"Tunneling+ hash: {tun_meta.get('model_hash','not found')}")
log(f"Deception+ hash: {dec_meta.get('model_hash','not found')}")

controls = ["log_pitches", "starter"]
test_pairs = [2022, 2023, 2024, 2025]

def run_oos(qdf, features, y_col, test_pairs):
    """Rolling-origin OOS evaluation. features are predictors in S, y_col is outcome in S+1."""
    seasons = sorted(qdf["season"].unique())
    all_preds = []
    for test_s in test_pairs:
        test_next = test_s + 1
        if test_next not in seasons: continue
        # Training pairs: all earlier consecutive non-2020 pairs
        train_rows = []
        for s in seasons:
            if s >= test_s or s == 2020 or s+1 == 2020: continue
            if s+1 not in seasons: continue
            # Get S features + S outcome (if y_col is last-season-outcome control)
            cols_s = list(set(["pitcher_id"] + features + [y_col]))
            ds = qdf[qdf["season"]==s][cols_s].dropna().copy()
            # Outcome is S+1 y_col
            ds_next = qdf[qdf["season"]==s+1][["pitcher_id", y_col]].dropna().rename(columns={y_col: y_col+"_next"})
            m = ds.merge(ds_next, on="pitcher_id")
            train_rows.append(m)
        if not train_rows: continue
        df_tr = pd.concat(train_rows).dropna(subset=features + [y_col+"_next"])

        # Test
        cols_s = list(set(["pitcher_id"] + features + [y_col]))
        ds_t = qdf[qdf["season"]==test_s][cols_s].dropna().copy()
        dn_t = qdf[qdf["season"]==test_next][["pitcher_id", y_col]].dropna().rename(columns={y_col: y_col+"_next"})
        df_te = ds_t.merge(dn_t, on="pitcher_id").dropna(subset=features + [y_col+"_next"])
        if len(df_tr) < 50 or len(df_te) < 20: continue

        X_tr = df_tr[features].values
        y_tr = df_tr[y_col+"_next"].values  # train on S+1 outcome
        # Wait — we need to train on S features to predict S+1 outcome
        # But y_col in features is S outcome (control)
        # y_col+"_next" is what we predict
        X_te = df_te[features].values
        y_te = df_te[y_col+"_next"].values

        # For M0: features = controls + [y_col], y_col is last-season outcome
        # Need to retrain on training pairs where we predict next-season from current-season
        # The train target should be y_col_next (next-season outcome)
        # but y_col is in features as last-season outcome control
        # So: fit model on X=features, y=y_col_next
        # This is correct!
        try:
            lr = LinearRegression().fit(X_tr, y_tr)
            y_pred = lr.predict(X_te)
            all_preds.append(pd.DataFrame({
                "pitcher_id": df_te["pitcher_id"].values.ravel(),
                "y_true": y_te.ravel(),
                "y_pred": y_pred.ravel(),
                "test_s": test_s,
            }))
        except Exception as e:
            log(f"    Error fitting {e}")
    return pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

results = {}
all_oos = {}

for outcome_col, out_lbl in [("xfip","xFIP"), ("era","ERA"), ("k_bb_pct","K-BB%")]:
    log(f"  Outcome: {out_lbl}")
    qdf_c = qdf.copy()
    qdf_c["stuff_x_loc"] = qdf_c["stuff_plus_v4"] * qdf_c["location_plus_v3"]

    has_tun = qdf["tunneling_plus"].notna().sum() > 100
    has_dec = "deception_plus" in qdf.columns and qdf["deception_plus"].notna().sum() > 100

    model_defs = {
        "M0": controls + [outcome_col],
        "M1": controls + ["stuff_plus_v4"],
        "M2": controls + ["location_plus_v3"],
        "M3": controls + ["stuff_plus_v4","location_plus_v3"],
        "M4": controls + [outcome_col,"stuff_plus_v4"],
        "M5": controls + [outcome_col,"location_plus_v3"],
        "M6": controls + [outcome_col,"stuff_plus_v4","location_plus_v3"],
        "M7": controls + [outcome_col,"stuff_plus_v4","location_plus_v3","stuff_x_loc"],
    }
    if has_tun:
        model_defs["M8"] = controls + [outcome_col,"stuff_plus_v4","location_plus_v3","tunneling_plus"]
    if has_dec:
        model_defs["M9"] = controls + [outcome_col,"stuff_plus_v4","location_plus_v3","deception_plus"]
    if has_tun and has_dec:
        model_defs["M10"] = controls + [outcome_col,"stuff_plus_v4","location_plus_v3","tunneling_plus","deception_plus"]

    oos_results = {}
    oos_preds_cache = {}
    for mname, feats in model_defs.items():
        df_use = qdf_c if "stuff_x_loc" in feats else qdf
        preds = run_oos(df_use, feats, outcome_col, test_pairs)
        if len(preds) == 0:
            oos_results[mname] = {"n": 0, "note": "no predictions"}
            continue
        r2 = r2_score(preds["y_true"], preds["y_pred"])
        rmse = np.sqrt(mean_squared_error(preds["y_true"], preds["y_pred"]))
        oos_results[mname] = {"r2_oos": round(float(r2),4), "rmse_oos": round(float(rmse),4), "n": int(len(preds))}
        oos_preds_cache[mname] = preds
        log(f"    {mname}: R²={r2:.4f}, RMSE={rmse:.4f} (n={len(preds)})")

    # CIs for key comparisons
    def get_ci(m, base):
        if m not in oos_preds_cache or base not in oos_preds_cache: return None
        pm = oos_preds_cache[m]
        pb = oos_preds_cache[base]
        merged = pm.merge(pb[["pitcher_id","test_s","y_true","y_pred"]].rename(columns={"y_pred":"y_base"}),
                          on=["pitcher_id","test_s","y_true"])
        if len(merged) < 20: return None
        delta, r2b, r2m, ci_lo, ci_hi = pitcher_clustered_bootstrap_ci(
            merged["y_true"], merged["y_pred"], merged["y_base"], merged["pitcher_id"])
        return {"delta_r2": round(float(delta),4), "ci_95_lo": round(float(ci_lo),4),
                "ci_95_hi": round(float(ci_hi),4), "n": int(len(merged)),
                "materiality_met": bool(ci_lo > 0 and delta >= 0.005)}

    ci_m3_m1 = get_ci("M3","M1")
    ci_m3_m2 = get_ci("M3","M2")
    ci_m6_m0 = get_ci("M6","M0")
    ci_m8_m6 = get_ci("M8","M6") if has_tun else None
    ci_m9_m6 = get_ci("M9","M6") if has_dec else None

    cond1 = bool(ci_m3_m1 and ci_m3_m1.get("materiality_met") and ci_m3_m2 and ci_m3_m2.get("materiality_met"))
    cond2 = bool(ci_m6_m0 and ci_m6_m0.get("materiality_met"))
    decision = "A" if (cond1 and cond2) else "B" if cond2 else "C"

    # In-sample M3 coefficients (standardized)
    rows = []
    for s in sorted(qdf["season"].unique()):
        if s==2020 or s+1==2020 or s+1 not in sorted(qdf["season"].unique()): continue
        ds = qdf[qdf["season"]==s][["pitcher_id","stuff_plus_v4","location_plus_v3","log_pitches","starter",outcome_col]].dropna()
        dn = qdf[qdf["season"]==s+1][["pitcher_id",outcome_col]].dropna().rename(columns={outcome_col:outcome_col+"_next"})
        rows.append(ds.merge(dn, on="pitcher_id"))
    m3_coef = {}
    if rows:
        df_all = pd.concat(rows).dropna()
        feats_m3 = ["stuff_plus_v4","location_plus_v3","log_pitches","starter"]
        for f in feats_m3:
            df_all[f+"_std"] = (df_all[f] - df_all[f].mean()) / (df_all[f].std() or 1.0)
        Xs = df_all[[f+"_std" for f in feats_m3]].values
        lr3 = LinearRegression().fit(Xs, df_all[outcome_col+"_next"].values)
        m3_coef = {f: round(float(c),4) for f,c in zip(feats_m3, lr3.coef_)}
        m3_coef["intercept"] = round(float(lr3.intercept_),4)
        m3_coef["in_sample_r2"] = round(float(r2_score(df_all[outcome_col+"_next"].values, lr3.predict(Xs))),4)

    all_oos[out_lbl] = {
        "model_table": oos_results,
        "ci_m3_vs_m1": ci_m3_m1, "ci_m3_vs_m2": ci_m3_m2,
        "ci_m6_vs_m0": ci_m6_m0,
        "ci_m8_vs_m6": ci_m8_m6, "ci_m9_vs_m6": ci_m9_m6,
        "condition_1_met": cond1, "condition_2_met": cond2,
        "decision": decision,
        "tunneling_predictive": bool(ci_m8_m6 and ci_m8_m6.get("materiality_met")),
        "deception_predictive": bool(ci_m9_m6 and ci_m9_m6.get("materiality_met")),
        "m3_insample_coefficients_standardized": m3_coef,
        "n_tunneling_available": int(qdf["tunneling_plus"].notna().sum()) if has_tun else 0
    }
    log(f"  {out_lbl}: Cond1={cond1}, Cond2={cond2} -> Decision={decision}")

results["models"] = all_oos
primary = all_oos.get("xFIP",{})
results["primary_decision"] = primary.get("decision","C")
results["tunneling_verdict"] = "predictive" if primary.get("tunneling_predictive") else "descriptive"
results["deception_verdict"] = "predictive" if primary.get("deception_predictive") else "descriptive"
results["limitation"] = ("Stuff+ and Location+ scored by models trained on overlapping seasons. "
    "Per-pitch R² ~0.001-0.003; negligible effect on pitcher-season regression coefficients. "
    "Acknowledged per addendum.")

# Sensitivity at 500 pitches
log("Sensitivity at 500 pitches...")
qdf_500 = main_df[main_df["n_pitches"] >= 500].copy()
sense = {}
for outcome_col, out_lbl in [("xfip","xFIP")]:
    p_m6 = run_oos(qdf_500, controls+[outcome_col,"stuff_plus_v4","location_plus_v3"], outcome_col, test_pairs)
    p_m0 = run_oos(qdf_500, controls+[outcome_col], outcome_col, test_pairs)
    if len(p_m6)>0 and len(p_m0)>0:
        sense[out_lbl] = {
            "M0_r2": round(float(r2_score(p_m0["y_true"],p_m0["y_pred"])),4),
            "M6_r2": round(float(r2_score(p_m6["y_true"],p_m6["y_pred"])),4),
            "n_500": int(len(qdf_500))
        }
        log(f"  500-pitch {out_lbl}: M0={sense[out_lbl]['M0_r2']:.4f}, M6={sense[out_lbl]['M6_r2']:.4f}")
results["sensitivity_500"] = sense

# Quadrant table
log("Building quadrant table...")
qdf_q = qdf.dropna(subset=["stuff_plus_v4","location_plus_v3"]).copy()
for col in ["stuff_plus_v4","location_plus_v3"]:
    qdf_q[col+"_t"] = pd.qcut(qdf_q[col], 3, labels=["Low","Mid","High"])

# Attach next-season xFIP
seasons = sorted(qdf_q["season"].unique())
qdf_q["xfip_next"] = np.nan
for s in seasons:
    if s==2020 or s+1==2020 or s+1 not in seasons: continue
    dn = qdf[qdf["season"]==s+1][["pitcher_id","xfip"]].dropna().rename(columns={"xfip":"xfip_n"})
    mask = qdf_q["season"]==s
    temp = qdf_q[mask].merge(dn, on="pitcher_id", how="left")
    qdf_q.loc[mask, "xfip_next"] = temp["xfip_n"].values

quads = (qdf_q.dropna(subset=["stuff_plus_v4_t","location_plus_v3_t"])
    .groupby(["stuff_plus_v4_t","location_plus_v3_t"], observed=True)
    .agg(n=("pitcher_id","count"),
         mean_stuff=("stuff_plus_v4","mean"),
         mean_loc=("location_plus_v3","mean"),
         mean_xfip=("xfip","mean"),
         mean_xfip_next=("xfip_next","mean"))
    .reset_index())
quads_path = f"{OUT}/quadrant_table.csv"
quads.to_csv(quads_path, index=False)
log(f"  Quadrant table: {len(quads)} cells")

results["quadrant_table_path"] = quads_path

# Decision JSON
decision_json = {
    "primary_outcome": "xFIP",
    "decision": results["primary_decision"],
    "tunneling_verdict": results["tunneling_verdict"],
    "deception_verdict": results["deception_verdict"],
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "pre_registered_rules": {
        "decision_A": "Both conditions met: blended Pitching+ earns a score",
        "decision_B": "Only Condition 2 met: axes add beyond last season; headline is quadrant view",
        "decision_C": "Neither condition met: both axes are descriptive"
    },
    "per_outcome": {out: {"decision": d.get("decision"), "cond1": d.get("condition_1_met"), "cond2": d.get("condition_2_met")}
                    for out, d in all_oos.items()}
}
with open(f"{OUT}/decision.json", "w") as f:
    json.dump(decision_json, f, indent=2, default=str)

out_path = f"{OUT}/part_c_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
log(f"Part C complete. Decision: {results['primary_decision']}")
with open(LOG, "a") as f:
    f.write("PART_C_COMPLETE\n")
print("DONE")

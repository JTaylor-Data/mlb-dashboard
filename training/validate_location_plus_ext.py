"""
Location+ v1 - Part D validation battery (reliability incl. a direct
Stuff+ comparison run on identical infrastructure, year-over-year,
cross-model correlation, fastest_fb_velo sanity check, leaderboard). The
count/pitch-type interaction sanity check (Part D.5) and the stand-ablation
(Part E.5) are separate scripts - see location_count_interaction_check.py
and location_stand_ablation.py. Out-of-sample predictive validity (Part
D.4) is also separate (location_predictive_validity.py) since it needs its
own model trained without 2024.

Usage:
    python training/validate_location_model.py
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import analytics.stuff_model as stuff_sm
import analytics.deception_model as deception_sm
import analytics.tunneling_v3_model as tunneling_sm
from analytics.location_features import shrinkage_params, shrunk_group_means, to_plus_scale

ARTIFACTS_DIR = Path("training/location_plus_ext_artifacts")
DB_PATH = "mlb_data.db"
SEASON = 2025  # most recent completed season
MIN_PITCHES_QUALIFY = 100
RELIABILITY_THRESHOLDS = [50, 100, 200, 400, 600]  # per-HALF thresholds, same convention as Deception+'s own split-half check


def _overall_split_half(raw_preds, season, seed=42):
    """Splits each pitcher's pitches for `season` into two random halves,
    computes the OVERALL (all pitch types pooled, unshrunk) plus-scale mean
    per half. Deliberately unshrunk and pooled across pitch types - this is
    testing whether the model's raw signal is stable, not how the shrunk
    display number behaves (shrinkage would only entangle the reliability
    test with the K's own overlap with these same thresholds). Same
    methodology as train_deception_model.py's split_half_reliability,
    generalized to work on either model's raw predictions."""
    df = raw_preds[raw_preds["season"].astype(str) == str(season)].copy()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["half"] = df.index % 2
    mean_, std_ = df["pred"].mean(), df["pred"].std()
    halves = {}
    for half in (0, 1):
        sub = df[df["half"] == half]
        g = sub.groupby("pitcher_id").agg(mean_pred=("pred", "mean"), n=("pred", "size"))
        g["score"] = to_plus_scale(g["mean_pred"], mean_, std_)
        halves[half] = g[["score", "n"]]
    return halves[0].join(halves[1], lsuffix="_0", rsuffix="_1", how="inner")


def reliability_curve(merged, thresholds):
    results = {}
    for min_p in thresholds:
        sub = merged[(merged.n_0 >= min_p) & (merged.n_1 >= min_p)]
        results[str(min_p)] = dict(
            n=int(len(sub)),
            correlation=float(sub["score_0"].corr(sub["score_1"])) if len(sub) > 2 else None,
        )
    return results


def first_threshold_crossing(curve, bar):
    for min_p in RELIABILITY_THRESHOLDS:
        r = curve[str(min_p)]["correlation"]
        if r is not None and r >= bar:
            return min_p
    return None


def official_location_plus(raw, k_by_type, fallback_k, type_means, season):
    """The 'real' displayed score: shrink each (pitcher, pitch_type) mean
    toward its own pitch type's league mean (same method as Stuff+'s
    compute_stuff_plus), then usage-weight-average across pitch types to
    one overall per-pitcher-season number. `raw` needs pitcher_id, season,
    pitch_type, pred."""
    season_raw = raw[raw["season"].astype(str) == str(season)]
    shrunk = shrunk_group_means(season_raw, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
    shrunk["score"] = to_plus_scale(shrunk["shrunk_mean"], season_raw["pred"].mean(), season_raw["pred"].std())

    def _wavg(g):
        return pd.Series({
            "score": np.average(g["score"], weights=g["n_pitches"]),
            "total_pitches": g["n_pitches"].sum(),
        })
    overall = shrunk.groupby("pitcher_id").apply(_wavg, include_groups=False).reset_index()
    overall["season"] = str(season)
    return overall


def year_over_year(all_scores):
    q = all_scores[all_scores["total_pitches"] >= MIN_PITCHES_QUALIFY]
    seasons = sorted(q["season"].unique())
    per_pair, pooled_t, pooled_t1 = [], [], []
    for i in range(len(seasons) - 1):
        t, t1 = seasons[i], seasons[i + 1]
        dt = q[q["season"] == t][["pitcher_id", "score"]].rename(columns={"score": "t"})
        dt1 = q[q["season"] == t1][["pitcher_id", "score"]].rename(columns={"score": "t1"})
        merged = dt.merge(dt1, on="pitcher_id", how="inner")
        if len(merged) < 5:
            continue
        r = merged["t"].corr(merged["t1"])
        per_pair.append({"season_t": t, "season_t1": t1, "n": len(merged), "r2": round(float(r ** 2), 3)})
        pooled_t.extend(merged["t"].tolist())
        pooled_t1.extend(merged["t1"].tolist())
    r = np.corrcoef(pooled_t, pooled_t1)[0, 1] if len(pooled_t) >= 5 else None
    return {"per_pair": per_pair, "pooled_r2": round(float(r ** 2), 3) if r is not None else None, "n_pooled": len(pooled_t)}


def cross_model_correlation(conn, loc_scores, season):
    q = loc_scores[(loc_scores["season"] == str(season)) & (loc_scores["total_pitches"] >= MIN_PITCHES_QUALIFY)]

    stuff = stuff_sm.compute_stuff_plus(conn, int(season)).reset_index()
    overall_stuff = stuff.groupby("pitcher_id").apply(
        lambda g: np.average(g["stuff_plus"], weights=g["n_pitches"]), include_groups=False
    ).rename("stuff_plus").reset_index()

    dec_raw = deception_sm._raw_predictions(conn, int(season))
    dec_mean, dec_std = dec_raw["pred"].mean(), dec_raw["pred"].std()
    dec = dec_raw.groupby("pitcher_id")["pred"].mean().reset_index()
    dec["deception_plus"] = deception_sm._to_plus_scale(dec["pred"], dec_mean, dec_std)

    tun_raw = tunneling_sm._raw_predictions(conn, int(season))
    tun_mean, tun_std = tun_raw["pred"].mean(), tun_raw["pred"].std()
    tun = tun_raw.groupby("pitcher_id")["pred"].mean().reset_index()
    tun["tunneling_plus"] = tunneling_sm._to_plus_scale(tun["pred"], tun_mean, tun_std)

    merged = (q.merge(overall_stuff, on="pitcher_id", how="inner")
                .merge(dec[["pitcher_id", "deception_plus"]], on="pitcher_id", how="inner")
                .merge(tun[["pitcher_id", "tunneling_plus"]], on="pitcher_id", how="inner"))
    return dict(
        n=int(len(merged)),
        r_stuff=round(float(merged["score"].corr(merged["stuff_plus"])), 3),
        r_deception=round(float(merged["score"].corr(merged["deception_plus"])), 3),
        r_tunneling=round(float(merged["score"].corr(merged["tunneling_plus"])), 3),
    )


def fastest_fb_velo_correlation(loc_scores, season):
    commit_table = pd.read_csv("project_artifacts/training_artifacts/tunneling_v2_artifacts/200ms/pitcher_season_commit_point.csv")
    commit_table["season"] = commit_table["season"].astype(str)
    q = loc_scores[(loc_scores["season"] == str(season)) & (loc_scores["total_pitches"] >= MIN_PITCHES_QUALIFY)]
    merged = q.merge(commit_table[["pitcher_id", "season", "fastest_fb_velo"]], on=["pitcher_id", "season"], how="inner")
    return round(float(merged["score"].corr(merged["fastest_fb_velo"])), 3)


def leaderboard(conn, loc_scores, season, top_n=15):
    q = loc_scores[(loc_scores["season"] == str(season)) & (loc_scores["total_pitches"] >= MIN_PITCHES_QUALIFY)]
    ids = q["pitcher_id"].astype(int).tolist()
    placeholders = ",".join("?" * len(ids))
    names = pd.read_sql(f"""
        SELECT p.player_id AS pitcher_id, p.name, t.abbreviation AS team_abbrev
        FROM players p LEFT JOIN teams t ON p.team_id = t.team_id
        WHERE p.player_id IN ({placeholders})
    """, conn, params=ids)
    merged = q.merge(names, on="pitcher_id", how="left")

    def _rows(d):
        return [
            dict(pitcher_id=int(r.pitcher_id), name=r.name, team_abbrev=r.team_abbrev,
                 total_pitches=int(r.total_pitches), score=round(float(r.score), 1))
            for r in d.itertuples()
        ]
    top = merged.sort_values("score", ascending=False).head(top_n)
    bottom = merged.sort_values("score", ascending=True).head(top_n)
    return {"n_qualified": int(len(q)), "top": _rows(top), "bottom": _rows(bottom)}


def main():
    conn = sqlite3.connect(DB_PATH)

    print("Loading Location+ per-pitch predictions + estimated shrinkage...", flush=True)
    loc_raw = pd.read_parquet(ARTIFACTS_DIR / "location_v1_per_pitch_preds.parquet")
    meta = json.loads((Path("models/pitching_plus_ext") / "location_plus_v1_ext_meta.json").read_text())
    k_by_type, fallback_k = meta["shrinkage_k_by_pitch_type"], meta["shrinkage_fallback_k"]
    _, _, type_means = shrinkage_params(loc_raw)

    print("\n=== Part D.1: split-half reliability, Location+ vs. Stuff+ (identical thresholds) ===", flush=True)
    loc_merged = _overall_split_half(loc_raw, SEASON)
    loc_curve = reliability_curve(loc_merged, RELIABILITY_THRESHOLDS)
    for t in RELIABILITY_THRESHOLDS:
        r = loc_curve[str(t)]
        print(f"  Location+  per-half>={t:>3}: n={r['n']:>4}  r={r['correlation']}")

    stuff_raw = stuff_sm._raw_predictions(conn, SEASON).rename(columns={})
    stuff_raw["season"] = str(SEASON)
    stuff_merged = _overall_split_half(stuff_raw, SEASON)
    stuff_curve = reliability_curve(stuff_merged, RELIABILITY_THRESHOLDS)
    for t in RELIABILITY_THRESHOLDS:
        r = stuff_curve[str(t)]
        print(f"  Stuff+     per-half>={t:>3}: n={r['n']:>4}  r={r['correlation']}")

    loc_cross_70 = first_threshold_crossing(loc_curve, 0.7)
    loc_cross_90 = first_threshold_crossing(loc_curve, 0.9)
    stuff_cross_70 = first_threshold_crossing(stuff_curve, 0.7)
    stuff_cross_90 = first_threshold_crossing(stuff_curve, 0.9)
    print(f"  Location+ crosses r=0.7 at per-half>={loc_cross_70}, r=0.9 at per-half>={loc_cross_90}")
    print(f"  Stuff+    crosses r=0.7 at per-half>={stuff_cross_70}, r=0.9 at per-half>={stuff_cross_90}")

    print("\n=== Part D.7 precursor: official (shrunk) per-pitcher-season scores, all seasons ===", flush=True)
    all_scores = pd.concat([
        official_location_plus(loc_raw, k_by_type, fallback_k, type_means, s)
        for s in sorted(loc_raw["season"].astype(str).unique())
    ], ignore_index=True)
    all_scores.to_parquet(ARTIFACTS_DIR / "location_v1_official_scores.parquet", index=False)

    print("\n=== Part D.2: year-over-year ===", flush=True)
    yoy = year_over_year(all_scores)
    for r in yoy["per_pair"]:
        print(f"  {r['season_t']}->{r['season_t1']}: n={r['n']}, r2={r['r2']}")
    print(f"  pooled r2={yoy['pooled_r2']} (n={yoy['n_pooled']})")

    print(f"\n=== Part D.3: cross-model correlation, {SEASON} ===", flush=True)
    cmc = cross_model_correlation(conn, all_scores, SEASON)
    print(f"  n={cmc['n']}  r_stuff={cmc['r_stuff']}  r_deception={cmc['r_deception']}  r_tunneling={cmc['r_tunneling']}")

    print("\n=== Part D.6: fastest_fb_velo correlation (leakage sanity check) ===", flush=True)
    velo_corr = fastest_fb_velo_correlation(all_scores, SEASON)
    print(f"  r={velo_corr}")

    print(f"\n=== Part D.7: leaderboard, {SEASON} (min {MIN_PITCHES_QUALIFY} pitches) ===", flush=True)
    lb = leaderboard(conn, all_scores, SEASON)
    print(f"  {lb['n_qualified']} qualified pitchers")
    print("\nTop 15:")
    for r in lb["top"]:
        print(f"  {r['name']:25s} {r['team_abbrev'] or '-':4s} {r['score']:6.1f}  ({r['total_pitches']} pitches)")
    print("\nBottom 15:")
    for r in lb["bottom"]:
        print(f"  {r['name']:25s} {r['team_abbrev'] or '-':4s} {r['score']:6.1f}  ({r['total_pitches']} pitches)")

    out = dict(
        reliability=dict(location_plus=loc_curve, stuff_plus=stuff_curve,
                          location_crosses_070=loc_cross_70, location_crosses_090=loc_cross_90,
                          stuff_crosses_070=stuff_cross_70, stuff_crosses_090=stuff_cross_90),
        year_over_year=yoy, cross_model_correlation=cmc,
        fastest_fb_velo_correlation=velo_corr, leaderboard=lb,
    )
    out_path = ARTIFACTS_DIR / "location_v1_validation.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

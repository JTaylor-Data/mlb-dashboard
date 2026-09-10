"""
Tunneling+ v3 - 150ms commit-point EXPERIMENT (validation battery).

Identical to tunneling_v3_validate.py - same sequential same-at-bat test
(model's own predicted-tercile methodology), same year-over-year, same
cross-model correlation, same fastest_fb_velo correlation, same leaderboard
construction - just pointed at the 150ms-trained model/artifacts instead of
the live 200ms ones. Nothing here touches models/ or the live-serving path.

Usage:
    python training/tunneling_v3_150ms_validate.py
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd

import analytics.stuff_model as stuff_sm
import analytics.deception_model as deception_sm
from training.train_tunneling_v3_150ms_model import ALL_FEATURES, CATEGORICAL_FEATURES, ARTIFACTS_DIR, load_feat

SEASON = 2025
MIN_PITCHES_QUALIFY = 100
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}


def sequential_same_at_bat_test():
    """Same corrected methodology as the live version - see
    tunneling_v3_validate.py's own docstring for the full reasoning. Not
    directly comparable to v1/v2 at face value; comparable to the live
    200ms v3 run since both use the identical methodology here."""
    with open(ARTIFACTS_DIR / "tunneling_v3_150ms_meta.json") as f:
        meta = json.load(f)
    feat = load_feat()
    for col in CATEGORICAL_FEATURES:
        feat[col] = pd.Categorical(feat[col], categories=meta["category_levels"][col])
    model = lgb.Booster(model_file=str(ARTIFACTS_DIR / "tunneling_v3_150ms.txt"))
    df = feat[["description"]].copy()
    df["pred"] = model.predict(feat[ALL_FEATURES])
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)

    df["tercile"] = pd.qcut(df["pred"], 3, labels=["low_pred", "mid_pred", "high_pred"], duplicates="drop")
    by_tercile = df.groupby("tercile", observed=True)["is_whiff"].agg(["mean", "count"])
    print(by_tercile)

    p_good, n_good = by_tercile.loc["low_pred", "mean"], by_tercile.loc["low_pred", "count"]
    p_bad, n_bad = by_tercile.loc["high_pred", "mean"], by_tercile.loc["high_pred", "count"]
    se = np.sqrt(p_good * (1 - p_good) / n_good + p_bad * (1 - p_bad) / n_bad)
    z = (p_good - p_bad) / se

    return dict(
        n_total=int(len(df)), whiff_rate_by_predicted_tercile={
            t: dict(whiff_rate=float(r["mean"]), n=int(r["count"])) for t, r in by_tercile.iterrows()
        },
        good_minus_bad_z_score=float(z),
    )


def year_over_year(scores):
    q = scores[scores["total_pitches"] >= MIN_PITCHES_QUALIFY]
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


def cross_model_correlation(conn, scores, season):
    q = scores[(scores["season"] == season) & (scores["total_pitches"] >= MIN_PITCHES_QUALIFY)]
    stuff = stuff_sm.compute_stuff_plus(conn, int(season)).reset_index()
    overall_stuff = stuff.groupby("pitcher_id").apply(
        lambda g: np.average(g["stuff_plus"], weights=g["n_pitches"]), include_groups=False
    ).rename("stuff_plus").reset_index()

    dec_raw = deception_sm._raw_predictions(conn, int(season))
    dec_mean, dec_std = dec_raw["pred"].mean(), dec_raw["pred"].std()
    dec = dec_raw.groupby("pitcher_id")["pred"].mean().reset_index()
    dec["deception_plus"] = deception_sm._to_plus_scale(dec["pred"], dec_mean, dec_std)

    merged = q.merge(overall_stuff, on="pitcher_id", how="inner").merge(dec[["pitcher_id", "deception_plus"]], on="pitcher_id", how="inner")
    return dict(
        n=int(len(merged)),
        r_stuff=round(float(merged["score"].corr(merged["stuff_plus"])), 3),
        r_deception=round(float(merged["score"].corr(merged["deception_plus"])), 3),
    )


def fastest_fb_velo_correlation(scores):
    commit_table = pd.read_csv("training/tunneling_v2_artifacts/150ms/pitcher_season_commit_point.csv")
    commit_table["season"] = commit_table["season"].astype(str)
    merged = scores.merge(commit_table[["pitcher_id", "season", "fastest_fb_velo"]], on=["pitcher_id", "season"], how="inner")
    return round(float(merged["score"].corr(merged["fastest_fb_velo"])), 3)


def leaderboard(conn, scores, season, top_n=15):
    q = scores[(scores["season"] == str(season)) & (scores["total_pitches"] >= MIN_PITCHES_QUALIFY)]
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
    conn = sqlite3.connect("mlb_data.db")
    scores = pd.read_parquet(ARTIFACTS_DIR / "tunneling_v3_150ms_scores.parquet")

    print("=== Sequential same-at-bat test (v3, 150ms) ===", flush=True)
    seq = sequential_same_at_bat_test()
    print(f"  n={seq['n_total']:,}  good-minus-bad z={seq['good_minus_bad_z_score']:.3f}")

    print("\n=== Year-over-year ===", flush=True)
    yoy = year_over_year(scores)
    for r in yoy["per_pair"]:
        print(f"  {r['season_t']}->{r['season_t1']}: n={r['n']}, r2={r['r2']}")
    print(f"  pooled r2={yoy['pooled_r2']} (n={yoy['n_pooled']})")

    print(f"\n=== Cross-model correlation, {SEASON} ===", flush=True)
    cmc = cross_model_correlation(conn, scores, str(SEASON))
    print(f"  n={cmc['n']}  r_stuff={cmc['r_stuff']}  r_deception={cmc['r_deception']}")

    print("\n=== fastest_fb_velo correlation ===", flush=True)
    velo_corr = fastest_fb_velo_correlation(scores)
    print(f"  r={velo_corr}")

    print(f"\n=== Leaderboard, {SEASON} (min {MIN_PITCHES_QUALIFY} pitches) ===", flush=True)
    lb = leaderboard(conn, scores, SEASON)
    print(f"  {lb['n_qualified']} qualified pitchers")
    print("\nTop 15:")
    for r in lb["top"]:
        print(f"  {r['name']:25s} {r['team_abbrev'] or '-':4s} {r['score']:6.1f}  ({r['total_pitches']} pitches)")
    print("\nBottom 15:")
    for r in lb["bottom"]:
        print(f"  {r['name']:25s} {r['team_abbrev'] or '-':4s} {r['score']:6.1f}  ({r['total_pitches']} pitches)")

    print(f"\n=== Score spread, {SEASON} (min {MIN_PITCHES_QUALIFY} pitches) ===", flush=True)
    q = scores[(scores["season"] == str(SEASON)) & (scores["total_pitches"] >= MIN_PITCHES_QUALIFY)]
    spread = dict(
        min=round(float(q["score"].min()), 1), max=round(float(q["score"].max()), 1),
        std=round(float(q["score"].std()), 2), p10=round(float(q["score"].quantile(0.10)), 1),
        p90=round(float(q["score"].quantile(0.90)), 1),
    )
    print(f"  min={spread['min']} max={spread['max']} std={spread['std']} p10={spread['p10']} p90={spread['p90']}")

    out = dict(
        sequential_same_at_bat_test=seq, year_over_year=yoy, cross_model_correlation=cmc,
        fastest_fb_velo_correlation=velo_corr, leaderboard=lb, score_spread=spread,
    )
    out_path = ARTIFACTS_DIR / "tunneling_v3_150ms_validation.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

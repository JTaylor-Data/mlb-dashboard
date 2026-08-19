"""
Predictive-validity comparison: v1 (legacy per-pitch-type OLS) vs. v2
(unified gradient-boosted model), unshrunk vs. shrunk.

## What this actually tests
Does each pitcher's 2024 Stuff+ predict their REAL 2025 outcomes (xwOBA
against, whiff%, swinging-strike%, K-BB%, chase%)? This is the gold-standard
test for a "skill" metric that same-season concurrent validity (checked
separately, see stuff_model.py's docstrings/conversation history) can't
provide - a metric could fit the current season's noise well without
actually capturing anything that persists.

## Why v2's 2024 model is trained WITHOUT 2024 data
The live site's production v2 model is trained on ALL seasons at once
(including 2024) - fine for the live site (more data = a better model),
but scoring 2024 with a model that has already seen 2024's own pitches
would make 2024's numbers optimistic in a way that's specific to this
predictive-validity test, not a fair comparison. So this script trains a
SEPARATE, one-off model on every season except 2024, uses THAT to score
2024 pitches, and never touches the production model files - this is
purely a benchmarking artifact.

v1 gets no such treatment - it never had a train/test split concept to
begin with (a fresh per-season OLS fit IS how v1 always actually worked
in production), so its 2024 numbers here are exactly what v1 would have
shown on the live site in 2024. That's a fair, if anything slightly
generous-to-v1 comparison: an in-sample fit that overfit to 2024-specific
noise would, if anything, hurt its 2025 predictive validity, not help it.
"""
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import stuff_model as v2
import stuff_model_v1_legacy as v1
from metrics import compute_vaa
from percentiles import pitcher_percentile_metrics
from stuff_features import ALL_FEATURES, CATEGORICAL_FEATURES, engineer_features, fit_vaa_expectation
from train_stuff_model import _fit_one_model, load_pitches

DB_PATH = "mlb_data.db"
REPORT_PATH = Path(__file__).parent / "models" / "predictive_validity_report.json"
TRAIN_SEASONS = [2021, 2022, 2023, 2025, 2026]  # deliberately excludes 2024
SCORE_SEASON = 2024
OUTCOME_SEASON = 2025
OUTCOMES = ["xwoba", "whiff_pct", "swstr_pct", "k_minus_bb_pct", "chase_pct"]


def train_out_of_sample_model(conn):
    print("Training a v2 model that has never seen 2024...", flush=True)
    train_raw = load_pitches(conn, TRAIN_SEASONS)
    vaa_coefs = fit_vaa_expectation(train_raw.assign(vaa=lambda d: compute_vaa(d.vy0, d.vz0, d.ay, d.az)))
    feat = engineer_features(train_raw, vaa_coefs)
    for col in CATEGORICAL_FEATURES:
        feat[col] = feat[col].astype(str).astype("category")
    category_levels = {col: feat[col].cat.categories.tolist() for col in CATEGORICAL_FEATURES}
    model = _fit_one_model(feat, ALL_FEATURES)
    print(f"  out-of-sample model best_iteration={model.best_iteration_}", flush=True)
    return model, vaa_coefs, category_levels


def score_2024_out_of_sample(conn, model, vaa_coefs, category_levels):
    print("Scoring 2024 pitches with the out-of-sample model...", flush=True)
    score_raw = load_pitches(conn, [SCORE_SEASON])
    feat = engineer_features(score_raw, vaa_coefs)
    for col, levels in category_levels.items():
        feat[col] = pd.Categorical(feat[col].astype(str), categories=levels)
    preds = model.predict(feat[ALL_FEATURES])
    out = feat[["pitcher_id", "pitch_type", "stand"]].copy()
    out["pred"] = preds
    return out


def v2_stuff_plus_table(raw, shrink):
    if shrink:
        k_by_type, fallback_k, type_means = v2._shrinkage_params(raw)
        g = v2._shrunk_group_means(raw, ["pitcher_id", "pitch_type"], k_by_type, fallback_k, type_means)
        value_col = "shrunk_mean"
    else:
        g = raw.groupby(["pitcher_id", "pitch_type"], observed=True)["pred"].agg(
            mean="mean", n_pitches="count"
        ).reset_index()
        value_col = "mean"
    g["stuff_plus"] = v2._to_stuff_plus(g[value_col], raw["pred"].mean(), raw["pred"].std())
    return g.set_index(["pitcher_id", "pitch_type"])[["n_pitches", "stuff_plus"]]


def overall_by_pitcher(stuff_table, col_name):
    flat = stuff_table.reset_index()
    out = flat.groupby("pitcher_id").apply(
        lambda g: pd.Series({
            col_name: np.average(g["stuff_plus"], weights=g["n_pitches"]),
            "n_2024_pitches": g["n_pitches"].sum(),
        })
    ).reset_index()
    return out


def main():
    conn = sqlite3.connect(DB_PATH)

    print("Computing v1 Stuff+ for 2024 (in-sample, as v1 always worked)...", flush=True)
    v1_overall = overall_by_pitcher(v1.compute_stuff_plus(conn, SCORE_SEASON), "v1_stuff")

    model, vaa_coefs, category_levels = train_out_of_sample_model(conn)
    raw_2024 = score_2024_out_of_sample(conn, model, vaa_coefs, category_levels)

    v2_unshrunk = overall_by_pitcher(v2_stuff_plus_table(raw_2024, shrink=False), "v2_unshrunk_stuff")
    v2_shrunk = overall_by_pitcher(v2_stuff_plus_table(raw_2024, shrink=True), "v2_shrunk_stuff")

    print("Loading real 2025 outcomes...", flush=True)
    pool = pitcher_percentile_metrics(conn, OUTCOME_SEASON, stand_filter=None).reset_index()
    id_col = pool.columns[0]
    pool = pool.rename(columns={id_col: "pitcher_id"})

    merged = (
        v1_overall[["pitcher_id", "v1_stuff", "n_2024_pitches"]]
        .merge(v2_unshrunk[["pitcher_id", "v2_unshrunk_stuff"]], on="pitcher_id")
        .merge(v2_shrunk[["pitcher_id", "v2_shrunk_stuff"]], on="pitcher_id")
        .merge(pool, on="pitcher_id")
    )
    print(f"\n{len(merged)} pitchers with both a 2024 Stuff+ (all three flavors) and 2025 outcomes.\n")

    results_by_min_pitches = {}
    for min_n in [100, 200, 300]:
        sub = merged[merged["n_2024_pitches"] >= min_n]
        print(f"=== min 2024 pitches = {min_n}  (n qualified = {len(sub)}) ===")
        header = f"{'2025 outcome':16s} {'v1 (legacy)':>14s} {'v2 (unshrunk)':>14s} {'v2 (shrunk)':>14s}"
        print(header)
        correlations = {}
        for outcome in OUTCOMES:
            c1 = sub["v1_stuff"].corr(sub[outcome])
            c2 = sub["v2_unshrunk_stuff"].corr(sub[outcome])
            c3 = sub["v2_shrunk_stuff"].corr(sub[outcome])
            print(f"{outcome:16s} {c1:14.3f} {c2:14.3f} {c3:14.3f}")
            correlations[outcome] = dict(v1=float(c1), v2_unshrunk=float(c2), v2_shrunk=float(c3))
        print()
        results_by_min_pitches[str(min_n)] = dict(n_qualified=int(len(sub)), correlations=correlations)

    report = dict(
        trained_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        score_season=SCORE_SEASON,
        outcome_season=OUTCOME_SEASON,
        description=(
            f"Does each pitcher's {SCORE_SEASON} Stuff+ predict their real {OUTCOME_SEASON} results? "
            f"v2 is scored by a model trained on every season EXCEPT {SCORE_SEASON} (honest out-of-sample "
            f"- the live site's model, unlike this one, is trained on all seasons including this one). "
            f"v1 is in-sample, exactly as it always worked in production - v1 never had a train/test split "
            f"concept, so this is not an unfair advantage to v2."
        ),
        n_total_pitchers=int(len(merged)),
        results_by_min_pitches=results_by_min_pitches,
    )
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()

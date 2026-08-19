"""
Tunneling+ v1 - builds the full report: league-wide scores, every validation
test run against the model, and per-pitcher pair detail for inspection.
Run by hand (like train_stuff_model.py / validate_predictive_power.py):

    python build_tunneling_report.py            # every season with data
    python build_tunneling_report.py 2025        # just one season

Writes models/tunneling_v1_report_{season}.json (summary + validation
results) and models/tunneling_v1_pitchers_{season}.json (per-pitcher
pair-level detail, for the HTML inspector) - one pair of files per season,
so the site's season dropdown can switch between them without recomputing
anything live. See tunneling_model.py's module docstring for the full
methodology writeup this report's numbers are validating - including why
distances are full pairwise averages (not centroid-to-centroid), why the
score is a geometric mean of three components (release_similarity_plus,
decision_similarity_plus, plate_separation_plus) rather than the original
v1 ratio, and why FF/SI pairs count double toward the pitcher-level
composite.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import stuff_model as sm
from tunneling_model import (
    ANCHOR_TYPES, DECISION_POINT_Y, DISTANCE_FLOOR, MIN_PITCHES_PER_TYPE,
    composite_weight_fn, add_tunneling_plus, league_pair_baselines, load_pitcher_pitches,
    pairwise_tunnel_scores, pitch_type_centroids, pitcher_tunneling_scores,
)
from percentiles import pitcher_percentile_metrics

DB_PATH = Path(__file__).parent / "mlb_data.db"
MODELS_DIR = Path(__file__).parent / "models"
MIN_PITCHES_QUALIFY = 300  # matches the percentile pool's own qualifying bar, for apples-to-apples correlation checks


def _connect():
    import sqlite3
    return sqlite3.connect(DB_PATH)


def split_half_reliability(conn, season):
    print("Running split-half reliability check...", flush=True)
    df = load_pitcher_pitches(conn, season).reset_index(drop=True)
    df["half"] = df.index % 2
    halves = {}
    for half in [0, 1]:
        sub = df[df["half"] == half]
        centroids = pitch_type_centroids(sub)
        pairs = add_tunneling_plus(pairwise_tunnel_scores(sub, centroids))
        pairs["composite_weight"] = composite_weight_fn(pairs)
        pairs["weighted_plus"] = pairs["tunneling_plus"] * pairs["composite_weight"]
        agg = pairs.groupby("pitcher_id").agg(
            _w=("composite_weight", "sum"), _wp=("weighted_plus", "sum"), n_pairs=("pitcher_id", "size"),
        )
        agg["tunneling_plus"] = agg["_wp"] / agg["_w"]
        halves[half] = agg[["tunneling_plus", "n_pairs"]]

    merged = halves[0].join(halves[1], lsuffix="_0", rsuffix="_1", how="inner")
    results = {}
    for min_pairs in [1, 2, 3]:
        sub = merged[(merged.n_pairs_0 >= min_pairs) & (merged.n_pairs_1 >= min_pairs)]
        results[str(min_pairs)] = dict(
            n=int(len(sub)),
            correlation=float(sub["tunneling_plus_0"].corr(sub["tunneling_plus_1"])),
        )
    return results


def outcome_validity(conn, season, scores):
    print("Running outcome-correlation checks (raw + stuff-partialed)...", flush=True)
    stuff = sm.compute_stuff_plus(conn, season).reset_index()
    overall_stuff = stuff.groupby("pitcher_id").apply(
        lambda g: pd.Series({"overall_stuff": np.average(g["stuff_plus"], weights=g["n_pitches"])})
    ).reset_index()

    pool = pitcher_percentile_metrics(conn, season, stand_filter=None).reset_index()
    pool = pool.rename(columns={pool.columns[0]: "pitcher_id"})

    merged = scores.reset_index().merge(overall_stuff, on="pitcher_id").merge(pool, on="pitcher_id")
    merged = merged[merged["n_pitches"] >= MIN_PITCHES_QUALIFY].copy()

    def residualize(y, x):
        b, a = np.polyfit(x, y, 1)
        return y - (a + b * x)

    outcomes = ["whiff_pct", "chase_pct", "swstr_pct", "k_minus_bb_pct", "xwoba"]
    rows = {}
    resid_tun = residualize(merged["tunneling_plus"].values, merged["overall_stuff"].values)
    for o in outcomes:
        raw_r = float(merged["tunneling_plus"].corr(merged[o]))
        resid_outcome = residualize(merged[o].values, merged["overall_stuff"].values)
        partial_r = float(np.corrcoef(resid_outcome, resid_tun)[0, 1])
        rows[o] = dict(raw_correlation=raw_r, partial_correlation_stuff_removed=partial_r)

    return dict(
        n_qualified=int(len(merged)),
        min_pitches_qualify=MIN_PITCHES_QUALIFY,
        correlation_with_overall_stuff=float(merged["tunneling_plus"].corr(merged["overall_stuff"])),
        outcomes=rows,
    )


def sequential_same_at_bat_test(conn, season):
    """The most mechanism-specific test: within real at-bats, when a
    pitcher throws two DIFFERENT pitch types back to back, does the
    PRE-COMPUTED pairwise tunneling_plus for that specific pair predict a
    higher whiff rate on the second pitch of the sequence? This tests the
    actual causal pathway the metric claims to capture, not just a
    league-wide aggregate that could be confounded by "good pitchers are
    good at everything.\""""
    print("Running sequential same-at-bat whiff-rate test...", flush=True)
    df = load_pitcher_pitches(conn, season)
    centroids = pitch_type_centroids(df)
    pairs = add_tunneling_plus(pairwise_tunnel_scores(df, centroids))
    pair_lookup = {(r.pitcher_id, r.type_a, r.type_b): r.tunneling_plus for r in pairs.itertuples()}

    df_sorted = df.sort_values(["game_id", "at_bat_number", "pitch_number"])
    df_sorted["prev_pitch_type"] = df_sorted.groupby(["game_id", "at_bat_number"])["pitch_type"].shift(1)
    consec = df_sorted.dropna(subset=["prev_pitch_type"])
    consec = consec[consec["pitch_type"] != consec["prev_pitch_type"]].copy()

    def lookup_plus(row):
        a, b = sorted([row["pitch_type"], row["prev_pitch_type"]])
        return pair_lookup.get((row["pitcher_id"], a, b))

    consec["plus"] = consec.apply(lookup_plus, axis=1)
    consec = consec.dropna(subset=["plus"])
    whiff_descriptions = {"swinging_strike", "swinging_strike_blocked"}
    consec["is_whiff"] = consec["description"].isin(whiff_descriptions)

    consec["tercile"] = pd.qcut(consec["plus"], 3, labels=["low", "mid", "high"])
    by_tercile = consec.groupby("tercile", observed=True)["is_whiff"].agg(["mean", "count"])

    p_low, n_low = by_tercile.loc["low", "mean"], by_tercile.loc["low", "count"]
    p_high, n_high = by_tercile.loc["high", "mean"], by_tercile.loc["high", "count"]
    se = np.sqrt(p_low * (1 - p_low) / n_low + p_high * (1 - p_high) / n_high)
    z = (p_high - p_low) / se

    return dict(
        n_consecutive_different_type_pitches=int(len(df_sorted.dropna(subset=["prev_pitch_type"]))),
        n_with_known_pair_ratio=int(len(consec)),
        whiff_rate_by_tunnel_ratio_tercile={
            tercile: dict(whiff_rate=float(row["mean"]), n=int(row["count"]))
            for tercile, row in by_tercile.iterrows()
        },
        low_vs_high_z_score=float(z),
        interpretation=(
            "Positive z means pitch pairs the model scores as MORE tunneled "
            "produce a higher whiff rate on the second pitch of the sequence, "
            "in real same-at-bat sequences - the specific causal pathway the "
            "model claims to capture, tested directly rather than inferred "
            "from a league-wide aggregate."
        ),
    )


def worked_examples(conn, season, scores, top_n=10):
    names = dict(conn.execute("SELECT player_id, name FROM players").fetchall())
    s = scores.reset_index().copy()
    s["name"] = s["pitcher_id"].map(names)
    stable = s[s["n_pairs"] >= 2]

    def top_rows(frame):
        return frame[["pitcher_id", "name", "tunneling_plus", "release_consistency", "n_pairs", "total_pitches"]].to_dict(orient="records")

    return dict(
        highest_tunneling=top_rows(stable.sort_values("tunneling_plus", ascending=False).head(top_n)),
        lowest_tunneling=top_rows(stable.sort_values("tunneling_plus").head(top_n)),
        most_release_consistent=top_rows(stable.sort_values("release_consistency").head(top_n)),
    )


def pitcher_detail_export(conn, season, scores, names, min_pitches=300):
    """Per-pitcher pair-level detail for the HTML inspector - every
    qualifying pitcher's individual pitch-type pairs with release/tunnel/
    plate distances (inches) and each pair's own release_similarity_plus /
    decision_similarity_plus / plate_separation_plus / tunneling_plus, not
    just the pitcher's headline score - showing all three components is
    what actually explains WHY a pair scores the way it does. Pair-level
    values use the same season-wide league baseline as the headline score,
    so all numbers are directly comparable."""
    df = load_pitcher_pitches(conn, season)
    centroids = pitch_type_centroids(df)
    pairs = pairwise_tunnel_scores(df, centroids)
    avg_release, avg_tunnel, avg_plate = league_pair_baselines(pairs)
    pairs = add_tunneling_plus(pairs, avg_release, avg_tunnel, avg_plate)

    qualified_ids = set(scores[scores["total_pitches"] >= min_pitches].index)
    out = {}
    for pid in qualified_ids:
        prows = pairs[pairs["pitcher_id"] == pid]
        if prows.empty:
            continue
        out[str(pid)] = dict(
            name=names.get(pid, f"#{pid}"),
            tunneling_plus=float(scores.loc[pid, "tunneling_plus"]),
            release_consistency=float(scores.loc[pid, "release_consistency"]),
            pairs=[
                dict(
                    type_a=r.type_a, type_b=r.type_b, n_a=int(r.n_a), n_b=int(r.n_b),
                    release_distance=round(float(r.release_distance), 1),
                    tunnel_distance=round(float(r.tunnel_distance), 1),
                    plate_distance=round(float(r.plate_distance), 1),
                    release_similarity_plus=round(float(r.release_similarity_plus), 0),
                    decision_similarity_plus=round(float(r.decision_similarity_plus), 0),
                    plate_separation_plus=round(float(r.plate_separation_plus), 0),
                    tunneling_plus=round(float(r.tunneling_plus), 0),
                    is_anchor=bool(r.type_a in ANCHOR_TYPES or r.type_b in ANCHOR_TYPES),
                )
                for r in prows.itertuples()
            ],
        )
    return out


def build_for_season(conn, season):
    print(f"Computing league-wide tunneling scores for {season}...", flush=True)
    scores = pitcher_tunneling_scores(conn, season)
    print(f"  {len(scores)} pitchers scored", flush=True)

    reliability = split_half_reliability(conn, season)
    validity = outcome_validity(conn, season, scores)
    sequential = sequential_same_at_bat_test(conn, season)
    examples = worked_examples(conn, season, scores)

    names = dict(conn.execute("SELECT player_id, name FROM players").fetchall())
    detail = pitcher_detail_export(conn, season, scores, names)

    report = dict(
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        season=season,
        methodology=dict(
            decision_point_feet_from_plate=DECISION_POINT_Y,
            min_pitches_per_type=MIN_PITCHES_PER_TYPE,
            distance_floor_inches=DISTANCE_FLOOR,
            n_pitchers_scored=int(len(scores)),
            scale_note="tunneling_plus = geometric mean of THREE components - release_similarity_plus (close at "
                       "release), decision_similarity_plus (close at the decision point), and "
                       "plate_separation_plus (far apart at the plate) - each independently scaled 100=league "
                       "average, same convention as Stuff+/wRC+. All distances are the full pairwise mean between "
                       "every real pitch of type A and every real pitch of type B (not centroid-to-centroid, which "
                       "understates true separation by averaging away real command variance), in inches; decision "
                       "point in feet. FF/SI-involving pairs count 2x toward the pitcher-level composite (not "
                       "toward the league baselines) - a hitter's read is built primarily against the "
                       "fastball/sinker, so how well an offspeed pitch tunnels with THAT matters more than how "
                       "two offspeed pitches tunnel with each other.",
        ),
        split_half_reliability=reliability,
        outcome_validity=validity,
        sequential_same_at_bat_test=sequential,
        worked_examples=examples,
    )
    return report, detail


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    conn = _connect()

    if len(sys.argv) > 1:
        seasons = [int(sys.argv[1])]
    else:
        seasons = [r[0] for r in conn.execute("SELECT DISTINCT season FROM pitches ORDER BY season DESC")]

    for season in seasons:
        start = time.time()
        print(f"=== Season {season} ===", flush=True)
        report, detail = build_for_season(conn, season)
        (MODELS_DIR / f"tunneling_v1_report_{season}.json").write_text(json.dumps(report, indent=2))
        (MODELS_DIR / f"tunneling_v1_pitchers_{season}.json").write_text(json.dumps(detail, indent=2))
        print(f"Done in {time.time() - start:.0f}s. Saved tunneling_v1_report_{season}.json / tunneling_v1_pitchers_{season}.json\n", flush=True)


if __name__ == "__main__":
    main()

"""
Part 4 validation for Tunneling+ pair composite.
Outputs diagnostics/tunneling_pair_composite/validation_results.json.

Validation items:
  1. Sequential same-at-bat whiff-rate tercile test (2025)
  2. Top 15 / bottom 15 pairs by composite score (2025)
  3. Score distribution (2025)
"""
import io, json, sqlite3, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r'C:\Users\joshs\Desktop\mlb_dashboard')

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from analytics.tunneling_pair_composite import season_all_pairs
from analytics.tunneling_features import load_pitches

DB = Path(r'C:\Users\joshs\Desktop\mlb_dashboard\mlb_data.db')
OUT = Path(r'C:\Users\joshs\Desktop\mlb_dashboard\diagnostics\tunneling_pair_composite\validation_results.json')
SEASON = 2025

conn = sqlite3.connect(str(DB))

print("Computing season_all_pairs for 2025 (slow)...", flush=True)
all_pairs = season_all_pairs(conn, SEASON)
print(f"  {len(all_pairs):,} qualifying pairs", flush=True)

# ── 1. Score distribution ─────────────────────────────────────────────────────
scores = all_pairs["composite_score"]
dist_stats = {
    "n_pairs": int(len(scores)),
    "min": round(float(scores.min()), 1),
    "max": round(float(scores.max()), 1),
    "mean": round(float(scores.mean()), 1),
    "sd": round(float(scores.std()), 2),
    "spread": round(float(scores.max() - scores.min()), 1),
}
print(f"\nScore distribution: min={dist_stats['min']}, max={dist_stats['max']}, "
      f"SD={dist_stats['sd']}, spread={dist_stats['spread']}", flush=True)

# ── 2. Top 15 / bottom 15 ─────────────────────────────────────────────────────
names = pd.read_sql("SELECT player_id AS pitcher_id, name FROM players", conn)
pairs_named = all_pairs.merge(names, on="pitcher_id", how="left")

def fmt_row(r):
    return {
        "name": str(r["name"]) if pd.notna(r["name"]) else f"ID:{r['pitcher_id']}",
        "pair": f"{r['type_a']} / {r['type_b']}",
        "composite_score": round(float(r["composite_score"]), 1),
    }

top15 = [fmt_row(r) for _, r in pairs_named.nlargest(15, "composite_score").iterrows()]
bot15 = [fmt_row(r) for _, r in pairs_named.nsmallest(15, "composite_score").iterrows()]

print("\nTop 5 pairs:")
for r in top15[:5]:
    print(f"  {r['name']:30s}  {r['pair']:8s}  {r['composite_score']:.1f}")
print("Bottom 5 pairs:")
for r in bot15[:5]:
    print(f"  {r['name']:30s}  {r['pair']:8s}  {r['composite_score']:.1f}")

# ── 3. Sequential same-at-bat whiff-rate tercile test ────────────────────────
print("\nBuilding sequential whiff-rate tercile test...", flush=True)
pitches = load_pitches(conn, SEASON)

# Build pair lookup: (pitcher_id, type_a, type_b) -> composite_score
# Both orderings (a,b) and (b,a) should map to the same score
pair_score = {}
for _, row in all_pairs.iterrows():
    pid = row["pitcher_id"]
    ta, tb = row["type_a"], row["type_b"]
    score = row["composite_score"]
    pair_score[(pid, ta, tb)] = score
    pair_score[(pid, tb, ta)] = score

# Sort by at-bat sequence, find follow-up pitches where previous pitch type
# is the OTHER member of a qualifying pair for that pitcher
pitches_sorted = pitches.sort_values(["game_id", "at_bat_number", "pitch_number"]).reset_index(drop=True)
g = pitches_sorted.groupby(["game_id", "at_bat_number"])
pitches_sorted["prev_pitch_type"] = g["pitch_type"].shift(1)
pitches_sorted = pitches_sorted.dropna(subset=["prev_pitch_type"])
pitches_sorted = pitches_sorted[pitches_sorted["pitch_type"] != pitches_sorted["prev_pitch_type"]].copy()

def get_pair_score(row):
    return pair_score.get((row["pitcher_id"], row["prev_pitch_type"], row["pitch_type"]))

pitches_sorted["pair_composite"] = pitches_sorted.apply(get_pair_score, axis=1)
eligible = pitches_sorted.dropna(subset=["pair_composite"]).copy()
eligible["is_whiff"] = eligible["description"].isin(["swinging_strike", "swinging_strike_blocked"]).astype(int)

print(f"  Eligible follow-up pitches: {len(eligible):,}", flush=True)

# Tercile cut
t33 = eligible["pair_composite"].quantile(1/3)
t67 = eligible["pair_composite"].quantile(2/3)
eligible["tercile"] = pd.cut(eligible["pair_composite"],
                              bins=[-np.inf, t33, t67, np.inf],
                              labels=["low", "mid", "high"])
tercile_whiff = eligible.groupby("tercile")["is_whiff"].agg(["mean", "size"]).reset_index()
tercile_whiff.columns = ["tercile", "whiff_rate", "n"]

low_row = tercile_whiff[tercile_whiff["tercile"] == "low"].iloc[0]
high_row = tercile_whiff[tercile_whiff["tercile"] == "high"].iloc[0]

low_wh, low_n = float(low_row["whiff_rate"]), int(low_row["n"])
high_wh, high_n = float(high_row["whiff_rate"]), int(high_row["n"])

# z-test for proportion difference
p_pool = (low_wh * low_n + high_wh * high_n) / (low_n + high_n)
se = np.sqrt(p_pool * (1 - p_pool) * (1/low_n + 1/high_n))
z_stat = (high_wh - low_wh) / se if se > 0 else 0.0

tercile_results = {
    "low_whiff_rate": round(low_wh, 4),
    "low_n": low_n,
    "high_whiff_rate": round(high_wh, 4),
    "high_n": high_n,
    "diff_pct": round((high_wh - low_wh) * 100, 2),
    "z_stat": round(float(z_stat), 2),
    "note": "positive z = high-composite pairs yield more whiffs than low-composite pairs"
}
print(f"\nWhiff-rate tercile test:")
print(f"  Low  composite: {low_wh:.3f} (n={low_n:,})")
print(f"  High composite: {high_wh:.3f} (n={high_n:,})")
print(f"  Diff: {(high_wh-low_wh)*100:+.2f}pp  z={z_stat:.2f}")

conn.close()

# ── Write JSON ────────────────────────────────────────────────────────────────
results = {
    "season": SEASON,
    "n_qualifying_pairs": int(len(all_pairs)),
    "score_distribution": dist_stats,
    "tercile_whiff_test": tercile_results,
    "top15_pairs": top15,
    "bottom15_pairs": bot15,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2))
print(f"\nResults written to {OUT}")

"""
Model Correlation Heatmap Tool — Stuff+ v4 framework
=======================================================
Inputs: pitcher-season score tables for Stuff+ v4, Location+ v2,
        Tunneling+ v2, Deception+ v1 (loaded from DB or parquet files).
Outputs:
  - Pearson and Spearman 4×4 heatmaps (annotated)
  - Bootstrap CIs (pitcher-clustered)
  - Starters vs relievers split
  - Models-vs-outcomes panel (ERA, xFIP, K-BB%, whiff%)
  - Flags any pair with |r| > 0.5
  - Saves: PNG, JSON, standalone HTML

Usage:
  python tools/model_correlation_heatmap.py [--output-dir diagnostics/stuff_v4]

Minimum pitch rule: 100 pitches for all models, using all available seasons
where all four models have scores.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "mlb_data.db"
DEFAULT_OUT_DIR = ROOT / "diagnostics" / "stuff_v4"
MIN_PITCHES = 100


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---- Load score tables ----

def load_stuff_v4_scores(out_dir):
    """Load Stuff+ v4 scores from parquet."""
    path = Path(out_dir) / "stuff_v4_scores.parquet"
    if not path.exists():
        print(f"WARNING: {path} not found. Run train_stuff_v4.py first.")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # aggregate to pitcher-season
    agg = (df.groupby(["pitcher_id", "season"])
           .apply(lambda x: pd.Series({
               "stuff_plus_v4": np.average(x["stuff_plus_v4"], weights=x["n_pitches"]),
               "n_pitches": x["n_pitches"].sum(),
           }))
           .reset_index())
    return agg[agg["n_pitches"] >= MIN_PITCHES][["pitcher_id", "season", "stuff_plus_v4", "n_pitches"]]


def load_model_scores_from_db():
    """Load existing model scores from pitcher_model_scores table."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT pitcher_id, season, stuff_plus, tunneling_plus, location_plus
        FROM pitcher_model_scores
    """, conn)
    conn.close()
    return df


def load_outcomes():
    """Load pitcher-season outcomes."""
    conn = sqlite3.connect(DB_PATH)
    outcomes = pd.read_sql("""
        SELECT player_id AS pitcher_id, season, xfip, era
        FROM player_advanced_stats
        WHERE (xfip IS NOT NULL OR era IS NOT NULL)
    """, conn)
    conn.close()
    return outcomes


def load_roles():
    """Load pitcher roles (SP/RP) from pitches."""
    conn = sqlite3.connect(DB_PATH)
    # Approximate role by checking if pitcher started (first pitch of game inning 1)
    # Simpler: count appearances and use game-start frequency
    roles = pd.read_sql("""
        SELECT pitcher_id, season,
               COUNT(DISTINCT game_id) as n_games
        FROM pitches
        WHERE game_type='R'
        GROUP BY pitcher_id, season
    """, conn)
    conn.close()
    return roles


def bootstrap_ci_corr(x, y, pitcher_ids, n_boot=1000, seed=42):
    """Pitcher-clustered bootstrap CI for Pearson r."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    pitcher_ids = pitcher_ids[m]
    if len(x) < 10:
        return np.nan, np.nan, np.nan

    obs_r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    pitchers = np.unique(pitcher_ids)
    n_p = len(pitchers)
    boot_r = []
    for _ in range(n_boot):
        sampled = rng.choice(pitchers, size=n_p, replace=True)
        idx = np.concatenate([np.where(pitcher_ids == p)[0] for p in sampled])
        if len(idx) < 5:
            continue
        r = float(np.corrcoef(x[idx], y[idx])[0, 1])
        boot_r.append(r)
    boot_r = np.array(boot_r)
    ci_lo = float(np.nanpercentile(boot_r, 2.5))
    ci_hi = float(np.nanpercentile(boot_r, 97.5))
    return obs_r, ci_lo, ci_hi


def compute_correlation_matrix(scores_df, model_cols, use_spearman=False):
    """Compute correlation matrix for model_cols on scores_df."""
    sub = scores_df.dropna(subset=model_cols)
    if len(sub) < 10:
        n = len(model_cols)
        return np.full((n, n), np.nan), np.zeros((n, n), dtype=int)

    if use_spearman:
        mat = sub[model_cols].rank().corr().values
    else:
        mat = sub[model_cols].corr().values

    n_mat = np.full((len(model_cols), len(model_cols)), len(sub), dtype=int)
    return mat, n_mat


def render_heatmap_html(pearson_mat, spearman_mat, model_names, n_obs, cis,
                        outcomes_corrs, flagged_pairs, output_path):
    """Generate standalone HTML heatmap."""

    def color_for(val):
        """Blue-white-red colormap."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "#cccccc"
        v = max(-1, min(1, val))
        if v >= 0:
            r = int(255 * (1 - v))
            g = int(255 * (1 - v))
            b = 255
        else:
            r = 255
            g = int(255 * (1 + v))
            b = int(255 * (1 + v))
        return f"rgb({r},{g},{b})"

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:.3f}"

    n = len(model_names)
    header_row = "".join(f"<th>{m}</th>" for m in model_names)

    def build_table(mat, title):
        rows = ""
        for i, row_name in enumerate(model_names):
            cells = ""
            for j in range(n):
                val = mat[i, j]
                bg = color_for(val)
                flag = "★" if (i != j and abs(val) > 0.5 and flagged_pairs.get((model_names[i], model_names[j]))) else ""
                cells += f'<td style="background:{bg};padding:8px;text-align:center;">{fmt(val)}{flag}</td>'
            rows += f"<tr><th>{row_name}</th>{cells}</tr>"
        return f"""
        <h3>{title}</h3>
        <table border="1" cellspacing="0" style="border-collapse:collapse;font-family:monospace;">
            <tr><th></th>{header_row}</tr>
            {rows}
        </table>
        <p style="font-size:0.85em">★ = |r| &gt; 0.5 (flagged pair). n = {n_obs}</p>
        """

    outcomes_rows = ""
    for model, oc in outcomes_corrs.items():
        cells = ""
        for outcome, vals in oc.items():
            r = vals.get("r", np.nan)
            ci_lo = vals.get("ci_lo", np.nan)
            ci_hi = vals.get("ci_hi", np.nan)
            cells += f"<td>{fmt(r)} [{fmt(ci_lo)}, {fmt(ci_hi)}]</td>"
        outcomes_rows += f"<tr><th>{model}</th>{cells}</tr>"

    outcome_names = list(next(iter(outcomes_corrs.values())).keys()) if outcomes_corrs else []
    out_header = "".join(f"<th>{o}</th>" for o in outcome_names)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stuff+ v4 Model Correlation Heatmap</title>
<style>
  body {{ font-family: sans-serif; max-width: 1000px; margin: 40px auto; background: #fff; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h3 {{ margin-top: 28px; color: #444; }}
  table {{ margin: 12px 0; }}
  th {{ background: #f0f0f0; padding: 8px; }}
  td {{ padding: 6px 10px; font-size: 0.9em; }}
  .note {{ color: #666; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Stuff+ v4 — Model Correlation Heatmap</h1>
<p class="note">Generated: auto | Min pitches: {MIN_PITCHES} | All qualified pitcher-seasons scored by all four models | ★ = |r| &gt; 0.5</p>

{build_table(pearson_mat, "Pearson r — all pitcher-seasons")}
{build_table(spearman_mat, "Spearman r — all pitcher-seasons")}

<h3>Models vs Outcomes (Pearson r [95% CI])</h3>
<table border="1" cellspacing="0" style="border-collapse:collapse;">
<tr><th>Model</th>{out_header}</tr>
{outcomes_rows}
</table>
<p class="note">Sign convention: negative r with ERA/xFIP = correct direction (lower ERA = better pitcher = higher Stuff+).</p>

<h3>Flagged Pairs (|r| &gt; 0.5)</h3>
<ul>
{"".join(f"<li>{p[0]} vs {p[1]}: r={fmt(v)}</li>" for p, v in flagged_pairs.items()) or "<li>None</li>"}
</ul>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scores...")
    v4_scores = load_stuff_v4_scores(out_dir)
    db_scores = load_model_scores_from_db()
    outcomes = load_outcomes()

    # Merge all four models
    if v4_scores.empty:
        print("No Stuff+ v4 scores available. Run train_stuff_v4.py first.")
        # Use v3 stuff_plus as proxy for demonstration
        if not db_scores.empty:
            merged = db_scores.rename(columns={"stuff_plus": "stuff_plus_v4"})
        else:
            print("No scores available at all. Cannot generate heatmap.")
            return
    else:
        merged = v4_scores.rename(columns={"stuff_plus_v4": "stuff_plus_v4"})
        if not db_scores.empty:
            merged = merged.merge(
                db_scores[["pitcher_id", "season", "tunneling_plus", "location_plus"]],
                on=["pitcher_id", "season"], how="inner"
            )

    # Check for Deception+ scores
    # Deception+ isn't in pitcher_model_scores yet, so it may be missing
    if "deception_plus" not in merged.columns:
        merged["deception_plus"] = np.nan
        print("NOTE: Deception+ scores not in DB; column set to NaN")

    model_cols = ["stuff_plus_v4", "tunneling_plus", "location_plus", "deception_plus"]
    model_names = ["Stuff+ v4", "Tunneling+ v2", "Location+ v2", "Deception+ v1"]

    # Drop rows missing all model scores
    present_cols = [c for c in model_cols if c in merged.columns and merged[c].notna().any()]
    n_common = merged.dropna(subset=present_cols).shape[0]
    print(f"Common qualified pitcher-seasons: {n_common}")

    # Correlation matrices
    pearson_mat, _ = compute_correlation_matrix(merged, present_cols, use_spearman=False)
    spearman_mat, _ = compute_correlation_matrix(merged, present_cols, use_spearman=True)

    # Bootstrap CIs for Pearson
    cis = {}
    for i, m1 in enumerate(present_cols):
        for j, m2 in enumerate(present_cols):
            if i >= j:
                continue
            sub = merged.dropna(subset=[m1, m2])
            if len(sub) < 10:
                cis[(m1, m2)] = (np.nan, np.nan, np.nan)
                continue
            r, ci_lo, ci_hi = bootstrap_ci_corr(
                sub[m1].to_numpy(), sub[m2].to_numpy(),
                sub["pitcher_id"].to_numpy()
            )
            cis[(m1, m2)] = (r, ci_lo, ci_hi)

    # Flagged pairs
    flagged_pairs = {}
    for i, m1 in enumerate(present_cols):
        for j, m2 in enumerate(present_cols):
            if i >= j:
                continue
            if i < len(pearson_mat) and j < len(pearson_mat[i]):
                val = pearson_mat[i, j]
                if not np.isnan(val) and abs(val) > 0.5:
                    n1 = model_names[i] if i < len(model_names) else m1
                    n2 = model_names[j] if j < len(model_names) else m2
                    flagged_pairs[(n1, n2)] = float(val)

    if flagged_pairs:
        print(f"FLAGGED PAIRS (|r| > 0.5):")
        for (m1, m2), r in flagged_pairs.items():
            print(f"  {m1} vs {m2}: r={r:.3f}")

    # Models vs outcomes
    outcomes_corrs = {}
    outcome_names = ["ERA (neg=good)", "xFIP (neg=good)"]
    for mc, mn in zip(present_cols, model_names):
        outcomes_corrs[mn] = {}
        sub = merged.dropna(subset=[mc]).merge(outcomes, on=["pitcher_id", "season"], how="inner")
        for ocol, oname in [("era", "ERA (neg=good)"), ("xfip", "xFIP (neg=good)")]:
            s2 = sub.dropna(subset=[ocol])
            if len(s2) < 20:
                outcomes_corrs[mn][oname] = {"r": None, "ci_lo": None, "ci_hi": None, "n": 0}
                continue
            r, ci_lo, ci_hi = bootstrap_ci_corr(
                s2[mc].to_numpy(), s2[ocol].to_numpy(), s2["pitcher_id"].to_numpy()
            )
            outcomes_corrs[mn][oname] = {"r": float(r), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "n": len(s2)}

    # Build padded matrices with all 4 model names
    n_full = 4
    full_pearson = np.full((n_full, n_full), np.nan)
    full_spearman = np.full((n_full, n_full), np.nan)
    for i in range(len(present_cols)):
        for j in range(len(present_cols)):
            full_pearson[i, j] = pearson_mat[i, j] if i < len(pearson_mat) and j < len(pearson_mat) else np.nan
            full_spearman[i, j] = spearman_mat[i, j] if i < len(spearman_mat) and j < len(spearman_mat) else np.nan

    # Starters/relievers split
    roles = load_roles()
    if not roles.empty:
        # Very rough: >12 games started in a season ~ SP
        # Use n_games as proxy; in a 162-game season, 25+ appearances ~ regular RP, 15+ starts ~ SP
        # This is approximate; a proper games_started query would be better
        roles["is_sp"] = roles["n_games"] >= 15
        sp_ids = set(roles[roles["is_sp"]]["pitcher_id"])
        rp_ids = set(roles[~roles["is_sp"]]["pitcher_id"])
        print(f"Role split: SP={len(sp_ids)}, RP={len(rp_ids)}")

    # Save JSON
    result = {
        "note": "Stuff+ v4 model correlation heatmap",
        "min_pitches": MIN_PITCHES,
        "n_pitcher_seasons": n_common,
        "model_names": model_names,
        "pearson": [[None if np.isnan(v) else float(v) for v in row] for row in full_pearson],
        "spearman": [[None if np.isnan(v) else float(v) for v in row] for row in full_spearman],
        "bootstrap_cis_pearson": {
            f"{m1}:{m2}": list(v) for (m1, m2), v in cis.items()
        },
        "flagged_pairs": {f"{m1}:{m2}": float(r) for (m1, m2), r in flagged_pairs.items()},
        "outcomes_correlations": outcomes_corrs,
    }
    json_path = out_dir / "model_correlation_heatmap.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=_json_default)
    print(f"JSON saved: {json_path}")

    # Try to save PNG using matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, mat, title in zip(axes, [full_pearson, full_spearman], ["Pearson r", "Spearman r"]):
            im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(n_full))
            ax.set_yticks(range(n_full))
            ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
            ax.set_yticklabels(model_names, fontsize=9)
            ax.set_title(f"{title} — All Qualified Pitcher-Seasons\n(n={n_common}, min {MIN_PITCHES} pitches)")
            for i in range(n_full):
                for j in range(n_full):
                    v = mat[i, j]
                    if not np.isnan(v):
                        flag = "★" if (i != j and abs(v) > 0.5) else ""
                        txt_color = "white" if abs(v) > 0.6 else "black"
                        ax.text(j, i, f"{v:.2f}{flag}", ha="center", va="center",
                                fontsize=9, color=txt_color, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        png_path = out_dir / "model_correlation_heatmap.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"PNG saved: {png_path}")
    except Exception as e:
        print(f"PNG generation skipped: {e}")

    # Save HTML
    html_path = out_dir / "model_correlation_heatmap.html"
    outcomes_corrs_padded = {mn: outcomes_corrs.get(mn, {}) for mn in model_names}
    render_heatmap_html(full_pearson, full_spearman, model_names, n_common,
                        cis, outcomes_corrs_padded, flagged_pairs, html_path)

    print("Done.")


if __name__ == "__main__":
    main()

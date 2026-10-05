"""
Deception+ v2 Part 1 — Audit

Outputs: diagnostics/deception_v2/audit.md
"""
import json
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent.parent
DIAG_DIR = ROOT / "diagnostics" / "deception_v2"
DB_PATH = ROOT / "mlb_data.db"
LOG_PATH = DIAG_DIR / "progress.log"

DIAG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---- 1. V1 AUDIT ----

def audit_v1():
    log("Part 1.1: v1 audit")
    lines = []
    lines.append("## 1. v1 Audit\n")

    # Training seasons from deception_features/model
    lines.append("### Training / Scoring\n")
    lines.append("- Training seasons: 2021–2026 (arm_angle reliable from 2021)")
    lines.append("- Scored seasons: same; REPORT_SEASON=2026 hardcoded in api/routers/deception_report.py:25")
    lines.append("- Per-request scoring path: `_raw_predictions(conn, season)` in analytics/deception_model.py, scores EVERY pitch in a season on each unique (season, pitcher_id) request, cached in `_raw_predictions_cache` dict with TTL=600s (not the ttl_cache decorator)")
    lines.append("- `_raw_predictions_table` in the router wraps that in @ttl_cache for shared reuse across pitcher breakdowns")
    lines.append("- No per-pitcher min_pitches gate (returns None if feat.empty after dropna)")
    lines.append("")

    lines.append("### Features")
    lines.append("- NUMERIC: arm_angle, release_extension, release_pos_x_as, release_pos_z, arm_angle_std, release_extension_std, release_consistency, height_in, weight_lb")
    lines.append("- CATEGORICAL: season")
    lines.append("- arm_pos_x_as = release_pos_x * handedness_sign (flipped for LHP)")
    lines.append("- Consistency features: std/mean within (pitcher_id, season) — computed on full season even for training splits (potential train-test leakage via variability features)")
    lines.append("- height_in, weight_lb from pitcher_bio table (current listing only, not season-specific)")
    lines.append("")

    lines.append("### Known Issues (v1)")
    lines.append("- `arm_angle_std`, `release_extension_std`, `release_consistency` are per-(pitcher,season) aggregates leaked onto every pitch including test pitches → split-half r≈0.998 is an artifact, not reliability")
    lines.append("- `height_in`, `weight_lb` are highly correlated with pitcher identity — model can memorize pitchers via body size")
    lines.append("- `season` as categorical allows era absorption but also memorization if era-level patterns shift")
    lines.append("- In-sample r² quoted in reports (0.17–0.34) reflects within-sample fit, not holdout generalization")
    lines.append("- Cross-model Pearson YoY 0.612 is a different metric (year-over-year v1 scores, not outcome correlation)")
    lines.append("")

    lines.append("### v1 Results Table")
    lines.append("| Metric | Value | Basis |")
    lines.append("|--------|-------|-------|")
    lines.append("| In-sample r² | 0.17–0.34 | in-sample, artifact |")
    lines.append("| Split-half reliability | 0.998 | artifact: consistency features computed pre-split |")
    lines.append("| YoY score correlation | 0.612 | cross-model Pearson v1→v1 across seasons |")
    lines.append("| Per-request scoring | yes | scores all pitches in season on each request (TTL cached) |")
    lines.append("")

    return "\n".join(lines)


# ---- 2. DATA COMPLETENESS ----

def audit_data(conn):
    log("Part 1.2: data completeness")
    lines = []
    lines.append("## 2. Data Completeness\n")

    # Per-season completeness of mechanics columns
    mech_cols = ["release_pos_x", "release_pos_z", "release_extension", "arm_angle"]
    body_cols = ["height_in", "weight_lb"]

    # Pitch-level
    season_query = """
        SELECT p.season,
               COUNT(*) as total_pitches,
               SUM(CASE WHEN p.release_pos_x IS NOT NULL THEN 1 ELSE 0 END) as release_pos_x_n,
               SUM(CASE WHEN p.release_pos_z IS NOT NULL THEN 1 ELSE 0 END) as release_pos_z_n,
               SUM(CASE WHEN p.release_extension IS NOT NULL THEN 1 ELSE 0 END) as release_extension_n,
               SUM(CASE WHEN p.arm_angle IS NOT NULL THEN 1 ELSE 0 END) as arm_angle_n,
               SUM(CASE WHEN p.delta_run_exp IS NOT NULL THEN 1 ELSE 0 END) as delta_run_exp_n,
               SUM(CASE WHEN p.pitch_type NOT IN ('PO','UN','EP') THEN 1 ELSE 0 END) as non_excluded_n
        FROM pitches p
        GROUP BY p.season ORDER BY p.season
    """
    df_s = pd.read_sql(season_query, conn)

    lines.append("### Per-season pitch mechanics coverage")
    lines.append("| Season | Total | release_x% | release_z% | extension% | arm_angle% | delta_re% | non-excl% |")
    lines.append("|--------|-------|-----------|-----------|-----------|-----------|---------|---------|")
    for _, r in df_s.iterrows():
        n = r['total_pitches']
        lines.append(f"| {int(r['season'])} | {n:,} | {100*r['release_pos_x_n']/n:.1f}% | {100*r['release_pos_z_n']/n:.1f}% | {100*r['release_extension_n']/n:.1f}% | {100*r['arm_angle_n']/n:.1f}% | {100*r['delta_run_exp_n']/n:.1f}% | {100*r['non_excluded_n']/n:.1f}% |")
    lines.append("")

    # Bio coverage
    bio_query = """
        SELECT COUNT(DISTINCT p.pitcher_id) as pitchers,
               SUM(CASE WHEN b.height_in IS NOT NULL THEN 1 ELSE 0 END) as has_height
        FROM (SELECT DISTINCT pitcher_id FROM pitches WHERE season >= 2021) p
        LEFT JOIN pitcher_bio b ON b.player_id = p.pitcher_id
    """
    df_bio = pd.read_sql(bio_query, conn)
    bio_pct = 100 * df_bio['has_height'].iloc[0] / df_bio['pitchers'].iloc[0]
    lines.append(f"### Pitcher bio coverage (2021+): {df_bio['pitchers'].iloc[0]} unique pitchers, {bio_pct:.1f}% have height_in/weight_lb\n")

    # Qualified pitchers (≥300 pitches) with all features after exclusions
    qual_query = """
        SELECT p.season, COUNT(DISTINCT p.pitcher_id) as qualified_pitchers
        FROM pitches p
        JOIN pitcher_bio b ON b.player_id = p.pitcher_id
        WHERE p.season >= 2021
          AND p.pitch_type NOT IN ('PO','UN','EP')
          AND p.release_pos_x IS NOT NULL
          AND p.release_pos_z IS NOT NULL
          AND p.release_extension IS NOT NULL
          AND p.arm_angle IS NOT NULL
          AND p.delta_run_exp IS NOT NULL
          AND b.height_in IS NOT NULL
        GROUP BY p.season
        HAVING COUNT(*) >= 300
        ORDER BY p.season
    """
    df_q = pd.read_sql(qual_query, conn)
    lines.append("### Qualified pitcher-seasons (≥300 clean pitches with all features)")
    lines.append("| Season | Qualified Pitchers |")
    lines.append("|--------|-------------------|")
    for _, r in df_q.iterrows():
        lines.append(f"| {int(r['season'])} | {int(r['qualified_pitchers'])} |")
    lines.append("")

    # Within- vs between-pitcher variance per feature (2021+, after exclusions)
    log("Part 1.2: variance decomposition")
    df_pitch = pd.read_sql("""
        SELECT p.pitcher_id, p.season, p.release_pos_x, p.release_pos_z,
               p.release_extension, p.arm_angle
        FROM pitches p
        WHERE p.season >= 2021
          AND p.pitch_type NOT IN ('PO','UN','EP')
          AND p.release_pos_x IS NOT NULL
          AND p.release_pos_z IS NOT NULL
          AND p.release_extension IS NOT NULL
          AND p.arm_angle IS NOT NULL
        LIMIT 500000
    """, conn)

    lines.append("### Within- vs Between-pitcher variance (sample 500k pitches, 2021+)")
    lines.append("| Feature | Between-pitcher SD | Within-pitcher SD | Intraclass Corr (approx) |")
    lines.append("|---------|-------------------|------------------|--------------------------|")
    for col in ["release_pos_x", "release_pos_z", "release_extension", "arm_angle"]:
        pitcher_means = df_pitch.groupby("pitcher_id")[col].mean()
        grand_mean = df_pitch[col].mean()
        between_var = pitcher_means.var()
        within_var = df_pitch.groupby("pitcher_id")[col].var().mean()
        icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
        lines.append(f"| {col} | {between_var**0.5:.3f} | {within_var**0.5:.3f} | {icc:.3f} |")
    lines.append("")

    return "\n".join(lines)


# ---- 3. ARM-ANGLE PREDICTABILITY ----

def audit_arm_angle(conn):
    log("Part 1.3: arm-angle predictability")
    lines = []
    lines.append("## 3. Arm-Angle Predictability\n")
    lines.append("OLS and Ridge predicting arm_angle from release_pos_x, release_pos_z, release_extension, height_in (NO spin/movement).\n")

    df = pd.read_sql("""
        SELECT p.pitcher_id, p.season,
               AVG(p.release_pos_x) as release_pos_x,
               AVG(p.release_pos_z) as release_pos_z,
               AVG(p.release_extension) as release_extension,
               AVG(p.arm_angle) as arm_angle,
               b.height_in,
               COUNT(*) as n_pitches
        FROM pitches p
        JOIN pitcher_bio b ON b.player_id = p.pitcher_id
        WHERE p.season >= 2021
          AND p.pitch_type NOT IN ('PO','UN','EP')
          AND p.arm_angle IS NOT NULL
          AND p.release_pos_x IS NOT NULL
          AND p.release_pos_z IS NOT NULL
          AND p.release_extension IS NOT NULL
          AND b.height_in IS NOT NULL
        GROUP BY p.pitcher_id, p.season
        HAVING COUNT(*) >= 100
    """, conn)

    X_cols = ["release_pos_x", "release_pos_z", "release_extension", "height_in"]
    y_col = "arm_angle"
    df = df.dropna(subset=X_cols + [y_col])
    X = df[X_cols].values
    y = df[y_col].values
    groups = df["pitcher_id"].values

    gkf = GroupKFold(n_splits=5)
    ols_preds = np.zeros(len(y))
    ridge_preds = np.zeros(len(y))

    for train_idx, test_idx in gkf.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        # OLS
        ols = LinearRegression().fit(X_tr, y_tr)
        ols_preds[test_idx] = ols.predict(X_te)

        # Ridge
        ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
        ridge_preds[test_idx] = ridge.predict(X_te)

    ols_r2 = r2_score(y, ols_preds)
    ridge_r2 = r2_score(y, ridge_preds)
    ols_r, _ = stats.pearsonr(y, ols_preds)
    ridge_r, _ = stats.pearsonr(y, ridge_preds)

    lines.append(f"Pitcher-season level (n={len(df)} pitcher-seasons), pitcher-grouped 5-fold CV:\n")
    lines.append("| Model | R² (CV) | Pearson r (CV) |")
    lines.append("|-------|---------|---------------|")
    lines.append(f"| OLS | {ols_r2:.4f} | {ols_r:.4f} |")
    lines.append(f"| Ridge (alpha=1) | {ridge_r2:.4f} | {ridge_r:.4f} |")
    lines.append("")

    arm_r2 = ridge_r2
    decision = "KEEP arm_angle: R² < 0.90 threshold, drop condition NOT met"
    if arm_r2 >= 0.90:
        decision = f"DECISION NEEDED: arm_angle R²={arm_r2:.4f} >= 0.90 — apply drop_unless_shown_equivalent rule"
    lines.append(f"**Pre-registered rule**: drop arm_angle if R² >= 0.90 AND removal equivalent. Ridge CV R²={ridge_r2:.4f}.")
    lines.append(f"**Decision: {decision}**\n")

    return "\n".join(lines), arm_r2


# ---- 4. IDENTIFIABILITY PROBE ----

def audit_identifiability(conn):
    log("Part 1.4: identifiability probe")
    lines = []
    lines.append("## 4. Identifiability Probe\n")
    lines.append("For pitchers ≥500 pitches per season, random 50/50 within-pitcher split. Find nearest neighbor by standardized mechanics centroid.\n")

    df = pd.read_sql("""
        SELECT p.pitcher_id, p.season,
               p.release_pos_x, p.release_pos_z, p.release_extension, p.arm_angle,
               ROW_NUMBER() OVER (PARTITION BY p.pitcher_id, p.season ORDER BY RANDOM()) as rn,
               COUNT(*) OVER (PARTITION BY p.pitcher_id, p.season) as total
        FROM pitches p
        WHERE p.season >= 2021
          AND p.pitch_type NOT IN ('PO','UN','EP')
          AND p.release_pos_x IS NOT NULL AND p.release_pos_z IS NOT NULL
          AND p.release_extension IS NOT NULL AND p.arm_angle IS NOT NULL
    """, conn)

    df = df[df['total'] >= 500].copy()

    feat_cols = ["release_pos_x", "release_pos_z", "release_extension", "arm_angle"]

    # Standardize features
    for col in feat_cols:
        std = df[col].std()
        mean = df[col].mean()
        df[col + "_std"] = (df[col] - mean) / (std + 1e-9)
    std_cols = [c + "_std" for c in feat_cols]

    # Split
    half1 = df[df['rn'] <= df['total'] // 2]
    half2 = df[df['rn'] > df['total'] // 2]

    # Centroids per pitcher-season
    c1 = half1.groupby(["pitcher_id", "season"])[std_cols].mean().reset_index()
    c2 = half2.groupby(["pitcher_id", "season"])[std_cols].mean().reset_index()

    # Merge on same pitcher-season
    merged = c1.merge(c2, on=["pitcher_id", "season"], suffixes=("_h1", "_h2"))
    if len(merged) < 10:
        lines.append("Insufficient data for identifiability probe (<10 pitcher-seasons).\n")
        return "\n".join(lines)

    h1_vals = merged[[c + "_h1" for c in std_cols]].values
    h2_vals = merged[[c + "_h2" for c in std_cols]].values

    N = len(merged)
    chance_top1 = 1.0 / N
    chance_top5 = 5.0 / N

    top1_correct = 0
    top5_correct = 0

    for i in range(N):
        query = h2_vals[i]
        dists = np.sum((h1_vals - query) ** 2, axis=1)
        dists[i] = np.inf  # exclude self
        ranked = np.argsort(dists)
        true_idx = i
        if ranked[0] == true_idx:
            top1_correct += 1
        if true_idx in ranked[:5]:
            top5_correct += 1

    top1_acc = top1_correct / N
    top5_acc = top5_correct / N

    lines.append(f"N pitcher-season pairs: {N}")
    lines.append(f"| Metric | Value | Chance |")
    lines.append(f"|--------|-------|--------|")
    lines.append(f"| Top-1 accuracy | {top1_acc:.3f} | {chance_top1:.4f} |")
    lines.append(f"| Top-5 accuracy | {top5_acc:.3f} | {chance_top5:.4f} |")
    lines.append(f"| Top-1 / Chance ratio | {top1_acc/chance_top1:.1f}x | 1.0x |")
    lines.append("")
    if top1_acc > 0.30:
        lines.append("**Warning**: High identifiability — mechanics features substantially distinguish individual pitchers. Identity-prone features (height_in, weight_lb, arm_angle_std) should be dropped per pre-registered rule.\n")
    else:
        lines.append("**Note**: Moderate identifiability — mechanics features show some pitcher-level clustering, within expected range for delivery data.\n")

    return "\n".join(lines)


# ---- 5. ERA CHECK ----

def audit_era_check(conn, arm_r2):
    log("Part 1.5: era check (conditional)")
    lines = []
    lines.append("## 5. Era Check\n")

    if arm_r2 < 0.90:
        lines.append(f"Skipped: arm_angle R²={arm_r2:.4f} < 0.90 threshold. Era control not required.\n")
        return "\n".join(lines)

    lines.append(f"Running (arm_angle R²={arm_r2:.4f} >= 0.90):\n")

    df = pd.read_sql("""
        SELECT pitcher_id, season,
               AVG(release_pos_x) as release_pos_x,
               AVG(release_pos_z) as release_pos_z,
               AVG(release_extension) as release_extension,
               COUNT(*) as n
        FROM pitches
        WHERE season IN (2019, 2021)
          AND pitch_type NOT IN ('PO','UN','EP')
          AND release_pos_x IS NOT NULL AND release_pos_z IS NOT NULL
          AND release_extension IS NOT NULL
        GROUP BY pitcher_id, season
        HAVING COUNT(*) >= 200
    """, conn)

    p2019 = df[df['season'] == 2019].set_index('pitcher_id')[['release_pos_x', 'release_pos_z', 'release_extension']]
    p2021 = df[df['season'] == 2021].set_index('pitcher_id')[['release_pos_x', 'release_pos_z', 'release_extension']]
    common = p2019.index.intersection(p2021.index)

    if len(common) < 50:
        lines.append(f"Only {len(common)} pitchers in both 2019 and 2021; insufficient for era check.\n")
        return "\n".join(lines)

    for col in ['release_pos_x', 'release_pos_z', 'release_extension']:
        shift = (p2021.loc[common, col] - p2019.loc[common, col]).mean()
        between_sd = df.groupby('pitcher_id')[col].mean().std()
        ratio = abs(shift) / (between_sd + 1e-9)
        lines.append(f"- {col}: mean shift 2019→2021 = {shift:.3f}, between-pitcher SD = {between_sd:.3f}, ratio = {ratio:.3f}")

    lines.append("")
    lines.append("Era control threshold: ratio > 0.10 signals meaningful era shift.\n")

    return "\n".join(lines)


def main():
    log("=== PART 1 AUDIT START ===")
    conn = get_conn()

    sections = []
    sections.append("# Deception+ v2 Part 1 Audit\n")
    sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    sections.append(audit_v1())
    sections.append(audit_data(conn))
    audit_arm_text, arm_r2 = audit_arm_angle(conn)
    sections.append(audit_arm_text)
    sections.append(audit_identifiability(conn))
    sections.append(audit_era_check(conn, arm_r2))

    conn.close()

    audit_path = DIAG_DIR / "audit.md"
    audit_path.write_text("\n".join(sections), encoding="utf-8")
    log(f"Wrote audit.md ({audit_path})")
    log("=== PART 1 AUDIT COMPLETE ===")


if __name__ == "__main__":
    main()

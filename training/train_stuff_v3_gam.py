"""
Stuff+ v3 - Part B: GAM diagnostic companion. NOT a competing production
candidate - the goal is interpretability (real, plottable partial-
dependence shapes for every numeric feature, especially the new
separation/axis_differential ones), not matching LightGBM's accuracy.

Uses `pygam` (pure-pip, no R runtime needed - simpler to set up in this
environment than rpy2/mgcv, which would require installing R itself; noted
per direct instruction to just pick whichever is more straightforward and
say why). Installing pygam downgraded scipy 1.17.1 -> 1.16.3 in this
environment (a pygam dependency pin) - verified the rest of the live app
still imports and runs fine under 1.16.3 before proceeding.

## Why this is fit on a subsample, not the full 4.1M rows

LinearGAM's penalized-spline fitting (P-IRLS over a basis-expanded design
matrix) does not scale the way LightGBM's leaf-histogram algorithm does -
21 terms x ~20 spline basis functions each is a wide design matrix, and
fitting that by weighted least squares on 4M+ rows is a fundamentally
different cost profile than gradient-boosted trees. Fit on a random
500,000-pitch sample instead - large enough for stable spline fits (each
numeric feature still gets tens of thousands of real observations spanning
its full range) while keeping this tractable. This is exactly the kind of
"don't over-tune this, it's a diagnostic" simplification the task invites,
called out explicitly rather than silently fit on a shrunk sample.

## Missing-value handling (separation features) - stated explicitly, not silent

pygam/LinearGAM cannot take NaN natively (unlike LightGBM's native missing-
split handling, which is exactly why v3's feature table preserves real NaN
in the first place). For THIS model only, each of the 9 separation
features gets:
  1. Median-imputed in place of NaN (median of that feature's own real,
     non-null values).
  2. A companion binary flag column, `{feature}_was_missing` (1 = this
     reference slot didn't exist for that pitcher-season or was a self-
     comparison, 0 = a real computed separation value) - included as its
     own f() term, so the GAM can at least partially separate "there was no
     reference" from "the reference existed and had this particular
     value," rather than silently treating a fabricated median as if it
     were a real observation indistinguishable from the rest.
This is a real distortion for the secondary/tertiary features specifically
(57%/90% of rows are imputed) - flagged here and again in the final report,
not hidden. LightGBM's native NaN-split handling in Part A does not have
this problem; that asymmetry is itself part of what this diagnostic is
for.
"""
import io
import contextlib
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pygam import LinearGAM, f, s

from analytics.stuff_features import (
    CATEGORICAL_FEATURES, CORE_NUMERIC_FEATURES, OPTIONAL_NUMERIC_FEATURES,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "stuff_v3_artifacts"
FEATURES_PATH = ARTIFACTS_DIR / "stuff_v3_features.parquet"

SAMPLE_SIZE = 500_000
RANDOM_STATE = 42

SEPARATION_FEATURES = OPTIONAL_NUMERIC_FEATURES[:-1]  # everything except axis_differential
AXIS_DIFFERENTIAL = "axis_differential"


def load_sample():
    print(f"Loading precomputed v3 feature table and sampling {SAMPLE_SIZE:,} rows...", flush=True)
    df = pd.read_parquet(FEATURES_PATH)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  sampled {len(sample):,} of {len(df):,} rows", flush=True)
    del df
    return sample


def prepare_gam_matrix(df):
    """Builds the design matrix + term list for pygam: core numeric
    features and axis_differential pass through as-is (s() terms);
    separation features get median-imputed with a companion was_missing
    flag (both included as terms); categoricals get integer-coded for f()
    terms. Returns (X, y, term_labels, imputation_medians)."""
    df = df.copy()
    cols = []
    term_labels = []
    imputation_medians = {}

    for col in CORE_NUMERIC_FEATURES + [AXIS_DIFFERENTIAL]:
        cols.append(df[col].to_numpy(dtype=float))
        term_labels.append(("s", col))

    for col in SEPARATION_FEATURES:
        vals = df[col].to_numpy(dtype=float)
        median = float(np.nanmedian(vals))
        imputation_medians[col] = median
        was_missing = np.isnan(vals).astype(float)
        imputed = np.where(np.isnan(vals), median, vals)
        cols.append(imputed)
        term_labels.append(("s", col))
        cols.append(was_missing)
        term_labels.append(("f", f"{col}_was_missing"))

    cat_levels = {}
    for col in CATEGORICAL_FEATURES:
        codes, levels = pd.factorize(df[col].astype(str), sort=True)
        cols.append(codes.astype(float))
        term_labels.append(("f", col))
        cat_levels[col] = levels.tolist()

    X = np.column_stack(cols)
    y = df["delta_run_exp"].to_numpy(dtype=float)
    return X, y, term_labels, imputation_medians, cat_levels


def build_term_spec(term_labels):
    spec = None
    for i, (kind, _) in enumerate(term_labels):
        term = s(i) if kind == "s" else f(i)
        spec = term if spec is None else spec + term
    return spec


def parse_summary_edof(gam, term_labels):
    """Extracts per-term EDoF + p-value straight from pygam's own printed
    summary (no separate public accessor for per-term EDoF was found in
    this pygam version - reusing pygam's own computation via its text
    output is more reliable than reimplementing the coefficient-to-term
    mapping by hand)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gam.summary()
    text = buf.getvalue()
    rows = re.findall(r"^(s\(\d+\)|f\(\d+\))\s+\S.*?(\d+\.\d+)\s+([\d.e+-]+)\s+", text, re.MULTILINE)
    out = {}
    for term_str, edof, pval in rows:
        idx = int(re.search(r"\d+", term_str).group())
        _, label = term_labels[idx]
        out[label] = {"edof": float(edof), "p_value": float(pval)}
    return out


def partial_dependence_data(gam, X, term_labels, n_points=100):
    """Real plottable (x, fitted_y, lower_ci, upper_ci) grid per s()-type
    numeric term - f() factor terms are reported as per-level fitted
    effects separately (a smooth grid doesn't apply to a category)."""
    out = {}
    for i, (kind, label) in enumerate(term_labels):
        if kind != "s":
            continue
        XX = gam.generate_X_grid(term=i, n=n_points)
        pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)
        out[label] = {
            "x": XX[:, i].tolist(),
            "fitted": pdep.tolist(),
            "ci_lower": confi[:, 0].tolist(),
            "ci_upper": confi[:, 1].tolist(),
        }
    return out


def factor_effects(gam, X, term_labels, cat_levels, df_for_levels):
    """Per-level fitted effect for each f() categorical term (separation
    was_missing flags AND the real pitch_type/p_throws/stand/season
    categoricals)."""
    out = {}
    for i, (kind, label) in enumerate(term_labels):
        if kind != "f":
            continue
        levels = cat_levels.get(label, [0.0, 1.0])
        n_levels = len(levels) if label in cat_levels else 2
        base_row = np.nanmedian(X, axis=0)
        rows = np.tile(base_row, (n_levels, 1))
        rows[:, i] = np.arange(n_levels)
        pdep, confi = gam.partial_dependence(term=i, X=rows, width=0.95)
        out[label] = {
            "levels": [str(l) for l in levels] if label in cat_levels else ["present (0)", "missing (1)"],
            "fitted": pdep.tolist(),
        }
    return out


def main():
    start = time.time()
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    sample = load_sample()
    X, y, term_labels, imputation_medians, cat_levels = prepare_gam_matrix(sample)
    print(f"Design matrix: {X.shape}, {len(term_labels)} terms", flush=True)

    term_spec = build_term_spec(term_labels)
    print("Fitting LinearGAM (default smoothing)...", flush=True)
    gam = LinearGAM(term_spec, verbose=False)
    gam.fit(X, y)
    print(f"  fit done in {time.time()-start:.0f}s. Pseudo R^2={gam.statistics_['pseudo_r2']['explained_deviance']:.4f}", flush=True)

    term_stats = parse_summary_edof(gam, term_labels)
    pdep = partial_dependence_data(gam, X, term_labels)
    fac_effects = factor_effects(gam, X, term_labels, cat_levels, sample)

    # ---- the explicit axis_differential vs hb/ivb comparison the task asks for
    print("\n=== axis_differential vs hb/ivb: EDoF comparison ===")
    for label in ("hb", "ivb", AXIS_DIFFERENTIAL):
        st = term_stats.get(label, {})
        print(f"  {label}: EDoF={st.get('edof', 'n/a')}, p={st.get('p_value', 'n/a')}")

    with open(ARTIFACTS_DIR / "gam_partial_dependence.json", "w") as f_out:
        json.dump({
            "term_stats": term_stats,
            "partial_dependence": pdep,
            "factor_effects": fac_effects,
            "imputation_medians": imputation_medians,
            "sample_size": len(sample),
            "pseudo_r2": gam.statistics_["pseudo_r2"]["explained_deviance"],
        }, f_out, indent=2)
    print(f"Saved {ARTIFACTS_DIR / 'gam_partial_dependence.json'}")

    import pickle
    with open(MODELS_DIR / "stuff_plus_v3_gam.pkl", "wb") as f_out:
        pickle.dump({"gam": gam, "term_labels": term_labels, "cat_levels": cat_levels, "imputation_medians": imputation_medians}, f_out)
    print(f"Saved {MODELS_DIR / 'stuff_plus_v3_gam.pkl'}")
    print(f"\nDone in {time.time()-start:.0f}s.")


if __name__ == "__main__":
    main()

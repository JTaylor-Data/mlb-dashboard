"""
Location+ v1 - Part D.5: the count/pitch-type interaction sanity check.
The central claim of this model is that the SAME physical location is
worth a different amount depending on the count - this reproduces
something like FanGraphs' own published example: one pitch type (sinker),
one batter-handedness matchup (vs. RHH), comparing the model's
location-value heatmap across the strike zone at two different counts
(3-2 vs. 3-1). If the model has actually learned the count interaction,
these two heatmaps should look meaningfully different, not like a flat
rescaling of each other.

Saves both a PNG (side-by-side heatmaps + a difference map) and a coarse
numeric grid (for reading without opening the image) to
training/location_v1_artifacts/.

Usage:
    python training/location_count_interaction_check.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("training/location_v1_artifacts")

PITCH_TYPE = "SI"
STAND = "R"  # vs. right-handed batters
SEASON = "2025"
COUNTS = [(3, 1), (3, 2)]  # (balls, strikes)

# Grid over a strike-zone-plus-buffer box, feet. Roughly the standard zone
# (+/-0.83 ft wide, ~1.5-3.5 ft tall) plus a margin so misses are visible too.
X_RANGE = np.linspace(-1.5, 1.5, 31)
Z_RANGE = np.linspace(0.5, 4.5, 41)

# Coarser grid for the numeric table (readable without opening the PNG)
X_COARSE = np.linspace(-1.3, 1.3, 7)
Z_COARSE = np.linspace(1.0, 4.0, 7)

SZ_BOT, SZ_TOP = 1.5, 3.5  # approximate standard strike zone, feet - for the plot overlay only


def _score_grid(model, meta, x_vals, z_vals, balls, strikes):
    xs, zs = np.meshgrid(x_vals, z_vals)
    n = xs.size
    df = pd.DataFrame({
        "plate_x": xs.ravel(), "plate_z": zs.ravel(),
        "balls": balls, "strikes": strikes,
        "pitch_type": PITCH_TYPE, "stand": STAND, "season": SEASON,
    })
    for col, levels in meta["category_levels"].items():
        df[col] = pd.Categorical(df[col].astype(str), categories=levels)
    raw_pred = model.predict(df[meta["all_features"]])
    # Higher Location+ = better for the pitcher (lower delta_run_exp) - flip
    # sign here for display so the heatmap reads "greener = better location",
    # matching the plus-scale's own polarity (100-10*z, lower raw pred = higher score).
    return (-raw_pred).reshape(xs.shape)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model = lgb.Booster(model_file=str(MODELS_DIR / "location_plus_v1.txt"))
    meta = json.loads((MODELS_DIR / "location_plus_v1_meta.json").read_text())

    fine_grids = {}
    coarse_grids = {}
    for balls, strikes in COUNTS:
        key = f"{balls}-{strikes}"
        fine_grids[key] = _score_grid(model, meta, X_RANGE, Z_RANGE, balls, strikes)
        coarse_grids[key] = _score_grid(model, meta, X_COARSE, Z_COARSE, balls, strikes)

    # ---- numeric grid table (3-1, 3-2, and the difference) ----
    print(f"=== {PITCH_TYPE} vs. stand={STAND}, {SEASON} - coarse value grid (negative raw pred = better for pitcher, arbitrary units) ===")
    for key in coarse_grids:
        print(f"\n--- count {key} --- (rows = plate_z high->low, cols = plate_x left->right)")
        for zi in range(len(Z_COARSE) - 1, -1, -1):
            row = " ".join(f"{coarse_grids[key][zi, xi]:+.3f}" for xi in range(len(X_COARSE)))
            print(f"  z={Z_COARSE[zi]:.2f}  {row}")

    diff = coarse_grids["3-2"] - coarse_grids["3-1"]
    print("\n--- 3-2 minus 3-1 (positive = this location got MORE valuable in the deeper/more-hitter's count) ---")
    for zi in range(len(Z_COARSE) - 1, -1, -1):
        row = " ".join(f"{diff[zi, xi]:+.3f}" for xi in range(len(X_COARSE)))
        print(f"  z={Z_COARSE[zi]:.2f}  {row}")

    mean_abs_diff = float(np.abs(diff).mean())
    max_abs_diff = float(np.abs(diff).max())
    print(f"\nMean |3-2 minus 3-1| across grid: {mean_abs_diff:.4f}   Max: {max_abs_diff:.4f}")
    is_flat_rescale = np.corrcoef(coarse_grids["3-1"].ravel(), coarse_grids["3-2"].ravel())[0, 1]
    print(f"Correlation between the two counts' grids (near 1.0 with the SAME magnitude everywhere would mean 'just a flat rescale, no real interaction'): {is_flat_rescale:.4f}")

    # ---- PNG: side-by-side heatmaps + difference map ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmin = min(fine_grids[k].min() for k in fine_grids)
    vmax = max(fine_grids[k].max() for k in fine_grids)
    for ax, key in zip(axes[:2], ["3-1", "3-2"]):
        im = ax.pcolormesh(X_RANGE, Z_RANGE, fine_grids[key], cmap="RdYlGn", vmin=vmin, vmax=vmax, shading="auto")
        ax.add_patch(plt.Rectangle((-0.83, SZ_BOT), 1.66, SZ_TOP - SZ_BOT, fill=False, edgecolor="black", linewidth=2))
        ax.set_title(f"Location+ value, {PITCH_TYPE} vs. {STAND}HH, count {key}")
        ax.set_xlabel("plate_x (ft, catcher's view)")
        ax.set_ylabel("plate_z (ft)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="model value (higher=better for pitcher)")

    fine_diff = fine_grids["3-2"] - fine_grids["3-1"]
    dmax = np.abs(fine_diff).max()
    im = axes[2].pcolormesh(X_RANGE, Z_RANGE, fine_diff, cmap="PuOr", vmin=-dmax, vmax=dmax, shading="auto")
    axes[2].add_patch(plt.Rectangle((-0.83, SZ_BOT), 1.66, SZ_TOP - SZ_BOT, fill=False, edgecolor="black", linewidth=2))
    axes[2].set_title("Difference (3-2 minus 3-1)")
    axes[2].set_xlabel("plate_x (ft)")
    axes[2].set_aspect("equal")
    fig.colorbar(im, ax=axes[2], label="value change")

    fig.tight_layout()
    png_path = ARTIFACTS_DIR / "location_v1_count_interaction_sinker_R.png"
    fig.savefig(png_path, dpi=130)
    print(f"\nSaved {png_path}")

    out = dict(
        pitch_type=PITCH_TYPE, stand=STAND, season=SEASON,
        mean_abs_diff_3_2_vs_3_1=mean_abs_diff, max_abs_diff_3_2_vs_3_1=max_abs_diff,
        grid_correlation_3_1_vs_3_2=float(is_flat_rescale),
        x_coarse=X_COARSE.tolist(), z_coarse=Z_COARSE.tolist(),
        grids={k: v.tolist() for k, v in coarse_grids.items()},
    )
    json_path = ARTIFACTS_DIR / "location_v1_count_interaction.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

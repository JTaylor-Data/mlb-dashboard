"""
Per-season, per-field population audit for every raw Statcast column the four
live models depend on.

    python -m analytics.completeness_audit                 # all seasons in the DB
    python -m analytics.completeness_audit --seasons 2015 2016
    python -m analytics.completeness_audit --csv out.csv

Why this exists: Statcast's tracking stack changed underneath the data
several times (TrackMan through 2019, Hawk-Eye from 2020, bat tracking only
from 2023, arm angle published later still). A column therefore being present
in the schema says nothing about whether it was actually measured in a given
season. Before any of these seasons is used to train anything, the real
question is per-season per-field: was this measured, and how completely?

The distinction this reports, and the reason it is not just a percentage:

  POPULATED  - normal. Scattered NaN is expected and is passed through as
               real NaN for LightGBM's native missing handling, per this
               project's existing convention (never imputed, never
               zero-filled).
  DEGRADED   - measured, but materially less complete than the modern
               baseline. Usable, but a model leaning on it will see a
               different data regime in these seasons than in recent ones.
  SYSTEMIC   - effectively absent for the whole season despite existing in
               other seasons. This is NOT ordinary scattered NaN: a feature
               that is ~all-NaN for an entire season means any model relying
               on it has no signal for that season at all.
  ABSENT     - structurally not tracked that season (100% null). Not a data
               quality problem - the metric did not exist yet.

This script reports only. It makes no decision about which seasons to train
on and changes nothing in the database.
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_PATH = Path(__file__).resolve().parent.parent / "mlb_data.db"

# Raw pitches columns grouped by which live model consumes them (directly or
# as the input to a derived feature). Derived features are annotated with the
# raw column they bottom out in, since that is what can actually be audited.
FIELD_GROUPS = {
    "Stuff+": [
        "release_speed",      # core
        "effective_speed",    # core
        "release_spin_rate",  # core
        "spin_axis",          # core; also the sole input to axis_differential
        "pfx_x",              # -> hb, and hb_sep_vs_primary_fb
        "pfx_z",              # -> ivb, and ivb_sep_vs_primary_fb
        "vx0", "vy0", "vz0", "ax", "ay", "az",  # -> vaa_residual
        "release_pos_y",      # -> vaa_residual
    ],
    "Deception+": [
        "arm_angle",          # core; also arm_angle_std
        "release_extension",  # core; also release_extension_std
        "release_pos_x",      # -> release_pos_x_as
        "release_pos_z",      # core; also release_consistency
    ],
    "Location+": [
        "plate_x", "plate_z", "balls", "strikes", "stand", "pitch_type",
        "sz_top", "sz_bot", "zone",
    ],
    "Targets / outcomes": [
        "delta_run_exp", "estimated_woba_using_speedangle", "woba_value",
        "launch_speed", "launch_angle", "description", "events",
    ],
    "Bat tracking": [
        "bat_speed", "swing_length", "attack_angle", "attack_direction",
        "swing_path_tilt",
    ],
    "Other": [
        "delta_home_win_exp", "hc_x", "hc_y", "p_throws", "inning_topbot",
    ],
}

ALL_FIELDS = [f for fields in FIELD_GROUPS.values() for f in fields]
FIELD_TO_GROUP = {f: g for g, fields in FIELD_GROUPS.items() for f in fields}

# A modern, fully-tracked season to measure every other season against, so
# "degraded" means degraded relative to the real baseline rather than to an
# arbitrary round number.
BASELINE_SEASON = 2025

# These are only ever populated on the subset of pitches that produced the
# relevant event (a swing, a batted ball, a completed PA), so their rate is a
# share of ALL pitches and is expected to sit far below 100% even when
# tracking is perfect. Only their movement ACROSS seasons is meaningful -
# an absolute 17% for hc_x is normal, not a gap.
EVENT_CONDITIONAL = frozenset({
    "launch_speed", "launch_angle", "events", "woba_value",
    "estimated_woba_using_speedangle", "hc_x", "hc_y",
    "bat_speed", "swing_length", "attack_angle", "attack_direction",
    "swing_path_tilt",
})

ABSENT_MAX_PCT = 0.1      # <=0.1% populated -> structurally absent
SYSTEMIC_MAX_PCT = 5.0    # <=5% populated  -> systemic whole-season gap
DEGRADED_DELTA = 10.0     # >10pp below baseline -> degraded


def classify(pct, baseline_pct):
    if pct <= ABSENT_MAX_PCT:
        return "ABSENT"
    if pct <= SYSTEMIC_MAX_PCT:
        return "SYSTEMIC"
    if baseline_pct is not None and baseline_pct - pct > DEGRADED_DELTA:
        return "DEGRADED"
    return "POPULATED"


def audit(conn, seasons=None):
    available = [r[0] for r in conn.execute(
        "SELECT DISTINCT season FROM pitches ORDER BY season")]
    seasons = [s for s in (seasons or available) if s in available]

    cols = {c[1] for c in conn.execute("PRAGMA table_info(pitches)")}
    fields = [f for f in ALL_FIELDS if f in cols]
    missing_from_schema = [f for f in ALL_FIELDS if f not in cols]

    # One pass per season computing every field's non-null count at once -
    # a per-field query would be one full scan of a multi-million row table
    # per field per season.
    counts = {}
    for season in seasons:
        select = ", ".join(f'COUNT("{f}")' for f in fields)
        row = conn.execute(
            f"SELECT COUNT(*), {select} FROM pitches WHERE season = ?", (season,)
        ).fetchone()
        total = row[0]
        counts[season] = {
            "total": total,
            "pct": {f: (100.0 * row[i + 1] / total if total else 0.0)
                    for i, f in enumerate(fields)},
        }

    baseline = counts.get(BASELINE_SEASON, {}).get("pct", {})

    records = []
    for season in seasons:
        for f in fields:
            pct = counts[season]["pct"][f]
            records.append({
                "season": season,
                "model": FIELD_TO_GROUP[f],
                "field": f,
                "pct_populated": round(pct, 2),
                "rows": counts[season]["total"],
                "status": classify(pct, baseline.get(f)),
            })
    return pd.DataFrame(records), missing_from_schema


def print_report(df, missing_from_schema):
    seasons = sorted(df["season"].unique())

    print("\n" + "=" * 78)
    print("PER-SEASON, PER-FIELD POPULATION RATE (% non-null)")
    print(f"baseline season for 'DEGRADED' comparison: {BASELINE_SEASON}")
    print("=" * 78)

    for model in FIELD_GROUPS:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        print(f"\n--- {model} ---")
        pivot = sub.pivot(index="field", columns="season", values="pct_populated")
        pivot = pivot.reindex([f for f in FIELD_GROUPS[model] if f in pivot.index])
        header = "field".ljust(34) + "".join(f"{s:>8}" for s in seasons)
        print(header)
        print("-" * len(header))
        for field, row in pivot.iterrows():
            cells = ""
            for s in seasons:
                v = row.get(s)
                cells += "     n/a" if pd.isna(v) else f"{v:>8.1f}"
            mark = " *" if field in EVENT_CONDITIONAL else ""
            print(field.ljust(34) + cells + mark)
        if any(f in EVENT_CONDITIONAL for f in pivot.index):
            print("  * event-conditional: only defined on swings/batted balls/"
                  "completed PAs, so a low rate is expected. Compare across "
                  "seasons, not against 100%.")

    print("\n" + "=" * 78)
    print("FLAGGED FIELDS (anything not simply POPULATED)")
    print("=" * 78)
    flagged = df[df["status"] != "POPULATED"]
    if flagged.empty:
        print("None - every audited field is normally populated in every season.")
    else:
        for status in ("ABSENT", "SYSTEMIC", "DEGRADED"):
            s = flagged[flagged["status"] == status]
            if s.empty:
                continue
            print(f"\n{status}:")
            for field, grp in s.groupby("field"):
                yrs = ", ".join(f"{int(r.season)} ({r.pct_populated:.1f}%)"
                                for r in grp.sort_values("season").itertuples())
                print(f"  {field:<32} {yrs}")

    if missing_from_schema:
        print(f"\nNOTE: audited field(s) absent from the pitches schema entirely: "
              f"{missing_from_schema}")

    print("\n" + "=" * 78)
    print("This report makes no training decision and changed nothing.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int)
    ap.add_argument("--csv", type=str, help="also write the full table to this CSV")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH, timeout=60)
    df, missing = audit(conn, args.seasons)
    print_report(df, missing)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nFull table written to {args.csv}")
    conn.close()


if __name__ == "__main__":
    main()

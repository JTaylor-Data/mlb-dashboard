"""
/api/tunneling/{pitcher_id}/trajectories - per-pitch-type 3D flight curves
for the Tunneling+ 3D viewer (tunnel3d.html). This is the VISUAL layer only
- every distance/score a user sees (release/tunnel/plate distances, the
three "+"-scaled components, tunneling_plus itself) still comes straight
from analytics/tunneling_model.py's validated pipeline via the existing
/api/tunneling/pitcher/{id} endpoint, unmodified. This endpoint exists only
to turn that same physics into a smooth curve instead of three isolated
points, using the exact same solve (position_at_y, Y0=50, DECISION_POINT_Y)
so the curve's decision-point marker and plate endpoint land exactly where
the validated numbers say they do - not a second, competing calculation.

Split by batter handedness (stand) for the visual, since real release point/
movement/location all genuinely differ by opponent side and that's real,
useful signal to look at - but note the pairwise "+" scores themselves are
NOT split by hand (analytics/tunneling_model.py scores a pitcher's whole
arsenal together), so don't expect the two to derive from identical subsets
of pitches. The curve's own decision_similarity/plate_separation numbers
here are informal (centroid-based, this pitcher/split only) - shown as
reference only, never as a substitute for the real tunneling_plus number,
which is always pulled from the validated endpoint.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query


from api.cache import ttl_cache
from api.db import get_conn
from analytics.tunneling_model import DECISION_POINT_Y, Y0, position_at_y

router = APIRouter(prefix="/api/tunneling")

PITCH_COLORS = {
    "FF": "#d62728", "CU": "#1f77b4", "FC": "#8c564b", "SI": "#ff7f0e",
    "CH": "#2ca02c", "FS": "#40e0d0", "ST": "#b35900", "SL": "#e6c200",
}
DEFAULT_COLOR = "#999999"
N_SAMPLES = 40
MIN_PITCHES_PER_TYPE_VISUAL = 10  # looser than the model's own 100 - this is a display filter, not a scoring one


@ttl_cache()
def _load_pitcher_pitches(conn, pitcher_id, season):
    df = pd.read_sql("""
        SELECT pitch_type, stand, release_pos_x, release_pos_y, release_pos_z,
               vx0, vy0, vz0, ax, ay, az, plate_x, plate_z
        FROM pitches
        WHERE pitcher_id = ? AND season = ? AND pitch_type IS NOT NULL AND stand IN ('L', 'R')
          AND release_pos_x IS NOT NULL AND release_pos_y IS NOT NULL AND release_pos_z IS NOT NULL
          AND vx0 IS NOT NULL AND vy0 IS NOT NULL AND vz0 IS NOT NULL
          AND ax IS NOT NULL AND ay IS NOT NULL AND az IS NOT NULL
          AND plate_x IS NOT NULL AND plate_z IS NOT NULL
    """, conn, params=(pitcher_id, season))
    df["pitch_type"] = df["pitch_type"].replace({"KC": "CU"})
    return df


def _build_split(df, stand):
    sub = df[df["stand"] == stand]
    total = len(sub)
    pitches = []
    if total == 0:
        return {"pitches": []}

    for pt, group in sub.groupby("pitch_type"):
        n = len(group)
        if n < MIN_PITCHES_PER_TYPE_VISUAL:
            continue
        rx, ry, rz = group["release_pos_x"].mean(), group["release_pos_y"].mean(), group["release_pos_z"].mean()
        vx0, vy0, vz0 = group["vx0"].mean(), group["vy0"].mean(), group["vz0"].mean()
        ax, ay, az = group["ax"].mean(), group["ay"].mean(), group["az"].mean()

        # Sample the curve using the SAME solve the validated model uses
        # (position_at_y, Y0=50) - t=0 at Y0 itself reproduces (rx, rz)
        # exactly, matching the model's own release_pos_x/z-at-Y0
        # simplification rather than introducing a second one.
        ys = np.linspace(Y0, 0.0, N_SAMPLES + 1)
        xs, zs = position_at_y(np.full(N_SAMPLES + 1, rx), np.full(N_SAMPLES + 1, rz),
                                np.full(N_SAMPLES + 1, vx0), np.full(N_SAMPLES + 1, vy0), np.full(N_SAMPLES + 1, vz0),
                                np.full(N_SAMPLES + 1, ax), np.full(N_SAMPLES + 1, ay), np.full(N_SAMPLES + 1, az), ys)
        points = [{"x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4)}
                  for x, y, z in zip(xs, ys, zs)]

        dx, dz = position_at_y(rx, rz, vx0, vy0, vz0, ax, ay, az, DECISION_POINT_Y)
        decision_point = {"x": round(float(dx), 4), "y": DECISION_POINT_Y, "z": round(float(dz), 4)}

        velo_mph = float((vx0 ** 2 + vy0 ** 2 + vz0 ** 2) ** 0.5) / 1.4667
        pitches.append({
            "pitch_type": pt,
            "color": PITCH_COLORS.get(pt, DEFAULT_COLOR),
            "n": n,
            "usage": round(n / total, 4),
            "velo_mph": round(velo_mph, 1),
            "release": {"x": round(float(rx), 3), "y": round(float(ry), 3), "z": round(float(rz), 3)},
            "points": points,
            "decision_point": decision_point,
            "plate_real": {"x": round(float(group["plate_x"].mean()), 4), "z": round(float(group["plate_z"].mean()), 4)},
        })
    return {"pitches": pitches}


@router.get("/{pitcher_id}/trajectories")
def get_trajectories(pitcher_id: int, season: int = Query(...), conn=Depends(get_conn)):
    name_row = conn.execute("SELECT name FROM players WHERE player_id=?", (pitcher_id,)).fetchone()
    name = name_row[0] if name_row else str(pitcher_id)
    df = _load_pitcher_pitches(conn, pitcher_id, season)
    return {
        "pitcher_id": pitcher_id,
        "name": name,
        "season": season,
        "decision_point_y": DECISION_POINT_Y,
        "splits": {stand: _build_split(df, stand) for stand in ("R", "L")},
    }
